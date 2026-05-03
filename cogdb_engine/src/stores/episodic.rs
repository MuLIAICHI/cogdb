use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use rusqlite::params;
use uuid::Uuid;

use crate::error::{CogError, Result};
use crate::storage::sql::{create_episodic_schema, dt_to_ms, ms_to_dt, open_connection};
use crate::types::{DecayScanRow, MemoryScope, MemoryUnit};
use crate::vector::filter::{
    candidate_ids, choose_strategy, EpisodicFilter, SearchStrategy, BRUTE_FORCE_THRESHOLD,
};
use crate::vector::hnsw::{uuid_to_label, HnswIndex};
use crate::wal::record::WalRecord;
use crate::wal::writer::WalWriter;

/// SQLite-backed episodic memory store with HNSW vector search and WAL.
///
/// Thread-safe: all public methods acquire internal locks as needed.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use std::sync::Arc;
/// use cogdb_engine::stores::episodic::EpisodicStore;
/// use cogdb_engine::wal::writer::WalWriter;
/// let wal = Arc::new(WalWriter::open(Path::new("/tmp/test.wal"), 0, 64 * 1024 * 1024).unwrap());
/// let store = EpisodicStore::open(Path::new("/tmp/ep"), 384, 16, 200, wal).unwrap();
/// ```
pub struct EpisodicStore {
    hnsw: HnswIndex,
    /// label (lower 64 bits of UUID) → full UUID, for resolving HNSW hits
    label_map: RwLock<HashMap<usize, Uuid>>,
    conn: Mutex<rusqlite::Connection>,
    wal: Arc<WalWriter>,
}

impl EpisodicStore {
    /// Open or create an EpisodicStore at `db_path`.
    ///
    /// Creates `{db_path}/episodic_meta.db` and initialises the schema.
    /// HNSW starts empty; call `load_snapshot` + `apply_wal_upsert` for recovery.
    pub fn open(
        db_path: &Path,
        dim: usize,
        m: usize,
        ef_construction: usize,
        wal: Arc<WalWriter>,
    ) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let conn = open_connection(&db_path.join("episodic_meta.db"))?;
        create_episodic_schema(&conn)?;
        Ok(Self {
            hnsw: HnswIndex::new(dim, m, ef_construction),
            label_map: RwLock::new(HashMap::new()),
            conn: Mutex::new(conn),
            wal,
        })
    }

    /// Release the SQLite file handle by swapping to an in-memory connection.
    ///
    /// Called by Engine::close() to unlock files before directory cleanup.
    /// Any operations after this will hit an empty in-memory database.
    pub fn close_connections(&self) {
        if let Ok(mem) = rusqlite::Connection::open_in_memory() {
            *self.conn.lock() = mem;
        }
    }

    /// Persist the HNSW snapshot to `snap_dir` (for WAL checkpoint).
    pub fn save_snapshot(&self, snap_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(snap_dir)?;
        self.hnsw.save(snap_dir)
    }

    /// Restore HNSW state from a snapshot directory and rebuild the label map.
    ///
    /// Called by Engine during WAL recovery after finding a Checkpoint record.
    /// The HNSW must be empty (freshly opened store) when this is called.
    pub fn restore_from_snapshot(&self, snap_dir: &Path) -> Result<()> {
        let loaded = HnswIndex::load(snap_dir, self.hnsw.dim, 16, 200)?;
        for (label, vec) in loaded.vector_pairs() {
            // Re-insert into our HNSW using the label_to_uuid inverse
            // (lossy UUID used only as a key for the HNSW label slot)
            let id = crate::vector::hnsw::label_to_uuid(label);
            self.hnsw.insert(id, &vec)?;
        }
        self.rebuild_label_map_from_db()
    }

    /// Rebuild the label_map by scanning all IDs from SQLite.
    ///
    /// Called after snapshot restore or cold-start WAL replay.
    pub fn rebuild_label_map_from_db(&self) -> Result<()> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id FROM episodic_meta")
            .map_err(CogError::Sqlite)?;
        let ids: Vec<Uuid> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(CogError::Sqlite)?
            .filter_map(|r| r.ok())
            .filter_map(|s| s.parse::<Uuid>().ok())
            .collect();

        let mut label_map = self.label_map.write();
        label_map.clear();
        for id in ids {
            label_map.insert(uuid_to_label(&id), id);
        }
        Ok(())
    }

    /// Number of vectors in the HNSW index (for testing and monitoring).
    pub fn hnsw_len(&self) -> usize {
        self.hnsw.len()
    }

    /// Replay a single WAL upsert record (called by Engine during recovery).
    pub fn apply_wal_upsert(&self, id_bytes: [u8; 16], embedding: Vec<f32>) -> Result<()> {
        let id = WalRecord::id_bytes_to_uuid(&id_bytes);
        let label = uuid_to_label(&id);
        self.hnsw.insert(id, &embedding)?;
        self.label_map.write().insert(label, id);
        Ok(())
    }

    /// Replay a WAL delete record (called by Engine during recovery).
    pub fn apply_wal_delete(&self, id_bytes: [u8; 16]) {
        let id = WalRecord::id_bytes_to_uuid(&id_bytes);
        let label = uuid_to_label(&id);
        self.label_map.write().remove(&label);
        // The HNSW does not support removal; the deleted entry is filtered out
        // naturally because it won't appear in SQLite lookups.
    }

    // ── Public store API ──────────────────────────────────────────────────────

    /// Store a memory unit. The embedding must be pre-computed and set on `unit`.
    ///
    /// Returns the UUID string of the stored record.
    pub fn add(&self, unit: &MemoryUnit) -> Result<String> {
        let id_str = unit.id.to_string();
        let metadata_json = serde_json::to_string(&unit.metadata)?;

        // Insert embedding into HNSW + record in WAL
        if let Some(embedding) = &unit.embedding {
            self.wal.append(WalRecord::EpisodicUpsert {
                seq: 0,
                id: WalRecord::uuid_to_id_bytes(&unit.id),
                embedding: embedding.clone(),
                metadata_json: metadata_json.clone(),
            })?;
            self.hnsw.insert(unit.id, embedding)?;
            self.label_map.write().insert(uuid_to_label(&unit.id), unit.id);
        }

        // Insert metadata into SQLite
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO episodic_meta
             (id, agent_id, scope, importance, content, memory_type, metadata_json,
              created_at, accessed_at, access_count, decay_score, team_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            params![
                id_str,
                unit.agent_id,
                unit.scope.to_string(),
                unit.importance,
                unit.content,
                unit.memory_type.to_string(),
                metadata_json,
                dt_to_ms(unit.created_at),
                dt_to_ms(unit.accessed_at),
                unit.access_count,
                unit.decay_score,
                unit.team_id,
            ],
        )
        .map_err(CogError::Sqlite)?;

        Ok(id_str)
    }

    /// Vector similarity search with metadata filtering.
    ///
    /// Always applies scope/agent access control. Returns up to `top_k` results
    /// sorted by `effective_importance` (importance × decay_score) descending.
    pub fn search(
        &self,
        query_embedding: &[f32],
        agent_id: &str,
        top_k: usize,
        scope_filter: Option<&MemoryScope>,
        min_importance: f64,
        time_range_start_ms: Option<i64>,
        time_range_end_ms: Option<i64>,
    ) -> Result<Vec<MemoryUnit>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        // Always apply at least the scope/agent filter
        let filter = EpisodicFilter {
            agent_id,
            scope_filter,
            min_importance,
            time_range_start_ms,
            time_range_end_ms,
        };
        let candidate_set = {
            let conn = self.conn.lock();
            candidate_ids(&conn, &filter)?
        };

        if candidate_set.is_empty() {
            return Ok(vec![]);
        }

        let strategy = choose_strategy(Some(candidate_set.len()), BRUTE_FORCE_THRESHOLD);
        let label_map = self.label_map.read();
        let hnsw_hits = match strategy {
            SearchStrategy::BruteForce => {
                // Use candidate UUIDs directly for brute-force search
                let candidates: Vec<Uuid> = candidate_set.iter().copied().collect();
                self.hnsw.search_filtered(query_embedding, &candidates, top_k)
            }
            SearchStrategy::HnswPostFilter | SearchStrategy::HnswDirect => {
                let over_fetch = (top_k * 10).max(50);
                let all_hits = self.hnsw.search(query_embedding, over_fetch);
                all_hits
                    .into_iter()
                    .filter(|h| {
                        let label = uuid_to_label(&h.id);
                        label_map
                            .get(&label)
                            .map(|full_id| candidate_set.contains(full_id))
                            .unwrap_or(false)
                    })
                    .take(top_k)
                    .collect()
            }
        };

        // Resolve labels → full UUIDs → load from SQLite
        let conn = self.conn.lock();
        let mut results = Vec::with_capacity(hnsw_hits.len());
        for hit in &hnsw_hits {
            let label = uuid_to_label(&hit.id);
            if let Some(&real_id) = label_map.get(&label) {
                if let Some(unit) = self.fetch_by_id_locked(&conn, real_id)? {
                    results.push(unit);
                }
            }
        }
        drop(label_map);

        results.sort_by(|a, b| {
            b.effective_importance()
                .partial_cmp(&a.effective_importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        Ok(results)
    }

    /// Fetch a single memory unit by UUID.
    pub fn get(&self, memory_id: Uuid) -> Result<Option<MemoryUnit>> {
        let conn = self.conn.lock();
        self.fetch_by_id_locked(&conn, memory_id)
    }

    /// Delete a memory unit by UUID.
    ///
    /// Returns `true` if a record was deleted, `false` if not found.
    pub fn delete(&self, memory_id: Uuid) -> Result<bool> {
        self.wal.append(WalRecord::EpisodicDelete {
            seq: 0,
            id: WalRecord::uuid_to_id_bytes(&memory_id),
        })?;

        // Remove from label map (HNSW entry stays but won't appear in SQLite lookups)
        self.label_map.write().remove(&uuid_to_label(&memory_id));

        let conn = self.conn.lock();
        let deleted = conn
            .execute(
                "DELETE FROM episodic_meta WHERE id = ?1",
                params![memory_id.to_string()],
            )
            .map_err(CogError::Sqlite)?;

        Ok(deleted > 0)
    }

    /// Update fields on a stored memory unit.
    ///
    /// `updates` is a JSON object whose keys map to column names or metadata keys.
    /// Recognised first-class columns: `accessed_at` (ms), `access_count`,
    /// `decay_score`, `importance`, `scope`, `team_id`.
    /// All other keys are merged into `metadata_json`.
    pub fn update_metadata(&self, memory_id: Uuid, updates: &serde_json::Value) -> Result<bool> {
        let conn = self.conn.lock();
        let id_str = memory_id.to_string();

        // Load current row
        let Some(mut unit) = self.fetch_by_id_locked(&conn, memory_id)? else {
            return Ok(false);
        };

        let obj = updates.as_object().ok_or_else(|| {
            CogError::InvalidArg("updates must be a JSON object".into())
        })?;

        let mut extra_meta = unit.metadata.as_object().cloned().unwrap_or_default();

        for (key, val) in obj {
            match key.as_str() {
                "accessed_at" => {
                    if let Some(ms) = val.as_i64() {
                        unit.accessed_at = ms_to_dt(ms);
                    }
                }
                "access_count" => {
                    if let Some(n) = val.as_i64() {
                        unit.access_count = n;
                    }
                }
                "decay_score" => {
                    if let Some(f) = val.as_f64() {
                        unit.decay_score = f;
                    }
                }
                "importance" => {
                    if let Some(f) = val.as_f64() {
                        unit.importance = f;
                    }
                }
                "scope" => {
                    if let Some(s) = val.as_str() {
                        if let Ok(scope) = s.parse() {
                            unit.scope = scope;
                        }
                    }
                }
                "team_id" => {
                    unit.team_id = val.as_str().map(str::to_string);
                }
                other => {
                    extra_meta.insert(other.to_string(), val.clone());
                }
            }
        }
        unit.metadata = serde_json::Value::Object(extra_meta);

        let updated = conn
            .execute(
                "UPDATE episodic_meta SET
                     importance    = ?2,
                     decay_score   = ?3,
                     accessed_at   = ?4,
                     access_count  = ?5,
                     scope         = ?6,
                     team_id       = ?7,
                     metadata_json = ?8
                 WHERE id = ?1",
                params![
                    id_str,
                    unit.importance,
                    unit.decay_score,
                    dt_to_ms(unit.accessed_at),
                    unit.access_count,
                    unit.scope.to_string(),
                    unit.team_id,
                    serde_json::to_string(&unit.metadata)?,
                ],
            )
            .map_err(CogError::Sqlite)?;

        Ok(updated > 0)
    }

    /// Count stored memory units, optionally filtered by agent.
    pub fn count(&self, agent_id: Option<&str>) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = match agent_id {
            None => conn
                .query_row("SELECT COUNT(*) FROM episodic_meta", [], |r| r.get(0))
                .map_err(CogError::Sqlite)?,
            Some(aid) => conn
                .query_row(
                    "SELECT COUNT(*) FROM episodic_meta WHERE agent_id = ?1",
                    params![aid],
                    |r| r.get(0),
                )
                .map_err(CogError::Sqlite)?,
        };
        Ok(n)
    }

    /// Paginated scan for decay processing — returns lightweight rows without embeddings.
    pub fn scan_batch(
        &self,
        agent_id: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<DecayScanRow>> {
        let conn = self.conn.lock();
        let sql = match agent_id {
            None => {
                "SELECT id, accessed_at, decay_score FROM episodic_meta
                  ORDER BY accessed_at ASC LIMIT ?1 OFFSET ?2"
                    .to_string()
            }
            Some(aid) => format!(
                "SELECT id, accessed_at, decay_score FROM episodic_meta
                  WHERE agent_id = '{aid}'
                  ORDER BY accessed_at ASC LIMIT ?1 OFFSET ?2"
            ),
        };

        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                let id_str: String = row.get(0)?;
                let accessed_ms: i64 = row.get(1)?;
                let decay: f64 = row.get(2)?;
                Ok((id_str, accessed_ms, decay))
            })
            .map_err(CogError::Sqlite)?;

        let mut result = Vec::new();
        for row in rows {
            let (id_str, accessed_ms, decay) = row.map_err(CogError::Sqlite)?;
            let id = id_str
                .parse::<Uuid>()
                .map_err(|e| CogError::Store(e.to_string()))?;
            result.push(DecayScanRow {
                id,
                accessed_at: ms_to_dt(accessed_ms),
                decay_score: decay,
            });
        }
        Ok(result)
    }

    /// Batch update decay scores (used by DecayEngine).
    pub fn bulk_update_decay(&self, updates: &[(Uuid, f64)]) -> Result<()> {
        let conn = self.conn.lock();
        let tx = conn.unchecked_transaction().map_err(CogError::Sqlite)?;
        for (id, decay) in updates {
            tx.execute(
                "UPDATE episodic_meta SET decay_score = ?2 WHERE id = ?1",
                params![id.to_string(), decay],
            )
            .map_err(CogError::Sqlite)?;
        }
        tx.commit().map_err(CogError::Sqlite)?;
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn fetch_by_id_locked(
        &self,
        conn: &rusqlite::Connection,
        id: Uuid,
    ) -> Result<Option<MemoryUnit>> {
        let id_str = id.to_string();
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, agent_id, scope, importance, content, memory_type,
                        metadata_json, created_at, accessed_at, access_count,
                        decay_score, team_id
                   FROM episodic_meta WHERE id = ?1",
            )
            .map_err(CogError::Sqlite)?;

        let mut rows = stmt.query(params![id_str]).map_err(CogError::Sqlite)?;
        match rows.next().map_err(CogError::Sqlite)? {
            Some(row) => Ok(Some(self.row_to_unit(row)?)),
            None => Ok(None),
        }
    }

    fn row_to_unit(&self, row: &rusqlite::Row<'_>) -> Result<MemoryUnit> {
        let id_str: String = row.get(0).map_err(CogError::Sqlite)?;
        let scope_str: String = row.get(2).map_err(CogError::Sqlite)?;
        let mt_str: String = row.get(5).map_err(CogError::Sqlite)?;
        let meta_str: String = row.get(6).map_err(CogError::Sqlite)?;

        Ok(MemoryUnit {
            id: id_str.parse().map_err(|e: uuid::Error| CogError::Store(e.to_string()))?,
            agent_id: row.get(1).map_err(CogError::Sqlite)?,
            scope: scope_str.parse().map_err(CogError::Store)?,
            importance: row.get(3).map_err(CogError::Sqlite)?,
            content: row.get(4).map_err(CogError::Sqlite)?,
            memory_type: mt_str.parse().map_err(CogError::Store)?,
            metadata: serde_json::from_str(&meta_str)?,
            embedding: None,
            created_at: ms_to_dt(row.get(7).map_err(CogError::Sqlite)?),
            accessed_at: ms_to_dt(row.get(8).map_err(CogError::Sqlite)?),
            access_count: row.get(9).map_err(CogError::Sqlite)?,
            decay_score: row.get(10).map_err(CogError::Sqlite)?,
            team_id: row.get(11).map_err(CogError::Sqlite)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryType;
    use chrono::Utc;
    use tempfile::tempdir;

    fn make_wal(dir: &Path) -> Arc<WalWriter> {
        Arc::new(WalWriter::open(&dir.join("test.wal"), 0, 64 * 1024 * 1024).unwrap())
    }

    fn make_unit(agent_id: &str, scope: MemoryScope, importance: f64, dim: usize) -> MemoryUnit {
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding: Vec<f32> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };
        MemoryUnit {
            id: Uuid::new_v4(),
            content: format!("Memory for {agent_id}"),
            memory_type: MemoryType::Episodic,
            agent_id: agent_id.to_string(),
            scope,
            importance,
            embedding: Some(embedding),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 0,
            decay_score: 1.0,
            team_id: None,
        }
    }

    #[test]
    fn add_and_count() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let u = make_unit("a1", MemoryScope::Private, 0.8, 4);
        store.add(&u).unwrap();
        assert_eq!(store.count(None).unwrap(), 1);
        assert_eq!(store.count(Some("a1")).unwrap(), 1);
        assert_eq!(store.count(Some("a2")).unwrap(), 0);
    }

    #[test]
    fn get_returns_stored_unit() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let u = make_unit("a1", MemoryScope::Private, 0.8, 4);
        let id = u.id;
        store.add(&u).unwrap();

        let fetched = store.get(id).unwrap().expect("should exist");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.agent_id, "a1");
        assert!((fetched.importance - 0.8).abs() < 1e-10);
    }

    #[test]
    fn get_missing_returns_none() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        assert!(store.get(Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn delete_existing_returns_true() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let u = make_unit("a1", MemoryScope::Private, 0.5, 4);
        let id = u.id;
        store.add(&u).unwrap();
        assert!(store.delete(id).unwrap());
        assert_eq!(store.count(None).unwrap(), 0);
        assert!(store.get(id).unwrap().is_none());
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        assert!(!store.delete(Uuid::new_v4()).unwrap());
    }

    #[test]
    fn search_respects_agent_scope() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();

        let u1 = make_unit("agent-1", MemoryScope::Private, 0.8, 4);
        let u2 = make_unit("agent-2", MemoryScope::Private, 0.8, 4);
        let u3 = make_unit("agent-3", MemoryScope::Org, 0.8, 4);
        store.add(&u1).unwrap();
        store.add(&u2).unwrap();
        store.add(&u3).unwrap();

        let query = vec![0.5f32, 0.5, 0.5, 0.5];
        let results = store.search(&query, "agent-1", 10, None, 0.0, None, None).unwrap();

        let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&u1.id), "own private memory must be returned");
        assert!(!ids.contains(&u2.id), "other agent private must NOT be returned");
        assert!(ids.contains(&u3.id), "org-scoped memory must be returned");
    }

    #[test]
    fn search_filters_by_importance() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();

        let mut low = make_unit("a1", MemoryScope::Private, 0.2, 4);
        let mut high = make_unit("a1", MemoryScope::Private, 0.9, 4);
        // Give them different embeddings so search can distinguish
        low.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
        high.embedding = Some(vec![0.0, 1.0, 0.0, 0.0]);
        store.add(&low).unwrap();
        store.add(&high).unwrap();

        let query = vec![0.5, 0.5, 0.0, 0.0];
        let results = store.search(&query, "a1", 10, None, 0.5, None, None).unwrap();
        let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        assert!(!ids.contains(&low.id), "low importance should be filtered out");
        assert!(ids.contains(&high.id), "high importance should be returned");
    }

    #[test]
    fn update_metadata_modifies_fields() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let u = make_unit("a1", MemoryScope::Private, 0.5, 4);
        let id = u.id;
        store.add(&u).unwrap();

        let updates = serde_json::json!({ "decay_score": 0.3, "access_count": 5 });
        assert!(store.update_metadata(id, &updates).unwrap());

        let fetched = store.get(id).unwrap().unwrap();
        assert!((fetched.decay_score - 0.3).abs() < 1e-10);
        assert_eq!(fetched.access_count, 5);
    }

    #[test]
    fn update_metadata_nonexistent_returns_false() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let updates = serde_json::json!({ "decay_score": 0.1 });
        assert!(!store.update_metadata(Uuid::new_v4(), &updates).unwrap());
    }

    #[test]
    fn scan_batch_pagination() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        for _ in 0..5 {
            store.add(&make_unit("a1", MemoryScope::Private, 0.5, 4)).unwrap();
        }
        let page0 = store.scan_batch(Some("a1"), 3, 0).unwrap();
        let page1 = store.scan_batch(Some("a1"), 3, 3).unwrap();
        assert_eq!(page0.len(), 3);
        assert_eq!(page1.len(), 2);
    }

    #[test]
    fn bulk_update_decay() {
        let dir = tempdir().unwrap();
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, make_wal(dir.path())).unwrap();
        let u1 = make_unit("a1", MemoryScope::Private, 0.5, 4);
        let u2 = make_unit("a1", MemoryScope::Private, 0.5, 4);
        let id1 = u1.id;
        let id2 = u2.id;
        store.add(&u1).unwrap();
        store.add(&u2).unwrap();

        store.bulk_update_decay(&[(id1, 0.4), (id2, 0.7)]).unwrap();

        assert!((store.get(id1).unwrap().unwrap().decay_score - 0.4).abs() < 1e-10);
        assert!((store.get(id2).unwrap().unwrap().decay_score - 0.7).abs() < 1e-10);
    }

    #[test]
    fn wal_records_written() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let wal = Arc::new(WalWriter::open(&wal_path, 0, 64 * 1024 * 1024).unwrap());
        let store = EpisodicStore::open(dir.path(), 4, 8, 50, wal).unwrap();
        store.add(&make_unit("a1", MemoryScope::Private, 0.5, 4)).unwrap();
        store.delete(Uuid::new_v4()).unwrap(); // no-op on missing, but still writes WAL

        // WAL file must be non-trivial (header + at least one record)
        let len = std::fs::metadata(&wal_path).unwrap().len();
        assert!(len > 64, "WAL should have records, got {len} bytes");
    }
}

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use rusqlite::params;
use uuid::Uuid;

use crate::error::{CogError, Result};
use crate::graph::kg::KnowledgeGraph;
use crate::storage::sql::{create_semantic_schema, dt_to_ms, ms_to_dt, open_connection};
use crate::types::SemanticTriple;
use crate::wal::record::WalRecord;
use crate::wal::writer::WalWriter;

/// SQLite-backed semantic triple store with in-memory petgraph for traversal and WAL.
///
/// Thread-safe: all public methods acquire internal locks as needed.
/// Contradiction detection is enabled by default and can be disabled via `contradiction_check`.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use std::sync::Arc;
/// use cogdb_engine::stores::semantic::SemanticStore;
/// use cogdb_engine::wal::writer::WalWriter;
/// let wal = Arc::new(WalWriter::open(Path::new("/tmp/s.wal"), 0, 64 * 1024 * 1024).unwrap());
/// let store = SemanticStore::open(Path::new("/tmp/sem"), true, wal).unwrap();
/// ```
pub struct SemanticStore {
    graph: RwLock<KnowledgeGraph>,
    conn: Mutex<rusqlite::Connection>,
    wal: Arc<WalWriter>,
    contradiction_check: bool,
}

impl SemanticStore {
    /// Open or create a SemanticStore at `db_path`.
    ///
    /// Creates `{db_path}/semantic.db`, initialises the schema, and rebuilds
    /// the in-memory graph from all active triples in SQLite.
    pub fn open(
        db_path: &Path,
        contradiction_check: bool,
        wal: Arc<WalWriter>,
    ) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let conn = open_connection(&db_path.join("semantic.db"))?;
        create_semantic_schema(&conn)?;

        // Rebuild in-memory graph from active triples
        let graph = Self::rebuild_graph_from_db(&conn)?;

        Ok(Self {
            graph: RwLock::new(graph),
            conn: Mutex::new(conn),
            wal,
            contradiction_check,
        })
    }

    /// Release the SQLite file handle by swapping to an in-memory connection.
    pub fn close_connections(&self) {
        if let Ok(mem) = rusqlite::Connection::open_in_memory() {
            *self.conn.lock() = mem;
        }
    }

    // ── Public store API ──────────────────────────────────────────────────────

    /// Add or update a semantic triple.
    ///
    /// When `contradiction_check` is enabled, any existing active triple with
    /// the same (subject, predicate) but a different object is superseded
    /// (its `valid_until` is set to now). This replicates the Python behaviour.
    ///
    /// Returns the UUID string of the stored triple.
    pub fn add_triple(&self, triple: &SemanticTriple) -> Result<String> {
        let id_str = triple.id.to_string();
        let now_ms = dt_to_ms(Utc::now());
        let source_json = serde_json::to_string(&triple.source_episodes)?;
        let meta_json = serde_json::to_string(&triple.metadata)?;

        let conn = self.conn.lock();

        // Contradiction detection
        if self.contradiction_check {
            self.supersede_contradictions(
                &conn,
                &triple.subject,
                &triple.predicate,
                &triple.object,
                &triple.agent_id,
                now_ms,
            )?;
        }

        // Insert the new triple
        conn.execute(
            "INSERT OR REPLACE INTO triples
             (id, subject, predicate, object, agent_id, confidence,
              valid_from, valid_until, source_episodes, metadata_json)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                id_str,
                triple.subject,
                triple.predicate,
                triple.object,
                triple.agent_id,
                triple.confidence,
                dt_to_ms(triple.valid_from),
                triple.valid_until.map(dt_to_ms),
                source_json,
                meta_json,
            ],
        )
        .map_err(CogError::Sqlite)?;

        // Update the in-memory graph (only for active triples)
        if triple.valid_until.is_none() {
            self.graph.write().add_edge(
                triple.id,
                &triple.subject,
                &triple.predicate,
                &triple.object,
                triple.confidence,
            );
        }

        // WAL record
        self.wal.append(WalRecord::TripleUpsert {
            seq: 0,
            id: WalRecord::uuid_to_id_bytes(&triple.id),
            subject: triple.subject.clone(),
            predicate: triple.predicate.clone(),
            object: triple.object.clone(),
            valid_until_ms: triple.valid_until.map(dt_to_ms),
        })?;

        Ok(id_str)
    }

    /// Query all active triples where `subject` matches, optionally scoped to an agent.
    pub fn query_subject(
        &self,
        subject: &str,
        active_only: bool,
        agent_id: Option<&str>,
    ) -> Result<Vec<SemanticTriple>> {
        let conn = self.conn.lock();
        let mut conditions = vec!["subject = ?1".to_string()];
        let mut extra_params: Vec<String> = Vec::new();

        if active_only {
            conditions.push("valid_until IS NULL".to_string());
        }
        if let Some(aid) = agent_id {
            conditions.push(format!("agent_id = ?{}", 2 + extra_params.len()));
            extra_params.push(aid.to_string());
        }

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT id, subject, predicate, object, agent_id, confidence,
                    valid_from, valid_until, source_episodes, metadata_json
               FROM triples WHERE {where_clause}"
        );

        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
        let mut all_params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(subject.to_string())];
        for p in extra_params {
            all_params.push(Box::new(p));
        }
        let param_refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|p| p.as_ref()).collect();

        self.collect_triples(&mut stmt, param_refs.as_slice())
    }

    /// BFS graph traversal from `entity` up to `depth` hops.
    ///
    /// Returns active triples reachable within the given depth, matching
    /// the Python `query_entity` behaviour (bidirectional traversal).
    pub fn query_entity(
        &self,
        entity: &str,
        depth: usize,
        active_only: bool,
    ) -> Result<Vec<SemanticTriple>> {
        let triple_ids: Vec<Uuid> = {
            let graph = self.graph.read();
            graph
                .bfs_edges(entity, depth)
                .into_iter()
                .map(|e| e.triple_id)
                .collect()
        };

        if triple_ids.is_empty() {
            return Ok(vec![]);
        }

        let conn = self.conn.lock();
        let mut results = Vec::with_capacity(triple_ids.len());
        for id in triple_ids {
            if let Some(triple) = self.fetch_by_id_locked(&conn, id, active_only)? {
                results.push(triple);
            }
        }
        Ok(results)
    }

    /// Full-text search across subject, predicate, and object using SQLite LIKE.
    pub fn search_text(&self, query: &str, active_only: bool) -> Result<Vec<SemanticTriple>> {
        let conn = self.conn.lock();
        let active_clause = if active_only { "AND valid_until IS NULL" } else { "" };
        let pattern = format!("%{query}%");
        let sql = format!(
            "SELECT id, subject, predicate, object, agent_id, confidence,
                    valid_from, valid_until, source_episodes, metadata_json
               FROM triples
              WHERE (subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1)
                {active_clause}"
        );
        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
        self.collect_triples(&mut stmt, &[&pattern])
    }

    /// All entity names currently in the active in-memory graph.
    pub fn get_entities(&self) -> Vec<String> {
        self.graph.read().get_entities()
    }

    /// Direct neighbors of an entity (both outgoing and incoming).
    pub fn get_neighbors(&self, entity: &str) -> Vec<String> {
        self.graph.read().get_neighbors(entity)
    }

    /// Delete a triple by UUID.
    ///
    /// Returns `true` if a record was deleted, `false` if not found.
    pub fn delete_triple(&self, triple_id: Uuid) -> Result<bool> {
        self.wal.append(WalRecord::TripleDelete {
            seq: 0,
            id: WalRecord::uuid_to_id_bytes(&triple_id),
        })?;

        self.graph.write().remove_edge_by_id(triple_id);

        let conn = self.conn.lock();
        let deleted = conn
            .execute(
                "DELETE FROM triples WHERE id = ?1",
                params![triple_id.to_string()],
            )
            .map_err(CogError::Sqlite)?;

        Ok(deleted > 0)
    }

    /// Count triples, optionally only active ones.
    pub fn count(&self, active_only: bool) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = if active_only {
            conn.query_row(
                "SELECT COUNT(*) FROM triples WHERE valid_until IS NULL",
                [],
                |r| r.get(0),
            )
        } else {
            conn.query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
        }
        .map_err(CogError::Sqlite)?;
        Ok(n)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Set `valid_until = now` on any active triple that contradicts
    /// the incoming (subject, predicate, ?object) where object differs.
    fn supersede_contradictions(
        &self,
        conn: &rusqlite::Connection,
        subject: &str,
        predicate: &str,
        new_object: &str,
        _agent_id: &str,
        now_ms: i64,
    ) -> Result<()> {
        // Find conflicting triples — no agent_id filter, matching Phase 0 Python behaviour
        // (any agent's triple with same subject+predicate but different object is superseded)
        let mut stmt = conn
            .prepare(
                "SELECT id FROM triples
                  WHERE subject = ?1
                    AND predicate = ?2
                    AND object != ?3
                    AND valid_until IS NULL",
            )
            .map_err(CogError::Sqlite)?;

        let conflicting_ids: Vec<String> = stmt
            .query_map(params![subject, predicate, new_object], |r| {
                r.get::<_, String>(0)
            })
            .map_err(CogError::Sqlite)?
            .filter_map(|r| r.ok())
            .collect();

        for id_str in conflicting_ids {
            conn.execute(
                "UPDATE triples SET valid_until = ?2,
                     metadata_json = json_patch(metadata_json, json_object('superseded', true))
                  WHERE id = ?1",
                params![id_str, now_ms],
            )
            .map_err(CogError::Sqlite)?;

            // Remove from in-memory graph
            if let Ok(id) = id_str.parse::<Uuid>() {
                self.graph.write().remove_edge_by_id(id);
            }
        }
        Ok(())
    }

    fn fetch_by_id_locked(
        &self,
        conn: &rusqlite::Connection,
        id: Uuid,
        active_only: bool,
    ) -> Result<Option<SemanticTriple>> {
        let active_clause = if active_only { "AND valid_until IS NULL" } else { "" };
        let sql = format!(
            "SELECT id, subject, predicate, object, agent_id, confidence,
                    valid_from, valid_until, source_episodes, metadata_json
               FROM triples WHERE id = ?1 {active_clause}"
        );
        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
        let id_str = id.to_string();
        let mut rows = stmt.query(params![id_str]).map_err(CogError::Sqlite)?;
        match rows.next().map_err(CogError::Sqlite)? {
            Some(row) => Ok(Some(Self::row_to_triple(row)?)),
            None => Ok(None),
        }
    }

    fn collect_triples(
        &self,
        stmt: &mut rusqlite::Statement<'_>,
        params: &[&dyn rusqlite::ToSql],
    ) -> Result<Vec<SemanticTriple>> {
        let rows = stmt
            .query_map(params, |row| {
                // Collect raw values; actual conversion done outside the closure
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, Option<i64>>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(CogError::Sqlite)?;

        let mut triples = Vec::new();
        for row in rows {
            let (id_s, subject, predicate, object, agent_id, confidence,
                 valid_from_ms, valid_until_ms, src_json, meta_json) =
                row.map_err(CogError::Sqlite)?;

            triples.push(SemanticTriple {
                id: id_s.parse().map_err(|e: uuid::Error| CogError::Store(e.to_string()))?,
                subject,
                predicate,
                object,
                agent_id,
                confidence,
                valid_from: ms_to_dt(valid_from_ms),
                valid_until: valid_until_ms.map(ms_to_dt),
                source_episodes: serde_json::from_str(&src_json)?,
                metadata: serde_json::from_str(&meta_json)?,
            });
        }
        Ok(triples)
    }

    fn row_to_triple(row: &rusqlite::Row<'_>) -> Result<SemanticTriple> {
        let id_s: String = row.get(0).map_err(CogError::Sqlite)?;
        let src_json: String = row.get(8).map_err(CogError::Sqlite)?;
        let meta_json: String = row.get(9).map_err(CogError::Sqlite)?;
        Ok(SemanticTriple {
            id: id_s.parse().map_err(|e: uuid::Error| CogError::Store(e.to_string()))?,
            subject: row.get(1).map_err(CogError::Sqlite)?,
            predicate: row.get(2).map_err(CogError::Sqlite)?,
            object: row.get(3).map_err(CogError::Sqlite)?,
            agent_id: row.get(4).map_err(CogError::Sqlite)?,
            confidence: row.get(5).map_err(CogError::Sqlite)?,
            valid_from: ms_to_dt(row.get(6).map_err(CogError::Sqlite)?),
            valid_until: row.get::<_, Option<i64>>(7).map_err(CogError::Sqlite)?.map(ms_to_dt),
            source_episodes: serde_json::from_str(&src_json)?,
            metadata: serde_json::from_str(&meta_json)?,
        })
    }

    fn rebuild_graph_from_db(conn: &rusqlite::Connection) -> Result<KnowledgeGraph> {
        let mut graph = KnowledgeGraph::new();
        let mut stmt = conn
            .prepare(
                "SELECT id, subject, predicate, object, confidence
                   FROM triples WHERE valid_until IS NULL",
            )
            .map_err(CogError::Sqlite)?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            })
            .map_err(CogError::Sqlite)?;

        for row in rows {
            let (id_s, subject, predicate, object, confidence) = row.map_err(CogError::Sqlite)?;
            if let Ok(id) = id_s.parse::<Uuid>() {
                graph.add_edge(id, &subject, &predicate, &object, confidence);
            }
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn make_wal(dir: &Path) -> Arc<WalWriter> {
        Arc::new(WalWriter::open(&dir.join("sem.wal"), 0, 64 * 1024 * 1024).unwrap())
    }

    fn make_triple(subject: &str, predicate: &str, object: &str, agent_id: &str) -> SemanticTriple {
        SemanticTriple {
            id: Uuid::new_v4(),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            agent_id: agent_id.to_string(),
            confidence: 0.9,
            valid_from: Utc::now(),
            valid_until: None,
            source_episodes: vec![],
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn add_and_count() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();
        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Bob", "lives_in", "NYC", "a1")).unwrap();
        assert_eq!(store.count(true).unwrap(), 2);
    }

    #[test]
    fn query_subject_returns_matching_triples() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();
        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Alice", "lives_in", "NYC", "a1")).unwrap();
        store.add_triple(&make_triple("Bob", "works_at", "Acme", "a1")).unwrap();

        let results = store.query_subject("Alice", true, None).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|t| t.subject == "Alice"));
    }

    #[test]
    fn contradiction_detection_supersedes_old_triple() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        // Same subject+predicate, different object — should supersede the first
        store.add_triple(&make_triple("Alice", "works_at", "Globex", "a1")).unwrap();

        assert_eq!(store.count(true).unwrap(), 1);
        let results = store.query_subject("Alice", true, None).unwrap();
        assert_eq!(results[0].object, "Globex");
    }

    #[test]
    fn no_contradiction_check_keeps_both() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), false, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Alice", "works_at", "Globex", "a1")).unwrap();

        assert_eq!(store.count(true).unwrap(), 2);
    }

    #[test]
    fn different_predicates_coexist() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Alice", "lives_in", "NYC", "a1")).unwrap();

        assert_eq!(store.count(true).unwrap(), 2);
    }

    #[test]
    fn query_entity_depth_1_and_2() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Acme", "located_in", "NYC", "a1")).unwrap();
        store.add_triple(&make_triple("NYC", "in_country", "USA", "a1")).unwrap();

        let d1 = store.query_entity("Alice", 1, true).unwrap();
        let d2 = store.query_entity("Alice", 2, true).unwrap();

        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 2);
    }

    #[test]
    fn get_neighbors_bidirectional() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        store.add_triple(&make_triple("Bob", "manages", "Alice", "a1")).unwrap();

        let neighbors = store.get_neighbors("Alice");
        assert!(neighbors.contains(&"Acme".to_string()));
        assert!(neighbors.contains(&"Bob".to_string()));
    }

    #[test]
    fn search_text_matches_any_field() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();

        store.add_triple(&make_triple("Alice", "works_at", "Acme Corp", "a1")).unwrap();
        store.add_triple(&make_triple("Bob", "knows", "Charlie", "a1")).unwrap();

        let results = store.search_text("Acme", true).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, "Acme Corp");
    }

    #[test]
    fn delete_triple_removes_from_db_and_graph() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();
        let t = make_triple("Alice", "works_at", "Acme", "a1");
        let id = t.id;
        store.add_triple(&t).unwrap();

        assert!(store.delete_triple(id).unwrap());
        assert_eq!(store.count(true).unwrap(), 0);
        assert!(!store.get_entities().contains(&"Alice".to_string()));
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();
        assert!(!store.delete_triple(Uuid::new_v4()).unwrap());
    }

    #[test]
    fn get_entities_lists_all_active() {
        let dir = tempdir().unwrap();
        let store = SemanticStore::open(dir.path(), true, make_wal(dir.path())).unwrap();
        store.add_triple(&make_triple("Alice", "knows", "Bob", "a1")).unwrap();
        let entities = store.get_entities();
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"Bob".to_string()));
    }

    #[test]
    fn graph_rebuilt_on_reopen() {
        let dir = tempdir().unwrap();
        {
            let wal = make_wal(dir.path());
            let store = SemanticStore::open(dir.path(), true, wal).unwrap();
            store.add_triple(&make_triple("Alice", "works_at", "Acme", "a1")).unwrap();
        }
        // Reopen — graph must be rebuilt from SQLite
        let wal2 = Arc::new(
            WalWriter::open(&dir.path().join("sem2.wal"), 0, 64 * 1024 * 1024).unwrap(),
        );
        let store2 = SemanticStore::open(dir.path(), true, wal2).unwrap();
        assert!(store2.get_entities().contains(&"Alice".to_string()));
    }
}

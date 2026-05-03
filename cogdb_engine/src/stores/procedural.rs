use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::Mutex;
use rusqlite::params;
use uuid::Uuid;

use crate::error::{CogError, Result};
use crate::storage::sql::{create_procedural_schema, dt_to_ms, ms_to_dt, open_connection};
use crate::types::{ProcedureStep, ProcedureTemplate};
use crate::wal::record::WalRecord;
use crate::wal::writer::WalWriter;

/// SQLite-backed procedure template store with WAL.
///
/// Thread-safe: all public methods acquire the internal Mutex.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use std::sync::Arc;
/// use cogdb_engine::stores::procedural::ProceduralStore;
/// use cogdb_engine::wal::writer::WalWriter;
/// let wal = Arc::new(WalWriter::open(Path::new("/tmp/p.wal"), 0, 64 * 1024 * 1024).unwrap());
/// let store = ProceduralStore::open(Path::new("/tmp/proc"), wal).unwrap();
/// ```
pub struct ProceduralStore {
    conn: Mutex<rusqlite::Connection>,
    wal: Arc<WalWriter>,
}

impl ProceduralStore {
    /// Open or create a ProceduralStore at `db_path`.
    pub fn open(db_path: &Path, wal: Arc<WalWriter>) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;
        let conn = open_connection(&db_path.join("procedures.db"))?;
        create_procedural_schema(&conn)?;
        Ok(Self { conn: Mutex::new(conn), wal })
    }

    /// Release the SQLite file handle by swapping to an in-memory connection.
    pub fn close_connections(&self) {
        if let Ok(mem) = rusqlite::Connection::open_in_memory() {
            *self.conn.lock() = mem;
        }
    }

    // ── Public store API ──────────────────────────────────────────────────────

    /// Store a procedure template.
    ///
    /// Returns the UUID string of the stored record.
    pub fn add(&self, proc: &ProcedureTemplate) -> Result<String> {
        let id_str = proc.id.to_string();
        let steps_json = serde_json::to_string(&proc.steps)?;
        let src_json = serde_json::to_string(&proc.source_episodes)?;
        let ctx_json = serde_json::to_string(&proc.applicable_contexts)?;

        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO procedures
             (id, name, description, steps_json, agent_id, success_rate,
              execution_count, source_episodes, applicable_contexts, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                id_str,
                proc.name,
                proc.description,
                steps_json,
                proc.agent_id,
                proc.success_rate,
                proc.execution_count,
                src_json,
                ctx_json,
                dt_to_ms(proc.created_at),
                dt_to_ms(proc.updated_at),
            ],
        )
        .map_err(CogError::Sqlite)?;

        self.wal.append(WalRecord::ProceduralUpsert {
            seq: 0,
            id: WalRecord::uuid_to_id_bytes(&proc.id),
        })?;

        Ok(id_str)
    }

    /// Fetch a procedure template by UUID.
    pub fn get(&self, procedure_id: Uuid) -> Result<Option<ProcedureTemplate>> {
        let conn = self.conn.lock();
        self.fetch_by_id_locked(&conn, procedure_id)
    }

    /// Search by applicable context, description, or name using LIKE.
    ///
    /// Results are sorted by `success_rate DESC, execution_count DESC`.
    pub fn search_by_context(
        &self,
        context: &str,
        agent_id: Option<&str>,
        min_success_rate: f64,
    ) -> Result<Vec<ProcedureTemplate>> {
        let conn = self.conn.lock();
        let pattern = format!("%{context}%");

        let agent_clause = if agent_id.is_some() { "AND agent_id = ?3" } else { "" };
        let sql = format!(
            "SELECT id, name, description, steps_json, agent_id, success_rate,
                    execution_count, source_episodes, applicable_contexts, created_at, updated_at
               FROM procedures
              WHERE (name LIKE ?1 OR description LIKE ?1 OR applicable_contexts LIKE ?1)
                AND success_rate >= ?2
                {agent_clause}
              ORDER BY success_rate DESC, execution_count DESC"
        );

        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;

        let procs = if let Some(aid) = agent_id {
            self.query_procs(&mut stmt, params![pattern, min_success_rate, aid])?
        } else {
            self.query_procs(&mut stmt, params![pattern, min_success_rate])?
        };
        Ok(procs)
    }

    /// Search by name using LIKE.
    pub fn search_by_name(&self, name: &str) -> Result<Vec<ProcedureTemplate>> {
        let conn = self.conn.lock();
        let pattern = format!("%{name}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, steps_json, agent_id, success_rate,
                        execution_count, source_episodes, applicable_contexts, created_at, updated_at
                   FROM procedures WHERE name LIKE ?1",
            )
            .map_err(CogError::Sqlite)?;
        self.query_procs(&mut stmt, params![pattern])
    }

    /// Record an execution outcome and update the EMA success rate.
    ///
    /// EMA formula: `new_rate = 0.3 * outcome + 0.7 * old_rate` (matches Python).
    ///
    /// Returns `true` if the record was found and updated.
    pub fn record_execution(&self, procedure_id: Uuid, success: bool) -> Result<bool> {
        let conn = self.conn.lock();
        let outcome = if success { 1.0_f64 } else { 0.0_f64 };
        let now_ms = dt_to_ms(Utc::now());

        let updated = conn
            .execute(
                "UPDATE procedures
                    SET success_rate    = 0.3 * ?2 + 0.7 * success_rate,
                        execution_count = execution_count + 1,
                        updated_at      = ?3
                  WHERE id = ?1",
                params![procedure_id.to_string(), outcome, now_ms],
            )
            .map_err(CogError::Sqlite)?;

        Ok(updated > 0)
    }

    /// List all procedures, optionally filtered by agent.
    pub fn list_all(
        &self,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ProcedureTemplate>> {
        let conn = self.conn.lock();
        let sql = match agent_id {
            None => format!(
                "SELECT id, name, description, steps_json, agent_id, success_rate,
                        execution_count, source_episodes, applicable_contexts, created_at, updated_at
                   FROM procedures ORDER BY created_at DESC LIMIT {limit}"
            ),
            Some(aid) => format!(
                "SELECT id, name, description, steps_json, agent_id, success_rate,
                        execution_count, source_episodes, applicable_contexts, created_at, updated_at
                   FROM procedures WHERE agent_id = '{aid}'
                  ORDER BY created_at DESC LIMIT {limit}"
            ),
        };
        let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
        self.query_procs(&mut stmt, params![])
    }

    /// Delete a procedure template by UUID.
    ///
    /// Returns `true` if a record was deleted, `false` if not found.
    pub fn delete(&self, procedure_id: Uuid) -> Result<bool> {
        let conn = self.conn.lock();
        let deleted = conn
            .execute(
                "DELETE FROM procedures WHERE id = ?1",
                params![procedure_id.to_string()],
            )
            .map_err(CogError::Sqlite)?;
        Ok(deleted > 0)
    }

    /// Count stored procedure templates, optionally filtered by agent.
    pub fn count(&self, agent_id: Option<&str>) -> Result<i64> {
        let conn = self.conn.lock();
        let n: i64 = match agent_id {
            None => conn
                .query_row("SELECT COUNT(*) FROM procedures", [], |r| r.get(0))
                .map_err(CogError::Sqlite)?,
            Some(aid) => conn
                .query_row(
                    "SELECT COUNT(*) FROM procedures WHERE agent_id = ?1",
                    params![aid],
                    |r| r.get(0),
                )
                .map_err(CogError::Sqlite)?,
        };
        Ok(n)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn fetch_by_id_locked(
        &self,
        conn: &rusqlite::Connection,
        id: Uuid,
    ) -> Result<Option<ProcedureTemplate>> {
        let mut stmt = conn
            .prepare_cached(
                "SELECT id, name, description, steps_json, agent_id, success_rate,
                        execution_count, source_episodes, applicable_contexts, created_at, updated_at
                   FROM procedures WHERE id = ?1",
            )
            .map_err(CogError::Sqlite)?;

        let mut rows = stmt.query(params![id.to_string()]).map_err(CogError::Sqlite)?;
        match rows.next().map_err(CogError::Sqlite)? {
            Some(row) => Ok(Some(Self::row_to_proc(row)?)),
            None => Ok(None),
        }
    }

    fn query_procs(
        &self,
        stmt: &mut rusqlite::Statement<'_>,
        params: impl rusqlite::Params,
    ) -> Result<Vec<ProcedureTemplate>> {
        let rows = stmt
            .query_map(params, Self::row_to_proc_raw)
            .map_err(CogError::Sqlite)?;

        let mut procs = Vec::new();
        for row in rows {
            let raw = row.map_err(CogError::Sqlite)?;
            let proc = Self::raw_to_proc(raw)?;
            procs.push(proc);
        }
        Ok(procs)
    }

    #[allow(clippy::type_complexity)]
    fn row_to_proc_raw(
        row: &rusqlite::Row<'_>,
    ) -> rusqlite::Result<(String, String, String, String, String, f64, i64, String, String, i64, i64)>
    {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
            row.get(5)?,
            row.get(6)?,
            row.get(7)?,
            row.get(8)?,
            row.get(9)?,
            row.get(10)?,
        ))
    }

    fn raw_to_proc(
        raw: (String, String, String, String, String, f64, i64, String, String, i64, i64),
    ) -> Result<ProcedureTemplate> {
        let (id_s, name, description, steps_json, agent_id, success_rate,
             exec_count, src_json, ctx_json, created_ms, updated_ms) = raw;
        Ok(ProcedureTemplate {
            id: id_s.parse().map_err(|e: uuid::Error| CogError::Store(e.to_string()))?,
            name,
            description,
            steps: serde_json::from_str::<Vec<ProcedureStep>>(&steps_json)?,
            agent_id,
            success_rate,
            execution_count: exec_count,
            source_episodes: serde_json::from_str(&src_json)?,
            applicable_contexts: serde_json::from_str(&ctx_json)?,
            created_at: ms_to_dt(created_ms),
            updated_at: ms_to_dt(updated_ms),
        })
    }

    fn row_to_proc(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProcedureTemplate> {
        let raw = Self::row_to_proc_raw(row)?;
        Self::raw_to_proc(raw).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn make_wal(dir: &Path) -> Arc<WalWriter> {
        Arc::new(WalWriter::open(&dir.join("proc.wal"), 0, 64 * 1024 * 1024).unwrap())
    }

    fn make_proc(name: &str, agent_id: &str, contexts: &[&str]) -> ProcedureTemplate {
        ProcedureTemplate {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: format!("Description of {name}"),
            steps: vec![ProcedureStep {
                action: "do_something".into(),
                tool: None,
                parameters: serde_json::json!({}),
                expected_output: None,
                fallback_action: None,
            }],
            agent_id: agent_id.to_string(),
            success_rate: 1.0,
            execution_count: 0,
            source_episodes: vec![],
            applicable_contexts: contexts.iter().map(|s| s.to_string()).collect(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn add_and_get() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        let p = make_proc("fetch_data", "a1", &["data_pipeline"]);
        let id = p.id;
        store.add(&p).unwrap();

        let fetched = store.get(id).unwrap().expect("should exist");
        assert_eq!(fetched.name, "fetch_data");
        assert_eq!(fetched.agent_id, "a1");
    }

    #[test]
    fn get_missing_returns_none() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        assert!(store.get(Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn search_by_context_matches_name_and_description() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        store.add(&make_proc("ingest_data", "a1", &["pipeline"])).unwrap();
        store.add(&make_proc("send_email", "a1", &["notifications"])).unwrap();

        let results = store.search_by_context("pipeline", None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "ingest_data");
    }

    #[test]
    fn search_by_context_filters_by_min_success_rate() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();

        let mut p_low = make_proc("low_rate", "a1", &["ctx"]);
        p_low.success_rate = 0.3;
        let mut p_high = make_proc("high_rate", "a1", &["ctx"]);
        p_high.success_rate = 0.9;
        store.add(&p_low).unwrap();
        store.add(&p_high).unwrap();

        let results = store.search_by_context("ctx", None, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "high_rate");
    }

    #[test]
    fn search_by_name() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        store.add(&make_proc("process_invoices", "a1", &[])).unwrap();
        store.add(&make_proc("send_invoice", "a1", &[])).unwrap();

        let results = store.search_by_name("invoice").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn record_execution_success_increases_rate() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        let mut p = make_proc("task", "a1", &[]);
        p.success_rate = 0.5;
        let id = p.id;
        store.add(&p).unwrap();

        store.record_execution(id, true).unwrap();
        let updated = store.get(id).unwrap().unwrap();
        // 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        assert!((updated.success_rate - 0.65).abs() < 1e-6);
        assert_eq!(updated.execution_count, 1);
    }

    #[test]
    fn record_execution_failure_decreases_rate() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        let mut p = make_proc("task", "a1", &[]);
        p.success_rate = 0.8;
        let id = p.id;
        store.add(&p).unwrap();

        store.record_execution(id, false).unwrap();
        let updated = store.get(id).unwrap().unwrap();
        // 0.3 * 0.0 + 0.7 * 0.8 = 0.56
        assert!((updated.success_rate - 0.56).abs() < 1e-6);
    }

    #[test]
    fn delete_existing_returns_true() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        let p = make_proc("task", "a1", &[]);
        let id = p.id;
        store.add(&p).unwrap();
        assert!(store.delete(id).unwrap());
        assert_eq!(store.count(None).unwrap(), 0);
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        assert!(!store.delete(Uuid::new_v4()).unwrap());
    }

    #[test]
    fn list_all_with_limit() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        for i in 0..5 {
            store.add(&make_proc(&format!("task_{i}"), "a1", &[])).unwrap();
        }
        let all = store.list_all(None, 10).unwrap();
        let limited = store.list_all(None, 3).unwrap();
        assert_eq!(all.len(), 5);
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn count_by_agent() {
        let dir = tempdir().unwrap();
        let store = ProceduralStore::open(dir.path(), make_wal(dir.path())).unwrap();
        store.add(&make_proc("t1", "a1", &[])).unwrap();
        store.add(&make_proc("t2", "a1", &[])).unwrap();
        store.add(&make_proc("t3", "a2", &[])).unwrap();
        assert_eq!(store.count(Some("a1")).unwrap(), 2);
        assert_eq!(store.count(Some("a2")).unwrap(), 1);
        assert_eq!(store.count(None).unwrap(), 3);
    }
}

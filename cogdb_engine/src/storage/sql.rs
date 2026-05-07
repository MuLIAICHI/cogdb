use std::path::Path;

use rusqlite::Connection;

use crate::error::{CogError, Result};

/// Open a SQLite connection with WAL mode, foreign keys, and busy timeout.
///
/// All three stores use this helper to get a consistent baseline configuration.
///
/// # Args
/// - `path`: Path to the `.db` file (created if it does not exist).
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use cogdb_engine::storage::sql::open_connection;
/// let conn = open_connection(Path::new("/tmp/test.db")).unwrap();
/// ```
pub fn open_connection(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).map_err(CogError::Sqlite)?;
    configure_connection(&conn)?;
    Ok(conn)
}

/// Apply pragmas to an open connection.
///
/// Called once per connection after open. Safe to call again (idempotent).
pub fn configure_connection(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous   = NORMAL;
         PRAGMA foreign_keys  = ON;
         PRAGMA busy_timeout  = 5000;
         PRAGMA cache_size    = -8000;",
    )
    .map_err(CogError::Sqlite)?;
    Ok(())
}

/// Create the episodic metadata table and indexes (idempotent).
///
/// # Schema
/// Stores all non-vector fields for episodic memory units.
pub fn create_episodic_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS episodic_meta (
            id            TEXT    PRIMARY KEY,
            agent_id      TEXT    NOT NULL,
            scope         TEXT    NOT NULL,
            importance    REAL    NOT NULL DEFAULT 0.5,
            content       TEXT    NOT NULL,
            memory_type   TEXT    NOT NULL DEFAULT 'episodic',
            metadata_json TEXT    NOT NULL DEFAULT '{}',
            created_at    INTEGER NOT NULL,
            accessed_at   INTEGER NOT NULL,
            access_count  INTEGER NOT NULL DEFAULT 0,
            decay_score   REAL    NOT NULL DEFAULT 1.0,
            team_id       TEXT
        );

        CREATE INDEX IF NOT EXISTS ep_agent    ON episodic_meta(agent_id);
        CREATE INDEX IF NOT EXISTS ep_scope    ON episodic_meta(scope);
        CREATE INDEX IF NOT EXISTS ep_import   ON episodic_meta(importance);
        CREATE INDEX IF NOT EXISTS ep_accessed ON episodic_meta(accessed_at);",
    )
    .map_err(CogError::Sqlite)?;
    Ok(())
}

/// Create the semantic triples table and indexes (idempotent).
pub fn create_semantic_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS triples (
            id               TEXT    PRIMARY KEY,
            subject          TEXT    NOT NULL,
            predicate        TEXT    NOT NULL,
            object           TEXT    NOT NULL,
            agent_id         TEXT    NOT NULL,
            confidence       REAL    NOT NULL DEFAULT 1.0,
            valid_from       INTEGER NOT NULL,
            valid_until      INTEGER,
            source_episodes  TEXT    NOT NULL DEFAULT '[]',
            metadata_json    TEXT    NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS tri_subject   ON triples(subject);
        CREATE INDEX IF NOT EXISTS tri_predicate ON triples(predicate);
        CREATE INDEX IF NOT EXISTS tri_agent     ON triples(agent_id);
        CREATE INDEX IF NOT EXISTS tri_active    ON triples(valid_until)
            WHERE valid_until IS NULL;",
    )
    .map_err(CogError::Sqlite)?;
    Ok(())
}

/// Create the procedures table and indexes (idempotent).
pub fn create_procedural_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS procedures (
            id                   TEXT    PRIMARY KEY,
            name                 TEXT    NOT NULL,
            description          TEXT    NOT NULL,
            steps_json           TEXT    NOT NULL DEFAULT '[]',
            agent_id             TEXT    NOT NULL,
            success_rate         REAL    NOT NULL DEFAULT 1.0,
            execution_count      INTEGER NOT NULL DEFAULT 0,
            source_episodes      TEXT    NOT NULL DEFAULT '[]',
            applicable_contexts  TEXT    NOT NULL DEFAULT '[]',
            created_at           INTEGER NOT NULL,
            updated_at           INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS proc_agent ON procedures(agent_id);
        CREATE INDEX IF NOT EXISTS proc_name  ON procedures(name);",
    )
    .map_err(CogError::Sqlite)?;
    Ok(())
}

/// Verify WAL mode is active for a connection.
pub fn assert_wal_mode(conn: &Connection) -> Result<()> {
    let mode: String = conn
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .map_err(CogError::Sqlite)?;
    if mode != "wal" {
        return Err(CogError::Store(format!(
            "expected WAL journal mode, got '{mode}'"
        )));
    }
    Ok(())
}

// ── Timestamp helpers ─────────────────────────────────────────────────────────

/// Convert a `chrono::DateTime<Utc>` to unix milliseconds for SQLite storage.
pub fn dt_to_ms(dt: chrono::DateTime<chrono::Utc>) -> i64 {
    dt.timestamp_millis()
}

/// Convert unix milliseconds (from SQLite) back to `chrono::DateTime<Utc>`.
pub fn ms_to_dt(ms: i64) -> chrono::DateTime<chrono::Utc> {
    use chrono::TimeZone;
    chrono::Utc
        .timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(chrono::Utc::now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn in_memory() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        configure_connection(&conn).unwrap();
        conn
    }

    #[test]
    fn wal_mode_active() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let conn = open_connection(&path).unwrap();
        assert_wal_mode(&conn).unwrap();
    }

    #[test]
    fn episodic_schema_idempotent() {
        let conn = in_memory();
        create_episodic_schema(&conn).unwrap();
        create_episodic_schema(&conn).unwrap(); // second call must not fail
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM episodic_meta", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn semantic_schema_idempotent() {
        let conn = in_memory();
        create_semantic_schema(&conn).unwrap();
        create_semantic_schema(&conn).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM triples", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn procedural_schema_idempotent() {
        let conn = in_memory();
        create_procedural_schema(&conn).unwrap();
        create_procedural_schema(&conn).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM procedures", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn episodic_insert_and_query() {
        let conn = in_memory();
        create_episodic_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO episodic_meta
             (id, agent_id, scope, importance, content, memory_type,
              metadata_json, created_at, accessed_at, access_count, decay_score)
             VALUES (?1,'a1','private',0.8,'hello','episodic','{}',1000,1000,0,1.0)",
            ["test-id-1"],
        )
        .unwrap();
        let imp: f64 = conn
            .query_row(
                "SELECT importance FROM episodic_meta WHERE id = 'test-id-1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((imp - 0.8).abs() < 1e-10);
    }

    #[test]
    fn timestamp_roundtrip() {
        let now = chrono::Utc::now();
        let ms = dt_to_ms(now);
        let back = ms_to_dt(ms);
        // Millisecond precision — within 1ms
        let diff = (now - back).num_milliseconds().abs();
        assert!(diff < 2, "diff={diff}ms");
    }

    #[test]
    fn open_close_reopen_preserves_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("persist.db");

        {
            let conn = open_connection(&path).unwrap();
            create_episodic_schema(&conn).unwrap();
            conn.execute(
                "INSERT INTO episodic_meta
                 (id, agent_id, scope, importance, content, memory_type,
                  metadata_json, created_at, accessed_at, access_count, decay_score)
                 VALUES ('id-1','a1','private',0.5,'content','episodic','{}',1,1,0,1.0)",
                [],
            )
            .unwrap();
        } // connection dropped here

        {
            let conn = open_connection(&path).unwrap();
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM episodic_meta", [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 1);
        }
    }
}

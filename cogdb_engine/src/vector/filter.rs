use std::collections::HashSet;

use rusqlite::Connection;
use uuid::Uuid;

use crate::error::{CogError, Result};
use crate::types::MemoryScope;

/// Threshold below which brute-force cosine search beats HNSW + post-filter.
pub const BRUTE_FORCE_THRESHOLD: usize = 5_000;

/// Parameters for filtering episodic candidates from SQLite.
#[derive(Debug, Default)]
pub struct EpisodicFilter<'a> {
    pub agent_id: &'a str,
    pub scope_filter: Option<&'a MemoryScope>,
    pub min_importance: f64,
    pub time_range_start_ms: Option<i64>,
    pub time_range_end_ms: Option<i64>,
}

/// Query SQLite for episodic IDs that satisfy the filter predicates.
///
/// Returns a set of UUID strings that can be fed into the HNSW search.
///
/// When `scope_filter` is `None`, the default scope rule applies:
///   `(agent_id = :agent_id) OR (scope IN ('team', 'org'))`
/// This replicates the existing Python EpisodicStore behaviour exactly.
pub fn candidate_ids(conn: &Connection, filter: &EpisodicFilter<'_>) -> Result<HashSet<Uuid>> {
    let mut conditions = Vec::<String>::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    // Scope / agent access control
    match filter.scope_filter {
        None => {
            conditions.push(format!(
                "(agent_id = ?{} OR scope IN ('team', 'org'))",
                params.len() + 1
            ));
            params.push(Box::new(filter.agent_id.to_string()));
        }
        Some(scope) => {
            conditions.push(format!("scope = ?{}", params.len() + 1));
            params.push(Box::new(scope.to_string()));
        }
    }

    // Importance floor
    if filter.min_importance > 0.0 {
        conditions.push(format!("importance >= ?{}", params.len() + 1));
        params.push(Box::new(filter.min_importance));
    }

    // Time range
    if let Some(start_ms) = filter.time_range_start_ms {
        conditions.push(format!("created_at >= ?{}", params.len() + 1));
        params.push(Box::new(start_ms));
    }
    if let Some(end_ms) = filter.time_range_end_ms {
        conditions.push(format!("created_at <= ?{}", params.len() + 1));
        params.push(Box::new(end_ms));
    }

    let where_clause = if conditions.is_empty() {
        "1=1".to_string()
    } else {
        conditions.join(" AND ")
    };

    let sql = format!("SELECT id FROM episodic_meta WHERE {where_clause}");
    let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql).map_err(CogError::Sqlite)?;
    let ids = stmt
        .query_map(params_refs.as_slice(), |row| row.get::<_, String>(0))
        .map_err(CogError::Sqlite)?
        .filter_map(|r| r.ok())
        .filter_map(|s| s.parse::<Uuid>().ok())
        .collect::<HashSet<Uuid>>();

    Ok(ids)
}

/// Decide which search strategy to use based on candidate set size.
#[derive(Debug, PartialEq)]
pub enum SearchStrategy {
    /// Candidate set is small — use brute-force cosine on these IDs.
    BruteForce,
    /// Candidate set is large — query HNSW for top_k*10, then post-filter.
    HnswPostFilter,
    /// No filter applied — query HNSW directly.
    HnswDirect,
}

pub fn choose_strategy(candidate_count: Option<usize>, threshold: usize) -> SearchStrategy {
    match candidate_count {
        None => SearchStrategy::HnswDirect,
        Some(n) if n < threshold => SearchStrategy::BruteForce,
        Some(_) => SearchStrategy::HnswPostFilter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::sql::{configure_connection, create_episodic_schema, dt_to_ms};
    use chrono::Utc;
    use rusqlite::Connection;
    use uuid::Uuid;

    fn setup_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        configure_connection(&conn).unwrap();
        create_episodic_schema(&conn).unwrap();
        conn
    }

    fn insert_row(conn: &Connection, id: &str, agent_id: &str, scope: &str, importance: f64) {
        conn.execute(
            "INSERT INTO episodic_meta
             (id, agent_id, scope, importance, content, memory_type,
              metadata_json, created_at, accessed_at, access_count, decay_score)
             VALUES (?1, ?2, ?3, ?4, 'content', 'episodic', '{}', 1000, 1000, 0, 1.0)",
            rusqlite::params![id, agent_id, scope, importance],
        )
        .unwrap();
    }

    #[test]
    fn default_scope_includes_agent_and_shared() {
        let conn = setup_conn();
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-1", "private", 0.5);
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-2", "private", 0.5);
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-1", "team", 0.5);
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-3", "org", 0.5);

        let filter = EpisodicFilter {
            agent_id: "agent-1",
            ..Default::default()
        };
        let ids = candidate_ids(&conn, &filter).unwrap();
        // Should return: agent-1 private, agent-1 team, agent-3 org
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn scope_filter_overrides_default() {
        let conn = setup_conn();
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-1", "private", 0.5);
        insert_row(&conn, &Uuid::new_v4().to_string(), "agent-1", "org", 0.5);

        let filter = EpisodicFilter {
            agent_id: "agent-1",
            scope_filter: Some(&MemoryScope::Org),
            ..Default::default()
        };
        let ids = candidate_ids(&conn, &filter).unwrap();
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn importance_filter() {
        let conn = setup_conn();
        insert_row(&conn, &Uuid::new_v4().to_string(), "a", "org", 0.3);
        insert_row(&conn, &Uuid::new_v4().to_string(), "a", "org", 0.8);
        insert_row(&conn, &Uuid::new_v4().to_string(), "a", "org", 0.9);

        let filter = EpisodicFilter {
            agent_id: "a",
            min_importance: 0.7,
            scope_filter: Some(&MemoryScope::Org),
            ..Default::default()
        };
        let ids = candidate_ids(&conn, &filter).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn time_range_filter() {
        let conn = setup_conn();
        let now_ms = dt_to_ms(Utc::now());
        let id1 = Uuid::new_v4().to_string();
        let id2 = Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO episodic_meta
             (id, agent_id, scope, importance, content, memory_type,
              metadata_json, created_at, accessed_at, access_count, decay_score)
             VALUES (?1, 'a', 'org', 0.5, 'c', 'episodic', '{}', ?2, ?2, 0, 1.0)",
            rusqlite::params![&id1, now_ms - 10_000],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO episodic_meta
             (id, agent_id, scope, importance, content, memory_type,
              metadata_json, created_at, accessed_at, access_count, decay_score)
             VALUES (?1, 'a', 'org', 0.5, 'c', 'episodic', '{}', ?2, ?2, 0, 1.0)",
            rusqlite::params![&id2, now_ms + 10_000],
        )
        .unwrap();

        let filter = EpisodicFilter {
            agent_id: "a",
            scope_filter: Some(&MemoryScope::Org),
            time_range_end_ms: Some(now_ms),
            ..Default::default()
        };
        let ids = candidate_ids(&conn, &filter).unwrap();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&id1.parse::<Uuid>().unwrap()));
    }

    #[test]
    fn strategy_selection() {
        assert_eq!(choose_strategy(None, 5000), SearchStrategy::HnswDirect);
        assert_eq!(choose_strategy(Some(100), 5000), SearchStrategy::BruteForce);
        assert_eq!(choose_strategy(Some(5000), 5000), SearchStrategy::HnswPostFilter);
        assert_eq!(choose_strategy(Some(10_000), 5000), SearchStrategy::HnswPostFilter);
    }
}

#![cfg(feature = "python")]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::Path;

use crate::engine::{Engine, EngineConfig};
use crate::types::MemoryScope;

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Python-facing engine class. All complex types cross as JSON strings;
/// primitives (bool, int, float, str, list[str]) cross natively.
///
/// One PyEngine instance should exist per db_path (enforced by the Python
/// _engine_cache module). Multiple instances on the same path would cause
/// WAL and file-lock conflicts.
#[pyclass]
pub struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// Open (or create + recover) the engine at `db_path`.
    #[new]
    pub fn open(
        db_path: &str,
        embedding_dim: usize,
        hnsw_m: usize,
        hnsw_ef_construction: usize,
        contradiction_check: bool,
    ) -> PyResult<Self> {
        let config = EngineConfig {
            embedding_dim,
            hnsw_m,
            hnsw_ef_construction,
            contradiction_check,
            ..EngineConfig::default()
        };
        let engine = Engine::open(Path::new(db_path), config).map_err(py_err)?;
        Ok(Self { inner: engine })
    }

    /// Flush a checkpoint and close all SQLite connections.
    pub fn close(&self) -> PyResult<()> {
        self.inner.close().map_err(py_err)
    }

    // ── Episodic ──────────────────────────────────────────────────────────────

    /// Add a memory unit. `unit_json` is a JSON-serialised MemoryUnit.
    /// Returns the UUID string of the stored record.
    pub fn episodic_add(&self, unit_json: &str) -> PyResult<String> {
        let unit: crate::types::MemoryUnit =
            serde_json::from_str(unit_json).map_err(py_err)?;
        self.inner.episodic.add(&unit).map_err(py_err)
    }

    /// Search episodic memories.
    /// `embedding_json`: JSON array of f32 values.
    /// `scope_filter`: "private"/"team"/"org"/"session" or None.
    /// Returns a JSON array of MemoryUnit objects.
    pub fn episodic_search(
        &self,
        embedding_json: &str,
        agent_id: &str,
        top_k: usize,
        scope_filter: Option<String>,
        min_importance: f64,
        time_range_start_ms: Option<i64>,
        time_range_end_ms: Option<i64>,
    ) -> PyResult<String> {
        let embedding: Vec<f32> = serde_json::from_str(embedding_json).map_err(py_err)?;
        let scope: Option<MemoryScope> = scope_filter
            .as_deref()
            .map(|s| s.parse::<MemoryScope>().map_err(py_err))
            .transpose()?;

        let results = self.inner.episodic.search(
            &embedding,
            agent_id,
            top_k,
            scope.as_ref(),
            min_importance,
            time_range_start_ms,
            time_range_end_ms,
        ).map_err(py_err)?;

        serde_json::to_string(&results).map_err(py_err)
    }

    /// Fetch a single memory unit by UUID string.
    /// Returns a JSON-serialised MemoryUnit, or the JSON string "null".
    pub fn episodic_get(&self, memory_id: &str) -> PyResult<String> {
        let id = parse_uuid(memory_id)?;
        let result = self.inner.episodic.get(id).map_err(py_err)?;
        serde_json::to_string(&result).map_err(py_err)
    }

    /// Delete a memory unit by UUID string. Returns true if deleted.
    pub fn episodic_delete(&self, memory_id: &str) -> PyResult<bool> {
        let id = parse_uuid(memory_id)?;
        self.inner.episodic.delete(id).map_err(py_err)
    }

    /// Update metadata fields. `updates_json` is a JSON object.
    pub fn episodic_update_metadata(
        &self,
        memory_id: &str,
        updates_json: &str,
    ) -> PyResult<bool> {
        let id = parse_uuid(memory_id)?;
        let updates: serde_json::Value = serde_json::from_str(updates_json).map_err(py_err)?;
        self.inner.episodic.update_metadata(id, &updates).map_err(py_err)
    }

    /// Count stored memories; pass None to count all agents.
    pub fn episodic_count(&self, agent_id: Option<&str>) -> PyResult<i64> {
        self.inner.episodic.count(agent_id).map_err(py_err)
    }

    /// Paginated scan for decay processing.
    /// Returns a JSON array of DecayScanRow objects.
    pub fn episodic_scan_batch(
        &self,
        agent_id: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> PyResult<String> {
        let results = self.inner.episodic.scan_batch(agent_id, limit, offset).map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// Batch update decay scores.
    /// `updates_json`: JSON array of [uuid_str, f64] pairs.
    pub fn episodic_bulk_update_decay(&self, updates_json: &str) -> PyResult<()> {
        let pairs: Vec<(String, f64)> =
            serde_json::from_str(updates_json).map_err(py_err)?;
        let updates: Vec<(uuid::Uuid, f64)> = pairs
            .into_iter()
            .map(|(s, d)| parse_uuid(&s).map(|id| (id, d)))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.episodic.bulk_update_decay(&updates).map_err(py_err)
    }

    // ── Semantic ──────────────────────────────────────────────────────────────

    /// Add a semantic triple. `triple_json` is a JSON-serialised SemanticTriple.
    /// Returns the UUID string.
    pub fn semantic_add_triple(&self, triple_json: &str) -> PyResult<String> {
        let triple: crate::types::SemanticTriple =
            serde_json::from_str(triple_json).map_err(py_err)?;
        self.inner.semantic.add_triple(&triple).map_err(py_err)
    }

    /// Query all active triples with the given subject.
    /// Returns a JSON array of SemanticTriple objects.
    pub fn semantic_query_subject(
        &self,
        subject: &str,
        active_only: bool,
        agent_id: Option<&str>,
    ) -> PyResult<String> {
        let results = self
            .inner
            .semantic
            .query_subject(subject, active_only, agent_id)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// BFS traversal from entity up to depth hops.
    /// Returns a JSON array of SemanticTriple objects.
    pub fn semantic_query_entity(
        &self,
        entity: &str,
        depth: usize,
        active_only: bool,
    ) -> PyResult<String> {
        let results = self
            .inner
            .semantic
            .query_entity(entity, depth, active_only)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// Full-text LIKE search across subject/predicate/object.
    /// Returns a JSON array of SemanticTriple objects.
    pub fn semantic_search_text(&self, query: &str, active_only: bool) -> PyResult<String> {
        let results = self
            .inner
            .semantic
            .search_text(query, active_only)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// All entity names in the active graph.
    pub fn semantic_get_entities(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.semantic.get_entities())
    }

    /// Direct neighbors (outgoing + incoming) of entity.
    pub fn semantic_get_neighbors(&self, entity: &str) -> PyResult<Vec<String>> {
        Ok(self.inner.semantic.get_neighbors(entity))
    }

    /// Delete a triple by UUID string. Returns true if deleted.
    pub fn semantic_delete_triple(&self, triple_id: &str) -> PyResult<bool> {
        let id = parse_uuid(triple_id)?;
        self.inner.semantic.delete_triple(id).map_err(py_err)
    }

    /// Count triples; set active_only=True to count only active ones.
    pub fn semantic_count(&self, active_only: bool) -> PyResult<i64> {
        self.inner.semantic.count(active_only).map_err(py_err)
    }

    // ── Procedural ────────────────────────────────────────────────────────────

    /// Add a procedure template. `proc_json` is a JSON-serialised ProcedureTemplate.
    /// Returns the UUID string.
    pub fn procedural_add(&self, proc_json: &str) -> PyResult<String> {
        let proc: crate::types::ProcedureTemplate =
            serde_json::from_str(proc_json).map_err(py_err)?;
        self.inner.procedural.add(&proc).map_err(py_err)
    }

    /// Fetch a procedure by UUID string.
    /// Returns a JSON-serialised ProcedureTemplate, or the JSON string "null".
    pub fn procedural_get(&self, procedure_id: &str) -> PyResult<String> {
        let id = parse_uuid(procedure_id)?;
        let result = self.inner.procedural.get(id).map_err(py_err)?;
        serde_json::to_string(&result).map_err(py_err)
    }

    /// Search by context/name/description. Returns a JSON array.
    pub fn procedural_search_by_context(
        &self,
        context: &str,
        agent_id: Option<&str>,
        min_success_rate: f64,
    ) -> PyResult<String> {
        let results = self
            .inner
            .procedural
            .search_by_context(context, agent_id, min_success_rate)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// Search by name using LIKE. Returns a JSON array.
    pub fn procedural_search_by_name(&self, name: &str) -> PyResult<String> {
        let results = self
            .inner
            .procedural
            .search_by_name(name)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// Record an execution outcome (updates EMA success rate).
    pub fn procedural_record_execution(
        &self,
        procedure_id: &str,
        success: bool,
    ) -> PyResult<bool> {
        let id = parse_uuid(procedure_id)?;
        self.inner.procedural.record_execution(id, success).map_err(py_err)
    }

    /// List all procedures up to limit. Returns a JSON array.
    pub fn procedural_list_all(
        &self,
        agent_id: Option<&str>,
        limit: usize,
    ) -> PyResult<String> {
        let results = self
            .inner
            .procedural
            .list_all(agent_id, limit)
            .map_err(py_err)?;
        serde_json::to_string(&results).map_err(py_err)
    }

    /// Delete a procedure by UUID string. Returns true if deleted.
    pub fn procedural_delete(&self, procedure_id: &str) -> PyResult<bool> {
        let id = parse_uuid(procedure_id)?;
        self.inner.procedural.delete(id).map_err(py_err)
    }

    /// Count procedures; pass None to count all agents.
    pub fn procedural_count(&self, agent_id: Option<&str>) -> PyResult<i64> {
        self.inner.procedural.count(agent_id).map_err(py_err)
    }
}

fn parse_uuid(s: &str) -> PyResult<uuid::Uuid> {
    s.parse::<uuid::Uuid>()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Register all PyO3 classes into the module. Called from lib.rs `#[pymodule]`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}

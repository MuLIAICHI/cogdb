# CogDB — Build Progress

Last updated: 2026-05-08 (session 11)

---

## Phase Overview

| Phase | Tag | Status | Summary |
|---|---|---|---|
| 0 — Python PoC | v0.2.0 | ✅ Complete | Tri-memory stores, pipeline, adapters, benchmarks |
| 1 — Rust engine | v0.3.0 | ✅ Complete | cogdb_engine crate, WAL, PyO3 bindings, 140 tests green |
| 2 — ML retrieval | v0.4.0 | ✅ Complete | ImportanceModel + HNSW blending + tokenised procedure retrieval; Suite 1 = 90.7/100 |
| 3 — Schema evolution | — | 🔄 In progress | Phase 3A: dynamic typed metadata schemas shipped |
| 4 — Distributed | — | future | Clustered WAL replication |

---

## Phase 0 — Python Proof-of-Concept ✅ (v0.2.0)

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `cogdb/utils/config.py` | ✅ | CogDBConfig dataclass |
| 2 | `cogdb/utils/tokenizer.py` | ✅ | tiktoken-based, fallback to char estimate |
| 3 | `cogdb/stores/episodic.py` | ✅ | ChromaDB wrapper → replaced by Rust in Phase 1 |
| 4 | `cogdb/stores/semantic.py` | ✅ | NetworkX + SQLite → replaced by Rust in Phase 1 |
| 5 | `cogdb/stores/procedural.py` | ✅ | SQLite → replaced by Rust in Phase 1 |
| 6 | `cogdb/models/importance.py` | ✅ | Heuristic scoring — **Phase 2 target** |
| 7 | `cogdb/pipeline/encoder.py` | ✅ | sentence-transformers, batch encode, metadata extraction |
| 8 | `cogdb/pipeline/retriever.py` | ✅ | Token-budget greedy fill, L0–L3 progressive loading |
| 9 | `cogdb/pipeline/decay.py` | ✅ | Exponential decay, `scan_batch` + `bulk_update_decay` |
| 10 | `cogdb/pipeline/consolidator.py` | ✅ | Regex SPO extraction — **Phase 2 target** |
| 11 | `cogdb/query/planner.py` | ✅ | Intent-based store routing, scope access control |
| 12 | `cogdb/core.py` | ✅ | CognitiveDB main class |
| 13 | `cogdb/adapters/autogen.py` | ✅ | AutoGen Memory protocol |
| 14 | `cogdb/adapters/langgraph.py` | ✅ | CogDBCheckpointer + CogDBStore, sync + async |
| 15 | `cogdb/adapters/mcp.py` | ✅ | 6 tools, `cogdb-mcp` CLI |

---

## Phase 1 — Rust Storage Engine ✅ (v0.3.0)

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 1 | `cogdb_engine/src/error.rs` + `types.rs` | ✅ | CogError, MemoryUnit, SemanticTriple, ProcedureTemplate, serde |
| 2 | `cogdb_engine/src/wal/` | ✅ | Append-only WAL, CRC32 framing, crash-truncated tail tolerance |
| 3 | `cogdb_engine/src/storage/sql.rs` | ✅ | rusqlite WAL mode, all 3 schemas + indexes |
| 4 | `cogdb_engine/src/vector/hnsw.rs` | ✅ | HnswIndex, parallel vector HashMap for snapshots |
| 5 | `cogdb_engine/src/vector/filter.rs` | ✅ | Metadata pre-filter, BruteForce / HnswPostFilter strategy |
| 6 | `cogdb_engine/src/graph/kg.rs` | ✅ | petgraph DiGraph, BFS, O(1) targeted edge removal |
| 7 | `cogdb_engine/src/stores/episodic.rs` | ✅ | HNSW + SQL + WAL, scan_batch, bulk_update_decay |
| 8 | `cogdb_engine/src/stores/semantic.rs` | ✅ | petgraph + SQL + WAL, contradiction detection |
| 9 | `cogdb_engine/src/stores/procedural.rs` | ✅ | SQL + WAL fence, EMA success rate (α=0.3) |
| 10 | `cogdb_engine/src/engine.rs` | ✅ | Engine::open with WAL recovery, checkpoint, close |
| 11 | `cogdb_engine/src/python.rs` | ✅ | PyEngine class (JSON string interchange), PyO3 0.28 abi3-py310 |
| 12 | `cogdb/_engine_cache.py` | ✅ | Singleton PyEngine per db_path, release_engine for teardown |
| 13 | `cogdb/stores/episodic.py` | ✅ | Rust-backed wrapper, _ClientShim, Encoder integration |
| 14 | `cogdb/stores/semantic.py` | ✅ | Rust-backed wrapper |
| 15 | `cogdb/stores/procedural.py` | ✅ | Rust-backed wrapper |

**Tests:** 91/91 Rust unit tests · 140/140 Python tests · `pytest tests/ -v`

---

## Benchmark Results

### v0.4.0 (Phase 2 — ML retrieval)

| Suite | Metric | Result |
|---|---|---|
| 1 — Tri-Memory | Overall quality (keyword scoring) | **90.7 / 100** |
| 1 — Tri-Memory | episodic+procedural | 94.6 / 100 |
| 1 — Tri-Memory | semantic+episodic | 87.3 / 100 |
| 1 — Tri-Memory | all | 90.2 / 100 |

### v0.3.0 (Phase 1 — Rust engine, baseline)

| Suite | Metric | Result |
|---|---|---|
| 1 — Tri-Memory | Overall quality (keyword scoring) | 87.9 / 100 |
| 3 — Consistency | consistency_score | 100% |
| 3 — Consistency | supersede_accuracy | 100% |
| 3 — Consistency | conflict_resolution | 100% |
| 4 — Throughput | Raw storage write | ~1.2 ms/op (~830 ops/s) |
| 4 — Throughput | Raw storage search | ~0.9 ms/op (~1100 ops/s) |
| 4 — Throughput | Full-pipeline write (incl. encoding) | ~12 ms/op (~85 ops/s) |
| 4 — Throughput | Encoding overhead | ~11–14 ms/op (sentence-transformers bottleneck) |

Run: `python -m benchmarks.cogdb_bench --suite all --no-llm`

---

## Phase 2 — ML Retrieval ✅ (v0.4.0)

| # | Target | Status | Notes |
|---|--------|--------|-------|
| 1 | `cogdb/models/importance.py` | ✅ | `ImportanceModel` (Ridge, 11 features, synthetic training data, online `partial_fit`, save/load JSON). Fallback to heuristic if sklearn absent. |
| 2 | `cogdb/pipeline/retriever.py` | ✅ | HNSW rank blending: `(1-α)×importance + α×hnsw_relevance`. α=0.2 via `config.hnsw_blend_alpha`. Cap at `max_procedures_per_query=1`. |
| 3 | `cogdb/pipeline/consolidator.py` | ✅ | Optional LLM extraction path via `config.use_llm_consolidation=True` + `OPENAI_API_KEY`. Regex is still default. |
| 4 | `cogdb/utils/config.py` | ✅ | Added `hnsw_blend_alpha: float = 0.2`, `use_llm_consolidation: bool = False`, `max_procedures_per_query: int = 1`. |
| 5 | `pyproject.toml` | ✅ | Added `ml = ["scikit-learn>=1.3.0"]` optional dep. |
| 6 | `tests/test_importance_model.py` | ✅ | 31 tests — feature extraction, training, prediction, save/load, partial_fit, singleton, public API. 31/31 green. |
| 7 | `cogdb/stores/procedural.py` | ✅ | `search_by_context` tokenises natural-language queries word-by-word (Rust LIKE per keyword), deduplicates by ID, sorts by `(match_count DESC, success_rate DESC)`. |
| 8 | Benchmark Suite 1 | ✅ | **90.7/100** overall (episodic+procedural 94.6, semantic+episodic 87.3, all 90.2). Target > 90 met. |

---

## Tests Status

| File | Count | Status |
|------|-------|--------|
| `tests/test_episodic_store.py` | 9 | ✅ |
| `tests/test_semantic_store.py` | 12 | ✅ |
| `tests/test_procedural_store.py` | 14 | ✅ |
| `tests/test_core.py` | — | ✅ |
| `tests/test_retriever.py` | — | ✅ |
| `tests/test_query_planner.py` | — | ✅ |
| `tests/test_autogen_adapter.py` | — | ✅ |
| `tests/test_langgraph_adapter.py` | — | ✅ |
| `tests/test_mcp_adapter.py` | — | ✅ |
| `tests/test_importance_model.py` | 31 | ✅ |
| `tests/test_schema_registry.py` | 36 | ✅ |
| **Total** | **207** | **✅ all green** |

---

## Phase 3 — Schema Evolution (In Progress)

### Phase 3A — Dynamic Typed Metadata Schemas ✅

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `cogdb/schema/__init__.py` | ✅ | `FieldSchema`, `MetadataSchema`, `SchemaValidationError`. Supported types: str, int, float, bool, list, dict, any. |
| 2 | `cogdb/schema/registry.py` | ✅ | `SchemaRegistry` — register/get/list/validate/persist. Stored at `{db_path}/schemas.json`. |
| 3 | `cogdb/utils/config.py` | ✅ | Added `strict_metadata_validation: bool = True`. |
| 4 | `cogdb/core.py` | ✅ | `register_schema()`, `get_schema()`, `list_schemas()`. Validation wired into `remember()`. |
| 5 | `tests/test_schema_registry.py` | ✅ | 36 tests — FieldSchema, MetadataSchema, error format, validate, strict/non-strict, persistence roundtrip, CognitiveDB integration. |

**Scope (3A only):** episodic memory metadata. Semantic triples and procedural step parameters out of scope until 3B/3C.

### Phase 3B — Metadata Indexing (future)
Add SQLite indexes on commonly queried metadata keys. Requires Rust store changes.

### Phase 3C — Schema Migration (future)
Safe field rename/add/drop with version tracking. Version counter already written to `schemas.json` as a hook.

---

## Architecture Decisions Log

### 2026-05-08 (Phase 3A — session 11)
- **Dynamic typed metadata schemas**: `cogdb/schema/` module — `FieldSchema` (7 supported types), `MetadataSchema` (keyed by agent_id), `SchemaValidationError` with field-level error list.
- **SchemaRegistry**: persist to `{db_path}/schemas.json`; validate per-field with type-aware checking (bool vs int subclass handled correctly); re-registration bumps `version` as migration hook for Phase 3C.
- **Strict mode** (default `True`): raises `SchemaValidationError` on violation; `False` runs validation silently — configurable via `config.strict_metadata_validation`.
- **Scope**: episodic-only. Semantic and procedural metadata validation deferred to later phases.
- **207/207 tests green** ✅

### 2026-05-07 (Phase 2 — session 10)
- **Root cause found (session 10)**: `search_by_context` Rust SQL uses `WHERE applicable_contexts LIKE '%{full_query}%'` — the entire question string is the pattern, which never matches single-word context lists. Procedures were silently absent from ALL retrieval results, producing the identical `84.2 → 84.2` score across baseline and Phase 2.
- **Fix — tokenised keyword search** (`stores/procedural.py`): split context into individual words (len≥4, minus stop words), call Rust LIKE once per keyword, dedup by ID, sort by `(match_count DESC, success_rate DESC)` so the most query-relevant procedure ranks first (not just the highest success_rate globally).
- **Fix — procedure cap** (`retriever.py` + `config.py`): `max_procedures_per_query=2` prevents procedures (each ~80–100 tokens) from crowding out episodic memories in the 500-token L2 budget. With 2 procs (~160 tokens), ~10–11 episodic memories still fit.
- **Diagnostic note**: first attempt (tokenisation only, no cap) made things worse — procedures matched broadly (4+ per query) and consumed ~320/500 L2 tokens, dropping `semantic+episodic` from 89.3 → 78.3.
- **171/171 tests green** after both fixes.
- Benchmark Suite 1 pending — run to confirm > 90/100.

### 2026-05-04 (Phase 2 — session 9)
- **ImportanceModel**: Ridge regression, 11 features (type, word_count, has_version, has_metric, high/low kw density, entity_density, specificity, tech_density, access_log, recency_decay). Trained on 60-example synthetic dataset embedded in the module — no file I/O at runtime. sklearn at train-time only; inference is `np.dot(scaled_features, coef) + intercept`.
- **Training data note**: Trained on content-feature labels, not real access patterns (no access data yet). Phase 2.5 target: call `partial_fit()` as memories accumulate real access counts.
- **HNSW rank blending in `retriever.py`**: `(1-α)×importance + α×hnsw_relevance` where hnsw_relevance = position in HNSW result list. α=0.2 via `config.hnsw_blend_alpha`.
- **Optional LLM consolidation**: `config.use_llm_consolidation=True` routes `_extract_triples_llm()` (OpenAI gpt-4o-mini) instead of regex. Regex is always the fallback (no key, import error, parse error). No changes to the `run()` signature.

### 2026-05-03 (Phase 1)
- PyO3 bumped 0.22 → 0.28 (Python 3.14 requires ≥0.23); `abi3-py310` stable ABI used
- JSON string interchange chosen for Python↔Rust boundary (no `#[pyclass]` for complex types)
- `_engine_cache.py` singleton prevents WAL conflicts when 3 stores share one db_path
- `Engine::close_connections()` swaps file-backed SQLite connections to in-memory before cleanup — fixes Windows TemporaryDirectory file lock errors in tests
- LangGraph `CogDBStore.delete()` switched to deterministic UUID5 (no more ChromaDB `_collection` query)
- Contradiction detection bug fixed: removed `AND agent_id = ?` from Rust SQL — matches Phase 0 behaviour (cross-agent contradictions are detected)
- Suite 4 Throughput benchmark added: encoding overhead (~11–14 ms) is the bottleneck, not Rust storage (~1 ms)

### 2026-04-16 (Phase 0)
- Converted `cogdb/models.py` → `cogdb/models/__init__.py` to allow importance submodule
- Importance scoring: heuristic-only for Phase 0 (keyword regex + recency + access count); ML model deferred to Phase 2
- Consolidator uses regex SPO extraction (no LLM) — rule: no external LLM calls in core engine
- Query planner uses keyword-intent routing; caller-specified `memory_types` always respected
- Decay: exponential model with λ = ln(2) / half_life; batch eviction at configurable threshold

---

## Known Issues / Blockers

- Shell commands via Bash tool fail on this machine due to spaces in the path (`AI Eng Mustapha LIAICHI`). Run manually with `! <command>` in the Claude Code prompt.
- Rust PATH must be set per PowerShell session: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`

### 2026-05-03 (session 8 — continued)
- Built PyO3 Python bindings (`cogdb_engine/src/python.rs`) — single `PyEngine` class, JSON string interchange for all complex types
- `cogdb_engine/pyproject.toml` (maturin) + built with `maturin develop --release` — `cogdb_engine-0.1.0-cp310-abi3-win_amd64.whl`
- Replaced `cogdb/stores/episodic.py`, `semantic.py`, `procedural.py` with Rust-backed wrappers
- Added `cogdb/_engine_cache.py` — singleton `PyEngine` per `db_path` (prevents WAL conflicts)
- Fixed `cogdb/pipeline/decay.py` — replaced ChromaDB `_collection.get/update/delete` with `scan_batch` + `bulk_update_decay`
- Fixed `cogdb/adapters/langgraph.py` — replaced `_collection` access in `delete()` with deterministic UUID5-based deletion
- Fixed `tests/conftest.py` — patch `TemporaryDirectory` to `ignore_cleanup_errors=True` (Windows file lock workaround)
- Added `close_connections()` to all Rust stores — called by `Engine::close()` to release SQLite file handles
- **PyO3 version bump**: 0.22 → 0.28.3 (Python 3.14 requires ≥0.23; used `abi3-py310` stable ABI)
- **140/140 Python tests green** ✅ — full `pytest tests/ -v` passing against Rust backend

### 2026-05-03 (session 8)
- **Phase 1 architecture plan** finalised — Rust crate replacing ChromaDB/NetworkX/SQLite, PyO3 bindings, WAL design
- **Built `cogdb_engine` Rust crate** (`cogdb_engine/`) — full storage engine, 91 unit tests + 9 doc tests, zero warnings
- Crate modules:
  - `error.rs` / `types.rs` — `CogError`, `MemoryUnit`, `SemanticTriple`, `ProcedureTemplate`, all enums with serde
  - `wal/` — append-only WAL with CRC32 framing, fsync on every record, crash-truncated tail tolerance
  - `storage/sql.rs` — rusqlite connections with WAL mode, all 3 schemas + indexes
  - `vector/hnsw.rs` — `HnswIndex` wrapping `hnsw_rs`, parallel vector HashMap for snapshot (avoids lifetime issue), brute-force filtered search
  - `vector/filter.rs` — metadata pre-filter from SQLite, `choose_strategy(BruteForce / HnswPostFilter / HnswDirect)`
  - `graph/kg.rs` — petgraph `DiGraph`, entity index, bidirectional BFS, targeted O(1) edge removal (no graph-clear-and-rebuild)
  - `stores/episodic.rs` — HNSW + SQL + WAL, all public methods matching Python EpisodicStore interface; adds `scan_batch` + `bulk_update_decay` for decay.py coupling fix
  - `stores/semantic.rs` — petgraph + SQL + WAL, contradiction detection, graph rebuilt from SQL on open
  - `stores/procedural.rs` — SQL + WAL fence, EMA success rate (α=0.3), all Python interface methods
  - `engine.rs` — `Engine::open` with full WAL recovery (find last Checkpoint → load HNSW snapshot → replay EpisodicUpsert/Delete), `checkpoint`, `close`
- **Key design decisions recorded:**
  - hnsw_rs `'static` lifetime issue resolved by storing raw vectors in parallel HashMap (safe, no unsafe transmute)
  - petgraph node removal uses name-based re-lookup after each swap-remove (fixes orphan cleanup bug)
  - WAL replay off-by-one fixed: `checkpoint_seq: Option<u64>` so "no checkpoint" replays ALL records (not just seq > 0)
  - Scope filter always applied on search (no unguarded HNSW path that could leak cross-agent data)
- **Rust toolchain:** rustc 1.95.0 installed; PATH must be set per-session (`$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`)

### 2026-04-18 (session 7 — continued)
- Built `benchmarks/cogdb_bench.py` — full 3-suite benchmark (700+ lines), all suites confirmed passing
- **Suite 1 results** (keyword scoring): overall 87.8/100 — episodic+procedural 84.2, semantic+episodic 90.6, all 88.5
- **Suite 2 results**: baseline wins on density with keyword scoring (expected artifact — LLM judge needed to differentiate progressive vs raw)
- **Suite 3 results**: consistency 100%, supersede_accuracy 100%, conflict_resolution 100% after fixing timing bug (replaced `sleep(0.05)` with `threading.Event`)
- **Suite 1 (Tri-Memory)**: 90 fixture records (50 episodic + 30 semantic + 10 procedural), 30 questions across 3 categories (episodic+procedural, semantic+episodic, all), LLM judge (OpenAI gpt-4o-mini) with keyword-match fallback
- **Suite 2 (Token Efficiency)**: same data, per-question comparison of progressive L0-L3 vs dump-all vs raw ChromaDB; reports `information_density = quality/tokens`
- **Suite 3 (Multi-Agent Consistency)**: 3 concurrent threads × 50 ops; measures read correctness, supersede accuracy, conflict resolution accuracy
- CLI: `python -m benchmarks.cogdb_bench --suite all|tri-memory|token-efficiency|consistency --no-llm --questions N --out PATH`
- Outputs JSON to `benchmarks/results/bench_<timestamp>.json` + formatted terminal report
- Added `benchmark = ["openai>=1.0.0"]` extras to `pyproject.toml`; install with `pip install -e ".[benchmark]"`

### 2026-04-18 (session 7)
- Audited `cogdb/adapters/autogen.py` — all 5 protocol methods now present (add/query/update_context/clear/close)
- Added `close()` async method — delegates to `db.close()` if available, safe to call multiple times
- Added `TestCogDBMemoryClose` in `tests/test_autogen_adapter.py` — covers no-crash and idempotent close
- **Full test suite confirmed green** ✅ — all test files passing

### 2026-04-16 (session 6)
- Rebuilt `cogdb/adapters/mcp.py` — 6 tools (was 4), full JSON schemas with LLM-friendly descriptions
- Added `learn` (semantic facts) and `learn_procedure` (workflows) tools — matching full CogDB API
- Added `get_context` tool — exposes progressive L0–L3 context loading over MCP
- Added `cogdb-mcp` entry point in `pyproject.toml` — users can run `cogdb-mcp --db-path ./mem`
- Extracted `dispatch()` method — single routing function, easier to test and extend
- Written `tests/test_mcp_adapter.py` — 9 test classes, covers schemas, all 6 handlers, dispatch, thread-safety

### 2026-04-16 (session 5)
- All 8 test files passing (full suite green)
- Fixed 3 failure classes: Python 3.12 `asyncio.run()`, Windows ChromaDB file locks, logic bugs
- `examples/multi_agent_demo.py` confirmed working — verified output:
  - 4 episodic / 5 semantic / 1 procedural memories stored correctly
  - CORS recall returned 3 results ranked 0.9 → 0.8 → 0.7 within 300-token budget
  - L2 progressive context: 5 critical facts + 2 relevant memories, 90/500 tokens used (18%)
  - Cross-agent scope: ui-agent correctly sees own private + org-scoped devops memory
- **Phase 0 complete** ✅ — all 3 core ideas proven

---

## Known Issues / Blockers

- Shell commands via Bash tool fail on this machine due to spaces in the working directory path (`AI Eng Mustapha LIAICHI`). User must run any shell commands manually using `! <command>` in the Claude Code prompt.

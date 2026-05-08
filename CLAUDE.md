# CLAUDE.md — CogDB Master Context

## What This Project Is

CogDB is a cognitive database engine for AI agents. It unifies episodic, semantic, and procedural memory into a single Python package with framework adapters for AutoGen and LangGraph.

**Current phase: Phase 3 — Schema Evolution**

Phases 0–2 are complete. Phase 2 delivered: ImportanceModel (Ridge, sklearn), HNSW rank blending, tokenised procedural retrieval, and optional LLM consolidation. Suite 1 benchmark improved from 87.9 → 90.7/100.

---

## Architecture

```
cogdb/                           ← Python package (public API)
├── core.py                      ← CognitiveDB main class, single entry point
├── stores/
│   ├── episodic.py              ← Thin wrapper over Rust EpisodicStore (PyO3)
│   ├── semantic.py              ← Thin wrapper over Rust SemanticStore (PyO3)
│   └── procedural.py           ← Thin wrapper over Rust ProceduralStore (PyO3)
├── _engine_cache.py             ← Singleton PyEngine per db_path
├── pipeline/
│   ├── encoder.py               ← sentence-transformers, text → embedding
│   ├── consolidator.py          ← Regex SPO extraction (Phase 2: optional LLM)
│   ├── retriever.py             ← Token-budget-aware retrieval, L0–L3 loading
│   └── decay.py                 ← Exponential decay, uses scan_batch + bulk_update_decay
├── query/
│   └── planner.py               ← Routes queries to stores, merges results
├── adapters/
│   ├── autogen.py               ← AutoGen Memory protocol
│   ├── langgraph.py             ← LangGraph BaseCheckpointSaver + BaseStore
│   └── mcp.py                   ← MCP server (6 tools, cogdb-mcp CLI)
├── models/
│   └── importance.py            ← ImportanceModel (Ridge regression, Phase 2)
├── schema/
│   ├── __init__.py              ← FieldSchema, MetadataSchema, SchemaValidationError
│   └── registry.py             ← SchemaRegistry (persist schemas.json, validate)
└── utils/
    ├── tokenizer.py             ← tiktoken token counting
    └── config.py                ← CogDBConfig dataclass

cogdb_engine/                    ← Rust crate (storage engine, compiled via maturin)
├── Cargo.toml                   ← pyo3 = "0.28" (abi3-py310), hnsw_rs, rusqlite, petgraph
├── pyproject.toml               ← maturin build config
└── src/
    ├── lib.rs                   ← PyO3 module entry point
    ├── python.rs                ← PyEngine class (JSON-string interchange)
    ├── engine.rs                ← Engine::open (WAL recovery), checkpoint, close
    ├── error.rs / types.rs      ← CogError, MemoryUnit, SemanticTriple, ProcedureTemplate
    ├── wal/                     ← Append-only WAL, CRC32 framing, crash recovery
    ├── storage/sql.rs           ← rusqlite, WAL mode, schema migrations
    ├── vector/hnsw.rs           ← HNSW index (hnsw_rs), filtered search
    ├── vector/filter.rs         ← Metadata pre-filter, BruteForce / HnswPostFilter
    ├── graph/kg.rs              ← petgraph DiGraph, BFS, O(1) edge removal
    └── stores/
        ├── episodic.rs          ← HNSW + SQLite + WAL
        ├── semantic.rs          ← petgraph + SQLite + WAL, contradiction detection
        └── procedural.rs        ← SQLite + WAL fence, EMA success rate
```

### Build the Rust extension

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cd cogdb_engine
maturin develop --release   # installs cogdb_engine.pyd into active venv
```

---

## Phase Roadmap

| Phase | Status | Summary |
|---|---|---|
| 0 — Python PoC | ✅ v0.2.0 | Tri-memory stores, pipeline, adapters |
| 1 — Rust engine | ✅ v0.3.0 | cogdb_engine crate, WAL, PyO3 bindings |
| 2 — ML retrieval | ✅ v0.4.0 | ImportanceModel, HNSW blend, tokenised proc retrieval; Suite 1 = 90.7/100 |
| 3 — Schema evolution | 🔄 current | 3A done: typed metadata schemas; 3B: indexing; 3C: migration |
| 4 — Distributed | future | Clustered deployment, WAL replication |

---

## How to Work on This Project

### Phase 2 build targets

1. `cogdb/models/importance.py` — Replace heuristic with a lightweight sklearn/onnx model trained on access patterns. Must still work with no external API key.
2. `cogdb/pipeline/consolidator.py` — Add optional LLM extraction path (caller opts in via config). Regex path remains the default (no API key required).
3. New test: `tests/test_importance_model.py` — training, scoring, persistence.
4. Benchmark: Suite 1 score should improve above 88 with the learned model.

### Phase 1 items (already built — do not rebuild)

The Rust crate in `cogdb_engine/` is the source of truth for storage. Do not modify `cogdb/stores/*.py` to add new storage logic — add it to the Rust store and expose via `python.rs`.

### Dependencies

```
chromadb>=0.5.0       ← REMOVED in Phase 1 (replaced by cogdb_engine)
networkx>=3.0         ← REMOVED in Phase 1 (replaced by petgraph in Rust)
tiktoken>=0.7.0
sentence-transformers>=3.0.0
pydantic>=2.0
```

For Rust extension:
```
maturin>=1.5  (build-time only)
cogdb_engine  (built from cogdb_engine/ via maturin develop --release)
```

For adapters (optional):
```
autogen-agentchat>=0.4.0
langgraph>=0.2.0
mcp>=1.0.0
```

---

## Hard Rules

1. **Every public method must have a docstring** with Args, Returns, and a usage example
2. **All store operations must be thread-safe** — Rust handles this; Python wrappers are stateless
3. **Never load all memories into RAM** — always paginate, always respect token budgets
4. **Embeddings are lazy** — computed in `encoder.py`, never in the Rust stores
5. **Tests for every new feature** — maintain 140/140 passing
6. **No external LLM calls in the core engine** — LLM features go in optional pipeline stages (consolidator, Phase 2 retrieval optimizer)
7. **Config via dataclass, not env vars**
8. **Type hints everywhere**

---

## Key Data Models

### MemoryUnit (the universal memory record)

```python
@dataclass
class MemoryUnit:
    id: str                          # UUID string
    content: str
    memory_type: MemoryType          # episodic | semantic | procedural
    agent_id: str
    scope: MemoryScope               # private | team | org | session
    importance: float                # 0.0 to 1.0
    embedding: Optional[list[float]] # computed by Encoder, stored in HNSW
    metadata: dict
    created_at: datetime
    accessed_at: datetime
    access_count: int
    decay_score: float
```

### SemanticTriple / ProcedureTemplate

See `cogdb/models/__init__.py` — unchanged from Phase 0.

---

## Multi-Agent Memory Scopes

```
┌─────────────────────────────┐
│      Organization Scope     │  All agents can read
│  ┌───────────────────────┐  │
│  │     Team Scope        │  │  Defined group, read-write
│  │  ┌─────────────────┐  │  │
│  │  │  Private Scope   │  │  │  Single agent only
│  │  └─────────────────┘  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
┌─────────────────────────────┐
│      Session Scope          │  Ephemeral, auto-deleted
└─────────────────────────────┘
```

Scope enforcement: `vector/filter.rs` applies `(agent_id = ? OR scope IN ('team','org'))` on every search. No unguarded HNSW path exists.

---

## Token-Cost-Aware Retrieval

```
Budget: 500 tokens
├── L0 Identity (50 tokens)        → always included
├── L1 Critical facts (150 tokens) → included, 300 remaining
├── L2 Task-relevant (250 tokens)  → included, 50 remaining
└── L3 Deep search (0 tokens)      → skipped, budget exhausted
```

---

## Benchmark Baseline (v0.3.0 — Rust backend)

| Suite | Metric | Value |
|---|---|---|
| 1 — Tri-Memory | Overall quality (v0.4.0) | **90.7 / 100** ✅ |
| 1 — Tri-Memory | episodic+procedural | 94.6 / 100 |
| 1 — Tri-Memory | semantic+episodic | 87.3 / 100 |
| 1 — Tri-Memory | all | 90.2 / 100 |
| 3 — Consistency | consistency_score | 100% |
| 3 — Consistency | supersede_accuracy | 100% |
| 3 — Consistency | conflict_resolution | 100% |
| 4 — Throughput | Raw storage write | ~1.2 ms/op (830 ops/s) |
| 4 — Throughput | Raw storage search | ~0.9 ms/op (1100 ops/s) |
| 4 — Throughput | Encoding overhead | ~11–14 ms/op (sentence-transformers) |

---

## Testing

```
tests/
├── test_episodic_store.py    ← CRUD, similarity search, scope filter
├── test_semantic_store.py    ← Triple CRUD, contradiction, BFS depth
├── test_procedural_store.py  ← Workflow CRUD, EMA success rate
├── test_retriever.py         ← Token budget, L0–L3 progressive loading
├── test_query_planner.py     ← Store routing, explain()
├── test_core.py              ← Integration tests
├── test_autogen_adapter.py   ← AutoGen protocol compliance
├── test_langgraph_adapter.py ← Checkpointer + Store, sync + async
└── test_mcp_adapter.py       ← 6 tools, thread-safety
```

Run: `pytest tests/ -v`  → 140 tests, all must pass.

---

## What NOT to Build Yet

- ❌ Rust WAL per-agent sharding — Phase 1.5
- ❌ Internal ML training pipeline — Phase 2 uses pre-trained or sklearn, not custom training infra
- ❌ Agent-driven schema evolution — Phase 3
- ❌ Production distributed deployment — Phase 4
- ❌ Custom embedding model training — use sentence-transformers off the shelf

---

## Known Constraints (this machine)

- Shell commands via Bash tool fail due to spaces in path (`AI Eng Mustapha LIAICHI`). Run commands manually with `! <command>` in the Claude Code prompt.
- Rust PATH must be set per PowerShell session: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`
- PyO3 requires pyo3 ≥ 0.23 for Python 3.14; currently pinned to 0.28 with `abi3-py310` stable ABI.

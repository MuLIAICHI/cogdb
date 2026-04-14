# CLAUDE.md — CogDB Master Context

## What This Project Is

CogDB is a cognitive database engine for AI agents. It unifies episodic, semantic, and procedural memory into a single Python package with framework adapters for AutoGen and LangGraph.

**Current phase: Python proof-of-concept (Phase 0)**

The goal is NOT a production database engine yet. The goal is a working prototype that proves three ideas:
1. Tri-memory unification (episodic + semantic + procedural) in one interface
2. Token-cost-aware retrieval (optimize for LLM consumption, not human readability)
3. Drop-in framework adapters (AutoGen + LangGraph)

---

## Architecture

```
cogdb/
├── core.py              ← CognitiveDB main class, the single entry point
├── stores/
│   ├── episodic.py      ← ChromaDB-backed vector store for interaction records
│   ├── semantic.py      ← NetworkX + SQLite knowledge graph with temporal facts
│   └── procedural.py    ← SQLite-backed workflow template store
├── pipeline/
│   ├── encoder.py       ← Memory encoding (text → embedding + metadata extraction)
│   ├── consolidator.py  ← Episodic → semantic distillation (background)
│   ├── retriever.py     ← Token-budget-aware retrieval across all stores
│   └── decay.py         ← Importance decay and memory eviction
├── query/
│   └── planner.py       ← Routes queries to the right store(s), merges results
├── adapters/
│   ├── autogen.py       ← Implements AutoGen Memory protocol
│   ├── langgraph.py     ← Implements LangGraph BaseCheckpointSaver + BaseStore
│   └── mcp.py           ← MCP server exposing remember/recall/relate/forget tools
├── models/
│   └── importance.py    ← Lightweight importance scoring (Phase 0: heuristic, Phase 2: ML)
└── utils/
    ├── tokenizer.py     ← Token counting and budget management
    └── config.py        ← Configuration dataclass
```

---

## How to Work on This Project

### Build order (follow this sequence)

1. `cogdb/utils/config.py` — Config dataclass first
2. `cogdb/utils/tokenizer.py` — Token counter (use tiktoken)
3. `cogdb/stores/episodic.py` — ChromaDB vector store wrapper
4. `cogdb/stores/semantic.py` — Knowledge graph (NetworkX + SQLite)
5. `cogdb/stores/procedural.py` — Workflow template store (SQLite)
6. `cogdb/models/importance.py` — Importance scoring (heuristic for now)
7. `cogdb/pipeline/encoder.py` — Text → embedding + metadata
8. `cogdb/pipeline/retriever.py` — Token-budget-aware retrieval
9. `cogdb/pipeline/decay.py` — Memory decay/eviction
10. `cogdb/pipeline/consolidator.py` — Episode → semantic distillation
11. `cogdb/query/planner.py` — Query routing and result merging
12. `cogdb/core.py` — CognitiveDB main class (composes everything)
13. `cogdb/adapters/autogen.py` — AutoGen adapter
14. `cogdb/adapters/langgraph.py` — LangGraph adapter
15. `cogdb/adapters/mcp.py` — MCP server

### Dependencies

```
chromadb>=0.5.0
networkx>=3.0
tiktoken>=0.7.0
sentence-transformers>=3.0.0
pydantic>=2.0
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
2. **All store operations must be thread-safe** — use threading.Lock where needed
3. **Never load all memories into RAM** — always paginate, always respect token budgets
4. **Embeddings are lazy** — don't compute embeddings until storage time
5. **Tests for every store** — each store must have unit tests before building the next layer
6. **No external LLM calls in the core engine** — the stores and pipeline must work without an API key. LLM-powered features (entity extraction, consolidation) go in optional pipeline stages
7. **Config via dataclass, not env vars** — keep it explicit and testable
8. **Type hints everywhere** — this codebase must be fully typed

---

## Key Data Models

### MemoryUnit (the universal memory record)

```python
@dataclass
class MemoryUnit:
    id: str                          # UUID
    content: str                     # Raw text content
    memory_type: MemoryType          # episodic | semantic | procedural
    agent_id: str                    # Owning agent
    scope: MemoryScope               # private | team | org | session
    importance: float                # 0.0 to 1.0
    embedding: Optional[list[float]] # Computed lazily
    metadata: dict                   # Flexible key-value
    created_at: datetime
    accessed_at: datetime
    access_count: int
    decay_score: float               # Current decay value
```

### SemanticTriple (knowledge graph fact)

```python
@dataclass
class SemanticTriple:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float       # 0.0 to 1.0
    valid_from: datetime
    valid_until: Optional[datetime]
    source_episodes: list[str]  # Provenance links
    agent_id: str
```

### ProcedureTemplate (learned workflow)

```python
@dataclass
class ProcedureTemplate:
    id: str
    name: str
    description: str
    steps: list[ProcedureStep]
    success_rate: float
    execution_count: int
    source_episodes: list[str]
    agent_id: str
    applicable_contexts: list[str]  # When to suggest this procedure
```

### ContextResponse (what the agent receives)

```python
@dataclass
class ContextResponse:
    level: int                  # 0-3
    token_count: int            # Actual tokens used
    token_budget: int           # Max tokens allowed
    identity: str               # L0: agent identity string
    critical_facts: list[str]   # L1: key knowledge
    relevant_memories: list[MemoryUnit]  # L2: task-relevant
    deep_results: list[MemoryUnit]       # L3: similarity search
```

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

Scope enforcement happens at the query planner level. Every query carries an `agent_id` and the planner filters results based on the agent's access rights.

---

## Token-Cost-Aware Retrieval

The retriever works in tiers. Given a token budget, it fills from L0 upward:

```
Budget: 500 tokens
├── L0 Identity (50 tokens)     → always included
├── L1 Critical facts (150 tokens) → included, 300 remaining
├── L2 Task-relevant (250 tokens)  → included, 50 remaining
└── L3 Deep search (0 tokens)     → skipped, budget exhausted
```

Each memory unit's text is measured via tiktoken before inclusion. The retriever greedily fills the budget with the highest-importance memories that fit.

---

## Testing Strategy

```
tests/
├── test_episodic_store.py    ← CRUD, similarity search, metadata filtering
├── test_semantic_store.py    ← Triple CRUD, temporal queries, contradiction detection
├── test_procedural_store.py  ← Workflow CRUD, context matching
├── test_retriever.py         ← Token budget enforcement, progressive loading
├── test_query_planner.py     ← Cross-store query routing
├── test_core.py              ← Integration tests for CognitiveDB
├── test_autogen_adapter.py   ← AutoGen Memory protocol compliance
└── test_langgraph_adapter.py ← LangGraph interface compliance
```

Run tests: `pytest tests/ -v`

---

## What NOT to Build (Phase 0 Scope)

- ❌ Rust storage engine — that's Phase 1
- ❌ Internal ML models (retrieval optimizer, consolidation model) — Phase 2
- ❌ Agent-driven schema evolution — Phase 3
- ❌ Production WAL/crash recovery — Phase 1
- ❌ Distributed/clustered deployment — future
- ❌ Custom embedding model training — use sentence-transformers off the shelf

Focus on proving the API design and the three core ideas work.

# CogDB — The Cognitive Database for AI Agents

> No existing system combines episodic, semantic, and procedural memory with internal ML-powered optimization in a single, agent-native database.

**CogDB** is an open-source cognitive database engine designed as a second brain for AI agents in multi-agent systems. It unifies three memory types — episodic (what happened), semantic (what is known), and procedural (how to do things) — into a single query interface that is optimized for LLM consumption, not human readability.

## Why CogDB?

Every production multi-agent deployment today stitches together 2–4 separate backends (vector DB + graph DB + relational store + memory framework). Memory — the most critical differentiator for intelligent agents — is an afterthought bolted on top.

CogDB treats memory as a **first-class database problem**, not an application-layer concern.

### What makes CogDB different

| Feature | Mem0 | Zep | Letta | MemPalace | **CogDB** |
|---|---|---|---|---|---|
| Episodic memory | ✓ | ✓✓ | ✓ | ✓ | ✓✓ |
| Semantic memory (knowledge graph) | ✓✓ | ✓✓ | ✓ | ✓ | ✓✓✓ |
| Procedural memory | Partial | ✗ | Partial | ✗ | **✓✓** |
| Unified tri-store engine | ✗ | ✗ | ✗ | ✗ | **✓** |
| Token-cost-aware retrieval | ✗ | ✗ | ✗ | Partial | **✓** |
| Multi-agent memory scopes | Scoped | Per-user | Shared blocks | Per-agent | **4 formal scopes** |
| Self-improving retrieval | ✗ | ✗ | ✗ | ✗ | **Planned (Phase 2)** |
| Framework adapters | SDK | SDK | SDK/ADE | MCP | **AutoGen + LangGraph + MCP** |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Interface                       │
│         MCP Tools  │  Python SDK  │  REST API            │
├─────────────────────────────────────────────────────────┤
│                    Query Planner                         │
│    Token-cost optimization  │  Progressive loading       │
├─────────────────────────────────────────────────────────┤
│                  Cognitive Pipeline                      │
│  Encoding → Storage → Consolidation → Retrieval → Decay │
├──────────┬──────────────────┬───────────────────────────┤
│ Episodic │    Semantic      │     Procedural            │
│  Store   │     Store        │      Store                │
│ (Vector) │ (Knowledge Graph)│  (Workflow Templates)     │
├──────────┴──────────────────┴───────────────────────────┤
│              Unified Storage Engine                      │
│     ChromaDB  │  NetworkX/SQLite  │  SQLite              │
└─────────────────────────────────────────────────────────┘
```

### Memory Types

**Episodic Memory** — Timestamped records of agent interactions, observations, and tool calls. Stored as embeddings with full metadata. Supports temporal queries ("what happened in the last 3 tasks?") and similarity search ("find similar past situations").

**Semantic Memory** — A temporal knowledge graph where entities and relationships carry validity windows, confidence scores, and provenance links. Facts have lifecycles: they can be confirmed, contradicted, superseded, or expired.

**Procedural Memory** — Learned workflows extracted from successful agent task completions. When an agent solves a multi-step problem, the solution pattern is captured as a reusable template. The least-addressed memory type in the current landscape.

### Memory Scopes (Multi-Agent)

- **Private** — Single agent, fully isolated
- **Team** — Defined agent group, read-write with conflict resolution
- **Organization** — All agents, read with permissioned write
- **Session** — Ephemeral, single conversation

### Token-Cost-Aware Retrieval

CogDB doesn't just find relevant memories — it returns them in the most token-efficient format for LLM consumption. Progressive loading delivers context in tiers:

- **L0 — Identity** (~50 tokens): Agent name, role, critical constraints
- **L1 — Critical facts** (~200 tokens): Key knowledge graph entities
- **L2 — Task-relevant** (~500 tokens): Memories matching current context
- **L3 — Deep search** (variable): Full similarity search across all stores

## Quick Start

### Installation

```bash
pip install cogdb
```

### Basic Usage

```python
from cogdb import CognitiveDB

db = CognitiveDB()

# Store an episodic memory
db.remember(
    content="User prefers dark mode and compact layouts",
    memory_type="episodic",
    agent_id="ui-agent",
    importance=0.8
)

# Store a semantic fact with validity
db.learn(
    subject="user_preference",
    predicate="theme",
    object="dark_mode",
    confidence=0.95,
    valid_from="2026-04-14"
)

# Store a procedural memory (learned workflow)
db.learn_procedure(
    name="deploy_frontend",
    steps=[
        {"action": "run_tests", "tool": "pytest"},
        {"action": "build", "tool": "npm run build"},
        {"action": "deploy", "tool": "vercel --prod"}
    ],
    success_rate=0.92,
    source_episodes=["ep_001", "ep_002", "ep_003"]
)

# Recall with token budget
memories = db.recall(
    query="What does the user prefer for UI?",
    agent_id="ui-agent",
    token_budget=500,
    memory_types=["episodic", "semantic"]
)

# Get progressive context for an agent
context = db.get_context(
    agent_id="ui-agent",
    level=2,  # L0 + L1 + L2
    task_hint="redesigning the settings page"
)
```

### AutoGen Integration

```python
from autogen import ConversableAgent
from cogdb.adapters.autogen import CogDBMemory

memory = CogDBMemory(db_path="./agent_memory")

agent = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    memory=[memory]
)
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from cogdb.adapters.langgraph import CogDBCheckpointer, CogDBStore

checkpointer = CogDBCheckpointer(db_path="./agent_memory")
store = CogDBStore(db_path="./agent_memory")

graph = StateGraph(State)
# ... define your graph ...
app = graph.compile(checkpointer=checkpointer, store=store)
```

## Benchmarks

Run the benchmark suite against Mem0 and baseline ChromaDB:

```bash
python -m benchmarks.run --suite all
```

Benchmark dimensions:
- **Retrieval accuracy** on LongMemEval
- **Token efficiency** (information density per token returned)
- **Multi-agent consistency** (concurrent read/write correctness)
- **Progressive loading** (latency at each context level)

## Project Roadmap

### Phase 0 — Python PoC (Current)
- [x] Repo structure and research documentation
- [ ] Tri-memory store (episodic + semantic + procedural)
- [ ] Token-cost-aware retrieval with progressive loading
- [ ] AutoGen Memory protocol adapter
- [ ] LangGraph checkpointer + store adapter
- [ ] MCP server interface
- [ ] Benchmarks vs Mem0 and MemPalace

### Phase 1 — Rust Engine
- [ ] Purpose-built storage engine with hybrid indexes (HNSW + B+Tree + adjacency lists)
- [ ] WAL + crash recovery
- [ ] Multi-agent memory scopes with COW isolation
- [ ] Native MCP server

### Phase 2 — Self-Improving
- [ ] Learned retrieval optimizer
- [ ] Memory consolidation model
- [ ] Access pattern predictor
- [ ] Importance scorer

### Phase 3 — Agent-Native Evolution
- [ ] Agent-driven schema evolution
- [ ] Token-cost query planner
- [ ] LLM-optimized internal representation

## Research

See [`docs/research.md`](docs/research.md) for the full landscape analysis covering:
- Comprehensive comparison of 15+ existing memory systems
- Academic foundations (CoALA, learned indexes, self-driving databases)
- The five whitespace opportunities CogDB addresses
- Traditional DB primitives mapped to cognitive equivalents

## Contributing

CogDB is in early development. If you're interested in AI agent memory infrastructure, we'd love your input:

1. **Research contributions** — Analysis of new memory systems, benchmarks, academic papers
2. **Framework adapters** — CrewAI, Semantic Kernel, OpenAI Agents SDK
3. **Benchmark scenarios** — Real-world multi-agent memory patterns
4. **Core engine** — Storage, query planning, memory pipeline

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

## Citation

```bibtex
@software{cogdb2026,
  title={CogDB: A Cognitive Database for AI Agents},
  author={Mustapha Liaichi},
  year={2026},
  url={https://github.com/mustaphaliaichi/cogdb}
}
```

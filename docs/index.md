# CogDB Documentation

**A second brain for AI agents** — episodic, semantic, and procedural memory in one engine.

---

## Quick install

```bash
pip install git+https://github.com/MuLIAICHI/cogdb.git
```

```python
from cogdb import CognitiveDB

db = CognitiveDB(db_path="./memory")

# Store what happened (episodic)
db.remember("Deployed v2.3 — CORS error on /users", agent_id="devops", importance=0.9)

# Store what's known (semantic)
db.learn("api_service", "version", "v2.3", agent_id="devops", confidence=1.0)

# Store how to fix things (procedural)
db.learn_procedure("fix_cors", steps=[...], agent_id="devops")

# Recall with a token budget
memories = db.recall("CORS error", agent_id="devops", token_budget=500)
```

---

## What's in here

<div class="grid cards" markdown>

- **[Getting Started](tutorial_getting_started.md)**

    Build a DevOps assistant agent with full tri-memory support in under 30 lines of Python.

- **[API Reference](api.md)**

    Complete reference for every public method — parameters, return types, and working examples.

- **[Migrating from Mem0](tutorial_migrating_from_mem0.md)**

    Side-by-side code comparison and a migration script if you're switching from Mem0.

- **[Research](research.md)**

    Academic foundations: CoALA, JEPA, learned index structures, multi-agent memory architecture.

</div>

---

## Why three memory types?

| Type | Question it answers | Example |
|---|---|---|
| **Episodic** | What happened? | "CORS error on /users at 2pm" |
| **Semantic** | What is true? | "api_service.version = v2.3" |
| **Procedural** | How do I do X? | Learned 4-step CORS fix workflow |

Most memory systems only handle episodic (vector search). CogDB is the only system that handles all three — and knows which type to pull from based on your query.

---

## Token-aware retrieval

```
Budget: 500 tokens
├── L0 Identity           ~50 tokens   ✓ always included
├── L1 Critical facts    ~150 tokens   ✓ included
├── L2 Task-relevant     ~250 tokens   ✓ included
└── L3 Deep search           0 tokens  ✗ budget exhausted
```

Set a budget. CogDB fills it from the top with the highest-importance memories that fit. No context window blowouts.

---

## Framework adapters

CogDB works natively with the frameworks you're already using:

```python
# AutoGen
from cogdb.adapters.autogen import CogDBMemory

# LangGraph
from cogdb.adapters.langgraph import CogDBCheckpointer, CogDBStore

# CrewAI
from cogdb.adapters.crewai import CogDBCrewAIStorage

# OpenAI Agents SDK
from cogdb.adapters.openai_agents import CogDBAgentMemory, make_memory_tools

# Semantic Kernel
from cogdb.adapters.semantic_kernel import CogDBMemoryStore

# LlamaIndex
from cogdb.adapters.llamaindex import CogDBChatMemory

# MCP (Claude Code, Cursor, Windsurf)
# cogdb-mcp --db-path ./memory
```

---

## Benchmark

Suite 1 — Tri-Memory Retrieval Quality: **90.7 / 100** on a synthetic 90-memory DevOps scenario across 3 agents.

[Run it yourself →](https://github.com/MuLIAICHI/cogdb/tree/main/benchmarks)

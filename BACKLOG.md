# CogDB Backlog

This file tracks everything on the CogDB roadmap — what's done, what's in progress, and what's planned. Updated as work happens.

For implementation status of the current phase, see `progress.md`. This file looks further out.

---

## Status Legend

- ✅ Done
- 🚧 In progress
- 📋 Planned (committed)
- 💭 Idea (not committed yet)
- ❌ Decided against

---

## Phase 0 — Python PoC ✅

The proof-of-concept that established the core ideas.

- ✅ Core data models (MemoryUnit, SemanticTriple, ProcedureTemplate, ContextResponse)
- ✅ Episodic store (ChromaDB-backed vector store)
- ✅ Semantic store (NetworkX + SQLite knowledge graph with temporal facts)
- ✅ Procedural store (SQLite workflow templates)
- ✅ Token-budget-aware retriever with progressive loading (L0 → L3)
- ✅ Multi-agent memory scopes (private, team, organization, session)
- ✅ Contradiction detection and fact supersession
- ✅ Importance scoring (heuristic)
- ✅ Memory decay
- ✅ AutoGen Memory protocol adapter
- ✅ LangGraph BaseCheckpointSaver + BaseStore adapter
- ✅ MCP server with 6 tools (remember, recall, learn, learn_procedure, get_context, forget)
- ✅ Test coverage across all stores and integration tests
- ✅ Multi-agent demo example
- ✅ Tagged v0.1.0

---

## Phase 0.5 — Validation & Visibility 🚧

Before going deeper on the engine, prove it works in production and build credibility.

- 🚧 CogBoard — multi-project status tracker dogfooding CogDB
- 📋 Custom benchmark suite (tri-memory accuracy, token efficiency, multi-agent consistency)
- 📋 LongMemEval benchmark adapter and run
- 📋 Comparison benchmarks vs Mem0, Zep, MemPalace, raw ChromaDB
- 📋 Public dashboard at cogboard.ayautomate.com showing live CogDB stats
- 📋 LinkedIn launch post with real production data
- 📋 Research blog post on the tri-memory architecture
- 📋 Reddit posts to r/LocalLLaMA, r/LangChain, r/LLMDevs

---

## Phase 1 — Production Storage Engine 📋

Replace the Python store backends with a purpose-built Rust engine. This is the months-long project.

### Storage layer
- 📋 Tri-store engine in Rust (single binary, not stitched-together backends)
- 📋 Hybrid index stack: HNSW (vectors) + B+Tree (metadata) + adjacency lists (graph)
- 📋 Page manager with typed pages (embedding, graph, metadata)
- 📋 Importance-weighted free list / garbage collection

### Durability
- 📋 Write-ahead log (WAL) per agent
- 📋 Crash recovery from WAL replay
- 📋 Snapshot isolation via copy-on-write
- 📋 Per-agent operation log for replay, rollback, audit

### Multi-agent coherence
- 📋 Formal consistency protocols for shared memory scopes
- 📋 Last-writer-wins-with-merge for knowledge graph triples
- 📋 Vector versioning for embedding updates
- 📋 Conflict resolution for concurrent writes

### Native interfaces
- 📋 Native Rust MCP server
- 📋 Python bindings (PyO3) so existing Python integrations keep working
- 📋 REST API for language-agnostic access

---

## Phase 2 — Self-Improving Memory 📋

Internal ML models that make the database get smarter from agent usage. This is the differentiator no other system has.

### Internal models
- 📋 Retrieval optimizer (learned ranker, distilled from LLM judgments)
- 📋 Consolidation model (knows when to distill episodes into semantic facts)
- 📋 Access pattern predictor (pre-fetches memories agents will need)
- 📋 Importance scorer (replaces heuristic with learned model)

### Adaptive optimization
- 📋 Learned query optimizer (Bao-style multi-armed bandit)
- 📋 Workload forecasting from agent interaction patterns
- 📋 Auto-indexing based on query history
- 📋 Memory consolidation schedules learned per agent

### Training infrastructure
- 📋 Synthetic training data generation pipeline
- 📋 On-line fine-tuning from real agent feedback
- 📋 A/B testing framework for model rollouts

---

## Phase 3 — Agent-Native Evolution 💭

Schema and representation that agents define and evolve through interaction.

- 💭 Agent-driven schema evolution (agents create new memory types and relationships)
- 💭 Token-cost-aware query planner (dual optimization: I/O cost + token cost)
- 💭 LLM-optimized internal representation (compressed semantic encoding)
- 💭 Cross-agent memory federation (agents share learnings across organizations)

---

## Phase 4 — Predictive Memory 💭

Adding consequence prediction — the LeCun JEPA-inspired direction. Research-stage.

### Research foundation
- 💭 Whitepaper: "Predictive Memory for Trustworthy Multi-Agent LLM Systems"
- 💭 Architecture spec for fourth memory type (predictive/prospective)
- 💭 Survey of existing world model approaches (JEPA, Dreamer, MuZero, WebDreamer)

### Implementation
- 💭 `cogdb-predict` extension package (separate from core)
- 💭 Action consequence prediction grounded in episodic + semantic + procedural memory
- 💭 Risk scoring for proposed agent actions
- 💭 Prediction-vs-reality comparison loop for self-improvement
- 💭 Integration with agent decision loops (proceed/modify/abort flow)

### Evaluation
- 💭 Benchmark on irreversible-action environments (web agents, deployment agents)
- 💭 Comparison to reactive agents on multi-agent failure rates

---

## Cross-Cutting Concerns

### Adapters & integrations 📋
- ✅ AutoGen 0.4 adapter
- ✅ LangGraph adapter
- ✅ MCP server
- 📋 CrewAI memory adapter
- 📋 OpenAI Agents SDK adapter
- 📋 Semantic Kernel adapter
- 📋 LlamaIndex adapter

### Documentation 📋
- ✅ README with architecture overview
- ✅ CLAUDE.md for AI-assisted development
- ✅ Research doc (docs/research.md)
- 📋 API reference docs
- 📋 Tutorial: "Building your first multi-agent system with CogDB"
- 📋 Tutorial: "Migrating from Mem0 to CogDB"
- 📋 Architecture deep-dive blog posts

### Operations 💭
- 💭 Observability (metrics, traces, structured logs)
- 💭 Cloud-hosted CogDB (managed offering)
- 💭 Backup and restore tooling
- 💭 Migration tooling for schema/embedding model changes
- 💭 Monitoring dashboard for self-hosted deployments

### Community 📋
- 📋 Discord or community channel
- 📋 Contributor guide expansion
- 📋 Issue templates and PR templates
- 📋 Roadmap voting (let users prioritize Phase 1+ items)

---

## Decided Against ❌

Things that came up but won't be built:

- ❌ SQL query interface — agents don't speak SQL, this would be human-centric
- ❌ Distributed/sharded architecture for v1 — single-node first, distribution later if needed
- ❌ Custom embedding model training — use sentence-transformers off the shelf
- ❌ GraphQL API — REST + MCP cover the use cases without the complexity

---

## How to contribute to the backlog

1. Open an issue with the label `backlog-suggestion`
2. Describe the use case and why CogDB needs it
3. If accepted, it gets added here with a 💭 status
4. When committed to, it moves to 📋
5. When work starts, it moves to 🚧
6. When merged, it moves to ✅

The maintainer (Mustapha) reviews backlog suggestions weekly.

# Getting Started with CogDB

In this tutorial you'll build a DevOps assistant agent with a complete memory system — episodic (what happened), semantic (what's known), and procedural (how to do things) — in under 30 lines of Python.

## What you'll build

A DevOps agent that:
- Remembers deployment incidents as episodic memories
- Tracks the current state of the system as semantic facts
- Knows how to fix common issues as procedural workflows
- Retrieves relevant context within a token budget so it never overflows the LLM's context window

## Prerequisites

- Python 3.10+
- CogDB installed:

```bash
pip install git+https://github.com/MuLIAICHI/cogdb.git
```

No API key required. CogDB runs entirely on your machine.

---

## Step 1 — Import

```python
from cogdb import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
```

`CognitiveDB` is the single entry point. `MemoryScope` and `MemoryType` are enums you'll use when you need fine-grained control over visibility and retrieval.

---

## Step 2 — Create a database

```python
db = CognitiveDB(db_path="./devops_memory")
```

`db_path` is a directory. CogDB creates it on first use and stores the Rust-backed SQLite engine, HNSW vector index, and WAL log there. Reusing the same path across restarts gives you persistence for free.

For multiple isolated agents sharing one database, you can also pass a config object:

```python
from cogdb.utils.config import CogDBConfig

config = CogDBConfig(
    db_path="./shared_memory",
    default_token_budget=800,
    default_agent_id="devops-agent",
)
db = CognitiveDB(config=config)
```

---

## Step 3 — Store episodic memories (what happened)

Episodic memories are timestamped records of events. Think of them as an agent's diary.

```python
# A high-importance incident
mem_id: str = db.remember(
    "Deployed v2.3 to production. CORS error on /users endpoint — "
    "missing Access-Control-Allow-Origin header in nginx config.",
    agent_id="devops-agent",
    importance=0.9,
    metadata={"service": "api", "env": "production", "error_type": "cors"},
)
# → returns a UUID string like 'f3a1b2c4-...'

# A routine deployment that went fine
db.remember(
    "Deployed v2.2 to staging. All health checks passed.",
    agent_id="devops-agent",
    importance=0.4,
    metadata={"service": "api", "env": "staging"},
)

# A database incident
db.remember(
    "Postgres replication lag hit 45 s on replica-2. "
    "Root cause: long-running analytics query held a lock.",
    agent_id="devops-agent",
    importance=0.85,
    metadata={"service": "postgres", "error_type": "replication_lag"},
)
```

**Importance** (`0.0` to `1.0`) determines retrieval priority. High-importance memories surface first when the token budget is tight.

**Metadata** is a free-form dict attached to the record. You can filter on it during recall or use it for observability. If you call `db.register_schema()` first (Step 9), the metadata is type-validated at write time.

---

## Step 4 — Store semantic facts (what's known)

Semantic memory is a temporal knowledge graph. Use it for facts about the system state: what version is deployed, who owns a service, what the current config is.

```python
# The API is currently on v2.3
db.learn(
    subject="api_service",
    predicate="deployed_version",
    object="v2.3",
    agent_id="devops-agent",
    confidence=1.0,
)

# If you later deploy v2.4, call learn() again with the same subject+predicate.
# CogDB automatically marks v2.3 as superseded — no dangling stale facts.
db.learn(
    subject="api_service",
    predicate="deployed_version",
    object="v2.4",
    agent_id="devops-agent",
)
# The v2.3 fact is now inactive. query_knowledge() returns only v2.4.

# Other system facts
db.learn("api_service", "owner_team", "platform", agent_id="devops-agent")
db.learn("postgres_replica_2", "status", "degraded", agent_id="devops-agent")
db.learn("nginx", "config_path", "/etc/nginx/conf.d/api.conf", agent_id="devops-agent")
```

Query the graph around any entity:

```python
facts = db.query_knowledge("api_service", depth=2)
for f in facts:
    print(f"{f.subject} → {f.predicate} → {f.object}")
# api_service → deployed_version → v2.4
# api_service → owner_team → platform
```

`depth=2` traverses two relationship hops from the entity, so you get both the direct facts and the facts about connected entities.

---

## Step 5 — Store procedural memory (how to do things)

Procedural memory captures reusable workflows. When your agent successfully resolves an incident, encode the solution so future runs can retrieve it.

```python
db.learn_procedure(
    name="fix_cors_error",
    description="Fix missing CORS headers on the API gateway",
    steps=[
        {
            "action": "inspect_nginx_config",
            "tool": "cat /etc/nginx/conf.d/api.conf",
            "expected_output": "nginx config contents",
        },
        {
            "action": "add_cors_headers",
            "tool": "sed -i 's/location \\//location \\/ { add_header Access-Control-Allow-Origin *;/' /etc/nginx/conf.d/api.conf",
        },
        {
            "action": "reload_nginx",
            "tool": "systemctl reload nginx",
            "expected_output": "exit code 0",
            "fallback_action": "systemctl restart nginx",
        },
        {
            "action": "verify",
            "tool": "curl -I https://api.example.com/users",
            "expected_output": "Access-Control-Allow-Origin: *",
        },
    ],
    agent_id="devops-agent",
    applicable_contexts=["cors", "nginx", "api", "headers", "403", "preflight"],
)

db.learn_procedure(
    name="resolve_replication_lag",
    description="Clear replication lag caused by long-running queries on Postgres replicas",
    steps=[
        {"action": "identify_blocking_queries", "tool": "psql -c \"SELECT pid, query FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 5;\""},
        {"action": "terminate_blocker", "tool": "psql -c \"SELECT pg_terminate_backend(<pid>);\""},
        {"action": "monitor_lag", "tool": "psql -c \"SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;\""},
    ],
    agent_id="devops-agent",
    applicable_contexts=["postgres", "replication", "lag", "lock", "replica"],
)
```

`applicable_contexts` is a list of keywords used for fuzzy matching when the agent searches for relevant procedures. Include common error terms, service names, and action verbs.

---

## Step 6 — Retrieve memories

### Recall by query

`recall()` searches across all requested memory types, ranks by importance and semantic similarity, and returns results that fit within the token budget.

```python
memories = db.recall(
    "How do we fix CORS errors on the API?",
    agent_id="devops-agent",
    token_budget=500,
)

for m in memories:
    print(f"[{m.memory_type.value}] (importance={m.importance}) {m.content[:80]}")
# [episodic]   (importance=0.9) Deployed v2.3 to production. CORS error on /users endpoint...
# [episodic]   (importance=0.85) Postgres replication lag hit 45 s...
```

Each item in the list is a `MemoryUnit`. Access `.content` for the text, `.id` for the UUID, `.metadata` for the attached dict.

### Filter by memory type

```python
from cogdb.models import MemoryType

# Only episodic memories about the API service
episodic_only = db.recall(
    "api deployment issues",
    agent_id="devops-agent",
    memory_types=[MemoryType.EPISODIC],
    min_importance=0.7,
    max_results=10,
)
```

### Get progressive context (L0–L3)

`get_context()` builds a structured context object in four tiers, filling each tier until the token budget runs out:

| Level | Contents | Use |
|---|---|---|
| L0 | Agent identity string | Always included |
| L1 | Critical facts (highest importance) | System state summary |
| L2 | Task-relevant memories | Semantic + episodic hits |
| L3 | Deep search results | Exhaustive similarity search |

```python
ctx = db.get_context(
    agent_id="devops-agent",
    level=2,                                        # fill L0 through L2
    task_hint="API gateway is returning 403 on preflight requests",
    token_budget=800,
    identity="DevOps assistant for the platform team",
)

print(f"Tokens used:    {ctx.token_count} / {ctx.token_budget}")
print(f"Budget remaining: {ctx.budget_remaining}")
print(f"Utilization:    {ctx.utilization:.0%}")
print(f"Critical facts: {ctx.critical_facts}")
print(f"Relevant memories ({len(ctx.relevant_memories)}):")
for m in ctx.relevant_memories:
    print(f"  - {m.content[:60]}")
```

### Inject context into an LLM prompt

```python
def build_system_prompt(task: str, agent_id: str) -> str:
    ctx = db.get_context(
        agent_id=agent_id,
        level=2,
        task_hint=task,
        token_budget=600,
        identity="DevOps assistant",
    )
    lines = [f"Identity: {ctx.identity}", ""]
    if ctx.critical_facts:
        lines.append("Known system state:")
        lines.extend(f"  • {f}" for f in ctx.critical_facts)
        lines.append("")
    if ctx.relevant_memories:
        lines.append("Relevant past events:")
        lines.extend(f"  • {m.content}" for m in ctx.relevant_memories)
    return "\n".join(lines)
```

---

## Step 7 — Multi-agent memory scopes

When multiple agents collaborate, they need different levels of memory access. CogDB enforces four formal scopes:

```
PRIVATE    → visible only to the storing agent
TEAM       → visible to agents in the same team_id group
ORGANIZATION → visible to all agents
SESSION    → ephemeral, intended for auto-deletion after a conversation
```

```python
from cogdb.models import MemoryScope

# Private to the devops agent — a working note
db.remember(
    "Nginx config backup saved at /tmp/api.conf.bak",
    agent_id="devops-agent",
    scope=MemoryScope.PRIVATE,
    importance=0.3,
)

# Shared with the whole platform team
db.remember(
    "Production incident 2026-06-06: API CORS error resolved in 12 min. "
    "Fix: added Access-Control-Allow-Origin header to nginx.",
    agent_id="devops-agent",
    scope=MemoryScope.TEAM,
    team_id="platform",
    importance=0.9,
)

# Visible to all agents in the org
db.learn(
    subject="incident_2026_06_06",
    predicate="status",
    object="resolved",
    agent_id="devops-agent",
)
# Semantic facts default to the agent's scope — use scope arg on remember()
# for org-wide episodic memories

# Now a monitoring agent on the same db can read the team-scoped incident
monitoring_memories = db.recall(
    "recent incidents",
    agent_id="monitoring-agent",
    # team-scoped memories from devops-agent ARE visible here
    # because monitoring-agent is in the same team_id group
)
```

Scope enforcement runs inside the Rust engine — there is no unguarded query path.

---

## Step 8 — Forgetting memories

Delete a specific memory by its UUID and type:

```python
# Forget an episodic memory
was_deleted: bool = db.forget(mem_id, MemoryType.EPISODIC)
print(was_deleted)  # True

# Forget a semantic fact — use the UUID returned by learn()
triple_id = db.learn("api_service", "status", "degraded", agent_id="devops-agent")
db.forget(triple_id, MemoryType.SEMANTIC)
```

To supersede a semantic fact without deleting it (preserving history), just call `db.learn()` again with a different object for the same subject+predicate. The old fact is marked inactive but remains queryable with `active_only=False`.

---

## Step 9 — Schema validation

For production agents you can enforce typed metadata schemas. Once registered, every `remember()` call for that agent validates the metadata and raises `SchemaValidationError` on mismatch.

```python
from cogdb.schema import MetadataSchema, FieldSchema, SchemaValidationError

db.register_schema(MetadataSchema(
    agent_id="devops-agent",
    fields={
        "service":    FieldSchema(type="str", required=True, description="Service name"),
        "env":        FieldSchema(type="str", required=True, description="Environment: production|staging|dev"),
        "error_type": FieldSchema(type="str", required=False),
        "exit_code":  FieldSchema(type="int", required=False, default=0),
    },
))

# This will pass
db.remember(
    "Deployment completed",
    agent_id="devops-agent",
    metadata={"service": "api", "env": "production"},
)

# This will raise SchemaValidationError — 'service' is required
try:
    db.remember("Deployment failed", agent_id="devops-agent", metadata={})
except SchemaValidationError as e:
    print(e.errors)
    # ["metadata.service: required field missing"]
```

You can evolve the schema with versioned migrations:

```python
from cogdb.schema import SchemaMigration, FieldSchema

migration = (
    SchemaMigration(agent_id="devops-agent", from_version=1, to_version=2)
    .add_field("priority", FieldSchema(type="int", default=0), default=0)
    .rename_field("error_type", "error_category")
)
db.migrate_schema(migration)
```

---

## Step 10 — Check stats

```python
counts = db.stats()
print(counts)
# {'episodic': 3, 'semantic': 6, 'procedural': 2, 'total': 11}
```

---

## Complete example

A self-contained, runnable script — no API keys, no external services:

```python
from cogdb import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.schema import FieldSchema, MetadataSchema


def build_devops_agent_memory(db_path: str = "./devops_demo") -> CognitiveDB:
    """Initialise a CognitiveDB instance with DevOps agent memory."""
    db = CognitiveDB(db_path=db_path)

    # Optional: enforce typed metadata
    db.register_schema(MetadataSchema(
        agent_id="devops-agent",
        fields={
            "service":    FieldSchema(type="str", required=True),
            "env":        FieldSchema(type="str", required=False),
            "error_type": FieldSchema(type="str", required=False),
        },
    ))

    # Episodic — incidents
    db.remember(
        "Deployed v2.3. CORS error on /users — missing Access-Control-Allow-Origin.",
        agent_id="devops-agent",
        importance=0.9,
        metadata={"service": "api", "env": "production", "error_type": "cors"},
    )
    db.remember(
        "Postgres replica-2 replication lag hit 45 s. Caused by analytics query lock.",
        agent_id="devops-agent",
        importance=0.85,
        metadata={"service": "postgres", "env": "production", "error_type": "replication_lag"},
    )
    db.remember(
        "Staging deploy v2.2 — all health checks passed.",
        agent_id="devops-agent",
        importance=0.3,
        metadata={"service": "api", "env": "staging"},
    )

    # Semantic — current system state
    db.learn("api_service", "deployed_version", "v2.3", agent_id="devops-agent")
    db.learn("api_service", "owner_team", "platform", agent_id="devops-agent")
    db.learn("nginx", "config_path", "/etc/nginx/conf.d/api.conf", agent_id="devops-agent")
    db.learn("postgres_replica_2", "status", "degraded", agent_id="devops-agent")

    # Procedural — known fixes
    db.learn_procedure(
        name="fix_cors_error",
        description="Fix missing CORS headers on the API nginx config",
        steps=[
            {"action": "inspect_config", "tool": "cat /etc/nginx/conf.d/api.conf"},
            {"action": "add_cors_header", "tool": "sed -i ... /etc/nginx/conf.d/api.conf"},
            {"action": "reload_nginx", "tool": "systemctl reload nginx",
             "fallback_action": "systemctl restart nginx"},
            {"action": "verify", "tool": "curl -I https://api.example.com/users"},
        ],
        agent_id="devops-agent",
        applicable_contexts=["cors", "nginx", "api", "preflight", "headers"],
    )
    db.learn_procedure(
        name="resolve_replication_lag",
        description="Terminate blocking queries to clear Postgres replication lag",
        steps=[
            {"action": "find_blocker", "tool": "psql -c 'SELECT pid, query FROM pg_stat_activity ...'"},
            {"action": "terminate", "tool": "psql -c 'SELECT pg_terminate_backend(<pid>);'"},
            {"action": "confirm_lag_cleared", "tool": "psql -c 'SELECT now() - pg_last_xact_replay_timestamp();'"},
        ],
        agent_id="devops-agent",
        applicable_contexts=["postgres", "replication", "lag", "replica"],
    )

    return db


def respond_to_incident(db: CognitiveDB, incident: str, agent_id: str = "devops-agent") -> None:
    """Retrieve context for a new incident and print what the agent knows."""
    ctx = db.get_context(
        agent_id=agent_id,
        level=2,
        task_hint=incident,
        token_budget=600,
        identity="DevOps assistant for the platform team",
    )

    print(f"\nIncident: {incident}")
    print(f"Tokens loaded: {ctx.token_count}/{ctx.token_budget} ({ctx.utilization:.0%})")

    if ctx.critical_facts:
        print("\nKnown system state:")
        for fact in ctx.critical_facts:
            print(f"  • {fact}")

    if ctx.relevant_memories:
        print("\nRelevant past events:")
        for m in ctx.relevant_memories:
            print(f"  [{m.memory_type.value}] {m.content[:90]}")


if __name__ == "__main__":
    db = build_devops_agent_memory()

    respond_to_incident(db, "API gateway returning 403 on preflight CORS requests")
    respond_to_incident(db, "Postgres replica falling behind, reads are slow")

    print("\nMemory stats:", db.stats())
    # Memory stats: {'episodic': 3, 'semantic': 4, 'procedural': 2, 'total': 9}
```

---

## What's next

- **[API Reference](./api.md)** — complete method signatures and return types
- **[Migrating from Mem0](./tutorial_migrating_from_mem0.md)** — side-by-side comparison for Mem0 users
- **[Framework Adapters](../cogdb/adapters/)** — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Semantic Kernel, LlamaIndex
- **[MCP Server](../cogdb/adapters/mcp.py)** — connect CogDB to Claude Code, Cursor, or any MCP host via `cogdb-mcp --db-path ./memory`

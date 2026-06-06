# Migrating from Mem0 to CogDB

CogDB is a drop-in upgrade if you're using Mem0 for agent memory. This guide walks through the concept mapping, shows side-by-side code, and includes a migration script you can run against a live Mem0 instance.

---

## Why migrate?

| Capability | Mem0 | CogDB |
|---|:---:|:---:|
| Episodic / event memory | ✓ | ✓ |
| Semantic / knowledge graph | ✓ | ✓ |
| Procedural memory (reusable workflows) | ✗ | ✓ |
| Token-aware retrieval | ✗ | ✓ |
| Multi-agent memory scopes | user-scoped | 4 formal scopes |
| Works without an API key | ✗ | ✓ |
| Fully local / on-premise | ✗ (cloud default) | ✓ |
| Schema validation on writes | ✗ | ✓ |
| Temporal fact lifecycle (supersedes) | partial | ✓ |
| Framework adapters | partial | AutoGen, LangGraph, CrewAI, OpenAI, SK, LlamaIndex |

The two biggest practical differences:

1. **Mem0 requires an OpenAI API key by default** for its LLM-driven entity extraction. CogDB runs fully offline — sentence-transformers handles embeddings locally, no API calls in the hot path.
2. **Mem0 is a managed cloud service** (with a self-hosted option that still requires Qdrant). CogDB is a single Python package backed by a purpose-built Rust engine — no external database to provision.

---

## Concept mapping

### `Memory.add()` → `db.remember()`

Mem0's `add()` accepts a string or a list of message dicts and stores everything as memories. CogDB's `remember()` stores a single episodic event with explicit metadata.

```python
# Mem0
from mem0 import Memory
m = Memory()
m.add("The API is returning 403 on preflight requests", user_id="devops-agent-1")

# CogDB
from cogdb import CognitiveDB
db = CognitiveDB(db_path="./memory")
db.remember(
    "The API is returning 403 on preflight requests",
    agent_id="devops-agent-1",
    importance=0.8,
)
```

### `Memory.search()` → `db.recall()`

```python
# Mem0
results = m.search("API errors", user_id="devops-agent-1")
for r in results:
    print(r["memory"])   # Mem0 returns dicts

# CogDB
memories = db.recall("API errors", agent_id="devops-agent-1", token_budget=500)
for m in memories:
    print(m.content)     # CogDB returns MemoryUnit objects
```

> **Gotcha:** `db.recall()` returns `list[MemoryUnit]`, not `list[dict]`. Use `.content` to get the string, `.id` for the UUID, `.importance` for the score, `.metadata` for attached data.

### Sessions → `MemoryScope.SESSION`

Mem0 uses `session_id` as a separate query parameter. CogDB models scope as a first-class enum on every memory:

```python
# Mem0 — session-scoped memory
m.add("User asked about dark mode", user_id="ui-agent", session_id="sess-001")
results = m.search("dark mode", user_id="ui-agent", session_id="sess-001")

# CogDB — equivalent
from cogdb.models import MemoryScope
db.remember(
    "User asked about dark mode",
    agent_id="ui-agent",
    scope=MemoryScope.SESSION,
    importance=0.4,
)
# Retrieve — SESSION-scoped memories are visible to the same agent_id
results = db.recall("dark mode", agent_id="ui-agent")
```

The four CogDB scopes map to Mem0 concepts as follows:

| Mem0 | CogDB equivalent |
|---|---|
| `user_id` only | `MemoryScope.PRIVATE` |
| `session_id` | `MemoryScope.SESSION` |
| Shared across agents in the same app | `MemoryScope.TEAM` (+ `team_id`) |
| Org-wide knowledge | `MemoryScope.ORGANIZATION` |

### Entity extraction → `db.learn()`

Mem0 calls an LLM to extract entities and relationships automatically when you call `add()`. CogDB separates episodic storage from semantic knowledge — entities go in the knowledge graph explicitly via `db.learn()`.

```python
# Mem0 — entities extracted automatically (requires OpenAI key)
m.add("Alice works at Acme Corp in the engineering department", user_id="agent-1")
# Mem0 internally creates: Alice → works_at → Acme Corp, Alice → department → engineering

# CogDB — explicit (no API key required)
db.remember(
    "Alice works at Acme Corp in the engineering department",
    agent_id="agent-1",
)
db.learn("alice", "works_at", "Acme Corp", agent_id="agent-1", confidence=1.0)
db.learn("alice", "department", "engineering", agent_id="agent-1", confidence=1.0)
```

The trade-off: CogDB is more verbose but more predictable. You control exactly what goes into the knowledge graph. Facts have explicit confidence scores and validity windows, and calling `learn()` again with a new value for the same `(subject, predicate)` pair automatically supersedes the old one — no stale facts.

```python
# The API was on v2.3, now it's v2.4 — CogDB handles supersession automatically
db.learn("api_service", "version", "v2.3", agent_id="agent-1")
db.learn("api_service", "version", "v2.4", agent_id="agent-1")

facts = db.query_knowledge("api_service")
# → only the v2.4 fact is returned (active_only=True by default)
```

### `Memory.get_all()` → `db.recall()` with high budget

Mem0 has a `get_all()` method that dumps every memory for a user. CogDB doesn't have an exact equivalent — retrieval is always query-driven. To approximate `get_all()`, pass a large `token_budget` and `max_results`:

```python
# Mem0
all_memories = m.get_all(user_id="agent-1")

# CogDB
all_memories = db.recall(
    query="",           # empty query returns by importance order
    agent_id="agent-1",
    token_budget=100_000,
    max_results=10_000,
)
```

### `Memory.delete()` → `db.forget()`

```python
# Mem0
m.delete(memory_id="abc123")

# CogDB — must specify the memory type
from cogdb.models import MemoryType
db.forget("abc123", MemoryType.EPISODIC)
```

---

## Migration script

The script below exports all memories from a Mem0 instance and imports them into CogDB.

```python
from __future__ import annotations

from cogdb import CognitiveDB
from cogdb.models import MemoryScope


def migrate_from_mem0(
    mem0_memory,
    cogdb_db: CognitiveDB,
    user_id: str,
    agent_id: str,
    default_importance: float = 0.7,
) -> dict[str, int]:
    """Migrate all memories from a Mem0 instance to CogDB.

    Args:
        mem0_memory: An initialised Mem0 Memory or MemoryClient instance.
        cogdb_db: An initialised CognitiveDB instance.
        user_id: The Mem0 user_id to export from.
        agent_id: The CogDB agent_id to import into.
        default_importance: Importance to assign migrated memories (0.0–1.0).

    Returns:
        Dict with 'migrated' and 'failed' counts.

    Example:
        >>> from mem0 import Memory
        >>> from cogdb import CognitiveDB
        >>> m = Memory()
        >>> db = CognitiveDB(db_path="./memory")
        >>> result = migrate_from_mem0(m, db, user_id="alice", agent_id="alice-agent")
        >>> print(f"Migrated {result['migrated']} memories")
    """
    memories = mem0_memory.get_all(user_id=user_id)

    migrated = 0
    failed = 0

    for mem in memories:
        try:
            content = mem.get("memory") or mem.get("content", "")
            if not content:
                continue

            cogdb_db.remember(
                content=content,
                agent_id=agent_id,
                importance=default_importance,
                scope=MemoryScope.PRIVATE,
                metadata={
                    "migrated_from_mem0": True,
                    "mem0_id": mem.get("id"),
                    "mem0_user_id": user_id,
                },
            )
            migrated += 1
        except Exception as exc:
            print(f"Failed to migrate memory {mem.get('id')}: {exc}")
            failed += 1

    return {"migrated": migrated, "failed": failed}
```

Run it:

```python
from mem0 import Memory
from cogdb import CognitiveDB

m = Memory()
db = CognitiveDB(db_path="./migrated_memory")

result = migrate_from_mem0(m, db, user_id="your-user-id", agent_id="your-agent-id")
print(result)
# {'migrated': 142, 'failed': 0}
```

> **Note:** The migration script copies episodic content only. Mem0's extracted entities are not transferred as CogDB semantic triples because Mem0 does not expose those in a structured form through `get_all()`. If you need the entity graph migrated, extract it manually from Mem0's entity endpoints and call `db.learn()` for each triple.

---

## Complete before/after example

The same customer support agent, implemented first in Mem0, then in CogDB.

### Mem0 version

```python
from mem0 import Memory

m = Memory()
AGENT = "support-agent"

# Store a ticket resolution
m.add(
    "Customer #4821 reported login failures after password reset. "
    "Root cause: token cache not invalidated. Fixed by flushing Redis.",
    user_id=AGENT,
)

# Store product knowledge
m.add("Premium plan includes 100 GB storage and priority support", user_id=AGENT)
m.add("Refunds are processed within 5 business days", user_id=AGENT)

# Retrieve for a new ticket
def handle_ticket(issue: str) -> list[dict]:
    return m.search(issue, user_id=AGENT, limit=5)

results = handle_ticket("customer can't log in after changing password")
for r in results:
    print(r["memory"])
```

### CogDB version

The CogDB version adds: procedural memory for common resolutions, token-aware context loading, and schema validation.

```python
from cogdb import CognitiveDB
from cogdb.models import ContextResponse, MemoryScope, MemoryType
from cogdb.schema import FieldSchema, MetadataSchema

db = CognitiveDB(db_path="./support_memory")
AGENT = "support-agent"

# Enforce typed metadata on every write
db.register_schema(MetadataSchema(
    agent_id=AGENT,
    fields={
        "ticket_id":   FieldSchema(type="str", required=False),
        "category":    FieldSchema(type="str", required=False),
        "resolved":    FieldSchema(type="bool", required=False, default=False),
    },
))

# Episodic — ticket resolutions
db.remember(
    "Customer #4821 reported login failures after password reset. "
    "Root cause: token cache not invalidated. Fixed by flushing Redis.",
    agent_id=AGENT,
    importance=0.9,
    metadata={"ticket_id": "4821", "category": "auth", "resolved": True},
)

# Semantic — product facts that don't change often
db.learn("premium_plan", "storage_gb", "100", agent_id=AGENT, confidence=1.0)
db.learn("premium_plan", "support_tier", "priority", agent_id=AGENT, confidence=1.0)
db.learn("refund_policy", "processing_days", "5", agent_id=AGENT, confidence=1.0)

# Procedural — reusable fix for auth issues
db.learn_procedure(
    name="fix_post_reset_login_failure",
    description="Resolve login failures after password reset by flushing the auth token cache",
    steps=[
        {"action": "identify_user", "tool": "lookup user in admin panel"},
        {"action": "flush_token_cache", "tool": "redis-cli DEL auth:token:<user_id>"},
        {"action": "confirm_login", "tool": "ask customer to retry"},
    ],
    agent_id=AGENT,
    applicable_contexts=["login", "password reset", "auth", "token", "cache"],
)


def handle_ticket(issue: str) -> ContextResponse:
    """Return structured context for a support ticket."""
    return db.get_context(
        agent_id=AGENT,
        level=2,
        task_hint=issue,
        token_budget=600,
        identity="Customer support assistant",
    )


ctx = handle_ticket("customer can't log in after changing password")

print(f"Tokens: {ctx.token_count}/{ctx.token_budget}")
print("Relevant past resolutions:")
for m in ctx.relevant_memories:
    print(f"  • {m.content[:80]}")
```

The CogDB version surfaces the resolved ticket (`importance=0.9`) at the top of the context, includes the `fix_post_reset_login_failure` procedure in the retrieval results, and respects the 600-token ceiling — no prompt bloat.

---

## Configuration differences

| Setting | Mem0 | CogDB |
|---|---|---|
| API key | `OPENAI_API_KEY` required | Not required |
| Storage backend | Qdrant (vector) + Supabase (graph) | Built-in Rust engine (HNSW + SQLite) |
| Hosting | Cloud (mem0.ai) or self-hosted | Fully local |
| Config | `config` dict at init | `CogDBConfig` dataclass |

```python
# Mem0 — requires environment variable
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
from mem0 import Memory
m = Memory()

# CogDB — no environment variables needed
from cogdb import CognitiveDB
from cogdb.utils.config import CogDBConfig

db = CognitiveDB(config=CogDBConfig(
    db_path="./memory",
    default_token_budget=800,
    default_agent_id="my-agent",
))
```

---

## Gotchas

**`recall()` returns objects, not dicts.**
Mem0's `search()` returns `list[dict]` with a `"memory"` key. CogDB returns `list[MemoryUnit]`. Use `.content` to get the string:

```python
# Mem0
for r in m.search("api errors", user_id="agent"):
    print(r["memory"])

# CogDB
for unit in db.recall("api errors", agent_id="agent"):
    print(unit.content)   # not unit["memory"]
```

**Importance scores are on a 0–1 scale, not Mem0's relevance score.**
Mem0 returns a `score` field from vector search (cosine similarity, higher is better). CogDB's `importance` field is set at write time by the caller. Both affect ranking, but they mean different things. When migrating, assigning `importance=0.7` as a baseline is reasonable — you can adjust individual memories after import.

**CogDB does not auto-extract entities.**
Mem0's key feature is LLM-driven entity extraction from every `add()` call. CogDB deliberately separates this: `remember()` stores the raw episodic text, and `learn()` stores explicit semantic triples. This means more code, but also no API calls, no extraction errors, and full control over what enters the knowledge graph.

If you want auto-extraction, you can add it at the application layer:

```python
def remember_with_extraction(db: CognitiveDB, text: str, agent_id: str) -> None:
    """Store text as episodic memory and extract entities with your own LLM call."""
    db.remember(text, agent_id=agent_id)
    # Call your own LLM to extract (subject, predicate, object) triples from text,
    # then store each with db.learn(subject, predicate, object, agent_id=agent_id)
```

**Semantic facts require explicit supersession awareness.**
In Mem0, updating a fact is done by calling `update(memory_id, data)`. In CogDB, you call `learn()` with the same `subject` + `predicate` and a new `object`. The old triple is automatically marked inactive. If you need the old value for auditing, query with `active_only=False`:

```python
all_versions = db.query_knowledge("api_service", depth=1, active_only=False)
for f in all_versions:
    status = "active" if f.is_active else "superseded"
    print(f"  [{status}] {f.subject} → {f.predicate} → {f.object}")
```

**`forget()` requires the memory type.**
Mem0's `delete(memory_id)` works on any memory. CogDB stores the three memory types in separate stores, so `forget()` needs to know which one:

```python
db.forget(some_id, MemoryType.EPISODIC)    # episodic store
db.forget(some_id, MemoryType.SEMANTIC)    # knowledge graph
db.forget(some_id, MemoryType.PROCEDURAL)  # procedure store
```

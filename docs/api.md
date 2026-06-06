# CogDB API Reference

> v0.4.0 — Last updated: 2026-06-06

## Contents

- [CognitiveDB](#cognitivedb) — main entry point
- [Memory Operations](#memory-operations)
- [Retrieval](#retrieval)
- [Context Assembly](#context-assembly)
- [Semantic Memory](#semantic-memory)
- [Procedural Memory](#procedural-memory)
- [Schema System](#schema-system)
- [Schema Migration](#schema-migration)
- [Configuration](#configuration)
- [Data Models](#data-models)
- [Memory Scopes](#memory-scopes)

---

## CognitiveDB

```python
class cogdb.CognitiveDB(config=None, db_path=None)
```

The single entry point for all CogDB operations. Composes episodic, semantic, and procedural stores with a token-cost-aware retrieval pipeline and multi-agent memory scopes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `CogDBConfig \| None` | `None` | Full configuration object. Uses defaults when omitted. |
| `db_path` | `str \| None` | `None` | Shortcut to set `config.db_path`. Overrides the path in `config` if both are provided. |

**Example:**

```python
from cogdb import CognitiveDB

# Minimal — uses all defaults
db = CognitiveDB(db_path="./my_agent_memory")

# Full config
from cogdb.utils.config import CogDBConfig
config = CogDBConfig(db_path="./my_agent_memory", default_token_budget=2000)
db = CognitiveDB(config=config)
```

**Direct store access** (advanced use):

```python
db.episodic    # → EpisodicStore
db.semantic    # → SemanticStore
db.procedural  # → ProceduralStore
```

---

## Memory Operations

### `remember(content, agent_id=None, importance=0.5, scope=MemoryScope.PRIVATE, metadata=None, memory_type=MemoryType.EPISODIC, team_id=None) → str`

Store an episodic memory. Returns the UUID of the stored record.

If a metadata schema is registered for this agent, the `metadata` dict is validated before storage. Raises `SchemaValidationError` on violation when `strict_metadata_validation=True`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | — | Text content to store. |
| `agent_id` | `str \| None` | `None` | Agent storing the memory. Falls back to `CogDBConfig.default_agent_id`. |
| `importance` | `float` | `0.5` | Importance score, `0.0` to `1.0`. Higher values surface this memory first in retrieval. |
| `scope` | `MemoryScope` | `PRIVATE` | Visibility scope for multi-agent access. |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata. Validated against registered schema if one exists. |
| `memory_type` | `MemoryType` | `EPISODIC` | Override the memory type. |
| `team_id` | `str \| None` | `None` | Team identifier, required when `scope=MemoryScope.TEAM`. |

**Returns:** `str` — UUID of the stored memory.

**Raises:** `SchemaValidationError` if metadata violates the registered schema and `strict_metadata_validation=True`.

**Example:**

```python
memory_id = db.remember(
    "Deployment failed due to missing env var DB_HOST",
    agent_id="devops-agent",
    importance=0.9,
    scope=MemoryScope.TEAM,
    team_id="backend-team",
    metadata={"error_type": "config", "service": "api"},
)
```

---

### `forget(memory_id, memory_type) → bool`

Delete a specific memory by ID.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_id` | `str` | — | UUID of the memory to delete. |
| `memory_type` | `MemoryType` | — | Which store to delete from (`EPISODIC`, `SEMANTIC`, or `PROCEDURAL`). |

**Returns:** `bool` — `True` if the record was found and deleted.

**Example:**

```python
deleted = db.forget(memory_id, MemoryType.EPISODIC)
```

---

### `stats() → dict[str, int]`

Get memory counts across all stores.

**Returns:** `dict[str, int]` with keys `episodic`, `semantic`, `procedural`, and `total`.

**Example:**

```python
>>> db.stats()
{'episodic': 142, 'semantic': 58, 'procedural': 12, 'total': 212}
```

---

## Retrieval

### `recall(query, agent_id=None, token_budget=None, memory_types=None, scope_filter=None, min_importance=0.0, max_results=20) → list[MemoryUnit]`

Recall memories matching a natural language query within a token budget.

Searches across all requested memory types, blends HNSW vector similarity with importance scores, and returns the highest-ranked results that fit within the token budget.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | Natural language query. |
| `agent_id` | `str \| None` | `None` | The querying agent. Scoped to private + team/org memories accessible to this agent. |
| `token_budget` | `int \| None` | `None` | Max tokens in the response. Falls back to `CogDBConfig.default_token_budget`. |
| `memory_types` | `list[MemoryType] \| None` | `None` | Which memory types to search. Defaults to `[EPISODIC, SEMANTIC]`. |
| `scope_filter` | `MemoryScope \| None` | `None` | Restrict results to a specific scope. |
| `min_importance` | `float` | `0.0` | Minimum importance threshold. Records below this score are excluded. |
| `max_results` | `int` | `20` | Max candidate results fetched per store before budget trimming. |

**Returns:** `list[MemoryUnit]` — sorted by effective importance (importance × decay), trimmed to fit the token budget.

**Example:**

```python
memories = db.recall(
    "deployment errors this week",
    agent_id="devops-agent",
    token_budget=500,
    memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
    min_importance=0.4,
)
for m in memories:
    print(f"[{m.importance:.2f}] {m.content}")
```

---

## Context Assembly

### `get_context(agent_id=None, level=2, task_hint=None, token_budget=None, identity=None) → ContextResponse`

Build progressive context for an agent, loading memory tiers from L0 (identity) through L3 (deep search) up to the requested level and token budget.

| Level | Name | Content |
|-------|------|---------|
| L0 | Identity | Agent identity string (always included) |
| L1 | Critical facts | High-importance recent memories |
| L2 | Task-relevant | Semantic search against `task_hint` |
| L3 | Deep search | Exhaustive search, used only when budget allows |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str \| None` | `None` | The agent requesting context. |
| `level` | `int` | `2` | Maximum context level to load (`0`–`3`). |
| `task_hint` | `str \| None` | `None` | Current task description, used for L2/L3 relevance search. |
| `token_budget` | `int \| None` | `None` | Override the default token budget. |
| `identity` | `str \| None` | `None` | Agent identity string injected at L0. |

**Returns:** `ContextResponse` — structured tiered context (see [ContextResponse](#contextresponse)).

**Example:**

```python
ctx = db.get_context(
    agent_id="ui-agent",
    level=2,
    task_hint="redesigning the settings page",
    token_budget=800,
)
print(f"Loaded {len(ctx.critical_facts)} critical facts")
print(f"Loaded {len(ctx.relevant_memories)} relevant memories")
print(f"Budget used: {ctx.token_count}/{ctx.token_budget} tokens")
```

---

## Semantic Memory

### `learn(subject, predicate, object, agent_id=None, confidence=1.0, valid_from=None, source_episodes=None) → str`

Store a fact as a `(subject, predicate, object)` triple in the knowledge graph.

If an existing triple with the same `subject` + `predicate` already exists for this agent, the old triple is **superseded** (its `valid_until` is set to now). The new triple becomes the active fact. This implements automatic contradiction resolution.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subject` | `str` | — | The entity this fact is about. |
| `predicate` | `str` | — | The relationship or property name. |
| `object` | `str` | — | The value or target entity. |
| `agent_id` | `str \| None` | `None` | Agent asserting this fact. |
| `confidence` | `float` | `1.0` | Confidence score, `0.0` to `1.0`. |
| `valid_from` | `datetime \| None` | `None` | When this fact becomes valid. Defaults to now. |
| `source_episodes` | `list[str] \| None` | `None` | UUIDs of episodic memories that support this fact. |

**Returns:** `str` — UUID of the stored triple.

**Example:**

```python
# Assert a fact
db.learn(
    subject="api_service",
    predicate="deployed_version",
    object="v2.3.1",
    agent_id="devops-agent",
    confidence=1.0,
)

# Calling learn() again with the same subject+predicate supersedes the old fact
db.learn("api_service", "deployed_version", "v2.4.0", agent_id="devops-agent")
```

---

### `query_knowledge(entity, depth=1, active_only=True) → list[SemanticTriple]`

Traverse the knowledge graph starting from an entity.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | `str` | — | Starting entity for the BFS traversal. |
| `depth` | `int` | `1` | Number of relationship hops to follow. |
| `active_only` | `bool` | `True` | When `True`, returns only currently valid (non-superseded) facts. |

**Returns:** `list[SemanticTriple]` — all triples reachable within `depth` hops.

**Example:**

```python
facts = db.query_knowledge("api_service", depth=2)
for f in facts:
    print(f"{f.subject} —[{f.predicate}]→ {f.object}  (confidence={f.confidence})")
```

---

## Procedural Memory

### `learn_procedure(name, steps, agent_id=None, description="", success_rate=1.0, source_episodes=None, applicable_contexts=None) → str`

Store a learned workflow template. Procedures are reusable step-by-step plans that agents can recall when facing similar tasks.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Procedure name (unique per agent). |
| `steps` | `list[dict]` | — | Ordered steps. Each dict may contain `action` (str), `tool` (str), `parameters` (dict), `expected_output` (str), `fallback_action` (str). |
| `agent_id` | `str \| None` | `None` | Agent that learned this procedure. |
| `description` | `str` | `""` | Human-readable description. |
| `success_rate` | `float` | `1.0` | Initial success rate, `0.0` to `1.0`. Updated via EMA after each execution. |
| `source_episodes` | `list[str] \| None` | `None` | UUIDs of episodes this was extracted from. |
| `applicable_contexts` | `list[str] \| None` | `None` | Keywords describing when to apply this procedure. Used for token-budget-aware recall. |

**Returns:** `str` — UUID of the stored procedure.

**Example:**

```python
proc_id = db.learn_procedure(
    name="fix_cors_error",
    description="Fix CORS errors in the API gateway",
    steps=[
        {"action": "check_config", "tool": "cat nginx.conf"},
        {"action": "add_headers", "tool": "sed -i"},
        {"action": "restart", "tool": "systemctl restart nginx"},
        {"action": "verify", "tool": "curl -I https://api.example.com"},
    ],
    agent_id="devops-agent",
    applicable_contexts=["cors", "api", "nginx", "headers"],
)
```

---

## Schema System

### `register_schema(schema) → None`

Register or update the typed metadata schema for an agent.

Once registered, every `remember()` call for this agent validates the provided `metadata` dict against the schema. Re-registering for the same `agent_id` overwrites the existing schema and increments the version counter.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `MetadataSchema` | — | Schema to register. |

**Returns:** `None`

**Example:**

```python
from cogdb.schema import MetadataSchema, FieldSchema

db.register_schema(MetadataSchema(
    agent_id="devops-agent",
    fields={
        "tool":      FieldSchema(type="str", required=True, description="CLI tool used"),
        "exit_code": FieldSchema(type="int", required=False, default=0),
        "service":   FieldSchema(type="str", required=False),
    },
))

# This will now validate metadata on every remember() for "devops-agent"
db.remember("task done", agent_id="devops-agent", metadata={"tool": "bash", "exit_code": 0})
```

---

### `get_schema(agent_id) → MetadataSchema | None`

Retrieve the registered metadata schema for an agent.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | — | Agent whose schema to retrieve. |

**Returns:** `MetadataSchema` if registered, `None` otherwise.

**Example:**

```python
schema = db.get_schema("devops-agent")
if schema:
    print(f"v{schema.version}: {list(schema.fields)}")
```

---

### `list_schemas() → list[MetadataSchema]`

Return all registered metadata schemas, sorted by `agent_id`.

**Returns:** `list[MetadataSchema]`

**Example:**

```python
for s in db.list_schemas():
    print(f"{s.agent_id}  v{s.version}  fields={list(s.fields)}")
```

---

### `migrate_schema(migration) → MetadataSchema`

Apply a `SchemaMigration` to an agent's registered schema. Validates version continuity and applies each `FieldChange` in order.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `migration` | `SchemaMigration` | — | Migration to apply. |

**Returns:** `MetadataSchema` — the updated schema (already persisted).

**Raises:**
- `ValueError` if `from_version` doesn't match the current schema version.
- `ValueError` if no schema is registered for the agent.
- `ValueError` if a field constraint is violated (e.g. dropping a required field, adding a duplicate).

**Example:**

```python
from cogdb.schema.migration import SchemaMigration
from cogdb.schema import FieldSchema

migration = (
    SchemaMigration(agent_id="devops-agent", from_version=1, to_version=2)
    .add_field("priority", FieldSchema(type="int", default=0), default=0)
    .rename_field("tags", "labels")
    .drop_field("deprecated_field")
)
new_schema = db.migrate_schema(migration)
print(new_schema.version)  # 2
```

---

## Schema Migration

### `class SchemaMigration(agent_id, from_version, to_version, changes=[], description="")`

A versioned, chainable migration descriptor for a single agent's metadata schema. Build one by chaining its builder methods, then pass it to `db.migrate_schema()` or `SchemaMigrator.apply()`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Agent whose schema is being migrated. |
| `from_version` | `int` | Schema version this migration starts from. Must match the current version. |
| `to_version` | `int` | Schema version this migration produces. |
| `changes` | `list[FieldChange]` | Ordered list of field operations (appended by builder methods). |
| `description` | `str` | Human-readable summary. |
| `created_at` | `datetime` | Timestamp set at construction. |

**Builder methods (all return `self` for chaining):**

#### `.add_field(name, schema, default=None) → SchemaMigration`

Add a new field to the schema.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the new field. |
| `schema` | `FieldSchema` | Definition of the new field. |
| `default` | `Any` | Value injected into existing records that lack this field. |

#### `.rename_field(old_name, new_name) → SchemaMigration`

Rename an existing field.

#### `.drop_field(name) → SchemaMigration`

Remove an optional field. Raises `ValueError` if the field is `required=True`.

#### `.change_type(name, new_schema) → SchemaMigration`

Replace the `FieldSchema` for an existing field.

**Full chaining example:**

```python
from cogdb.schema.migration import SchemaMigration
from cogdb.schema import FieldSchema

migration = (
    SchemaMigration(
        agent_id="my-agent",
        from_version=1,
        to_version=2,
        description="Add priority field, rename tags → labels",
    )
    .add_field("priority", FieldSchema(type="int", default=0), default=0)
    .rename_field("tags", "labels")
    .change_type("score", FieldSchema(type="float", required=False))
)
new_schema = db.migrate_schema(migration)
```

---

### `class SchemaMigrator(registry)`

Low-level migrator that operates directly on a `SchemaRegistry`. Use `db.migrate_schema()` for the high-level path; use `SchemaMigrator` directly when you need `migrate_metadata()` or `plan()`.

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `registry` | `SchemaRegistry` | The registry to read and write schemas. |

---

#### `SchemaMigrator.apply(migration) → MetadataSchema`

Apply a migration to the stored schema and persist the result.

**Raises:** `ValueError` for version mismatches, missing agents, and constraint violations (see `db.migrate_schema()`).

**Example:**

```python
from cogdb.schema.registry import SchemaRegistry
from cogdb.schema.migration import SchemaMigrator, SchemaMigration

registry = SchemaRegistry("./my_db", strict=True)
migrator = SchemaMigrator(registry)
new_schema = migrator.apply(migration)
```

---

#### `SchemaMigrator.migrate_metadata(migration, records) → list[dict]`

Backfill a list of raw metadata dicts to match the migration's changes. Returns a new list; originals are not mutated. Call after `apply()` to bring existing memory metadata in line with the new schema.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `migration` | `SchemaMigration` | The migration whose changes to replay. |
| `records` | `list[dict]` | Raw metadata dicts to update. |

**Returns:** `list[dict]` — updated copies of the input records.

**Example:**

```python
old_records = [{"tags": "deploy"}, {"tags": "test", "score": 0.8}]
updated = migrator.migrate_metadata(migration, old_records)
# old "tags" key is now "labels"; "priority" is added with default 0
```

---

#### `SchemaMigrator.plan(migration) → list[str]`

Return a human-readable dry-run description of what the migration will do. Useful for previewing changes before applying.

**Returns:** `list[str]` — one line per operation.

**Example:**

```python
for line in migrator.plan(migration):
    print(line)
# Migration: my-agent v1 → v2
# Description: Add priority field, rename tags → labels
# Changes:
#   ADD: 'priority' (type=int, default=0)
#   RENAME: 'tags' → 'labels'
```

---

## Configuration

### `class CogDBConfig`

Dataclass configuring a `CognitiveDB` instance. All fields have sensible defaults.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `db_path` | `str` | `"./cogdb_data"` | Root directory for all storage files. |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformers model name. |
| `embedding_dim` | `int` | `384` | Embedding dimensionality (must match the chosen model). |
| `default_token_budget` | `int` | `1000` | Default max tokens for `recall()` and `get_context()` responses. |
| `l0_token_budget` | `int` | `50` | Token allocation for L0 (identity) context tier. |
| `l1_token_budget` | `int` | `200` | Token allocation for L1 (critical facts) tier. |
| `l2_token_budget` | `int` | `500` | Token allocation for L2 (task-relevant) tier. |
| `decay_half_life_hours` | `float` | `168.0` | Hours until a memory's decay score halves (default: 1 week). |
| `consolidation_threshold` | `int` | `10` | Min episodic memories before consolidation triggers. |
| `contradiction_check` | `bool` | `True` | Check for contradicting semantic facts on write. |
| `max_results_per_store` | `int` | `20` | Max candidates fetched per store during retrieval. |
| `hnsw_blend_alpha` | `float` | `0.2` | HNSW similarity weight in final retrieval ranking. `0.0` = pure importance, `1.0` = pure HNSW rank. |
| `max_procedures_per_query` | `int` | `1` | Max procedural memories included per `recall()` call. |
| `strict_metadata_validation` | `bool` | `True` | Raise `SchemaValidationError` on schema violations. Set `False` for warn-only mode. |
| `default_agent_id` | `str` | `"default"` | Fallback `agent_id` when none is provided. |
| `use_llm_consolidation` | `bool` | `False` | Enable LLM-powered SPO extraction in the consolidation pipeline (requires API key). |

**Example:**

```python
from cogdb.utils.config import CogDBConfig
from cogdb import CognitiveDB

config = CogDBConfig(
    db_path="./agent_memory",
    embedding_model="all-MiniLM-L6-v2",
    default_token_budget=2000,
    hnsw_blend_alpha=0.3,
    strict_metadata_validation=True,
)
db = CognitiveDB(config=config)
```

---

## Data Models

### `MemoryUnit`

The universal memory record — the atom of CogDB. Every stored memory (episodic event, semantic fact, procedural context) is represented as a `MemoryUnit`.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | — | Text content of the memory. |
| `memory_type` | `MemoryType` | — | One of `EPISODIC`, `SEMANTIC`, `PROCEDURAL`. |
| `agent_id` | `str` | — | Agent that owns this memory. |
| `importance` | `float` | `0.5` | Importance score, `0.0` to `1.0`. |
| `scope` | `MemoryScope` | `PRIVATE` | Visibility scope. |
| `id` | `str` | auto UUID | Unique identifier (UUID4 string). |
| `embedding` | `list[float] \| None` | `None` | Vector embedding (computed by the encoder, stored in HNSW). |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata. |
| `created_at` | `datetime` | now (UTC) | Creation timestamp. |
| `accessed_at` | `datetime` | now (UTC) | Last access timestamp. Updated on every retrieval. |
| `access_count` | `int` | `0` | Number of times this memory has been retrieved. |
| `decay_score` | `float` | `1.0` | Exponential decay factor. Approaches 0 over `decay_half_life_hours`. |
| `team_id` | `str \| None` | `None` | Team identifier for `TEAM`-scoped memories. |

**Methods:**

```python
unit.touch()                  # Update accessed_at and increment access_count
unit.effective_importance()   # → float: importance × decay_score (used for ranking)
unit.to_dict()                # → dict: serialize for storage/transport
MemoryUnit.from_dict(data)    # → MemoryUnit: deserialize from storage
```

**Example:**

```python
from cogdb.models import MemoryUnit, MemoryType, MemoryScope

unit = MemoryUnit(
    content="User prefers dark mode",
    memory_type=MemoryType.EPISODIC,
    agent_id="ui-agent",
    importance=0.8,
    scope=MemoryScope.PRIVATE,
)
print(unit.id)                    # auto-generated UUID
print(unit.effective_importance()) # 0.8 (decay_score starts at 1.0)
```

---

### `SemanticTriple`

A fact in the temporal knowledge graph. Triples carry validity windows — facts have lifecycles and can be superseded.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `subject` | `str` | — | Entity this fact is about. |
| `predicate` | `str` | — | Relationship or property name. |
| `object` | `str` | — | Value or target entity. |
| `agent_id` | `str` | — | Agent that asserted this fact. |
| `confidence` | `float` | `1.0` | Confidence score, `0.0` to `1.0`. |
| `id` | `str` | auto UUID | Unique identifier. |
| `valid_from` | `datetime` | now (UTC) | When this fact becomes valid. |
| `valid_until` | `datetime \| None` | `None` | Expiry time. Set automatically when superseded. |
| `source_episodes` | `list[str]` | `[]` | UUIDs of episodic memories supporting this fact. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata (includes `superseded_by` key when superseded). |

**Properties and methods:**

```python
triple.is_active            # bool: True if now is within [valid_from, valid_until]
triple.supersede(new_triple) # Sets valid_until=now and records superseded_by
triple.to_dict()             # → dict
SemanticTriple.from_dict(d)  # → SemanticTriple
```

**Example:**

```python
from cogdb.models import SemanticTriple

triple = SemanticTriple(
    subject="user_settings",
    predicate="theme",
    object="dark_mode",
    confidence=0.95,
    agent_id="ui-agent",
)
print(triple.is_active)  # True
```

---

### `ProcedureTemplate`

A learned workflow extracted from successful agent task completions. Stores ordered steps with tool/parameter bindings and tracks execution success via EMA.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Procedure name. |
| `description` | `str` | — | Human-readable description. |
| `steps` | `list[ProcedureStep]` | — | Ordered list of steps. |
| `agent_id` | `str` | — | Agent that owns this procedure. |
| `id` | `str` | auto UUID | Unique identifier. |
| `success_rate` | `float` | `1.0` | EMA success rate over all executions. |
| `execution_count` | `int` | `0` | Total number of executions recorded. |
| `source_episodes` | `list[str]` | `[]` | Episodes this procedure was extracted from. |
| `applicable_contexts` | `list[str]` | `[]` | Keywords for context-match retrieval. |
| `created_at` | `datetime` | now (UTC) | Creation timestamp. |
| `updated_at` | `datetime` | now (UTC) | Last update timestamp. |

**Methods:**

```python
proc.record_execution(success: bool)  # Update success_rate via EMA (α=0.3)
proc.to_dict()                         # → dict
ProcedureTemplate.from_dict(d)         # → ProcedureTemplate
```

---

### `ProcedureStep`

A single step within a `ProcedureTemplate`.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action` | `str` | — | Action name or description. |
| `tool` | `str \| None` | `None` | Tool or command to invoke. |
| `parameters` | `dict[str, Any]` | `{}` | Parameters to pass to the tool. |
| `expected_output` | `str \| None` | `None` | Expected output pattern for validation. |
| `fallback_action` | `str \| None` | `None` | Alternative action if this step fails. |

---

### `ContextResponse`

Structured result from `get_context()`. Contains tiered memory contents and budget tracking.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `level` | `int` | Highest context level loaded (`0`–`3`). |
| `token_count` | `int` | Tokens consumed across all loaded tiers. |
| `token_budget` | `int` | Total token budget provided. |
| `identity` | `str` | Agent identity string (L0 content). |
| `critical_facts` | `list[str]` | High-importance fact strings loaded at L1. |
| `relevant_memories` | `list[MemoryUnit]` | Task-relevant memories loaded at L2. |
| `deep_results` | `list[MemoryUnit]` | Deep search results loaded at L3. |

**Properties:**

```python
ctx.budget_remaining  # int: max(0, token_budget - token_count)
ctx.utilization       # float: fraction of budget used (0.0 to 1.0)
```

**Example:**

```python
ctx = db.get_context(agent_id="ui-agent", level=2, task_hint="settings page")
print(f"Identity: {ctx.identity}")
print(f"Budget: {ctx.token_count}/{ctx.token_budget} ({ctx.utilization:.0%})")
for fact in ctx.critical_facts:
    print(f"  • {fact}")
```

---

### `RecallQuery`

Internal query structure built by `recall()`. Exposed for direct use with lower-level APIs.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | — | Natural language query. |
| `agent_id` | `str` | — | Querying agent. |
| `token_budget` | `int` | `1000` | Max tokens in the response. |
| `memory_types` | `list[MemoryType]` | `[EPISODIC, SEMANTIC]` | Which memory types to search. |
| `scope_filter` | `MemoryScope \| None` | `None` | Restrict to a specific scope. |
| `time_range_start` | `datetime \| None` | `None` | Only return memories created after this time. |
| `time_range_end` | `datetime \| None` | `None` | Only return memories created before this time. |
| `min_importance` | `float` | `0.0` | Minimum importance threshold. |
| `max_results` | `int` | `20` | Max candidates per store. |

---

### `FieldSchema`

Definition of a single typed field within a `MetadataSchema`.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | — | Expected Python type. One of: `"str"`, `"int"`, `"float"`, `"bool"`, `"list"`, `"dict"`, `"any"`. |
| `required` | `bool` | `False` | If `True`, the field must be present on every `remember()` call. |
| `default` | `Any` | `None` | Documented default when the field is absent and not required. |
| `description` | `str` | `""` | Human-readable description for tooling. |

**Raises:** `ValueError` on construction if `type` is not one of the supported values.

**Example:**

```python
from cogdb.schema import FieldSchema

tool_field = FieldSchema(type="str", required=True, description="CLI tool used")
exit_code_field = FieldSchema(type="int", required=False, default=0)
```

---

### `MetadataSchema`

Typed schema for episodic memory metadata, scoped to a specific agent.

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | — | Agent this schema applies to. |
| `fields` | `dict[str, FieldSchema]` | `{}` | Mapping of field name to `FieldSchema`. |
| `version` | `int` | `1` | Schema version. Auto-incremented by `SchemaRegistry` on re-registration. |
| `created_at` | `str` | `""` | ISO-8601 timestamp set by `SchemaRegistry.register()`. |
| `name` | `str` | `""` | Optional human-readable schema name. |

**Example:**

```python
from cogdb.schema import MetadataSchema, FieldSchema

schema = MetadataSchema(
    agent_id="devops-agent",
    name="DevOps Memory Schema",
    fields={
        "tool":      FieldSchema(type="str", required=True),
        "exit_code": FieldSchema(type="int", required=False, default=0),
        "service":   FieldSchema(type="str", required=False),
    },
)
db.register_schema(schema)
```

---

### `SchemaValidationError`

Raised by `remember()` when metadata violates a registered schema and `strict_metadata_validation=True`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `errors` | `list[str]` | Field-level error messages, one per violation. |

**Example:**

```python
from cogdb.schema import SchemaValidationError

try:
    db.remember(
        "task failed",
        agent_id="devops-agent",
        metadata={"exit_code": "oops"},  # wrong type
    )
except SchemaValidationError as e:
    print(e.errors)
    # ['metadata.exit_code: expected int, got str']
```

---

### `FieldChange`

Describes a single field operation within a `SchemaMigration`. Constructed automatically by the builder methods on `SchemaMigration`; rarely instantiated directly.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | One of `"add"`, `"rename"`, `"drop"`, `"change_type"`, `"change_required"`. |
| `field_name` | `str` | Target field name. |
| `new_name` | `str \| None` | New name for rename operations. |
| `new_schema` | `FieldSchema \| None` | Replacement schema for add/change_type operations. |
| `default_value` | `Any` | Fill value for existing records when adding a field. |

---

## Memory Scopes

`MemoryScope` controls which agents can read and write a memory. Scope enforcement is applied at the vector index level on every search — no unguarded path exists.

| Value | String | Description |
|-------|--------|-------------|
| `MemoryScope.PRIVATE` | `"private"` | Fully isolated to the owning agent. |
| `MemoryScope.TEAM` | `"team"` | Accessible to a defined group of agents. Requires `team_id`. |
| `MemoryScope.ORGANIZATION` | `"org"` | Readable by all agents in the organization. |
| `MemoryScope.SESSION` | `"session"` | Ephemeral; auto-deleted after the conversation ends. |

```
┌─────────────────────────────┐
│      Organization Scope     │  all agents can read
│  ┌───────────────────────┐  │
│  │     Team Scope        │  │  defined group, read-write
│  │  ┌─────────────────┐  │  │
│  │  │  Private Scope   │  │  │  single agent only
│  │  └─────────────────┘  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
┌─────────────────────────────┐
│      Session Scope          │  ephemeral, auto-deleted
└─────────────────────────────┘
```

**Example:**

```python
from cogdb.models import MemoryScope

# Private (default)
db.remember("my private note", agent_id="agent-1")

# Team-shared
db.remember(
    "Shared deployment config",
    agent_id="agent-1",
    scope=MemoryScope.TEAM,
    team_id="backend-team",
)

# Org-wide reference
db.remember(
    "Company-wide API rate limit is 10k req/min",
    agent_id="admin-agent",
    scope=MemoryScope.ORGANIZATION,
    importance=0.9,
)
```

---

## Memory Types

| Value | String | Description |
|-------|--------|-------------|
| `MemoryType.EPISODIC` | `"episodic"` | Timestamped events and observations (what happened). |
| `MemoryType.SEMANTIC` | `"semantic"` | Factual knowledge graph triples (what is true). |
| `MemoryType.PROCEDURAL` | `"procedural"` | Learned step-by-step workflows (how to do things). |

```python
from cogdb.models import MemoryType

# recall() defaults to episodic + semantic
memories = db.recall("query", memory_types=[MemoryType.EPISODIC])

# forget() requires the type to route to the correct store
db.forget(memory_id, MemoryType.EPISODIC)
```

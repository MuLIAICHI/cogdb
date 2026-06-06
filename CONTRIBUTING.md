# Contributing to CogDB

CogDB is a cognitive database engine for AI agents — tri-memory (episodic, semantic, procedural), Rust-backed, and designed for production multi-agent systems. We welcome contributions of all sizes.

---

## Table of contents

1. [Getting started](#1-getting-started)
2. [Project structure](#2-project-structure)
3. [Adding a new framework adapter](#3-adding-a-new-framework-adapter)
4. [Writing tests](#4-writing-tests)
5. [Benchmarks](#5-benchmarks)
6. [Code style](#6-code-style)
7. [Community](#7-community)

---

## 1. Getting started

### Prerequisites

- Python 3.10+
- Rust toolchain (for building the storage engine) — install via [rustup.rs](https://rustup.rs)
- [maturin](https://github.com/PyO3/maturin) — Rust/Python build bridge

### Clone and install

```bash
git clone https://github.com/mustaphaliaichi/cogdb.git
cd cogdb

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Build the Rust engine

The storage engine lives in `cogdb_engine/` and must be compiled before any tests run.

```bash
cd cogdb_engine
maturin develop --release   # compiles and installs cogdb_engine into the active venv
cd ..
```

> **Windows note:** Add Cargo to PATH first:
> ```powershell
> $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
> ```

### Run the test suite

```bash
pytest tests/ -v   # must show 140/140 passing
```

### Lint and type-check

```bash
ruff check . && mypy cogdb/
```

---

## 2. Project structure

```
cogdb/
├── core.py          ← CognitiveDB main class — public entry point, start here
├── pipeline/
│   ├── encoder.py       ← text → embedding (sentence-transformers)
│   ├── consolidator.py  ← SPO extraction; optional LLM path
│   ├── retriever.py     ← token-budget-aware L0–L3 retrieval
│   └── decay.py         ← exponential importance decay
├── stores/
│   ├── episodic.py      ← thin Python wrapper over Rust EpisodicStore
│   ├── semantic.py      ← thin Python wrapper over Rust SemanticStore
│   └── procedural.py   ← thin Python wrapper over Rust ProceduralStore
├── adapters/
│   ├── autogen.py       ← AutoGen Memory protocol
│   ├── langgraph.py     ← LangGraph BaseCheckpointSaver + BaseStore
│   ├── mcp.py           ← MCP server (6 tools, cogdb-mcp CLI)
│   └── __init__.py
├── schema/
│   ├── __init__.py      ← FieldSchema, MetadataSchema, SchemaValidationError
│   └── registry.py      ← SchemaRegistry — persist schemas.json, validate on write
├── models/
│   └── importance.py    ← ImportanceModel (Ridge regression, sklearn)
└── utils/
    ├── config.py        ← CogDBConfig dataclass
    └── tokenizer.py     ← tiktoken token counting

cogdb_engine/            ← Rust PyO3 extension (WAL, HNSW, knowledge graph)
├── src/
│   ├── stores/          ← episodic.rs, semantic.rs, procedural.rs
│   ├── vector/hnsw.rs   ← HNSW index
│   ├── graph/kg.rs      ← petgraph knowledge graph
│   └── wal/             ← append-only WAL, crash recovery
benchmarks/              ← benchmark suites (not part of pytest)
tests/                   ← pytest test suite (140 tests)
docs/                    ← API reference and tutorials
```

**Rule:** do not add storage logic in `cogdb/stores/*.py`. Add it to the Rust store and expose via `cogdb_engine/src/python.rs`.

---

## 3. Adding a new framework adapter

Adapters translate CogDB's tri-memory API into whatever interface a framework expects. Existing adapters are in `cogdb/adapters/`.

### Step-by-step

**1. Create the file**

```bash
touch cogdb/adapters/<framework>.py
```

**2. Follow the lazy-import pattern**

Adapters must work even when the target framework is not installed. Use a `try/except ImportError` guard and provide a stub base class:

```python
# cogdb/adapters/<framework>.py
from __future__ import annotations

from cogdb.core import CognitiveDB
from cogdb.utils.config import CogDBConfig

try:
    from <framework> import SomeBaseClass, SomeRequiredType

    _FRAMEWORK_AVAILABLE = True
except ImportError:
    _FRAMEWORK_AVAILABLE = False

    class SomeBaseClass:  # type: ignore[no-redef]
        """Stub — installed only when <framework> is available."""


class CogDB<Framework>Adapter(SomeBaseClass):
    """CogDB adapter for <Framework>.

    Args:
        agent_id: Identifier for this agent's memory namespace.
        db: An existing CognitiveDB instance. If None, a new one is created.
        config: CogDBConfig used when creating a new instance.

    Example:
        >>> adapter = CogDB<Framework>Adapter(agent_id="agent-1", db_path="./memory")
    """

    def __init__(
        self,
        agent_id: str,
        db: CognitiveDB | None = None,
        config: CogDBConfig | None = None,
    ) -> None:
        if not _FRAMEWORK_AVAILABLE:
            raise ImportError(
                "Install <framework> to use this adapter: pip install <framework>"
            )
        self._db = db or CognitiveDB(config=config or CogDBConfig())
        self._agent_id = agent_id

    # Implement the framework's required interface methods here
```

**3. Add tests**

Create `tests/test_<framework>_adapter.py`. Tests must pass with and without the framework installed — mock the import if needed.

**4. Register the adapter**

Add it to `cogdb/adapters/__init__.py` under a `TYPE_CHECKING` guard so it doesn't force an import:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogdb.adapters.<framework> import CogDB<Framework>Adapter
```

**5. Update the README**

Add a row to the framework comparison table in `README.md`.

---

## 4. Writing tests

### Location and naming

- All tests go in `tests/`
- One file per module: `test_episodic_store.py`, `test_autogen_adapter.py`, etc.
- Benchmark scripts go in `benchmarks/`, not `tests/`

### db_path

Always use a temp directory — never hardcode a path:

```python
import tempfile
import pytest
from cogdb import CognitiveDB

@pytest.fixture
def db(tmp_path):
    return CognitiveDB(db_path=str(tmp_path))
```

### Optional dependencies

Tests for adapters must not fail when the adapter's framework is not installed. Skip or mock as appropriate:

```python
autogen = pytest.importorskip("autogen_core")
```

### Async adapters

Use `pytest-asyncio` with the `@pytest.mark.asyncio` decorator:

```python
import pytest

@pytest.mark.asyncio
async def test_langgraph_checkpointer(db):
    ...
```

### Gate

All 140 existing tests must pass before a PR can merge. Do not delete or weaken existing tests.

---

## 5. Benchmarks

Benchmark suites live in `benchmarks/` and are run separately from the test suite.

### Available suites

| Suite | What it measures |
|---|---|
| Suite 1 — Tri-Memory | End-to-end memory quality across all three stores |
| Suite 3 — Consistency | Contradiction detection and supersede accuracy |
| Suite 4 — Throughput | Raw storage write/search latency |

### Running benchmarks

```bash
# Run all suites, skip LLM-dependent steps
python -m benchmarks.cogdb_bench --suite all --no-llm

# Run a specific suite
python -m benchmarks.cogdb_bench --suite 1
```

### Baseline

The Phase 2 baseline is **90.7 / 100** on Suite 1. Performance regressions in Suite 1 below this baseline will block a PR.

---

## 6. Code style

### Comments

Write no comments by default. Add one only when the **why** is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific external bug. If removing the comment wouldn't confuse a future reader, skip it.

### Docstrings

Every **public** method needs a docstring with `Args:`, `Returns:`, and a short `Example:`. Private helpers (leading underscore) do not need docstrings.

### Import order

```python
# 1. stdlib
import os
import tempfile

# 2. third-party
import numpy as np

# 3. cogdb-internal
from cogdb.core import CognitiveDB
from cogdb.utils.config import CogDBConfig
```

### Line length

100 characters (enforced by ruff).

### Types

- Annotate all public API (function signatures, class attributes)
- Skip annotations for short private helpers where the type is obvious

### Formatting and linting

```bash
ruff check .          # linting
ruff format .         # formatting
mypy cogdb/           # type checking
```

---

## 7. Community

- **Bug reports** → [GitHub Issues](https://github.com/mustaphaliaichi/cogdb/issues) using the bug report template
- **Feature requests** → [GitHub Issues](https://github.com/mustaphaliaichi/cogdb/issues) using the feature request template
- **Questions and discussion** → [GitHub Discussions](https://github.com/mustaphaliaichi/cogdb/discussions)
- **Pull requests** → open against `main`; fill out the PR template, describe what you changed and why

All contributions are released under the MIT License.

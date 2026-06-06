---
name: Bug report
about: Report a bug in CogDB
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the bug

What happened, and what did you expect to happen instead?

## To reproduce

Provide a minimal, self-contained code snippet that reproduces the issue:

```python
from cogdb import CognitiveDB

db = CognitiveDB(db_path="/tmp/test_cogdb")
# ... minimal repro
```

## Environment

| Field | Value |
|---|---|
| Python version | e.g. 3.11.4 |
| CogDB version | e.g. 0.4.0 |
| OS | e.g. macOS 14.2, Ubuntu 22.04, Windows 11 |
| Rust engine compiled? | Yes / No / Using pre-built wheel |
| cogdb_engine version | e.g. 0.4.0 (from `pip show cogdb`) |

## Error output

Paste the full traceback here. Do not truncate it.

```
Traceback (most recent call last):
  ...
```

## Memory type affected

- [ ] Episodic (`EpisodicStore`)
- [ ] Semantic (`SemanticStore`)
- [ ] Procedural (`ProceduralStore`)
- [ ] Schema system (`cogdb/schema/`)
- [ ] Framework adapter (AutoGen / LangGraph / MCP)
- [ ] All / unknown

## Additional context

Any other context that might help: related issues, workarounds you tried, relevant parts of your config, etc.

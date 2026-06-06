---
name: Framework adapter request
about: Request a new framework adapter (LangChain, CrewAI, etc.)
title: "[Adapter] Add <FrameworkName> adapter"
labels: adapter, enhancement
assignees: ''
---

## Framework

Name and link to the framework's documentation or memory interface:

- **Name**: e.g. LangChain
- **Docs**: e.g. https://python.langchain.com/docs/...
- **Version**: e.g. langchain-core >= 0.2.0

## Current workaround

How are you using CogDB with this framework today, if at all? A short snippet helps us understand what the adapter needs to do.

```python
# e.g. calling cogdb directly and manually injecting context
```

## Interface to implement

Which base class, protocol, or interface does the framework expose for external memory?

Example: "LangChain expects a `BaseMemory` subclass with `load_memory_variables()` and `save_context()` methods."

## Priority

- [ ] Blocking my project — I cannot ship without this
- [ ] Nice to have — would clean up my integration
- [ ] Exploratory — just raising it for discussion

## Additional context

Any other details: framework version constraints, patterns we should follow, links to similar adapters in other projects, etc.

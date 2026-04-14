# Contributing to CogDB

CogDB is in early development (Phase 0 — Python proof-of-concept). We welcome contributions across several areas.

## Areas of Contribution

### Research
- Analysis of new agent memory systems as they emerge
- Benchmark design for multi-agent memory scenarios
- Academic paper reviews relevant to the cognitive memory stack

### Framework Adapters
- CrewAI memory interface adapter
- Semantic Kernel adapter
- OpenAI Agents SDK adapter

### Core Engine
- Memory consolidation pipeline (episodic → semantic distillation)
- Importance decay algorithms
- Contradiction detection improvements

### Benchmarks
- LongMemEval integration
- Custom multi-agent memory consistency benchmarks
- Token efficiency benchmarks vs Mem0/Zep/MemPalace

## Development Setup

```bash
# Clone the repo
git clone https://github.com/mustaphaliaichi/cogdb.git
cd cogdb

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check cogdb/

# Run type checker
mypy cogdb/
```

## Code Standards

- **Type hints everywhere** — the codebase is fully typed
- **Docstrings on all public methods** — include Args, Returns, and Example
- **Tests before merging** — every new feature needs unit tests
- **Thread safety** — all store operations must be thread-safe

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for your changes
3. Ensure `pytest`, `ruff`, and `mypy` pass
4. Submit a PR with a clear description of what and why

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

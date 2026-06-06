## Summary

<!-- What does this PR do? One or two sentences. -->

## Type

- [ ] Bug fix
- [ ] New framework adapter
- [ ] New feature
- [ ] Performance improvement
- [ ] Docs / examples
- [ ] Tests / benchmarks

## Related issue

Closes #<!-- issue number -->

## Changes

<!-- Bullet list of what changed and why. Focus on the "why" — the diff shows the "what". -->

- 

## Testing

- [ ] Added tests for new code
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Ran the benchmark suite (`python -m benchmarks.cogdb_bench --suite all --no-llm`)
- [ ] Checked for regressions in existing adapters

## Checklist

- [ ] No unnecessary comments — only added a comment where the WHY is non-obvious
- [ ] No new external dependencies added without good reason (note them in the summary if you did)
- [ ] If adding an adapter: follows the lazy-import pattern (see `cogdb/adapters/autogen.py`)
- [ ] If adding a schema change: included a migration test
- [ ] Public methods have docstrings with Args, Returns, and an example
- [ ] Type hints on all new public API

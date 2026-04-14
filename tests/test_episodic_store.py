"""Tests for the episodic memory store."""

import tempfile
from datetime import datetime, timezone

import pytest

from cogdb.models import MemoryScope, MemoryType, MemoryUnit
from cogdb.stores.episodic import EpisodicStore
from cogdb.utils.config import CogDBConfig


@pytest.fixture
def store():
    """Create a fresh EpisodicStore with a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CogDBConfig(db_path=tmpdir)
        config.ensure_dirs()
        yield EpisodicStore(config)


class TestEpisodicStoreAdd:
    def test_add_returns_id(self, store):
        unit = MemoryUnit(
            content="Test memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        )
        result_id = store.add(unit)
        assert result_id == unit.id

    def test_add_increments_count(self, store):
        assert store.count() == 0
        unit = MemoryUnit(
            content="First memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        )
        store.add(unit)
        assert store.count() == 1

    def test_add_multiple_agents(self, store):
        for agent in ["agent-1", "agent-2", "agent-3"]:
            store.add(MemoryUnit(
                content=f"Memory from {agent}",
                memory_type=MemoryType.EPISODIC,
                agent_id=agent,
            ))
        assert store.count() == 3
        assert store.count(agent_id="agent-1") == 1


class TestEpisodicStoreSearch:
    def test_search_returns_relevant(self, store):
        store.add(MemoryUnit(
            content="The user prefers dark mode for the UI",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        ))
        store.add(MemoryUnit(
            content="API deployment completed successfully",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        ))

        results = store.search("dark mode", agent_id="agent-1", top_k=5)
        assert len(results) > 0
        assert any("dark mode" in r.content for r in results)

    def test_search_respects_agent_scope(self, store):
        store.add(MemoryUnit(
            content="Private memory for agent-1",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
            scope=MemoryScope.PRIVATE,
        ))
        store.add(MemoryUnit(
            content="Org-wide announcement",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-2",
            scope=MemoryScope.ORGANIZATION,
        ))

        # agent-1 should see its own + org-scoped
        results = store.search("memory", agent_id="agent-1", top_k=10)
        assert len(results) >= 1

    def test_search_filters_by_importance(self, store):
        store.add(MemoryUnit(
            content="Low importance memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
            importance=0.1,
        ))
        store.add(MemoryUnit(
            content="High importance memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
            importance=0.9,
        ))

        results = store.search(
            "memory",
            agent_id="agent-1",
            min_importance=0.5,
            top_k=10,
        )
        for r in results:
            assert r.importance >= 0.5


class TestEpisodicStoreGetDelete:
    def test_get_by_id(self, store):
        unit = MemoryUnit(
            content="Retrievable memory",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        )
        store.add(unit)
        retrieved = store.get(unit.id)
        assert retrieved is not None
        assert retrieved.content == "Retrievable memory"

    def test_get_nonexistent_returns_none(self, store):
        result = store.get("nonexistent-id")
        assert result is None

    def test_delete(self, store):
        unit = MemoryUnit(
            content="Memory to delete",
            memory_type=MemoryType.EPISODIC,
            agent_id="agent-1",
        )
        store.add(unit)
        assert store.count() == 1

        store.delete(unit.id)
        assert store.get(unit.id) is None

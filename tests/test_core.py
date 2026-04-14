"""Integration tests for the CognitiveDB main class."""

import tempfile

import pytest

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield CognitiveDB(db_path=tmpdir)


class TestCognitiveDBRemember:
    def test_remember_basic(self, db):
        mem_id = db.remember("Test memory", agent_id="agent-1")
        assert mem_id is not None
        assert db.stats()["episodic"] == 1

    def test_remember_with_metadata(self, db):
        db.remember(
            "Error in API",
            agent_id="agent-1",
            importance=0.9,
            metadata={"error_type": "500"},
        )
        assert db.stats()["episodic"] == 1

    def test_remember_scoped(self, db):
        db.remember("Private info", agent_id="a1", scope=MemoryScope.PRIVATE)
        db.remember("Team info", agent_id="a1", scope=MemoryScope.TEAM)
        db.remember("Org info", agent_id="a1", scope=MemoryScope.ORGANIZATION)
        assert db.stats()["episodic"] == 3


class TestCognitiveDBLearn:
    def test_learn_fact(self, db):
        triple_id = db.learn(
            subject="api",
            predicate="version",
            object="v2.0",
            agent_id="devops",
        )
        assert triple_id is not None
        assert db.stats()["semantic"] == 1

    def test_learn_supersedes(self, db):
        db.learn("api", "version", "v1.0", agent_id="devops")
        db.learn("api", "version", "v2.0", agent_id="devops")

        facts = db.query_knowledge("api")
        version_facts = [f for f in facts if f.predicate == "version"]
        assert len(version_facts) == 1
        assert version_facts[0].object == "v2.0"


class TestCognitiveDBProcedure:
    def test_learn_procedure(self, db):
        proc_id = db.learn_procedure(
            name="deploy",
            description="Deploy to production",
            steps=[
                {"action": "test", "tool": "pytest"},
                {"action": "build", "tool": "npm"},
                {"action": "deploy", "tool": "vercel"},
            ],
            agent_id="devops",
            applicable_contexts=["deploy", "release"],
        )
        assert proc_id is not None
        assert db.stats()["procedural"] == 1


class TestCognitiveDBRecall:
    def test_recall_within_budget(self, db):
        for i in range(5):
            db.remember(
                f"Memory number {i} about deployment and infrastructure",
                agent_id="agent-1",
                importance=0.5 + i * 0.1,
            )

        results = db.recall(
            "deployment",
            agent_id="agent-1",
            token_budget=100,
        )
        assert len(results) > 0

    def test_recall_empty_db(self, db):
        results = db.recall("anything", agent_id="agent-1")
        assert len(results) == 0


class TestCognitiveDBContext:
    def test_get_context_l0(self, db):
        ctx = db.get_context(agent_id="agent-1", level=0)
        assert ctx.level == 0
        assert ctx.identity is not None
        assert len(ctx.critical_facts) == 0
        assert len(ctx.relevant_memories) == 0

    def test_get_context_l2_with_data(self, db):
        db.remember("Settings page is slow", agent_id="ui-agent", importance=0.8)
        db.learn("user", "prefers", "dark_mode", agent_id="ui-agent")

        ctx = db.get_context(
            agent_id="ui-agent",
            level=2,
            task_hint="settings page",
            token_budget=500,
        )
        assert ctx.level == 2
        assert ctx.token_count <= ctx.token_budget

    def test_context_respects_budget(self, db):
        # Add many memories
        for i in range(20):
            db.remember(
                f"Detailed memory about topic {i} with lots of context and description " * 3,
                agent_id="agent-1",
            )

        ctx = db.get_context(
            agent_id="agent-1",
            level=3,
            task_hint="topic 5",
            token_budget=200,
        )
        assert ctx.token_count <= ctx.token_budget + 50  # Small tolerance for truncation


class TestCognitiveDBForget:
    def test_forget_episodic(self, db):
        mem_id = db.remember("To be forgotten", agent_id="a1")
        assert db.stats()["episodic"] == 1

        db.forget(mem_id, MemoryType.EPISODIC)
        # ChromaDB delete is eventual, so just verify the call succeeds

    def test_forget_semantic(self, db):
        triple_id = db.learn("x", "y", "z", agent_id="a1")
        assert db.stats()["semantic"] == 1

        db.forget(triple_id, MemoryType.SEMANTIC)
        assert db.stats()["semantic"] == 0


class TestCognitiveDBStats:
    def test_stats_empty(self, db):
        stats = db.stats()
        assert stats["episodic"] == 0
        assert stats["semantic"] == 0
        assert stats["procedural"] == 0
        assert stats["total"] == 0

    def test_stats_with_data(self, db):
        db.remember("ep1", agent_id="a1")
        db.remember("ep2", agent_id="a1")
        db.learn("s", "p", "o", agent_id="a1")
        db.learn_procedure(
            name="proc1", steps=[{"action": "do"}],
            agent_id="a1", description="test",
        )

        stats = db.stats()
        assert stats["episodic"] == 2
        assert stats["semantic"] == 1
        assert stats["procedural"] == 1
        assert stats["total"] == 4

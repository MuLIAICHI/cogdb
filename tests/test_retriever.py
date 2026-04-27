"""Tests for the token-budget-aware Retriever."""

import gc
import shutil
import tempfile

import pytest

from cogdb.models import MemoryScope, MemoryType, MemoryUnit, RecallQuery, SemanticTriple
from cogdb.pipeline.retriever import Retriever
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.procedural import ProceduralStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig
from cogdb.utils.tokenizer import count_tokens


@pytest.fixture
def components():
    tmpdir = tempfile.mkdtemp()
    config = CogDBConfig(db_path=tmpdir)
    config.ensure_dirs()
    episodic = EpisodicStore(config)
    semantic = SemanticStore(config)
    procedural = ProceduralStore(config)
    retriever = Retriever(episodic, semantic, procedural, config)
    yield episodic, semantic, procedural, retriever, config
    try:
        episodic._client.reset()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _add_unit(episodic, content, agent_id="agent-1", importance=0.5):
    unit = MemoryUnit(
        content=content,
        memory_type=MemoryType.EPISODIC,
        agent_id=agent_id,
        importance=importance,
    )
    episodic.add(unit)
    return unit


class TestRetrieverRecall:
    def test_recall_returns_results(self, components):
        episodic, _, _, retriever, _ = components
        _add_unit(episodic, "The deployment pipeline uses GitHub Actions")

        results = retriever.recall(RecallQuery(
            query="deployment pipeline",
            agent_id="agent-1",
            token_budget=500,
        ))
        assert len(results) > 0

    def test_recall_empty_store_returns_empty(self, components):
        _, _, _, retriever, _ = components
        results = retriever.recall(RecallQuery(
            query="anything",
            agent_id="agent-1",
            token_budget=500,
        ))
        assert results == []

    def test_token_budget_is_respected(self, components):
        episodic, _, _, retriever, _ = components
        for i in range(10):
            _add_unit(episodic, f"Memory {i}: " + "word " * 30, importance=0.5)

        budget = 100
        results = retriever.recall(RecallQuery(
            query="memory",
            agent_id="agent-1",
            token_budget=budget,
        ))

        total_tokens = sum(count_tokens(m.content) for m in results)
        assert total_tokens <= budget + 10

    def test_results_sorted_by_importance(self, components):
        episodic, _, _, retriever, _ = components
        _add_unit(episodic, "Low priority memory about topic A", importance=0.2)
        _add_unit(episodic, "High priority memory about topic A", importance=0.9)
        _add_unit(episodic, "Medium priority memory about topic A", importance=0.5)

        results = retriever.recall(RecallQuery(
            query="topic A",
            agent_id="agent-1",
            token_budget=2000,
        ))

        importances = [m.effective_importance() for m in results]
        assert importances == sorted(importances, reverse=True)

    def test_recall_touches_memory(self, components):
        episodic, _, _, retriever, _ = components
        _add_unit(episodic, "Touchable memory about recall")

        results = retriever.recall(RecallQuery(
            query="touchable memory",
            agent_id="agent-1",
            token_budget=500,
        ))

        assert len(results) > 0
        assert results[0].access_count >= 1

    def test_recall_filters_by_memory_type(self, components):
        episodic, _, _, retriever, _ = components
        _add_unit(episodic, "Episodic memory about the build")

        results = retriever.recall(RecallQuery(
            query="build",
            agent_id="agent-1",
            token_budget=500,
            memory_types=[MemoryType.SEMANTIC],
        ))
        assert results == []


class TestRetrieverProgressiveContext:
    def test_l0_always_includes_identity(self, components):
        _, _, _, retriever, _ = components
        ctx = retriever.get_context(agent_id="agent-1", level=0)
        assert ctx.level == 0
        assert "agent-1" in ctx.identity
        assert ctx.token_count > 0

    def test_l1_includes_critical_facts(self, components):
        _, semantic, _, retriever, _ = components
        triple = SemanticTriple(
            subject="user", predicate="prefers", object="dark_mode",
            agent_id="agent-1", confidence=0.95,
        )
        semantic.add_triple(triple)  # correct method name

        ctx = retriever.get_context(agent_id="agent-1", level=1)
        assert ctx.level == 1
        assert any("user" in f for f in ctx.critical_facts)

    def test_l2_includes_relevant_memories(self, components):
        episodic, _, _, retriever, _ = components
        _add_unit(episodic, "Deploy the frontend service using Vercel")

        ctx = retriever.get_context(
            agent_id="agent-1",
            level=2,
            task_hint="deploy frontend",
            token_budget=500,
        )
        assert ctx.level == 2
        assert len(ctx.relevant_memories) > 0

    def test_context_never_exceeds_budget(self, components):
        episodic, _, _, retriever, _ = components
        for i in range(15):
            _add_unit(episodic, f"Detailed memory {i}: " + "context word " * 20)

        budget = 300
        ctx = retriever.get_context(
            agent_id="agent-1",
            level=3,
            task_hint="memory",
            token_budget=budget,
        )
        assert ctx.token_count <= budget + 20

    def test_l3_no_duplicate_memories(self, components):
        episodic, _, _, retriever, _ = components
        for i in range(5):
            _add_unit(episodic, f"Relevant fact {i} about the frontend deployment")

        ctx = retriever.get_context(
            agent_id="agent-1",
            level=3,
            task_hint="frontend deployment",
            token_budget=2000,
        )
        l2_ids = {m.id for m in ctx.relevant_memories}
        l3_ids = {m.id for m in ctx.deep_results}
        assert l2_ids.isdisjoint(l3_ids)

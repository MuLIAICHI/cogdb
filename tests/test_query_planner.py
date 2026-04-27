"""Tests for the QueryPlanner — store routing and scope enforcement."""

import gc
import shutil
import tempfile

import pytest

from cogdb.models import MemoryScope, MemoryType, MemoryUnit, RecallQuery
from cogdb.pipeline.retriever import Retriever
from cogdb.query.planner import QueryPlanner
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.procedural import ProceduralStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig


@pytest.fixture
def planner():
    tmpdir = tempfile.mkdtemp()
    config = CogDBConfig(db_path=tmpdir)
    config.ensure_dirs()
    episodic = EpisodicStore(config)
    semantic = SemanticStore(config)
    procedural = ProceduralStore(config)
    retriever = Retriever(episodic, semantic, procedural, config)
    qp = QueryPlanner(episodic, semantic, procedural, retriever, config)
    yield qp, episodic, semantic, procedural, config
    try:
        episodic._client.reset()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


_ALL_TYPES = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]


def _add(episodic, content, agent_id="agent-1", importance=0.5, scope=MemoryScope.PRIVATE):
    unit = MemoryUnit(
        content=content,
        memory_type=MemoryType.EPISODIC,
        agent_id=agent_id,
        importance=importance,
        scope=scope,
    )
    episodic.add(unit)
    return unit


class TestQueryPlannerRouting:
    def test_execute_returns_results(self, planner):
        qp, episodic, *_ = planner
        _add(episodic, "The API deployed successfully last Tuesday")

        results = qp.execute(RecallQuery(
            query="API deployment",
            agent_id="agent-1",
            token_budget=500,
        ))
        assert len(results) > 0

    def test_execute_empty_store(self, planner):
        qp, *_ = planner
        results = qp.execute(RecallQuery(
            query="something",
            agent_id="agent-1",
            token_budget=500,
        ))
        assert results == []

    def test_procedural_hint_routes_to_procedural(self, planner):
        qp, *_ = planner
        # Pass all 3 types so the planner can route freely
        query = RecallQuery(
            query="how to deploy the frontend step by step",
            agent_id="agent-1",
            token_budget=500,
            memory_types=_ALL_TYPES,
        )
        plan = qp.explain(query)
        assert MemoryType.PROCEDURAL.value in plan["stores"]

    def test_episodic_hint_routes_to_episodic(self, planner):
        qp, *_ = planner
        query = RecallQuery(
            query="what happened last time we deployed",
            agent_id="agent-1",
            token_budget=500,
            memory_types=_ALL_TYPES,
        )
        plan = qp.explain(query)
        assert MemoryType.EPISODIC.value in plan["stores"]

    def test_semantic_hint_routes_to_semantic(self, planner):
        qp, *_ = planner
        query = RecallQuery(
            query="what is the relationship between api and database",
            agent_id="agent-1",
            token_budget=500,
            memory_types=_ALL_TYPES,
        )
        plan = qp.explain(query)
        assert MemoryType.SEMANTIC.value in plan["stores"]

    def test_explicit_single_type_not_overridden(self, planner):
        qp, *_ = planner
        # Caller restricts to ONE type — must not be overridden
        query = RecallQuery(
            query="how to deploy",
            agent_id="agent-1",
            token_budget=500,
            memory_types=[MemoryType.EPISODIC],
        )
        plan = qp.explain(query)
        assert plan["stores"] == [MemoryType.EPISODIC.value]


class TestQueryPlannerExplain:
    def test_explain_returns_required_keys(self, planner):
        qp, *_ = planner
        plan = qp.explain(RecallQuery(
            query="how to fix CORS errors",
            agent_id="agent-1",
            token_budget=300,
            memory_types=_ALL_TYPES,
        ))
        assert "stores" in plan
        assert "reasoning" in plan
        assert "token_budget" in plan
        assert plan["token_budget"] == 300

    def test_explain_no_strong_signal_queries_all(self, planner):
        qp, *_ = planner
        plan = qp.explain(RecallQuery(
            query="xyz abc 123",
            agent_id="agent-1",
            token_budget=500,
            memory_types=_ALL_TYPES,
        ))
        assert len(plan["stores"]) == 3


class TestQueryPlannerContext:
    def test_get_context_delegates_to_retriever(self, planner):
        qp, episodic, *_ = planner
        _add(episodic, "Frontend build takes 3 minutes on average")

        ctx = qp.get_context(
            agent_id="agent-1",
            level=2,
            task_hint="frontend build time",
            token_budget=500,
        )
        assert ctx.level == 2
        assert ctx.token_budget == 500

    def test_get_context_l0_works_on_empty_store(self, planner):
        qp, *_ = planner
        ctx = qp.get_context(agent_id="agent-1", level=0)
        assert ctx.identity is not None
        assert ctx.token_count > 0

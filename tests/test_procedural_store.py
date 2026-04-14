"""Tests for the procedural memory store."""

import tempfile

import pytest

from cogdb.models import ProcedureStep, ProcedureTemplate
from cogdb.stores.procedural import ProceduralStore
from cogdb.utils.config import CogDBConfig


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CogDBConfig(db_path=tmpdir)
        config.ensure_dirs()
        yield ProceduralStore(config)


def _make_procedure(name="test_proc", agent_id="agent-1", **kwargs):
    return ProcedureTemplate(
        name=name,
        description=kwargs.get("description", f"Test procedure: {name}"),
        steps=[
            ProcedureStep(action="step1", tool="tool_a"),
            ProcedureStep(action="step2", tool="tool_b"),
        ],
        agent_id=agent_id,
        applicable_contexts=kwargs.get("applicable_contexts", ["test"]),
        success_rate=kwargs.get("success_rate", 1.0),
    )


class TestProceduralStoreAdd:
    def test_add_procedure(self, store):
        proc = _make_procedure()
        result_id = store.add(proc)
        assert result_id == proc.id
        assert store.count() == 1

    def test_add_multiple(self, store):
        for name in ["deploy", "test", "rollback"]:
            store.add(_make_procedure(name=name))
        assert store.count() == 3


class TestProceduralStoreGet:
    def test_get_by_id(self, store):
        proc = _make_procedure(name="deploy_app")
        store.add(proc)

        retrieved = store.get(proc.id)
        assert retrieved is not None
        assert retrieved.name == "deploy_app"
        assert len(retrieved.steps) == 2

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None


class TestProceduralStoreSearch:
    def test_search_by_context(self, store):
        store.add(_make_procedure(
            name="fix_cors",
            description="Fix CORS errors in nginx",
            applicable_contexts=["cors", "nginx", "api"],
        ))
        store.add(_make_procedure(
            name="deploy_frontend",
            description="Deploy frontend to Vercel",
            applicable_contexts=["deploy", "vercel", "frontend"],
        ))

        results = store.search_by_context("cors")
        assert len(results) >= 1
        assert results[0].name == "fix_cors"

    def test_search_by_context_no_match(self, store):
        store.add(_make_procedure(name="unrelated"))
        results = store.search_by_context("quantum_computing")
        assert len(results) == 0

    def test_search_filters_by_success_rate(self, store):
        store.add(_make_procedure(name="reliable", success_rate=0.95))
        store.add(_make_procedure(name="flaky", success_rate=0.3))

        results = store.search_by_context("test", min_success_rate=0.5)
        names = [r.name for r in results]
        assert "reliable" in names
        assert "flaky" not in names

    def test_search_by_name(self, store):
        store.add(_make_procedure(name="deploy_api"))
        store.add(_make_procedure(name="deploy_frontend"))
        store.add(_make_procedure(name="run_tests"))

        results = store.search_by_name("deploy")
        assert len(results) == 2


class TestProceduralStoreExecution:
    def test_record_execution_success(self, store):
        proc = _make_procedure(success_rate=0.5)
        store.add(proc)

        store.record_execution(proc.id, success=True)
        updated = store.get(proc.id)
        assert updated is not None
        assert updated.execution_count == 1
        assert updated.success_rate > 0.5  # Should increase

    def test_record_execution_failure(self, store):
        proc = _make_procedure(success_rate=0.8)
        store.add(proc)

        store.record_execution(proc.id, success=False)
        updated = store.get(proc.id)
        assert updated is not None
        assert updated.success_rate < 0.8  # Should decrease

    def test_record_nonexistent(self, store):
        result = store.record_execution("fake-id", success=True)
        assert result is False


class TestProceduralStoreDelete:
    def test_delete(self, store):
        proc = _make_procedure()
        store.add(proc)
        assert store.count() == 1

        store.delete(proc.id)
        assert store.count() == 0

    def test_delete_nonexistent(self, store):
        result = store.delete("fake-id")
        assert result is False


class TestProceduralStoreList:
    def test_list_all(self, store):
        for name in ["a", "b", "c"]:
            store.add(_make_procedure(name=name))

        results = store.list_all()
        assert len(results) == 3

    def test_list_by_agent(self, store):
        store.add(_make_procedure(name="a", agent_id="agent-1"))
        store.add(_make_procedure(name="b", agent_id="agent-2"))

        results = store.list_all(agent_id="agent-1")
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"

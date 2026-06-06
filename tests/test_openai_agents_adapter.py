"""Tests for cogdb/adapters/openai_agents.py.

All tests run without openai-agents installed.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestCogDBAgentMemory:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def _make_memory(self):
        from cogdb.adapters.openai_agents import CogDBAgentMemory

        return CogDBAgentMemory(agent_id="test-oai", db_path=self.tmp)

    def test_import_without_sdk(self):
        from cogdb.adapters import openai_agents

        assert hasattr(openai_agents, "CogDBAgentMemory")

    def test_remember_returns_id(self):
        m = self._make_memory()
        result = m.remember("The API rate limit was hit at 2pm")
        assert isinstance(result, str) and len(result) > 0

    def test_recall_returns_list_of_strings(self):
        m = self._make_memory()
        m.remember("Database connection pool exhausted")
        results = m.recall("database connection")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, str)

    def test_context_returns_string(self):
        m = self._make_memory()
        m.remember("User is working on a Python web app")
        ctx = m.context(task_hint="web development")
        assert isinstance(ctx, str)

    def test_make_memory_tools(self):
        from cogdb.adapters.openai_agents import CogDBAgentMemory, make_memory_tools

        mem = CogDBAgentMemory(agent_id="tool-test", db_path=self.tmp)
        tools = make_memory_tools(mem)
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_agent_id_property(self):
        m = self._make_memory()
        assert m.agent_id == "test-oai"

    def test_close_safe(self):
        m = self._make_memory()
        m.close()  # Should not raise

    def test_remember_and_recall_roundtrip(self):
        m = self._make_memory()
        m.remember(
            "CORS errors fixed by adding Access-Control-Allow-Origin header",
            importance=0.8,
        )
        results = m.recall("CORS header fix")
        assert isinstance(results, list)

    def test_db_property(self):
        from cogdb.core import CognitiveDB

        m = self._make_memory()
        assert isinstance(m.db, CognitiveDB)

    def test_remember_importance_range(self):
        m = self._make_memory()
        low = m.remember("low-importance note", importance=0.1)
        high = m.remember("critical system alert", importance=1.0)
        assert isinstance(low, str) and len(low) > 0
        assert isinstance(high, str) and len(high) > 0

    def test_recall_empty_store_returns_list(self):
        # Fresh store with no memories — recall should return an empty list, not raise
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from cogdb.adapters.openai_agents import CogDBAgentMemory

            m = CogDBAgentMemory(agent_id="empty-agent", db_path=d)
            results = m.recall("anything")
            assert isinstance(results, list)

    def test_context_empty_store_returns_string(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from cogdb.adapters.openai_agents import CogDBAgentMemory

            m = CogDBAgentMemory(agent_id="empty-ctx", db_path=d)
            ctx = m.context()
            assert isinstance(ctx, str)

    def test_tools_are_callable(self):
        from cogdb.adapters.openai_agents import CogDBAgentMemory, make_memory_tools

        mem = CogDBAgentMemory(agent_id="callable-test", db_path=self.tmp)
        tools = make_memory_tools(mem)
        for t in tools:
            assert callable(t)

    def test_custom_token_budget(self):
        from cogdb.adapters.openai_agents import CogDBAgentMemory

        m = CogDBAgentMemory(agent_id="budget-test", db_path=self.tmp, token_budget=200)
        m.remember("Budget-constrained memory test")
        results = m.recall("budget", token_budget=200)
        assert isinstance(results, list)

    def test_existing_db_injection(self):
        from cogdb.adapters.openai_agents import CogDBAgentMemory
        from cogdb.core import CognitiveDB
        from cogdb.utils.config import CogDBConfig

        cfg = CogDBConfig(db_path=self.tmp)
        shared_db = CognitiveDB(config=cfg)
        m = CogDBAgentMemory(agent_id="injected", db=shared_db)
        assert m.db is shared_db
        mid = m.remember("Shared DB injection test")
        assert isinstance(mid, str)

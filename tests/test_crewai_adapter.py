"""Tests for cogdb.adapters.crewai — run without crewai installed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

import pytest


class TestCogDBCrewAIStorage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def _make_storage(self):
        from cogdb.adapters.crewai import CogDBCrewAIStorage

        return CogDBCrewAIStorage(agent_id="test-crew", db_path=self.tmp)

    def test_import_without_crewai(self):
        """Module must import successfully even if crewai is not installed."""
        from cogdb.adapters import crewai

        assert hasattr(crewai, "CogDBCrewAIStorage")

    def test_save_and_search(self):
        s = self._make_storage()
        s.save("The deployment failed at 3pm due to OOM error")
        results = s.search("deployment failure")
        assert isinstance(results, list)

    def test_save_multiple_and_search(self):
        s = self._make_storage()
        s.save("User prefers dark mode")
        s.save("Deployment pipeline uses GitHub Actions")
        results = s.search("deployment")
        assert isinstance(results, list)

    def test_reset(self):
        s = self._make_storage()
        s.save("Some memory to be cleared")
        s.reset()  # Should not raise

    def test_agent_id_property(self):
        s = self._make_storage()
        assert s.agent_id == "test-crew"

    def test_db_property(self):
        s = self._make_storage()
        from cogdb.core import CognitiveDB

        assert isinstance(s.db, CognitiveDB)

    def test_search_returns_strings(self):
        s = self._make_storage()
        s.save("Python is great for ML")
        results = s.search("Python")
        for r in results:
            assert isinstance(r, str)

    def test_save_with_metadata(self):
        s = self._make_storage()
        s.save("Critical bug in prod", importance=0.9)
        results = s.search("critical bug")
        assert isinstance(results, list)

    def test_search_limit_respected(self):
        s = self._make_storage()
        for i in range(5):
            s.save(f"Memory entry number {i}")
        results = s.search("memory entry", limit=3)
        assert len(results) <= 3

    def test_reset_empties_memories(self):
        s = self._make_storage()
        s.save("This should be gone after reset")
        s.reset()
        # After reset, a targeted search should return nothing
        results = s.search("gone after reset", limit=10)
        assert results == []

    def test_separate_agents_are_isolated(self):
        """Memories saved under one agent_id must not appear under another."""
        from cogdb.adapters.crewai import CogDBCrewAIStorage
        from cogdb.utils.config import CogDBConfig
        from cogdb.core import CognitiveDB

        shared_db = CognitiveDB(config=CogDBConfig(db_path=self.tmp))
        s1 = CogDBCrewAIStorage(agent_id="agent-a", db=shared_db)
        s2 = CogDBCrewAIStorage(agent_id="agent-b", db=shared_db)

        s1.save("Secret only for agent-a")
        results = s2.search("Secret only for agent-a")
        assert results == []

    def test_default_scope_is_private(self):
        from cogdb.adapters.crewai import CogDBCrewAIStorage
        from cogdb.models import MemoryScope

        s = CogDBCrewAIStorage(agent_id="test-crew", db_path=self.tmp)
        assert s._scope == MemoryScope.PRIVATE

    def test_custom_token_budget(self):
        from cogdb.adapters.crewai import CogDBCrewAIStorage

        s = CogDBCrewAIStorage(agent_id="test-crew", db_path=self.tmp, token_budget=200)
        assert s._token_budget == 200

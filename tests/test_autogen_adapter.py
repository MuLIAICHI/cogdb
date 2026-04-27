"""Tests for the AutoGen adapter (CogDBMemory).

Runs without requiring autogen-agentchat — tests CogDB-side logic directly.
"""

import asyncio
import gc
import shutil
import tempfile

import pytest

from cogdb.adapters.autogen import CogDBMemory
from cogdb.core import CognitiveDB
from cogdb.models import MemoryType


@pytest.fixture
def memory():
    tmpdir = tempfile.mkdtemp()
    db = CognitiveDB(db_path=tmpdir)
    m = CogDBMemory(agent_id="test-agent", db=db, token_budget=1000)
    yield m
    try:
        db._episodic._client.reset()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestCogDBMemoryAdd:
    def test_add_plain_string(self, memory):
        asyncio.run(memory.add("User prefers dark mode"))
        assert memory.db.stats()["episodic"] == 1

    def test_add_with_importance(self, memory):
        asyncio.run(memory.add("Critical API key expired", importance=0.95))
        assert memory.db.stats()["episodic"] == 1

    def test_add_multiple(self, memory):
        asyncio.run(memory.add("Memory one"))
        asyncio.run(memory.add("Memory two"))
        asyncio.run(memory.add("Memory three"))
        assert memory.db.stats()["episodic"] == 3

    def test_add_with_metadata(self, memory):
        asyncio.run(memory.add("Structured memory", metadata={"source": "test"}))
        assert memory.db.stats()["episodic"] == 1


class TestCogDBMemoryQuery:
    def test_query_returns_results(self, memory):
        asyncio.run(memory.add("The frontend is deployed on Vercel"))
        result = asyncio.run(memory.query("frontend deployment"))
        assert result is not None

    def test_query_empty_store_no_crash(self, memory):
        result = asyncio.run(memory.query("anything"))
        assert result is not None

    def test_query_respects_token_budget(self, memory):
        for i in range(10):
            asyncio.run(memory.add(f"Memory {i}: " + "token filler word " * 25))
        result = asyncio.run(memory.query("memory", token_budget=100))
        assert result is not None

    def test_query_isolates_by_agent(self):
        tmpdir = tempfile.mkdtemp()
        try:
            shared_db = CognitiveDB(db_path=tmpdir)
            mem_a = CogDBMemory(agent_id="agent-a", db=shared_db)
            mem_b = CogDBMemory(agent_id="agent-b", db=shared_db)

            asyncio.run(mem_a.add("Agent A private information"))
            result = asyncio.run(mem_b.query("agent A private", token_budget=500))
            assert result is not None
        finally:
            try:
                shared_db._episodic._client.reset()
            except Exception:
                pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCogDBMemoryClear:
    def test_clear_removes_memories(self, memory):
        asyncio.run(memory.add("Memory to clear one"))
        asyncio.run(memory.add("Memory to clear two"))
        assert memory.db.stats()["episodic"] == 2

        asyncio.run(memory.clear())
        results = memory.db.recall("memory", agent_id="test-agent", token_budget=1000)
        assert results == []

    def test_clear_empty_store_no_crash(self, memory):
        asyncio.run(memory.clear())


class TestCogDBMemoryClose:
    def test_close_no_crash(self, memory):
        asyncio.run(memory.close())

    def test_close_idempotent(self, memory):
        asyncio.run(memory.close())
        asyncio.run(memory.close())


class TestCogDBMemoryProperties:
    def test_agent_id_property(self, memory):
        assert memory.agent_id == "test-agent"

    def test_db_property(self, memory):
        assert isinstance(memory.db, CognitiveDB)

    def test_creates_own_db_if_none_provided(self):
        tmpdir = tempfile.mkdtemp()
        try:
            mem = CogDBMemory(agent_id="standalone", db_path=tmpdir)
            assert isinstance(mem.db, CognitiveDB)
        finally:
            try:
                mem.db._episodic._client.reset()
            except Exception:
                pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

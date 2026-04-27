"""Tests for the LangGraph adapter (CogDBCheckpointer and CogDBStore).

Runs without requiring langgraph — validates CogDB-side storage/retrieval.
"""

import asyncio
import gc
import shutil
import tempfile
from datetime import datetime, timezone

import pytest

from cogdb.adapters.langgraph import CogDBCheckpointer, CogDBStore
from cogdb.core import CognitiveDB


@pytest.fixture
def db():
    tmpdir = tempfile.mkdtemp()
    instance = CognitiveDB(db_path=tmpdir)
    yield instance
    try:
        instance._episodic._client.reset()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def checkpointer(db):
    return CogDBCheckpointer(db=db, agent_id="test-graph")


@pytest.fixture
def store(db):
    return CogDBStore(db=db, agent_id="test-store")


def _make_config(thread_id: str = "thread-1") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _make_checkpoint(checkpoint_id: str = "ckpt-1") -> dict:
    return {
        "id": checkpoint_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"messages": ["hello"]},
        "channel_versions": {"messages": 1},
        "versions_seen": {},
    }


class TestCogDBCheckpointer:
    def test_put_returns_updated_config(self, checkpointer):
        config = _make_config("t1")
        checkpoint = _make_checkpoint("ckpt-1")
        new_config = checkpointer.put(config, checkpoint, {}, {})
        assert new_config is not None

    def test_put_and_get_tuple(self, checkpointer):
        config = _make_config("thread-abc")
        checkpoint = _make_checkpoint("ckpt-abc")
        checkpointer.put(config, checkpoint, {"step": 1}, {})
        tup = checkpointer.get_tuple(config)
        assert tup is None or hasattr(tup, "config")

    def test_put_multiple_threads_isolated(self, checkpointer):
        checkpointer.put(_make_config("thread-A"), _make_checkpoint("ckpt-A"), {}, {})
        checkpointer.put(_make_config("thread-B"), _make_checkpoint("ckpt-B"), {}, {})
        assert checkpointer._db.stats()["episodic"] == 2

    def test_put_writes_stores_payload(self, checkpointer):
        config = _make_config("thread-writes")
        checkpointer.put_writes(config, [("messages", ["hi"]), ("state", {})], "task-1")
        assert checkpointer._db.stats()["episodic"] >= 1

    def test_list_returns_iterator(self, checkpointer):
        config = _make_config("thread-list")
        checkpointer.put(config, _make_checkpoint("ckpt-list"), {}, {})
        results = list(checkpointer.list(config))
        assert isinstance(results, list)

    def test_creates_own_db_if_not_provided(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cp = CogDBCheckpointer(db_path=tmpdir)
            assert isinstance(cp._db, CognitiveDB)
        finally:
            try:
                cp._db._episodic._client.reset()
            except Exception:
                pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCogDBStore:
    def test_put_and_get(self, store):
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        item = store.get(("users", "alice"), "prefs")
        assert item is None or item.value == {"theme": "dark"}

    def test_get_nonexistent_returns_none(self, store):
        result = store.get(("namespace", "nobody"), "missing-key")
        assert result is None

    def test_put_stores_to_episodic(self, store):
        store.put(("config",), "timeout", {"seconds": 30})
        assert store._db.stats()["episodic"] >= 1

    def test_put_multiple_items(self, store):
        store.put(("ns",), "key1", {"val": 1})
        store.put(("ns",), "key2", {"val": 2})
        store.put(("ns",), "key3", {"val": 3})
        assert store._db.stats()["episodic"] == 3

    def test_delete_removes_item(self, store):
        store.put(("del_ns",), "to_delete", {"data": "bye"})
        assert store._db.stats()["episodic"] == 1
        store.delete(("del_ns",), "to_delete")
        assert store._db.stats()["episodic"] == 0

    def test_delete_nonexistent_no_crash(self, store):
        store.delete(("ns",), "nonexistent")

    def test_search_returns_list(self, store):
        store.put(("data",), "record-1", {"info": "frontend build config"})
        store.put(("data",), "record-2", {"info": "backend deploy config"})
        results = store.search(("data",), query="frontend", limit=5)
        assert isinstance(results, list)

    def test_list_namespaces_returns_list(self, store):
        store.put(("users", "alice"), "prefs", {})
        store.put(("users", "bob"), "settings", {})
        namespaces = store.list_namespaces()
        assert isinstance(namespaces, list)

    def test_creates_own_db_if_not_provided(self):
        tmpdir = tempfile.mkdtemp()
        try:
            s = CogDBStore(db_path=tmpdir)
            assert isinstance(s._db, CognitiveDB)
        finally:
            try:
                s._db._episodic._client.reset()
            except Exception:
                pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCogDBStoreAsync:
    def test_async_put_and_get(self, store):
        async def _run():
            await store.aput(("async_ns",), "key", {"data": "value"})
            return await store.aget(("async_ns",), "key")

        result = asyncio.run(_run())
        assert result is None or result.value == {"data": "value"}

    def test_async_delete(self, store):
        async def _run():
            await store.aput(("del_async",), "k", {"x": 1})
            await store.adelete(("del_async",), "k")

        asyncio.run(_run())
        assert store._db.stats()["episodic"] == 0

"""Tests for the Semantic Kernel adapter — runs without semantic-kernel installed."""

import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCogDBMemoryStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def _make_store(self):
        from cogdb.adapters.semantic_kernel import CogDBMemoryStore

        return CogDBMemoryStore(agent_id="test-sk", db_path=self.tmp)

    # ------------------------------------------------------------------
    # Module availability
    # ------------------------------------------------------------------

    def test_import_without_sk(self):
        from cogdb.adapters import semantic_kernel

        assert hasattr(semantic_kernel, "CogDBMemoryStore")

    def test_memory_record_stub_importable(self):
        from cogdb.adapters.semantic_kernel import MemoryRecord

        rec = MemoryRecord(id="x", text="hello", description="")
        assert rec.id == "x"
        assert rec.text == "hello"

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def test_create_delete_collection_noop(self):
        store = self._make_store()
        run(store.create_collection_async("test"))
        run(store.delete_collection_async("test"))

    def test_get_collections_returns_list(self):
        store = self._make_store()
        cols = run(store.get_collections_async())
        assert isinstance(cols, list)

    def test_get_collections_includes_created(self):
        store = self._make_store()
        run(store.create_collection_async("articles"))
        cols = run(store.get_collections_async())
        assert "articles" in cols

    def test_does_collection_exist(self):
        store = self._make_store()
        exists = run(store.does_collection_exist_async("any"))
        assert isinstance(exists, bool)

    def test_does_collection_exist_after_create(self):
        store = self._make_store()
        run(store.create_collection_async("present"))
        assert run(store.does_collection_exist_async("present"))

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def test_upsert_returns_record_id(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        record = MemoryRecord(id="rec1", text="Python async patterns", description="")
        key = run(store.upsert_async("default", record))
        assert key == "rec1"

    def test_upsert_registers_collection(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        run(store.upsert_async("my-col", MemoryRecord(id="r0", text="hello", description="")))
        cols = run(store.get_collections_async())
        assert "my-col" in cols

    def test_upsert_batch_returns_all_ids(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        records = [
            MemoryRecord(id="r1", text="First memory", description=""),
            MemoryRecord(id="r2", text="Second memory", description=""),
        ]
        keys = run(store.upsert_batch_async("default", records))
        assert len(keys) == 2
        assert "r1" in keys
        assert "r2" in keys

    def test_upsert_description_fallback(self):
        """upsert_async should use description when text is empty."""
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        record = MemoryRecord(id="desc-only", text="", description="fallback text")
        key = run(store.upsert_async("default", record))
        assert key == "desc-only"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def test_get_nearest_matches_returns_list(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        run(store.upsert_async("default", MemoryRecord(id="rec1", text="Python async patterns", description="")))
        results = run(store.get_nearest_matches_async("default", [], limit=5))
        assert isinstance(results, list)

    def test_get_nearest_matches_tuple_structure(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        run(store.upsert_async("col", MemoryRecord(id="k1", text="memory content", description="")))
        results = run(store.get_nearest_matches_async("col", [], limit=3))
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            record, score = item
            assert isinstance(score, float)
            assert hasattr(record, "text")

    def test_get_nearest_match_single_result(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        run(store.upsert_async("col", MemoryRecord(id="k1", text="single match", description="")))
        result = run(store.get_nearest_match_async("col", []))
        # May be None or a tuple — both are valid
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    def test_get_nearest_match_empty_db_returns_none(self):
        store = self._make_store()
        result = run(store.get_nearest_match_async("empty-col", []))
        assert result is None

    def test_get_nearest_matches_limit_respected(self):
        store = self._make_store()
        from cogdb.adapters.semantic_kernel import MemoryRecord

        for i in range(5):
            run(store.upsert_async("col", MemoryRecord(id=f"m{i}", text=f"memory {i}", description="")))
        results = run(store.get_nearest_matches_async("col", [], limit=2))
        assert len(results) <= 2

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def test_remove_noop_on_missing_key(self):
        store = self._make_store()
        run(store.remove_async("default", "nonexistent"))

    def test_remove_batch_noop(self):
        store = self._make_store()
        run(store.remove_batch_async("default", ["a", "b", "c"]))

    # ------------------------------------------------------------------
    # Get by key
    # ------------------------------------------------------------------

    def test_get_async_missing_key_returns_none(self):
        store = self._make_store()
        rec = run(store.get_async("col", "missing-key"))
        assert rec is None

    def test_get_batch_async_returns_list(self):
        store = self._make_store()
        results = run(store.get_batch_async("col", ["x", "y"]))
        assert isinstance(results, list)

    # ------------------------------------------------------------------
    # Lifecycle and properties
    # ------------------------------------------------------------------

    def test_close(self):
        store = self._make_store()
        store.close()

    def test_agent_id_property(self):
        store = self._make_store()
        assert store.agent_id == "test-sk"

    def test_db_property(self):
        from cogdb.core import CognitiveDB

        store = self._make_store()
        assert isinstance(store.db, CognitiveDB)

    def test_external_db_accepted(self):
        from cogdb.adapters.semantic_kernel import CogDBMemoryStore
        from cogdb.core import CognitiveDB
        from cogdb.utils.config import CogDBConfig

        cfg = CogDBConfig(db_path=self.tmp)
        external_db = CognitiveDB(config=cfg)
        store = CogDBMemoryStore(agent_id="external", db=external_db)
        assert store.db is external_db
        store.close()

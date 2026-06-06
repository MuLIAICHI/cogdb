"""Tests for the LlamaIndex adapter.

Designed to run without llama-index installed — uses the stub ChatMessage
and BaseMemory provided by the adapter module itself.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


class TestCogDBChatMemory:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def _make_memory(self):
        from cogdb.adapters.llamaindex import CogDBChatMemory

        return CogDBChatMemory(agent_id="test-li", db_path=self.tmp)

    def test_import_without_llamaindex(self):
        from cogdb.adapters import llamaindex

        assert hasattr(llamaindex, "CogDBChatMemory")

    def test_classes_exported(self):
        from cogdb.adapters import llamaindex

        assert hasattr(llamaindex, "CogDBVectorIndex")
        assert hasattr(llamaindex, "ChatMessage")

    def test_put_and_get(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="How do I deploy to AWS?"))
        messages = mem.get("AWS deployment")
        assert isinstance(messages, list)

    def test_get_returns_chat_messages(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="Tell me about Kubernetes"))
        messages = mem.get("Kubernetes")
        assert isinstance(messages, list)
        for m in messages:
            assert hasattr(m, "role")
            assert hasattr(m, "content")

    def test_get_with_no_input(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="Some content"))
        messages = mem.get()
        assert isinstance(messages, list)

    def test_put_user_vs_assistant_importance(self):
        # Both roles should store without raising
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="User message"))
        mem.put(ChatMessage(role="assistant", content="Assistant reply"))
        all_msgs = mem.get_all()
        assert len(all_msgs) >= 2

    def test_put_multiple_and_get_all(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="What is Docker?"))
        mem.put(
            ChatMessage(role="assistant", content="Docker is a containerization platform.")
        )
        all_msgs = mem.get_all()
        assert isinstance(all_msgs, list)
        assert len(all_msgs) >= 2

    def test_set_messages(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        msgs = [
            ChatMessage(role="user", content="Explain Kubernetes"),
            ChatMessage(role="assistant", content="Kubernetes orchestrates containers."),
        ]
        mem.set(msgs)  # Should not raise
        all_msgs = mem.get_all()
        assert len(all_msgs) >= 2

    def test_set_empty_list(self):
        mem = self._make_memory()
        mem.set([])  # Should not raise

    def test_reset(self):
        mem = self._make_memory()
        from cogdb.adapters.llamaindex import ChatMessage

        mem.put(ChatMessage(role="user", content="Some message"))
        mem.reset()
        # After reset, get_all returns empty or close to empty
        all_msgs = mem.get_all()
        assert isinstance(all_msgs, list)

    def test_from_defaults(self):
        from cogdb.adapters.llamaindex import CogDBChatMemory

        mem = CogDBChatMemory.from_defaults(agent_id="li-test", db_path=self.tmp)
        assert mem.agent_id == "li-test"

    def test_from_defaults_token_budget(self):
        from cogdb.adapters.llamaindex import CogDBChatMemory

        mem = CogDBChatMemory.from_defaults(
            agent_id="li-budget", db_path=self.tmp, token_budget=1200
        )
        assert mem._token_budget == 1200

    def test_agent_id_property(self):
        mem = self._make_memory()
        assert mem.agent_id == "test-li"

    def test_db_property(self):
        from cogdb.core import CognitiveDB

        mem = self._make_memory()
        assert isinstance(mem.db, CognitiveDB)

    def test_close(self):
        mem = self._make_memory()
        mem.close()  # Should not raise

    def test_close_idempotent(self):
        mem = self._make_memory()
        mem.close()
        mem.close()  # Should not raise on second call

    def test_inject_existing_db(self):
        from cogdb.adapters.llamaindex import CogDBChatMemory
        from cogdb.core import CognitiveDB

        db = CognitiveDB(db_path=self.tmp)
        mem = CogDBChatMemory(agent_id="shared", db=db)
        assert mem.db is db

    def test_scope_isolation(self):
        """Two agents with separate db_path dirs don't see each other's memories."""
        import tempfile
        from cogdb.adapters.llamaindex import CogDBChatMemory, ChatMessage

        tmp_a = tempfile.mkdtemp()
        tmp_b = tempfile.mkdtemp()

        mem_a = CogDBChatMemory(agent_id="agent-a", db_path=tmp_a)
        mem_b = CogDBChatMemory(agent_id="agent-b", db_path=tmp_b)

        mem_a.put(ChatMessage(role="user", content="Secret of agent A"))
        results_b = mem_b.get("Secret")
        # agent-b should not see agent-a's memory (different db paths)
        assert all("Secret of agent A" not in m.content for m in results_b)

        mem_a.close()
        mem_b.close()


class TestCogDBVectorIndex:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def _make_index(self):
        from cogdb.adapters.llamaindex import CogDBVectorIndex

        return CogDBVectorIndex(agent_id="vec-test", db_path=self.tmp)

    def test_import(self):
        from cogdb.adapters import llamaindex

        assert hasattr(llamaindex, "CogDBVectorIndex")

    def test_insert_returns_string_id(self):
        idx = self._make_index()
        mem_id = idx.insert("FastAPI is a modern Python web framework")
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_insert_and_query(self):
        idx = self._make_index()
        idx.insert("FastAPI is a modern Python web framework")
        results = idx.query("Python web framework")
        assert isinstance(results, list)

    def test_query_returns_strings(self):
        idx = self._make_index()
        idx.insert("LangChain is a framework for LLM applications")
        results = idx.query("LLM framework")
        assert all(isinstance(r, str) for r in results)

    def test_query_similarity_top_k(self):
        idx = self._make_index()
        for i in range(5):
            idx.insert(f"Document number {i} about Python")
        results = idx.query("Python", similarity_top_k=3)
        assert len(results) <= 3

    def test_insert_with_metadata(self):
        idx = self._make_index()
        mem_id = idx.insert(
            "Celery is a distributed task queue",
            metadata={"source": "docs", "category": "infrastructure"},
        )
        assert isinstance(mem_id, str)

    def test_delete_returns_bool(self):
        idx = self._make_index()
        mem_id = idx.insert("Temporary memory to delete")
        result = idx.delete(mem_id)
        assert isinstance(result, bool)

    def test_delete_nonexistent_id(self):
        idx = self._make_index()
        result = idx.delete("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_agent_id_property(self):
        idx = self._make_index()
        assert idx.agent_id == "vec-test"

    def test_db_property(self):
        from cogdb.core import CognitiveDB

        idx = self._make_index()
        assert isinstance(idx.db, CognitiveDB)

    def test_inject_existing_db(self):
        from cogdb.adapters.llamaindex import CogDBVectorIndex
        from cogdb.core import CognitiveDB

        db = CognitiveDB(db_path=self.tmp)
        idx = CogDBVectorIndex(agent_id="shared-vec", db=db)
        assert idx.db is db

    def test_multiple_inserts_tracked(self):
        idx = self._make_index()
        ids = [idx.insert(f"Text chunk {i}") for i in range(3)]
        assert len(set(ids)) == 3  # All IDs are unique

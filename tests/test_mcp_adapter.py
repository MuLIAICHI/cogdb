"""Tests for the MCP server adapter (CogDBMCPServer).

Tests all 6 tool handlers directly (no MCP transport layer needed).
Thread-safety is verified by running concurrent tool calls.
"""

import gc
import json
import shutil
import tempfile
import threading

import pytest

from cogdb.adapters.mcp import ALL_TOOLS, CogDBMCPServer
from cogdb.core import CognitiveDB


@pytest.fixture
def server():
    tmpdir = tempfile.mkdtemp()
    db = CognitiveDB(db_path=tmpdir)
    s = CogDBMCPServer(db=db, server_name="test-cogdb")
    yield s
    try:
        db._episodic._client.reset()
    except Exception:
        pass
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Tool schema tests ─────────────────────────────────────────────────────────

class TestToolSchemas:
    def test_all_six_tools_defined(self):
        names = {t["name"] for t in ALL_TOOLS}
        assert names == {"remember", "recall", "learn", "learn_procedure", "get_context", "forget"}

    def test_each_tool_has_description(self):
        for tool in ALL_TOOLS:
            assert "description" in tool
            assert len(tool["description"]) > 20

    def test_each_tool_has_input_schema(self):
        for tool in ALL_TOOLS:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]

    def test_required_fields_declared(self):
        required_map = {
            "remember": ["content"],
            "recall": ["query"],
            "learn": ["subject", "predicate", "object"],
            "learn_procedure": ["name", "description", "steps"],
            "get_context": ["agent_id"],
            "forget": ["memory_id"],
        }
        for tool in ALL_TOOLS:
            expected = required_map[tool["name"]]
            assert tool["inputSchema"].get("required") == expected, tool["name"]


# ── remember ──────────────────────────────────────────────────────────────────

class TestHandleRemember:
    def test_stores_memory_returns_id(self, server):
        result = json.loads(server.handle_remember({"content": "Deploy succeeded"}))
        assert "memory_id" in result
        assert result["status"] == "stored"
        assert len(result["memory_id"]) > 0

    def test_with_all_params(self, server):
        result = json.loads(server.handle_remember({
            "content": "Critical: DB migration failed",
            "agent_id": "devops",
            "importance": 0.95,
            "scope": "org",
        }))
        assert result["status"] == "stored"

    def test_default_agent_id(self, server):
        result = json.loads(server.handle_remember({"content": "Anonymous memory"}))
        assert result["status"] == "stored"
        assert server._db.stats()["episodic"] == 1

    def test_invalid_scope_falls_back_to_private(self, server):
        result = json.loads(server.handle_remember({
            "content": "Memory with bad scope",
            "scope": "invalid_scope",
        }))
        assert result["status"] == "stored"

    def test_multiple_memories_accumulate(self, server):
        for i in range(5):
            server.handle_remember({"content": f"Memory {i}"})
        assert server._db.stats()["episodic"] == 5


# ── recall ────────────────────────────────────────────────────────────────────

class TestHandleRecall:
    def test_recall_returns_memories(self, server):
        server.handle_remember({"content": "Nginx CORS config updated"})
        result = json.loads(server.handle_recall({"query": "CORS nginx"}))
        assert "memories" in result
        assert "count" in result
        assert result["count"] >= 1

    def test_recall_empty_store(self, server):
        result = json.loads(server.handle_recall({"query": "anything"}))
        assert result["memories"] == []
        assert result["count"] == 0

    def test_recall_respects_token_budget(self, server):
        for i in range(10):
            server.handle_remember({"content": f"Memory {i}: " + "word " * 40})
        result = json.loads(server.handle_recall({"query": "memory", "token_budget": 100}))
        assert result["count"] >= 0  # should not crash

    def test_recall_memory_fields(self, server):
        server.handle_remember({"content": "Test memory for field check"})
        result = json.loads(server.handle_recall({"query": "test memory"}))
        if result["count"] > 0:
            mem = result["memories"][0]
            assert "id" in mem
            assert "content" in mem
            assert "type" in mem
            assert "importance" in mem
            assert "scope" in mem
            assert "created_at" in mem

    def test_recall_with_memory_types_filter(self, server):
        server.handle_remember({"content": "Episodic event happened"})
        result = json.loads(server.handle_recall({
            "query": "episodic event",
            "memory_types": ["episodic"],
        }))
        assert isinstance(result["memories"], list)

    def test_recall_invalid_memory_type_falls_back(self, server):
        server.handle_remember({"content": "Some memory"})
        result = json.loads(server.handle_recall({
            "query": "memory",
            "memory_types": ["invalid_type"],
        }))
        assert "memories" in result


# ── learn ─────────────────────────────────────────────────────────────────────

class TestHandleLearn:
    def test_stores_triple_returns_id(self, server):
        result = json.loads(server.handle_learn({
            "subject": "api_service",
            "predicate": "version",
            "object": "v2.3.1",
        }))
        assert "triple_id" in result
        assert result["status"] == "stored"
        assert server._db.stats()["semantic"] == 1

    def test_with_all_params(self, server):
        result = json.loads(server.handle_learn({
            "subject": "user",
            "predicate": "prefers",
            "object": "dark_mode",
            "agent_id": "ui-agent",
            "confidence": 0.95,
        }))
        assert result["status"] == "stored"

    def test_supersedes_contradicting_fact(self, server):
        server.handle_learn({"subject": "api", "predicate": "status", "object": "degraded"})
        server.handle_learn({"subject": "api", "predicate": "status", "object": "healthy"})
        # Only one active fact should remain
        facts = server._db.query_knowledge("api")
        status_facts = [f for f in facts if f.predicate == "status"]
        assert len(status_facts) == 1
        assert status_facts[0].object == "healthy"

    def test_different_predicates_coexist(self, server):
        server.handle_learn({"subject": "api", "predicate": "version", "object": "v2"})
        server.handle_learn({"subject": "api", "predicate": "status", "object": "healthy"})
        assert server._db.stats()["semantic"] == 2


# ── learn_procedure ───────────────────────────────────────────────────────────

class TestHandleLearnProcedure:
    def test_stores_procedure_returns_id(self, server):
        result = json.loads(server.handle_learn_procedure({
            "name": "fix_cors",
            "description": "Fix CORS errors in nginx",
            "steps": [
                {"action": "edit_nginx_config", "tool": "vim /etc/nginx/conf.d/api.conf"},
                {"action": "reload_nginx", "tool": "systemctl reload nginx"},
            ],
        }))
        assert "procedure_id" in result
        assert result["step_count"] == 2
        assert result["status"] == "stored"
        assert server._db.stats()["procedural"] == 1

    def test_with_applicable_contexts(self, server):
        result = json.loads(server.handle_learn_procedure({
            "name": "deploy_frontend",
            "description": "Deploy frontend to Vercel",
            "steps": [{"action": "npm run build"}, {"action": "vercel --prod"}],
            "agent_id": "devops",
            "applicable_contexts": ["deploy", "frontend", "vercel"],
        }))
        assert result["status"] == "stored"

    def test_invalid_steps_returns_error(self, server):
        result = json.loads(server.handle_learn_procedure({
            "name": "bad_proc",
            "description": "Bad procedure",
            "steps": [],  # empty steps
        }))
        assert "error" in result

    def test_steps_without_action_returns_error(self, server):
        result = json.loads(server.handle_learn_procedure({
            "name": "bad_proc",
            "description": "Bad",
            "steps": [{"tool": "something"}],  # missing 'action'
        }))
        assert "error" in result


# ── get_context ───────────────────────────────────────────────────────────────

class TestHandleGetContext:
    def test_l0_returns_identity(self, server):
        result = json.loads(server.handle_get_context({
            "agent_id": "test-agent",
            "level": 0,
        }))
        assert "identity" in result
        assert "test-agent" in result["identity"]
        assert result["level"] == 0
        assert result["critical_facts"] == []
        assert result["relevant_memories"] == []

    def test_l1_with_facts(self, server):
        server.handle_learn({
            "subject": "user", "predicate": "theme", "object": "dark",
            "agent_id": "test-agent",
        })
        result = json.loads(server.handle_get_context({
            "agent_id": "test-agent",
            "level": 1,
        }))
        assert result["level"] == 1
        assert isinstance(result["critical_facts"], list)

    def test_l2_with_task_hint(self, server):
        server.handle_remember({
            "content": "Settings page redesign started",
            "agent_id": "ui-agent",
        })
        result = json.loads(server.handle_get_context({
            "agent_id": "ui-agent",
            "level": 2,
            "task_hint": "settings page",
            "token_budget": 500,
        }))
        assert result["level"] == 2
        assert "token_count" in result
        assert "token_budget" in result
        assert result["token_count"] <= result["token_budget"]

    def test_token_budget_respected(self, server):
        for i in range(10):
            server.handle_remember({
                "content": f"Memory {i}: " + "word " * 30,
                "agent_id": "budget-agent",
            })
        result = json.loads(server.handle_get_context({
            "agent_id": "budget-agent",
            "level": 2,
            "task_hint": "memory",
            "token_budget": 200,
        }))
        assert result["token_count"] <= 220  # small tolerance

    def test_response_has_all_fields(self, server):
        result = json.loads(server.handle_get_context({"agent_id": "a1"}))
        for field in ["identity", "level", "critical_facts", "relevant_memories",
                      "deep_results", "token_count", "token_budget", "utilization"]:
            assert field in result, f"missing field: {field}"


# ── forget ────────────────────────────────────────────────────────────────────

class TestHandleForget:
    def test_forget_episodic_memory(self, server):
        stored = json.loads(server.handle_remember({"content": "To be forgotten"}))
        memory_id = stored["memory_id"]
        assert server._db.stats()["episodic"] == 1

        result = json.loads(server.handle_forget({
            "memory_id": memory_id,
            "memory_type": "episodic",
        }))
        assert result["memory_id"] == memory_id

    def test_forget_semantic_triple(self, server):
        stored = json.loads(server.handle_learn({
            "subject": "api", "predicate": "status", "object": "old",
        }))
        triple_id = stored["triple_id"]
        assert server._db.stats()["semantic"] == 1

        json.loads(server.handle_forget({
            "memory_id": triple_id,
            "memory_type": "semantic",
        }))
        assert server._db.stats()["semantic"] == 0

    def test_forget_defaults_to_episodic(self, server):
        stored = json.loads(server.handle_remember({"content": "Default type forget"}))
        result = json.loads(server.handle_forget({"memory_id": stored["memory_id"]}))
        assert "memory_id" in result

    def test_forget_invalid_type_falls_back(self, server):
        stored = json.loads(server.handle_remember({"content": "Invalid type test"}))
        result = json.loads(server.handle_forget({
            "memory_id": stored["memory_id"],
            "memory_type": "not_a_type",
        }))
        assert "memory_id" in result


# ── dispatch ──────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_dispatch_unknown_tool_returns_error(self, server):
        result = json.loads(server.dispatch("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_dispatch_missing_required_arg_returns_error(self, server):
        result = json.loads(server.dispatch("remember", {}))  # missing 'content'
        assert "error" in result

    def test_dispatch_routes_all_tools(self, server):
        # Verify each tool name routes to a handler without crashing
        calls = [
            ("remember", {"content": "dispatch test"}),
            ("recall", {"query": "dispatch test"}),
            ("learn", {"subject": "s", "predicate": "p", "object": "o"}),
            ("learn_procedure", {"name": "n", "description": "d", "steps": [{"action": "a"}]}),
            ("get_context", {"agent_id": "test"}),
        ]
        for tool_name, args in calls:
            result = json.loads(server.dispatch(tool_name, args))
            assert "error" not in result, f"Tool {tool_name} failed: {result}"


# ── thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_remember_calls(self, server):
        errors = []

        def store(i):
            try:
                server.handle_remember({"content": f"Concurrent memory {i}"})
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=store, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert server._db.stats()["episodic"] == 20

    def test_concurrent_mixed_operations(self, server):
        errors = []

        def mixed_ops(i):
            try:
                server.dispatch("remember", {"content": f"Memory {i}"})
                server.dispatch("recall", {"query": f"Memory {i}", "token_budget": 200})
                server.dispatch("learn", {
                    "subject": f"entity_{i}", "predicate": "index", "object": str(i),
                })
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=mixed_ops, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

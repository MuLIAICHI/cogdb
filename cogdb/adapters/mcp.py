"""MCP server adapter — exposes CognitiveDB as Model Context Protocol tools.

Provides 6 tools that any MCP-compatible client (Claude, Cursor, etc.)
can call to interact with agent memory:

  remember(content, agent_id, importance, scope)                    → memory_id
  recall(query, agent_id, token_budget, memory_types)               → memories[]
  learn(subject, predicate, object, agent_id, confidence)           → triple_id
  learn_procedure(name, description, steps, agent_id, contexts)     → procedure_id
  get_context(agent_id, level, task_hint, token_budget)             → context
  forget(memory_id, memory_type)                                    → bool

Install: pip install mcp>=1.0.0
Run:     cogdb-mcp --db-path ./my_memory
         python -m cogdb.adapters.mcp --db-path ./my_memory
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# ── Tool schemas ──────────────────────────────────────────────────────────────

REMEMBER_TOOL = {
    "name": "remember",
    "description": (
        "Store an episodic memory — a timestamped record of an event, observation, "
        "or interaction. Use this when something important happened and should be "
        "retrievable later (e.g. 'deployment failed', 'user complained about X', "
        "'task completed successfully'). Returns the memory ID."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content to remember. Be descriptive — this is what gets searched later.",
            },
            "agent_id": {
                "type": "string",
                "description": "Identifier of the agent storing this memory.",
                "default": "default",
            },
            "importance": {
                "type": "number",
                "description": "How important is this memory? 0.0 = trivial, 1.0 = critical. Default 0.5.",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "scope": {
                "type": "string",
                "description": (
                    "Who can see this memory: "
                    "'private' (only this agent), "
                    "'team' (agents in same team), "
                    "'org' (all agents), "
                    "'session' (ephemeral, auto-deleted)."
                ),
                "enum": ["private", "team", "org", "session"],
                "default": "private",
            },
        },
        "required": ["content"],
    },
}

RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Retrieve memories relevant to a query, ranked by importance and fit within "
        "a token budget. Use this before starting a task to load relevant context, "
        "or when you need to find what was previously stored about a topic. "
        "Returns a list of memories sorted by relevance."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query describing what you want to recall.",
            },
            "agent_id": {
                "type": "string",
                "description": "Agent whose memories to search.",
                "default": "default",
            },
            "token_budget": {
                "type": "integer",
                "description": "Maximum tokens to return. Higher = more memories. Default 1000.",
                "default": 1000,
                "minimum": 50,
            },
            "memory_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural"],
                },
                "description": (
                    "Which memory types to search. "
                    "episodic = past events, semantic = facts, procedural = workflows. "
                    "Omit to search episodic + semantic."
                ),
            },
        },
        "required": ["query"],
    },
}

LEARN_TOOL = {
    "name": "learn",
    "description": (
        "Store a structured fact in the knowledge graph as a (subject, predicate, object) triple. "
        "Use this for facts that should be queryable as structured knowledge, not just text "
        "(e.g. 'api_service deployed_version v2.3', 'user prefers dark_mode', "
        "'database hosted_on aws_rds'). "
        "If a fact with the same subject+predicate already exists, it is automatically superseded. "
        "Returns the triple ID."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The entity this fact is about (e.g. 'api_service', 'user', 'nginx').",
            },
            "predicate": {
                "type": "string",
                "description": "The relationship or property (e.g. 'version', 'prefers', 'hosted_on').",
            },
            "object": {
                "type": "string",
                "description": "The value or target entity (e.g. 'v2.3.1', 'dark_mode', 'aws_rds').",
            },
            "agent_id": {
                "type": "string",
                "description": "Agent asserting this fact.",
                "default": "default",
            },
            "confidence": {
                "type": "number",
                "description": "How confident is the agent in this fact? 0.0–1.0. Default 1.0.",
                "default": 1.0,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["subject", "predicate", "object"],
    },
}

LEARN_PROCEDURE_TOOL = {
    "name": "learn_procedure",
    "description": (
        "Store a learned workflow or multi-step procedure so it can be reused on similar tasks. "
        "Use this after successfully solving a multi-step problem to capture the solution pattern "
        "(e.g. 'fix_cors_error', 'deploy_to_production', 'rollback_database'). "
        "The procedure will be suggested when similar contexts are encountered. "
        "Returns the procedure ID."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short identifier for the procedure (e.g. 'fix_cors_error').",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what this procedure does.",
            },
            "steps": {
                "type": "array",
                "description": "Ordered list of steps. Each step has 'action' (required) and optional 'tool', 'parameters'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "What to do in this step.",
                        },
                        "tool": {
                            "type": "string",
                            "description": "CLI command or tool to use (optional).",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Key-value parameters for this step (optional).",
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "What a successful output looks like (optional).",
                        },
                    },
                    "required": ["action"],
                },
                "minItems": 1,
            },
            "agent_id": {
                "type": "string",
                "description": "Agent that learned this procedure.",
                "default": "default",
            },
            "applicable_contexts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keywords describing when to suggest this procedure (e.g. ['cors', 'nginx', 'api']).",
            },
        },
        "required": ["name", "description", "steps"],
    },
}

GET_CONTEXT_TOOL = {
    "name": "get_context",
    "description": (
        "Load progressive memory context for an agent before starting a task. "
        "Returns tiered context: identity (L0), critical facts (L1), "
        "task-relevant memories (L2), deep search results (L3). "
        "Use this at the start of a conversation or task to load all relevant agent memory. "
        "Higher levels include more context but use more tokens."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The agent requesting context.",
                "default": "default",
            },
            "level": {
                "type": "integer",
                "description": (
                    "How deep to load context: "
                    "0 = identity only, "
                    "1 = + critical facts, "
                    "2 = + task-relevant memories (recommended), "
                    "3 = + deep similarity search."
                ),
                "default": 2,
                "minimum": 0,
                "maximum": 3,
            },
            "task_hint": {
                "type": "string",
                "description": "Brief description of the current task. Improves relevance at L2/L3.",
            },
            "token_budget": {
                "type": "integer",
                "description": "Max tokens for the context response. Default 1000.",
                "default": 1000,
                "minimum": 50,
            },
        },
        "required": ["agent_id"],
    },
}

FORGET_TOOL = {
    "name": "forget",
    "description": (
        "Permanently delete a specific memory by its ID. "
        "Use this to remove outdated, incorrect, or sensitive information. "
        "Requires the memory_id (returned by remember/learn/learn_procedure) "
        "and the memory_type to identify which store to delete from."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The UUID of the memory to delete.",
            },
            "memory_type": {
                "type": "string",
                "description": "Which store to delete from: episodic | semantic | procedural.",
                "enum": ["episodic", "semantic", "procedural"],
                "default": "episodic",
            },
        },
        "required": ["memory_id"],
    },
}

ALL_TOOLS = [
    REMEMBER_TOOL,
    RECALL_TOOL,
    LEARN_TOOL,
    LEARN_PROCEDURE_TOOL,
    GET_CONTEXT_TOOL,
    FORGET_TOOL,
]


# ── Server ────────────────────────────────────────────────────────────────────


class CogDBMCPServer:
    """MCP server wrapping CognitiveDB with 6 memory tools.

    Thread-safe: a single CognitiveDB instance is shared across all tool calls.
    The underlying stores use threading.Lock internally.

    Args:
        db: An existing CognitiveDB instance. If None, creates one from db_path.
        db_path: Storage path for a new CognitiveDB instance.
        server_name: MCP server name shown in client UIs.

    Example:
        >>> server = CogDBMCPServer(db_path="./my_memory")
        >>> import asyncio
        >>> asyncio.run(server.run())
    """

    def __init__(
        self,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_mcp",
        server_name: str = "cogdb",
    ) -> None:
        self._db = db or CognitiveDB(db_path=db_path)
        self._server_name = server_name
        # Extra lock for handler dispatch — stores are already thread-safe,
        # but this protects any multi-step operations in handlers.
        self._lock = threading.Lock()

    # ── Tool handlers ─────────────────────────────────────────────────────────

    def handle_remember(self, args: dict[str, Any]) -> str:
        """Store an episodic memory.

        Args:
            args: Tool arguments (content, agent_id, importance, scope).

        Returns:
            JSON with memory_id and status.

        Example:
            >>> server.handle_remember({"content": "Deploy succeeded", "importance": 0.8})
            '{"memory_id": "...", "status": "stored"}'
        """
        try:
            scope = MemoryScope(args.get("scope", "private"))
        except ValueError:
            scope = MemoryScope.PRIVATE

        memory_id = self._db.remember(
            content=args["content"],
            agent_id=args.get("agent_id", "default"),
            importance=float(args.get("importance", 0.5)),
            scope=scope,
        )
        return json.dumps({"memory_id": memory_id, "status": "stored"})

    def handle_recall(self, args: dict[str, Any]) -> str:
        """Retrieve memories matching a query within a token budget.

        Args:
            args: Tool arguments (query, agent_id, token_budget, memory_types).

        Returns:
            JSON with memories list and count.

        Example:
            >>> server.handle_recall({"query": "CORS errors", "token_budget": 500})
        """
        raw_types = args.get("memory_types")
        if raw_types:
            try:
                memory_types = [MemoryType(t) for t in raw_types]
            except ValueError:
                memory_types = None
        else:
            memory_types = None

        memories = self._db.recall(
            query=args["query"],
            agent_id=args.get("agent_id", "default"),
            token_budget=int(args.get("token_budget", 1000)),
            memory_types=memory_types,
        )

        return json.dumps({
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": round(m.importance, 3),
                    "scope": m.scope.value,
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ],
            "count": len(memories),
        })

    def handle_learn(self, args: dict[str, Any]) -> str:
        """Store a semantic fact in the knowledge graph.

        Args:
            args: Tool arguments (subject, predicate, object, agent_id, confidence).

        Returns:
            JSON with triple_id and status.

        Example:
            >>> server.handle_learn({"subject": "api", "predicate": "version", "object": "v2"})
        """
        triple_id = self._db.learn(
            subject=args["subject"],
            predicate=args["predicate"],
            object=args["object"],
            agent_id=args.get("agent_id", "default"),
            confidence=float(args.get("confidence", 1.0)),
        )
        return json.dumps({"triple_id": triple_id, "status": "stored"})

    def handle_learn_procedure(self, args: dict[str, Any]) -> str:
        """Store a learned workflow template.

        Args:
            args: Tool arguments (name, description, steps, agent_id, applicable_contexts).

        Returns:
            JSON with procedure_id, step_count, and status.

        Example:
            >>> server.handle_learn_procedure({
            ...     "name": "fix_cors", "description": "Fix CORS in nginx",
            ...     "steps": [{"action": "edit_config"}, {"action": "reload_nginx"}]
            ... })
        """
        steps = args.get("steps", [])
        # Validate steps are dicts with at least an 'action' key
        if not steps or not all(isinstance(s, dict) and "action" in s for s in steps):
            return json.dumps({"error": "steps must be a non-empty list of objects with 'action' key"})

        procedure_id = self._db.learn_procedure(
            name=args["name"],
            description=args["description"],
            steps=steps,
            agent_id=args.get("agent_id", "default"),
            applicable_contexts=args.get("applicable_contexts", []),
        )
        return json.dumps({
            "procedure_id": procedure_id,
            "step_count": len(steps),
            "status": "stored",
        })

    def handle_get_context(self, args: dict[str, Any]) -> str:
        """Load progressive memory context for an agent.

        Args:
            args: Tool arguments (agent_id, level, task_hint, token_budget).

        Returns:
            JSON with identity, critical_facts, relevant_memories, token usage.

        Example:
            >>> server.handle_get_context({"agent_id": "ui-agent", "level": 2, "task_hint": "settings page"})
        """
        ctx = self._db.get_context(
            agent_id=args.get("agent_id", "default"),
            level=int(args.get("level", 2)),
            task_hint=args.get("task_hint"),
            token_budget=int(args.get("token_budget", 1000)),
        )

        return json.dumps({
            "identity": ctx.identity,
            "level": ctx.level,
            "critical_facts": ctx.critical_facts,
            "relevant_memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": round(m.importance, 3),
                }
                for m in ctx.relevant_memories
            ],
            "deep_results": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": round(m.importance, 3),
                }
                for m in ctx.deep_results
            ],
            "token_count": ctx.token_count,
            "token_budget": ctx.token_budget,
            "utilization": round(ctx.utilization, 3),
        })

    def handle_forget(self, args: dict[str, Any]) -> str:
        """Delete a specific memory.

        Args:
            args: Tool arguments (memory_id, memory_type).

        Returns:
            JSON with deleted status and memory_id.

        Example:
            >>> server.handle_forget({"memory_id": "abc-123", "memory_type": "episodic"})
        """
        try:
            memory_type = MemoryType(args.get("memory_type", "episodic"))
        except ValueError:
            memory_type = MemoryType.EPISODIC

        deleted = self._db.forget(args["memory_id"], memory_type)
        return json.dumps({"deleted": deleted, "memory_id": args["memory_id"]})

    # ── Dispatch table ────────────────────────────────────────────────────────

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the correct handler.

        Args:
            name: Tool name.
            arguments: Tool arguments from the MCP client.

        Returns:
            JSON result string.

        Example:
            >>> server.dispatch("remember", {"content": "test"})
        """
        handlers = {
            "remember": self.handle_remember,
            "recall": self.handle_recall,
            "learn": self.handle_learn,
            "learn_procedure": self.handle_learn_procedure,
            "get_context": self.handle_get_context,
            "forget": self.handle_forget,
        }
        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            with self._lock:
                return handler(arguments)
        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── MCP server lifecycle ──────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the MCP server over stdio.

        Blocks until the client disconnects. Registers all 6 tools
        and handles list_tools / call_tool MCP requests.

        Example:
            >>> import asyncio
            >>> asyncio.run(CogDBMCPServer(db_path="./mem").run())
        """
        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "mcp package not installed. Run: pip install 'cogdb[mcp]'"
            )

        server = Server(self._server_name)

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [Tool(**t) for t in ALL_TOOLS]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            result = self.dispatch(name, arguments or {})
            return [TextContent(type="text", text=result)]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — registered as `cogdb-mcp` in pyproject.toml.

    Usage:
        cogdb-mcp --db-path ./my_memory
        cogdb-mcp --db-path ./my_memory --server-name my-agent-memory
    """
    import asyncio

    parser = argparse.ArgumentParser(
        description="CogDB MCP Server — expose agent memory as MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tools exposed:\n"
            "  remember          Store an episodic memory\n"
            "  recall            Retrieve memories by query\n"
            "  learn             Store a semantic fact (knowledge graph)\n"
            "  learn_procedure   Store a learned workflow\n"
            "  get_context       Load progressive memory context\n"
            "  forget            Delete a memory by ID\n"
        ),
    )
    parser.add_argument(
        "--db-path",
        default="./cogdb_mcp",
        help="Path to CogDB storage directory (default: ./cogdb_mcp)",
    )
    parser.add_argument(
        "--server-name",
        default="cogdb",
        help="MCP server name shown in client UIs (default: cogdb)",
    )
    args = parser.parse_args()

    server = CogDBMCPServer(db_path=args.db_path, server_name=args.server_name)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

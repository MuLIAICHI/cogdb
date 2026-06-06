"""OpenAI Agents SDK adapter — wraps CognitiveDB as tool-based memory for OpenAI agents.

Provides a ``CogDBAgentMemory`` class and ``make_memory_tools`` factory that expose
CognitiveDB's tri-memory retrieval as ``@function_tool`` decorated functions, ready to
be passed directly to an OpenAI Agents SDK ``Agent``.

Install: ``pip install openai-agents>=0.0.3``

Example::

    from agents import Agent
    from cogdb.adapters.openai_agents import CogDBAgentMemory, make_memory_tools

    memory = CogDBAgentMemory(agent_id="assistant", db_path="./agent_memory")
    tools = make_memory_tools(memory)

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant with persistent memory.",
        tools=tools,
    )

    # The agent can now call remember_tool, recall_tool, and get_context_tool.
"""

from __future__ import annotations

from typing import Any, List, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from agents import function_tool  # type: ignore[import-untyped]

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

    def function_tool(fn: Any) -> Any:  # type: ignore[misc]
        """No-op stand-in when openai-agents is not installed."""
        return fn


class CogDBAgentMemory:
    """CognitiveDB memory backend for OpenAI Agents SDK agents.

    Stores memories as episodic records and retrieves them using
    token-budget-aware ranking. Designed to be wired into an agent
    via ``make_memory_tools``.

    Args:
        agent_id: Unique identifier for the agent that owns these memories.
        db: An existing CognitiveDB instance. If None, one is created.
        db_path: Storage path used when creating a new CognitiveDB instance.
        token_budget: Default token budget for recall and context operations.
        scope: Memory visibility scope for stored memories.

    Example::

        >>> from cogdb.adapters.openai_agents import CogDBAgentMemory
        >>> memory = CogDBAgentMemory(agent_id="planner", db_path="./mem")
        >>> mid = memory.remember("Deploy is blocked by flaky test suite")
        >>> hits = memory.recall("deployment blocker")
        >>> ctx = memory.context(task_hint="CI/CD pipeline")
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_openai_agents",
        token_budget: int = 800,
        scope: MemoryScope = MemoryScope.PRIVATE,
    ) -> None:
        self._agent_id = agent_id
        self._token_budget = token_budget
        self._scope = scope

        if db is not None:
            self._db = db
        else:
            cfg = CogDBConfig(db_path=db_path)
            self._db = CognitiveDB(config=cfg)

    def remember(self, content: str, importance: float = 0.5) -> str:
        """Store a memory and return its ID.

        Args:
            content: The text to remember.
            importance: Salience score in [0.0, 1.0].

        Returns:
            The UUID string of the stored memory unit.

        Example::

            >>> mid = memory.remember("Rate limit hit at 14:00 UTC", importance=0.8)
        """
        memory_id = self._db.remember(
            content=content,
            agent_id=self._agent_id,
            importance=importance,
            scope=self._scope,
            metadata={"source": "openai_agents"},
        )
        return str(memory_id)

    def recall(self, query: str, token_budget: Optional[int] = None) -> List[str]:
        """Retrieve memories relevant to a query as plain strings.

        Args:
            query: Natural-language search query.
            token_budget: Token limit for this query; defaults to the instance budget.

        Returns:
            List of content strings ordered by relevance.

        Example::

            >>> hits = memory.recall("rate limit error")
            >>> for h in hits:
            ...     print(h)
        """
        budget = token_budget if token_budget is not None else self._token_budget
        units = self._db.recall(
            query=query,
            agent_id=self._agent_id,
            token_budget=budget,
            max_results=20,
        )
        return [u.content for u in units]

    def context(self, task_hint: str = "") -> str:
        """Return a formatted context string built from progressive memory retrieval.

        Uses L0–L3 loading so the most important memories are always included
        within the token budget.

        Args:
            task_hint: Optional task description to bias retrieval.

        Returns:
            Newline-delimited memory context ready to inject into a prompt.

        Example::

            >>> ctx = memory.context(task_hint="deploy the web app")
            >>> print(ctx)
        """
        response = self._db.get_context(
            agent_id=self._agent_id,
            task_hint=task_hint,
            token_budget=self._token_budget,
        )
        if not response:
            return ""
        if isinstance(response, str):
            return response
        # ContextResponse: assemble from its tiers
        lines: list[str] = []
        if getattr(response, "identity", ""):
            lines.append(f"Identity: {response.identity}")
        for fact in getattr(response, "critical_facts", []):
            lines.append(f"Fact: {fact}")
        for m in getattr(response, "relevant_memories", []):
            lines.append(m.content)
        for m in getattr(response, "deep_results", []):
            lines.append(m.content)
        return "\n".join(lines)

    def close(self) -> None:
        """Release resources held by the underlying CognitiveDB instance.

        Safe to call multiple times.

        Example::

            >>> memory.close()
        """
        if hasattr(self._db, "close"):
            self._db.close()

    @property
    def agent_id(self) -> str:
        """The agent ID this memory instance is bound to."""
        return self._agent_id

    @property
    def db(self) -> CognitiveDB:
        """Direct access to the underlying CognitiveDB instance."""
        return self._db


def make_memory_tools(memory: CogDBAgentMemory) -> List[Any]:
    """Create OpenAI Agents SDK-compatible tools backed by a CogDBAgentMemory.

    Returns a list of three ``@function_tool`` decorated functions that the
    OpenAI Agents SDK can discover and call automatically:

    * ``remember_tool`` — store a new memory
    * ``recall_tool``   — retrieve relevant memories
    * ``get_context_tool`` — get full progressive context

    If ``openai-agents`` is not installed the functions are returned as plain
    callables with docstrings so the module remains usable in test environments.

    Args:
        memory: A ``CogDBAgentMemory`` instance to back the tools.

    Returns:
        List of three tool callables (decorated or plain).

    Example::

        >>> tools = make_memory_tools(memory)
        >>> agent = Agent(name="bot", instructions="...", tools=tools)
    """

    @function_tool
    def remember_tool(content: str, importance: float = 0.5) -> str:
        """Store a memory for the current agent.

        Args:
            content: The information to remember.
            importance: Salience score between 0.0 and 1.0.

        Returns:
            The UUID of the stored memory.
        """
        return memory.remember(content, importance=importance)

    @function_tool
    def recall_tool(query: str) -> str:
        """Retrieve memories relevant to a query.

        Args:
            query: Natural-language search query.

        Returns:
            Newline-delimited list of relevant memory contents.
        """
        results = memory.recall(query)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {r}" for r in results)

    @function_tool
    def get_context_tool(task_hint: str = "") -> str:
        """Get progressive context from memory for the current task.

        Args:
            task_hint: Optional description of the current task to bias retrieval.

        Returns:
            Formatted memory context string.
        """
        ctx = memory.context(task_hint=task_hint)
        return ctx if ctx else "No context available."

    return [remember_tool, recall_tool, get_context_tool]

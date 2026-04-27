"""AutoGen adapter — wraps CognitiveDB as an AutoGen Memory component.

Implements the AutoGen Memory protocol so CognitiveDB can be dropped
into any AutoGen agent as a plug-in memory backend.

AutoGen Memory protocol requires three methods:
    add(content, **kwargs)   → store a memory
    query(query, **kwargs)   → retrieve memories as a string
    clear()                  → wipe all memories for this agent

Install: pip install autogen-agentchat>=0.4.0
"""

from __future__ import annotations

from typing import Any, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult
    from autogen_core.model_context import ChatCompletionContext

    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False
    # Provide stub base class so the module still imports without autogen installed
    class Memory:  # type: ignore[no-redef]
        pass


class CogDBMemory(Memory):
    """AutoGen Memory adapter backed by CognitiveDB.

    Drop-in replacement for AutoGen's built-in memory components.
    Stores agent interactions as episodic memories and retrieves
    them with token-budget-aware ranking.

    Args:
        agent_id: The AutoGen agent's identifier.
        db: An existing CognitiveDB instance. If None, creates one.
        config: CogDBConfig for creating a new instance (ignored if db provided).
        token_budget: Max tokens per memory query response.
        scope: Memory visibility scope.

    Example:
        >>> from cogdb.adapters.autogen import CogDBMemory
        >>> memory = CogDBMemory(agent_id="assistant", db_path="./agent_memory")
        >>>
        >>> # Use with an AutoGen AssistantAgent
        >>> agent = AssistantAgent(
        ...     name="assistant",
        ...     memory=[memory],
        ...     model_client=model_client,
        ... )
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        config: Optional[CogDBConfig] = None,
        db_path: str = "./cogdb_autogen",
        token_budget: int = 1000,
        scope: MemoryScope = MemoryScope.PRIVATE,
    ) -> None:
        self._agent_id = agent_id
        self._token_budget = token_budget
        self._scope = scope

        if db is not None:
            self._db = db
        else:
            cfg = config or CogDBConfig(db_path=db_path)
            self._db = CognitiveDB(config=cfg)

    async def add(self, content: Any, **kwargs: Any) -> None:
        """Store a memory from an AutoGen interaction.

        Args:
            content: MemoryContent object or plain string.
            **kwargs: Forwarded metadata (importance, metadata dict, etc.).

        Example:
            >>> await memory.add(MemoryContent(content="User asked about dark mode"))
        """
        if _AUTOGEN_AVAILABLE and isinstance(content, MemoryContent):
            text = _memory_content_to_text(content)
            metadata = {"source": "autogen", "mime_type": str(content.mime_type)}
        else:
            text = str(content)
            metadata = {"source": "autogen"}

        importance = kwargs.get("importance", 0.5)
        extra_meta = kwargs.get("metadata", {})
        metadata.update(extra_meta)

        self._db.remember(
            content=text,
            agent_id=self._agent_id,
            importance=importance,
            scope=self._scope,
            metadata=metadata,
        )

    async def query(
        self,
        query: Any,
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> "MemoryQueryResult":
        """Retrieve memories relevant to a query.

        Args:
            query: MemoryContent or plain string query.
            cancellation_token: AutoGen cancellation token (unused).
            **kwargs: Optional overrides: token_budget, max_results.

        Returns:
            MemoryQueryResult with a list of MemoryContent items.

        Example:
            >>> result = await memory.query("dark mode preferences")
            >>> for item in result.results:
            ...     print(item.content)
        """
        if _AUTOGEN_AVAILABLE and isinstance(query, MemoryContent):
            query_text = _memory_content_to_text(query)
        else:
            query_text = str(query)

        token_budget = kwargs.get("token_budget", self._token_budget)
        max_results = kwargs.get("max_results", 20)

        memories = self._db.recall(
            query=query_text,
            agent_id=self._agent_id,
            token_budget=token_budget,
            max_results=max_results,
        )

        if not _AUTOGEN_AVAILABLE:
            return memories  # type: ignore[return-value]

        results = [
            MemoryContent(content=m.content, mime_type="text/plain")
            for m in memories
        ]
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: "ChatCompletionContext") -> None:
        """Inject relevant memories into the model's chat context.

        Called by AutoGen before each model inference. Retrieves recent
        memories and prepends them as a system-style context message.

        Args:
            model_context: The AutoGen ChatCompletionContext to update.

        Example:
            >>> await memory.update_context(model_context)
        """
        if not _AUTOGEN_AVAILABLE:
            return

        messages = await model_context.get_messages()
        if not messages:
            return

        # Use the last user message as the retrieval query
        last_user = next(
            (m for m in reversed(messages) if hasattr(m, "content") and m.role == "user"),
            None,
        )
        query_text = str(last_user.content) if last_user else ""

        if not query_text:
            return

        memories = self._db.recall(
            query=query_text,
            agent_id=self._agent_id,
            token_budget=self._token_budget // 2,
        )

        if not memories:
            return

        memory_block = "Relevant memories:\n" + "\n".join(
            f"- {m.content}" for m in memories
        )

        from autogen_core.models import SystemMessage
        await model_context.add_message(SystemMessage(content=memory_block))

    async def clear(self) -> None:
        """Clear all memories for this agent.

        Example:
            >>> await memory.clear()
        """
        # Fetch all episodic memories for this agent and delete them
        memories = self._db.recall(
            query="",
            agent_id=self._agent_id,
            token_budget=999999,
            max_results=10000,
        )
        for m in memories:
            self._db.forget(m.id, MemoryType.EPISODIC)

    async def close(self) -> None:
        """Release resources held by this memory instance.

        Calls cleanup on the underlying CognitiveDB if it was created
        internally. Safe to call multiple times.

        Example:
            >>> await memory.close()
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


def _memory_content_to_text(content: "MemoryContent") -> str:
    """Extract plain text from an AutoGen MemoryContent object."""
    if hasattr(content, "content"):
        return str(content.content)
    return str(content)

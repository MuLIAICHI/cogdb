"""LlamaIndex adapter — wraps CognitiveDB as a LlamaIndex BaseMemory component.

Implements the LlamaIndex BaseMemory protocol so CognitiveDB can be dropped
into any LlamaIndex agent as a plug-in memory backend. Also provides
CogDBVectorIndex, a lightweight shim that exposes CogDB's episodic store
as a LlamaIndex-style vector index.

Install: pip install llama-index-core>=0.10

Example::

    from cogdb.adapters.llamaindex import CogDBChatMemory

    memory = CogDBChatMemory.from_defaults(
        agent_id="research-agent",
        db_path="./agent_memory",
        token_budget=800,
    )

    # Wire into a LlamaIndex ReActAgent
    from llama_index.core.agent import ReActAgent
    from llama_index.core.llms import OpenAI

    agent = ReActAgent.from_tools(
        tools=[...],
        llm=OpenAI(model="gpt-4o"),
        memory=memory,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from llama_index.core.memory import BaseMemory
    from llama_index.core.llms import ChatMessage

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False

    # Stub base class so the module imports without llama-index installed
    class BaseMemory:  # type: ignore[no-redef]
        pass

    @dataclass
    class ChatMessage:  # type: ignore[no-redef]
        """Minimal stub matching llama_index.core.llms.ChatMessage."""

        role: str = "user"
        content: str = ""
        additional_kwargs: dict = field(default_factory=dict)


class CogDBChatMemory(BaseMemory):
    """LlamaIndex BaseMemory backed by CognitiveDB.

    Drop-in replacement for LlamaIndex's built-in chat memory components.
    Stores messages as episodic memories and retrieves them with
    token-budget-aware semantic ranking.

    Args:
        agent_id: Identifier for the agent owning these memories.
        db: An existing CognitiveDB instance. If None, creates one at db_path.
        db_path: Path for a new CognitiveDB (ignored when db is provided).
        token_budget: Max tokens per retrieval response.
        scope: Memory visibility scope for multi-agent access.

    Example:
        >>> from cogdb.adapters.llamaindex import CogDBChatMemory
        >>> memory = CogDBChatMemory.from_defaults(
        ...     agent_id="assistant",
        ...     db_path="./agent_memory",
        ... )
        >>> from cogdb.adapters.llamaindex import ChatMessage
        >>> memory.put(ChatMessage(role="user", content="What is RAG?"))
        >>> messages = memory.get("RAG")
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_llamaindex",
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

    # ── BaseMemory interface ─────────────────────────────────────

    def get(self, input: Optional[str] = None) -> List["ChatMessage"]:
        """Recall memories relevant to the current input.

        Args:
            input: Query string to guide retrieval. If None or empty,
                falls back to recent memories.

        Returns:
            List of ChatMessage objects with role="assistant".

        Example:
            >>> messages = memory.get("deployment pipeline")
            >>> for m in messages:
            ...     print(m.content)
        """
        query = input or ""
        memories = self._db.recall(
            query=query,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
        )
        return [
            ChatMessage(role="assistant", content=m.content)
            for m in memories
        ]

    def get_all(self) -> List["ChatMessage"]:
        """Return all stored memories as ChatMessages.

        Uses a large token budget to fetch as many memories as possible.

        Returns:
            List of ChatMessage objects representing all stored memories.

        Example:
            >>> all_messages = memory.get_all()
            >>> print(f"Total: {len(all_messages)} memories")
        """
        memories = self._db.recall(
            query="",
            agent_id=self._agent_id,
            token_budget=999999,
            max_results=10000,
        )
        return [
            ChatMessage(role="assistant", content=m.content)
            for m in memories
        ]

    def put(self, message: "ChatMessage") -> None:
        """Store a single ChatMessage as an episodic memory.

        User messages are stored with importance 0.6; assistant messages
        with 0.5, matching typical retrieval priority in chat workflows.

        Args:
            message: The ChatMessage to persist.

        Example:
            >>> memory.put(ChatMessage(role="user", content="How do I deploy?"))
        """
        role = getattr(message, "role", "user")
        content = getattr(message, "content", str(message))

        importance = 0.6 if role == "user" else 0.5

        self._db.remember(
            content=content,
            agent_id=self._agent_id,
            importance=importance,
            scope=self._scope,
            metadata={"source": "llamaindex", "role": role},
        )

    def set(self, messages: List["ChatMessage"]) -> None:
        """Persist a list of ChatMessages, replacing prior history.

        Stores each message via put(). Does not wipe existing memories
        before inserting; call reset() first if a clean slate is needed.

        Args:
            messages: List of ChatMessage objects to store.

        Example:
            >>> memory.reset()
            >>> memory.set([
            ...     ChatMessage(role="user", content="Hello"),
            ...     ChatMessage(role="assistant", content="Hi there!"),
            ... ])
        """
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        """Forget all stored memories for this agent.

        Example:
            >>> memory.reset()
        """
        memories = self._db.recall(
            query="",
            agent_id=self._agent_id,
            token_budget=999999,
            max_results=10000,
        )
        for m in memories:
            self._db.forget(m.id, MemoryType.EPISODIC)

    def close(self) -> None:
        """Release resources held by the underlying CognitiveDB.

        Safe to call multiple times.

        Example:
            >>> memory.close()
        """
        if hasattr(self._db, "close"):
            self._db.close()

    # ── Class-method constructor ─────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        agent_id: str = "default",
        db_path: str = "./cogdb_llamaindex",
        token_budget: int = 800,
        scope: MemoryScope = MemoryScope.PRIVATE,
    ) -> "CogDBChatMemory":
        """Create a CogDBChatMemory with default settings.

        Follows the LlamaIndex ``from_defaults`` class-method convention.

        Args:
            agent_id: Identifier for the agent owning these memories.
            db_path: Filesystem path for the CognitiveDB database.
            token_budget: Max tokens per retrieval response.
            scope: Memory visibility scope.

        Returns:
            A fully configured CogDBChatMemory instance.

        Example:
            >>> memory = CogDBChatMemory.from_defaults(
            ...     agent_id="research-agent",
            ...     db_path="./research_memory",
            ... )
        """
        return cls(
            agent_id=agent_id,
            db_path=db_path,
            token_budget=token_budget,
            scope=scope,
        )

    # ── Properties ───────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        """The agent ID this memory instance is bound to."""
        return self._agent_id

    @property
    def db(self) -> CognitiveDB:
        """Direct access to the underlying CognitiveDB instance."""
        return self._db


class CogDBVectorIndex:
    """Lightweight shim exposing CogDB's episodic store as a vector index.

    Provides a LlamaIndex-style query/insert/delete interface backed by
    CognitiveDB's HNSW-based episodic store. Useful when you need a
    drop-in vector store without the full LlamaIndex VectorStoreIndex stack.

    Args:
        agent_id: Agent identifier for scoping stored memories.
        db: An existing CognitiveDB instance. If None, creates one at db_path.
        db_path: Filesystem path for a new CognitiveDB.
        token_budget: Max tokens per query response.

    Example:
        >>> from cogdb.adapters.llamaindex import CogDBVectorIndex
        >>> index = CogDBVectorIndex(agent_id="retriever", db_path="./vec_mem")
        >>> mem_id = index.insert("FastAPI is a modern Python web framework")
        >>> results = index.query("Python web framework", similarity_top_k=3)
        >>> for text in results:
        ...     print(text)
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_li_vec",
        token_budget: int = 1000,
    ) -> None:
        self._agent_id = agent_id
        self._token_budget = token_budget

        if db is not None:
            self._db = db
        else:
            cfg = CogDBConfig(db_path=db_path)
            self._db = CognitiveDB(config=cfg)

        # Track inserted ids for delete support
        self._stored_ids: list[str] = []

    def query(self, query_str: str, similarity_top_k: int = 5) -> List[str]:
        """Retrieve memory content strings semantically similar to the query.

        Args:
            query_str: Natural language query.
            similarity_top_k: Number of top results to return.

        Returns:
            List of memory content strings, ranked by relevance.

        Example:
            >>> results = index.query("Python async frameworks", similarity_top_k=3)
        """
        memories = self._db.recall(
            query=query_str,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
            max_results=similarity_top_k,
        )
        return [m.content for m in memories]

    def insert(self, text: str, metadata: Optional[dict[str, Any]] = None) -> str:
        """Store a text chunk as an episodic memory.

        Args:
            text: The text content to store.
            metadata: Optional key-value metadata.

        Returns:
            The UUID of the stored memory.

        Example:
            >>> mem_id = index.insert(
            ...     "LangChain is a framework for LLM applications",
            ...     metadata={"source": "docs"},
            ... )
        """
        mem_metadata = {"source": "llamaindex_vector"}
        if metadata:
            mem_metadata.update(metadata)

        mem_id = self._db.remember(
            content=text,
            agent_id=self._agent_id,
            importance=0.5,
            metadata=mem_metadata,
        )
        self._stored_ids.append(mem_id)
        return mem_id

    def delete(self, memory_id: str) -> bool:
        """Delete a stored memory by its UUID.

        Args:
            memory_id: The UUID returned by insert().

        Returns:
            True if the memory was found and deleted, False otherwise.

        Example:
            >>> success = index.delete(mem_id)
        """
        result = self._db.forget(memory_id, MemoryType.EPISODIC)
        if result and memory_id in self._stored_ids:
            self._stored_ids.remove(memory_id)
        return result

    @property
    def agent_id(self) -> str:
        """The agent ID this index is bound to."""
        return self._agent_id

    @property
    def db(self) -> CognitiveDB:
        """Direct access to the underlying CognitiveDB instance."""
        return self._db

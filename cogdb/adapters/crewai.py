"""CrewAI adapter — wraps CognitiveDB as a CrewAI Storage component.

Implements the CrewAI Storage interface so CognitiveDB can be used as a
plug-in memory backend for any CrewAI agent.

CrewAI Storage interface requires three methods:
    save(value: str, **kwargs)        → store a memory
    search(query: str, limit: int)    → retrieve relevant memories as list[str]
    reset()                           → wipe all memories for this agent

Install: pip install crewai>=0.28

Usage::

    from cogdb.adapters.crewai import CogDBCrewAIStorage

    storage = CogDBCrewAIStorage(agent_id="researcher", db_path="./crew_memory")

    # Wire into a CrewAI agent via memory_config
    from crewai import Agent
    agent = Agent(
        role="Researcher",
        goal="Find insights",
        backstory="...",
        memory=True,
        memory_config={
            "provider": "custom",
            "config": {"storage": storage},
        },
    )
"""

from __future__ import annotations

from typing import Any, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from crewai.memory.storage.interface import Storage

    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

    class Storage:  # type: ignore[no-redef]
        """Stub base class used when crewai is not installed."""

        def save(self, value: str, **kwargs: Any) -> None:
            raise NotImplementedError

        def search(self, query: str) -> list:
            raise NotImplementedError

        def reset(self) -> None:
            raise NotImplementedError


class CogDBCrewAIStorage(Storage):
    """CrewAI Storage adapter backed by CognitiveDB.

    Drop-in replacement for CrewAI's built-in storage components.
    Stores agent memories as episodic entries and retrieves them with
    token-budget-aware ranking.

    Args:
        agent_id: Identifier for the CrewAI agent owning these memories.
        db: An existing CognitiveDB instance. If None, one is created lazily.
        db_path: Filesystem path for the CognitiveDB store (used when db is None).
        token_budget: Max tokens to consume per search query.
        scope: Memory visibility scope (default: PRIVATE).

    Example:
        >>> from cogdb.adapters.crewai import CogDBCrewAIStorage
        >>> storage = CogDBCrewAIStorage(agent_id="researcher", db_path="./mem")
        >>> storage.save("The API rate limit is 100 req/min")
        >>> results = storage.search("API limits")
        >>> print(results)
        ['The API rate limit is 100 req/min']
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_crewai",
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

    def save(self, value: str, **kwargs: Any) -> None:
        """Store a memory string in CognitiveDB.

        Args:
            value: The text content to remember.
            **kwargs: Optional overrides — importance (float 0–1), metadata (dict).

        Example:
            >>> storage.save("User prefers concise answers", importance=0.8)
        """
        importance: float = kwargs.get("importance", 0.5)
        metadata: dict = {"source": "crewai"}
        metadata.update(kwargs.get("metadata", {}))

        self._db.remember(
            content=value,
            agent_id=self._agent_id,
            importance=importance,
            scope=self._scope,
            metadata=metadata,
        )

    def search(self, query: str, limit: int = 10) -> list[str]:
        """Retrieve memories relevant to a query string.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results to return.

        Returns:
            Ordered list of memory content strings, most relevant first.

        Example:
            >>> results = storage.search("deployment pipeline", limit=5)
            >>> for r in results:
            ...     print(r)
        """
        memories = self._db.recall(
            query=query,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
            max_results=limit,
        )
        return [m.content for m in memories]

    def reset(self) -> None:
        """Delete all episodic memories for this agent.

        Fetches every stored memory and removes each one individually.
        Safe to call on an empty store.

        Example:
            >>> storage.reset()
        """
        memories = self._db.recall(
            query="",
            agent_id=self._agent_id,
            token_budget=999_999,
            max_results=10_000,
        )
        for m in memories:
            self._db.forget(m.id, MemoryType.EPISODIC)

    @property
    def agent_id(self) -> str:
        """The agent ID this storage instance is bound to."""
        return self._agent_id

    @property
    def db(self) -> CognitiveDB:
        """Direct access to the underlying CognitiveDB instance."""
        return self._db

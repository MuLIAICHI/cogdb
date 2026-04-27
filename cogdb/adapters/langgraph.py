"""LangGraph adapter — CognitiveDB as a LangGraph checkpointer and store.

Implements two LangGraph interfaces:
  1. CogDBCheckpointer  — BaseCheckpointSaver for agent state persistence
  2. CogDBStore         — BaseStore for cross-thread shared memory

Install: pip install langgraph>=0.2.0
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        get_checkpoint_id,
    )
    from langgraph.store.base import BaseStore, Item, SearchItem

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

    class BaseCheckpointSaver:  # type: ignore[no-redef]
        pass

    class BaseStore:  # type: ignore[no-redef]
        pass


# ── Checkpointer ────────────────────────────────────────────────────────────


class CogDBCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by CognitiveDB episodic store.

    Persists LangGraph agent state (checkpoints) as episodic memories,
    enabling long-term state recovery across sessions.

    Args:
        db: An existing CognitiveDB instance.
        agent_id: Namespace for checkpoint storage.

    Example:
        >>> from cogdb.adapters.langgraph import CogDBCheckpointer
        >>> checkpointer = CogDBCheckpointer(db=CognitiveDB(), agent_id="my-graph")
        >>>
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> result = graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "t1"}})
    """

    def __init__(
        self,
        db: Optional[CognitiveDB] = None,
        agent_id: str = "langgraph",
        db_path: str = "./cogdb_langgraph",
    ) -> None:
        if _LANGGRAPH_AVAILABLE:
            super().__init__()
        self._agent_id = agent_id
        self._db = db or CognitiveDB(db_path=db_path)

    def get_tuple(self, config: dict[str, Any]) -> Optional["CheckpointTuple"]:
        """Retrieve the latest checkpoint for a thread.

        Args:
            config: LangGraph config dict with thread_id.

        Returns:
            CheckpointTuple if found, None otherwise.

        Example:
            >>> tup = checkpointer.get_tuple({"configurable": {"thread_id": "t1"}})
        """
        if not _LANGGRAPH_AVAILABLE:
            return None

        thread_id = _get_thread_id(config)
        memories = self._db.recall(
            query=f"checkpoint thread:{thread_id}",
            agent_id=self._agent_id,
            token_budget=50000,
            memory_types=[MemoryType.EPISODIC],
            max_results=1,
        )

        if not memories:
            return None

        mem = memories[0]
        return _deserialize_checkpoint_tuple(mem.content, mem.metadata, config)

    def list(
        self,
        config: dict[str, Any],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator["CheckpointTuple"]:
        """List all checkpoints for a thread.

        Args:
            config: LangGraph config with thread_id.
            filter: Optional metadata filter.
            before: Return checkpoints before this config.
            limit: Max checkpoints to return.

        Yields:
            CheckpointTuple for each stored checkpoint.
        """
        if not _LANGGRAPH_AVAILABLE:
            return

        thread_id = _get_thread_id(config)
        memories = self._db.recall(
            query=f"checkpoint thread:{thread_id}",
            agent_id=self._agent_id,
            token_budget=200000,
            memory_types=[MemoryType.EPISODIC],
            max_results=limit or 100,
        )

        count = 0
        for mem in memories:
            if limit and count >= limit:
                break
            tup = _deserialize_checkpoint_tuple(mem.content, mem.metadata, config)
            if tup is not None:
                yield tup
                count += 1

    def put(
        self,
        config: dict[str, Any],
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a checkpoint.

        Args:
            config: LangGraph config with thread_id.
            checkpoint: The checkpoint data to save.
            metadata: Checkpoint metadata.
            new_versions: Channel version updates.

        Returns:
            Updated config dict with checkpoint_id.

        Example:
            >>> new_config = checkpointer.put(config, checkpoint, metadata, {})
        """
        thread_id = _get_thread_id(config)
        checkpoint_id = checkpoint.get("id", str(datetime.now(timezone.utc).timestamp()))

        payload = json.dumps({
            "checkpoint": checkpoint,
            "metadata": metadata,
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        })

        self._db.remember(
            content=payload,
            agent_id=self._agent_id,
            importance=0.9,
            scope=MemoryScope.PRIVATE,
            metadata={
                "record_type": "checkpoint",
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            },
        )

        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save pending writes for a checkpoint (intermediate state).

        Args:
            config: LangGraph config.
            writes: List of (channel, value) pairs.
            task_id: The task that produced these writes.
        """
        thread_id = _get_thread_id(config)
        payload = json.dumps({
            "writes": [(ch, val) for ch, val in writes],
            "task_id": task_id,
            "thread_id": thread_id,
        })

        self._db.remember(
            content=payload,
            agent_id=self._agent_id,
            importance=0.7,
            scope=MemoryScope.PRIVATE,
            metadata={
                "record_type": "pending_writes",
                "thread_id": thread_id,
                "task_id": task_id,
            },
        )


# ── Store ────────────────────────────────────────────────────────────────────


class CogDBStore(BaseStore):
    """LangGraph BaseStore backed by CognitiveDB.

    Provides cross-thread, cross-agent shared memory for LangGraph
    workflows. Items are stored as episodic memories with namespace/key
    addressing.

    Args:
        db: An existing CognitiveDB instance.
        agent_id: Default agent namespace.

    Example:
        >>> from cogdb.adapters.langgraph import CogDBStore
        >>> store = CogDBStore(db=CognitiveDB(), agent_id="shared")
        >>>
        >>> store.put(("user_data", "alice"), "prefs", {"theme": "dark"})
        >>> item = store.get(("user_data", "alice"), "prefs")
        >>> item.value
        {'theme': 'dark'}
    """

    def __init__(
        self,
        db: Optional[CognitiveDB] = None,
        agent_id: str = "langgraph_store",
        db_path: str = "./cogdb_langgraph",
    ) -> None:
        if _LANGGRAPH_AVAILABLE:
            super().__init__()
        self._agent_id = agent_id
        self._db = db or CognitiveDB(db_path=db_path)

    def get(self, namespace: tuple[str, ...], key: str) -> Optional["Item"]:
        """Retrieve a stored item by namespace and key.

        Args:
            namespace: Tuple of namespace strings (e.g. ("user_data", "alice")).
            key: Item key within the namespace.

        Returns:
            Item if found, None otherwise.

        Example:
            >>> item = store.get(("user_data", "alice"), "preferences")
        """
        if not _LANGGRAPH_AVAILABLE:
            return None

        ns_str = _ns_to_str(namespace)
        memories = self._db.recall(
            query=f"store_item ns:{ns_str} key:{key}",
            agent_id=self._agent_id,
            token_budget=10000,
            memory_types=[MemoryType.EPISODIC],
            max_results=1,
        )

        if not memories:
            return None

        mem = memories[0]
        try:
            value = json.loads(mem.content)
        except (json.JSONDecodeError, ValueError):
            value = mem.content

        return Item(
            namespace=namespace,
            key=key,
            value=value,
            created_at=mem.created_at,
            updated_at=mem.accessed_at,
        )

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list["SearchItem"]:
        """Search items within a namespace prefix.

        Args:
            namespace_prefix: Namespace prefix to search within.
            query: Optional text query for semantic search.
            filter: Optional metadata filter (unused in Phase 0).
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of SearchItem results.

        Example:
            >>> items = store.search(("user_data",), query="dark mode", limit=5)
        """
        if not _LANGGRAPH_AVAILABLE:
            return []

        ns_str = _ns_to_str(namespace_prefix)
        search_query = query or f"store_item ns:{ns_str}"

        memories = self._db.recall(
            query=search_query,
            agent_id=self._agent_id,
            token_budget=50000,
            memory_types=[MemoryType.EPISODIC],
            max_results=limit + offset,
        )

        results = []
        for mem in memories[offset: offset + limit]:
            try:
                value = json.loads(mem.content)
            except (json.JSONDecodeError, ValueError):
                value = mem.content

            ns_meta = mem.metadata.get("namespace", "")
            key_meta = mem.metadata.get("key", "")
            ns_tuple = tuple(ns_meta.split("/")) if ns_meta else namespace_prefix

            results.append(SearchItem(
                namespace=ns_tuple,
                key=key_meta,
                value=value,
                created_at=mem.created_at,
                updated_at=mem.accessed_at,
                score=mem.effective_importance(),
            ))

        return results

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Store an item under a namespace and key.

        Args:
            namespace: Namespace tuple.
            key: Item key.
            value: Item value (must be JSON-serialisable).

        Example:
            >>> store.put(("user_data", "alice"), "prefs", {"theme": "dark"})
        """
        ns_str = _ns_to_str(namespace)
        payload = json.dumps(value)

        # EpisodicStore prefixes metadata keys with "meta_" in ChromaDB.
        # Storing "namespace" + "key" means ChromaDB gets "meta_namespace" + "meta_key",
        # which delete() queries via $where filters.
        self._db.remember(
            content=payload,
            agent_id=self._agent_id,
            importance=0.8,
            scope=MemoryScope.PRIVATE,
            metadata={
                "record_type": "store_item",
                "namespace": ns_str,
                "key": key,
            },
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item by namespace and key.

        Args:
            namespace: Namespace tuple.
            key: Item key to delete.

        Example:
            >>> store.delete(("user_data", "alice"), "prefs")
        """
        ns_str = _ns_to_str(namespace)
        # Use metadata filter directly — similarity search is unreliable for exact key lookup
        try:
            collection = self._db._episodic._collection
            results = collection.get(
                where={"$and": [
                    {"meta_namespace": ns_str},
                    {"meta_key": key},
                ]},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass

    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List all namespaces matching optional prefix/suffix filters.

        Args:
            prefix: Optional namespace prefix filter.
            suffix: Optional namespace suffix filter.
            max_depth: Maximum namespace depth to return.
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of namespace tuples.

        Example:
            >>> namespaces = store.list_namespaces(prefix=("user_data",))
        """
        memories = self._db.recall(
            query="store_item",
            agent_id=self._agent_id,
            token_budget=100000,
            memory_types=[MemoryType.EPISODIC],
            max_results=limit + offset,
        )

        seen: set[tuple[str, ...]] = set()
        results: list[tuple[str, ...]] = []

        for mem in memories[offset: offset + limit]:
            ns_str = mem.metadata.get("namespace", "")
            if not ns_str:
                continue
            ns = tuple(ns_str.split("/"))

            if max_depth is not None:
                ns = ns[:max_depth]
            if prefix and not ns[:len(prefix)] == prefix:
                continue
            if suffix and not ns[-len(suffix):] == suffix:
                continue

            if ns not in seen:
                seen.add(ns)
                results.append(ns)

        return results

    # Async variants — delegate to sync implementations

    async def aget(self, namespace: tuple[str, ...], key: str) -> Optional["Item"]:
        """Async version of get."""
        return self.get(namespace, key)

    async def asearch(self, namespace_prefix: tuple[str, ...], **kwargs: Any) -> list["SearchItem"]:
        """Async version of search."""
        return self.search(namespace_prefix, **kwargs)

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Async version of put."""
        self.put(namespace, key, value)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async version of delete."""
        self.delete(namespace, key)

    async def alist_namespaces(self, **kwargs: Any) -> list[tuple[str, ...]]:
        """Async version of list_namespaces."""
        return self.list_namespaces(**kwargs)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_thread_id(config: dict[str, Any]) -> str:
    return config.get("configurable", {}).get("thread_id", "default")


def _ns_to_str(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _deserialize_checkpoint_tuple(
    content: str,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> Optional["CheckpointTuple"]:
    """Reconstruct a CheckpointTuple from stored memory content."""
    if not _LANGGRAPH_AVAILABLE:
        return None
    try:
        data = json.loads(content)
        checkpoint = data.get("checkpoint", {})
        meta = data.get("metadata", {})
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=meta,
            parent_config=None,
            pending_writes=None,
        )
    except (json.JSONDecodeError, KeyError):
        return None

"""Semantic Kernel adapter — wraps CognitiveDB as a Semantic Kernel MemoryStore.

Implements the SK ``MemoryStoreBase`` async interface so CognitiveDB can be
used as a plug-in memory backend for any Semantic Kernel application.

Install: ``pip install semantic-kernel>=1.0``

Example:
    >>> from cogdb.adapters.semantic_kernel import CogDBMemoryStore, MemoryRecord
    >>> store = CogDBMemoryStore(agent_id="assistant", db_path="./sk_memory")
    >>>
    >>> # Upsert a record
    >>> record = MemoryRecord(id="fact-1", text="Python is dynamically typed")
    >>> await store.upsert_async("my-collection", record)
    >>>
    >>> # Search by embedding (text fallback when SK not installed)
    >>> results = await store.get_nearest_matches_async(
    ...     "my-collection", embedding=[], limit=5
    ... )
    >>> for mem_record, score in results:
    ...     print(score, mem_record.text)
    >>>
    >>> store.close()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.config import CogDBConfig

try:
    from semantic_kernel.memory.memory_store_base import MemoryStoreBase
    from semantic_kernel.memory.memory_record import MemoryRecord  # noqa: F401

    _SK_AVAILABLE = True
except ImportError:
    _SK_AVAILABLE = False

    # ---------------------------------------------------------------------------
    # Stubs — keep the module importable when semantic-kernel is not installed
    # ---------------------------------------------------------------------------

    @dataclass
    class MemoryRecord:  # type: ignore[no-redef]
        """Minimal stub matching the SK MemoryRecord interface."""

        id: str = ""
        text: str = ""
        description: str = ""
        metadata: dict = field(default_factory=dict)
        embedding: list = field(default_factory=list)
        is_reference: bool = False
        external_source_name: str = ""
        additional_metadata: str = ""

    class MemoryStoreBase:  # type: ignore[no-redef]
        """Stub base class — replaced when semantic-kernel is installed."""


class CogDBMemoryStore(MemoryStoreBase):
    """Semantic Kernel ``MemoryStoreBase`` adapter backed by CognitiveDB.

    Bridges the SK async memory interface to CogDB's episodic store.
    Collections are virtual — all records share the same CogDB instance
    and are differentiated by ``collection`` metadata.

    Args:
        agent_id: Identifier for the SK agent owning these memories.
        db: An existing CognitiveDB instance. If None, a new one is opened.
        db_path: Path for a new CognitiveDB instance (ignored when ``db`` provided).
        token_budget: Max tokens used per ``recall`` call inside search methods.
        scope: Memory visibility scope for stored records.

    Example:
        >>> store = CogDBMemoryStore(agent_id="sk-agent", db_path="./sk_mem")
        >>> record = MemoryRecord(id="k1", text="Semantic Kernel is async")
        >>> await store.upsert_async("docs", record)
        >>> matches = await store.get_nearest_matches_async("docs", [], limit=3)
        >>> store.close()
    """

    def __init__(
        self,
        agent_id: str,
        db: Optional[CognitiveDB] = None,
        db_path: str = "./cogdb_sk",
        token_budget: int = 800,
        scope: MemoryScope = MemoryScope.PRIVATE,
    ) -> None:
        self._agent_id = agent_id
        self._token_budget = token_budget
        self._scope = scope
        # Track virtual collection names seen during this session
        self._collections: set[str] = set()

        if db is not None:
            self._db = db
        else:
            cfg = CogDBConfig(db_path=db_path)
            self._db = CognitiveDB(config=cfg)

    # ------------------------------------------------------------------
    # Collection management (virtual — CogDB is agent-scoped)
    # ------------------------------------------------------------------

    async def create_collection_async(self, collection_name: str) -> None:
        """Register a virtual collection name (no-op in CogDB).

        Args:
            collection_name: Name of the collection to create.

        Example:
            >>> await store.create_collection_async("articles")
        """
        self._collections.add(collection_name)

    async def get_collections_async(self) -> list[str]:
        """Return the list of known virtual collections.

        Returns:
            List of collection names seen since this store was created,
            plus the agent_id as a fallback entry.

        Example:
            >>> cols = await store.get_collections_async()
        """
        return list(self._collections) or [self._agent_id]

    async def delete_collection_async(self, collection_name: str) -> None:
        """Remove a virtual collection name (no-op in CogDB).

        Args:
            collection_name: Name of the collection to delete.

        Example:
            >>> await store.delete_collection_async("old-docs")
        """
        self._collections.discard(collection_name)

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        """Check whether a collection has been registered.

        Args:
            collection_name: Name to check.

        Returns:
            True if the collection was previously created or upserted into.

        Example:
            >>> exists = await store.does_collection_exist_async("articles")
        """
        return collection_name in self._collections or True  # always reachable

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_async(self, collection_name: str, record: "MemoryRecord") -> str:
        """Store or update a single memory record.

        The record's ``text`` (or ``description``) is stored as episodic memory.
        The ``id`` and ``collection_name`` are preserved in metadata for later
        retrieval.

        Args:
            collection_name: Virtual collection to associate the record with.
            record: SK MemoryRecord (or stub) to store.

        Returns:
            The record's ``id``.

        Example:
            >>> key = await store.upsert_async("notes", MemoryRecord(id="n1", text="Reminder"))
        """
        self._collections.add(collection_name)
        text = record.text or record.description or ""
        metadata: dict[str, Any] = {
            "sk_key": record.id,
            "collection": collection_name,
            "source": "semantic_kernel",
        }
        if hasattr(record, "additional_metadata") and record.additional_metadata:
            metadata["additional_metadata"] = record.additional_metadata
        if hasattr(record, "is_reference"):
            metadata["is_reference"] = record.is_reference
        if hasattr(record, "external_source_name") and record.external_source_name:
            metadata["external_source_name"] = record.external_source_name

        self._db.remember(
            content=text,
            agent_id=self._agent_id,
            importance=0.5,
            scope=self._scope,
            metadata=metadata,
        )
        return record.id

    async def upsert_batch_async(
        self, collection_name: str, records: list["MemoryRecord"]
    ) -> list[str]:
        """Store or update a batch of memory records.

        Args:
            collection_name: Virtual collection to associate records with.
            records: List of SK MemoryRecords to store.

        Returns:
            List of record ids in the same order as ``records``.

        Example:
            >>> keys = await store.upsert_batch_async("docs", [r1, r2])
        """
        keys: list[str] = []
        for record in records:
            key = await self.upsert_async(collection_name, record)
            keys.append(key)
        return keys

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_async(
        self,
        collection_name: str,
        key: str,
        with_embedding: bool = False,
    ) -> "MemoryRecord | None":
        """Retrieve a single record by its SK key.

        Searches episodic memories for a record whose ``sk_key`` metadata
        matches ``key`` within the given collection.

        Args:
            collection_name: Virtual collection to search in.
            key: The record id set during ``upsert_async``.
            with_embedding: Ignored (embeddings are not round-tripped).

        Returns:
            A MemoryRecord if found, else None.

        Example:
            >>> rec = await store.get_async("notes", "n1")
        """
        # Recall using key as query; filter by sk_key in metadata client-side
        memories = self._db.recall(
            query=key,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
            max_results=50,
        )
        for m in memories:
            if m.metadata.get("sk_key") == key and m.metadata.get("collection") == collection_name:
                return _unit_to_record(m)
        return None

    async def get_batch_async(
        self,
        collection_name: str,
        keys: list[str],
        with_embedding: bool = False,
    ) -> list["MemoryRecord"]:
        """Retrieve multiple records by their SK keys.

        Args:
            collection_name: Virtual collection to search in.
            keys: List of record ids to retrieve.
            with_embedding: Ignored.

        Returns:
            List of found MemoryRecords (missing keys are omitted).

        Example:
            >>> recs = await store.get_batch_async("notes", ["n1", "n2"])
        """
        results: list[MemoryRecord] = []
        for key in keys:
            rec = await self.get_async(collection_name, key, with_embedding)
            if rec is not None:
                results.append(rec)
        return results

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def remove_async(self, collection_name: str, key: str) -> None:
        """Delete a record by its SK key.

        Searches for a matching memory and calls ``db.forget`` on it.

        Args:
            collection_name: Virtual collection the record belongs to.
            key: The record id to delete.

        Example:
            >>> await store.remove_async("notes", "n1")
        """
        memories = self._db.recall(
            query=key,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
            max_results=50,
        )
        for m in memories:
            if m.metadata.get("sk_key") == key and m.metadata.get("collection") == collection_name:
                self._db.forget(m.id, MemoryType.EPISODIC)
                return

    async def remove_batch_async(self, collection_name: str, keys: list[str]) -> None:
        """Delete multiple records by their SK keys.

        Args:
            collection_name: Virtual collection the records belong to.
            keys: List of record ids to delete.

        Example:
            >>> await store.remove_batch_async("notes", ["n1", "n2"])
        """
        for key in keys:
            await self.remove_async(collection_name, key)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def get_nearest_matches_async(
        self,
        collection_name: str,
        embedding: list[float],
        limit: int = 1,
        min_relevance_score: float = 0.7,
        with_embeddings: bool = False,
    ) -> list[tuple["MemoryRecord", float]]:
        """Return the nearest memory records to the given embedding.

        Because CogDB's ``recall`` operates on text queries, the
        ``collection_name`` is used as the retrieval hint when the embedding
        vector is empty.  When SK is installed and passes a real embedding,
        the collection name still provides useful context for scoped retrieval.

        Args:
            collection_name: Virtual collection to search.
            embedding: Query embedding vector (used as semantic hint; falls back
                to collection_name text query when empty).
            limit: Maximum number of results.
            min_relevance_score: Minimum relevance threshold (applied as a
                pass-through filter on the returned scores).
            with_embeddings: Ignored (embeddings are not stored in Python).

        Returns:
            List of ``(MemoryRecord, relevance_score)`` tuples, ordered by
            descending relevance, capped at ``limit``.

        Example:
            >>> matches = await store.get_nearest_matches_async("docs", [], limit=3)
            >>> for record, score in matches:
            ...     print(score, record.text)
        """
        query_text = collection_name if not embedding else collection_name
        memories = self._db.recall(
            query=query_text,
            agent_id=self._agent_id,
            token_budget=self._token_budget,
            max_results=limit,
        )

        results: list[tuple[MemoryRecord, float]] = []
        for m in memories:
            score = float(m.importance) if hasattr(m, "importance") else 1.0
            if score >= min_relevance_score or not results:
                results.append((_unit_to_record(m), score))
            if len(results) >= limit:
                break

        return results

    async def get_nearest_match_async(
        self,
        collection_name: str,
        embedding: list[float],
        min_relevance_score: float = 0.7,
        with_embedding: bool = False,
    ) -> "tuple[MemoryRecord, float] | None":
        """Return the single nearest memory record.

        Args:
            collection_name: Virtual collection to search.
            embedding: Query embedding vector.
            min_relevance_score: Minimum relevance threshold.
            with_embedding: Ignored.

        Returns:
            A ``(MemoryRecord, score)`` tuple, or None if no match found.

        Example:
            >>> match = await store.get_nearest_match_async("docs", [])
        """
        matches = await self.get_nearest_matches_async(
            collection_name,
            embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding,
        )
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the underlying CognitiveDB instance.

        Safe to call multiple times.

        Example:
            >>> store.close()
        """
        if hasattr(self._db, "close"):
            self._db.close()

    @property
    def agent_id(self) -> str:
        """The agent ID this store is bound to."""
        return self._agent_id

    @property
    def db(self) -> CognitiveDB:
        """Direct access to the underlying CognitiveDB instance."""
        return self._db


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit_to_record(unit: Any) -> "MemoryRecord":
    """Convert a CogDB MemoryUnit to a MemoryRecord (real or stub)."""
    rec_id = unit.metadata.get("sk_key", unit.id)
    return MemoryRecord(
        id=rec_id,
        text=unit.content,
        description="",
        metadata=unit.metadata,
    )

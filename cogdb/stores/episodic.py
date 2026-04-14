"""Episodic memory store — ChromaDB-backed vector storage.

Stores timestamped records of agent interactions, observations,
and tool calls as embeddings with full metadata. Supports
similarity search and temporal filtering.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings

from cogdb.models import MemoryScope, MemoryType, MemoryUnit
from cogdb.utils.config import CogDBConfig


class EpisodicStore:
    """Vector store for episodic memories using ChromaDB.

    Thread-safe wrapper around ChromaDB that handles embedding storage,
    similarity search, and metadata filtering for episodic memories.

    Args:
        config: CogDB configuration.

    Example:
        >>> store = EpisodicStore(config)
        >>> unit = MemoryUnit(
        ...     content="User asked about dark mode",
        ...     memory_type=MemoryType.EPISODIC,
        ...     agent_id="ui-agent",
        ... )
        >>> store.add(unit)
        >>> results = store.search("dark mode preferences", agent_id="ui-agent", top_k=5)
    """

    def __init__(self, config: CogDBConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(
            path=config.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, unit: MemoryUnit) -> str:
        """Store an episodic memory.

        Args:
            unit: The memory unit to store.

        Returns:
            The ID of the stored memory.

        Example:
            >>> unit = MemoryUnit(content="test", memory_type=MemoryType.EPISODIC, agent_id="a1")
            >>> store.add(unit)
            'a1b2c3d4-...'
        """
        metadata = {
            "agent_id": unit.agent_id,
            "scope": unit.scope.value,
            "importance": unit.importance,
            "created_at": unit.created_at.isoformat(),
            "accessed_at": unit.accessed_at.isoformat(),
            "access_count": unit.access_count,
            "decay_score": unit.decay_score,
            "memory_type": unit.memory_type.value,
        }
        # Add any extra metadata (flatten to string values for ChromaDB)
        for k, v in unit.metadata.items():
            metadata[f"meta_{k}"] = str(v)

        if unit.team_id:
            metadata["team_id"] = unit.team_id

        with self._lock:
            if unit.embedding is not None:
                self._collection.upsert(
                    ids=[unit.id],
                    documents=[unit.content],
                    embeddings=[unit.embedding],
                    metadatas=[metadata],
                )
            else:
                self._collection.upsert(
                    ids=[unit.id],
                    documents=[unit.content],
                    metadatas=[metadata],
                )

        return unit.id

    def search(
        self,
        query: str,
        agent_id: str,
        top_k: int = 10,
        scope_filter: Optional[MemoryScope] = None,
        min_importance: float = 0.0,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[MemoryUnit]:
        """Search episodic memories by similarity.

        Args:
            query: Natural language query string.
            agent_id: The querying agent's ID.
            top_k: Maximum results to return.
            scope_filter: Filter by memory scope.
            min_importance: Minimum importance threshold.
            time_range_start: Only return memories after this time.
            time_range_end: Only return memories before this time.
            query_embedding: Pre-computed query embedding (optional).

        Returns:
            List of matching MemoryUnits, ranked by relevance.
        """
        where_conditions: list[dict] = []

        # Scope-based access control
        if scope_filter:
            where_conditions.append({"scope": scope_filter.value})
        else:
            # Default: return private memories for this agent + team + org
            where_conditions.append(
                {"$or": [
                    {"agent_id": agent_id},
                    {"scope": MemoryScope.TEAM.value},
                    {"scope": MemoryScope.ORGANIZATION.value},
                ]}
            )

        if min_importance > 0:
            where_conditions.append({"importance": {"$gte": min_importance}})

        if time_range_start:
            where_conditions.append(
                {"created_at": {"$gte": time_range_start.isoformat()}}
            )
        if time_range_end:
            where_conditions.append(
                {"created_at": {"$lte": time_range_end.isoformat()}}
            )

        # Build the where clause
        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        with self._lock:
            if query_embedding is not None:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                )
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where,
                )

        return self._results_to_units(results)

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory's UUID.

        Returns:
            The MemoryUnit if found, None otherwise.
        """
        with self._lock:
            results = self._collection.get(ids=[memory_id])

        if not results["ids"]:
            return None

        units = self._results_to_units_from_get(results)
        return units[0] if units else None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory's UUID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                return False

    def update_metadata(self, memory_id: str, updates: dict) -> bool:
        """Update metadata fields on a stored memory.

        Args:
            memory_id: The memory's UUID.
            updates: Dict of metadata fields to update.

        Returns:
            True if updated successfully.
        """
        with self._lock:
            try:
                self._collection.update(ids=[memory_id], metadatas=[updates])
                return True
            except Exception:
                return False

    def count(self, agent_id: Optional[str] = None) -> int:
        """Count stored episodic memories.

        Args:
            agent_id: If provided, count only this agent's memories.

        Returns:
            Number of stored memories.
        """
        with self._lock:
            if agent_id:
                results = self._collection.get(
                    where={"agent_id": agent_id},
                )
                return len(results["ids"])
            return self._collection.count()

    def _results_to_units(self, results: dict) -> list[MemoryUnit]:
        """Convert ChromaDB query results to MemoryUnits."""
        units = []
        if not results["ids"] or not results["ids"][0]:
            return units

        for i, memory_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""

            unit = MemoryUnit(
                id=memory_id,
                content=doc,
                memory_type=MemoryType(meta.get("memory_type", "episodic")),
                agent_id=meta.get("agent_id", ""),
                scope=MemoryScope(meta.get("scope", "private")),
                importance=float(meta.get("importance", 0.5)),
                created_at=datetime.fromisoformat(meta["created_at"])
                if "created_at" in meta
                else datetime.now(timezone.utc),
                accessed_at=datetime.fromisoformat(meta["accessed_at"])
                if "accessed_at" in meta
                else datetime.now(timezone.utc),
                access_count=int(meta.get("access_count", 0)),
                decay_score=float(meta.get("decay_score", 1.0)),
                team_id=meta.get("team_id"),
            )
            units.append(unit)

        return units

    def _results_to_units_from_get(self, results: dict) -> list[MemoryUnit]:
        """Convert ChromaDB get results to MemoryUnits (different format than query)."""
        units = []
        if not results["ids"]:
            return units

        for i, memory_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            doc = results["documents"][i] if results["documents"] else ""

            unit = MemoryUnit(
                id=memory_id,
                content=doc,
                memory_type=MemoryType(meta.get("memory_type", "episodic")),
                agent_id=meta.get("agent_id", ""),
                scope=MemoryScope(meta.get("scope", "private")),
                importance=float(meta.get("importance", 0.5)),
                created_at=datetime.fromisoformat(meta["created_at"])
                if "created_at" in meta
                else datetime.now(timezone.utc),
                accessed_at=datetime.fromisoformat(meta["accessed_at"])
                if "accessed_at" in meta
                else datetime.now(timezone.utc),
                access_count=int(meta.get("access_count", 0)),
                decay_score=float(meta.get("decay_score", 1.0)),
                team_id=meta.get("team_id"),
            )
            units.append(unit)

        return units

"""Episodic memory store — Rust-backed via cogdb_engine PyO3 bindings.

Public API is identical to the previous ChromaDB-backed implementation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from cogdb.models import MemoryScope, MemoryType, MemoryUnit
from cogdb.pipeline.encoder import Encoder
from cogdb.utils.config import CogDBConfig
from cogdb._engine_cache import get_engine, release_engine


def _dt_iso(dt: datetime) -> str:
    """Return ISO 8601 string with explicit UTC offset (chrono requires it)."""
    if dt.tzinfo is None:
        return dt.isoformat() + "+00:00"
    return dt.isoformat()


def _unit_to_json(unit: MemoryUnit) -> str:
    """Serialize a Python MemoryUnit to JSON for Rust consumption."""
    d = unit.to_dict()
    # Ensure timezone-aware ISO strings
    d["created_at"] = _dt_iso(unit.created_at)
    d["accessed_at"] = _dt_iso(unit.accessed_at)
    # Add embedding (to_dict() omits it)
    d["embedding"] = unit.embedding
    return json.dumps(d)


def _json_to_unit(data: dict) -> MemoryUnit:
    """Deserialize a JSON dict from Rust to a Python MemoryUnit."""
    return MemoryUnit.from_dict(data)


class _ClientShim:
    """Backward-compatible shim — test teardown calls _client.reset()."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def reset(self) -> None:
        """Close and evict the Rust engine; releases all SQLite file locks."""
        release_engine(self._db_path)


class EpisodicStore:
    """Rust-backed episodic memory store. Thread-safe; API matches ChromaDB version.

    Args:
        config: CogDB configuration.

    Example:
        >>> store = EpisodicStore(config)
        >>> unit = MemoryUnit(content="User asked about dark mode",
        ...                   memory_type=MemoryType.EPISODIC, agent_id="ui-agent")
        >>> store.add(unit)
        >>> results = store.search("dark mode preferences", agent_id="ui-agent", top_k=5)
    """

    def __init__(self, config: CogDBConfig) -> None:
        self._config = config
        self._db_path = config.db_path
        self._engine = get_engine(
            db_path=self._db_path,
            embedding_dim=config.embedding_dim,
            contradiction_check=config.contradiction_check,
        )
        self._encoder = Encoder(config)
        # Shim keeps test teardown (instance._episodic._client.reset()) working.
        self._client = _ClientShim(self._db_path)

    def close(self) -> None:
        """Explicit close — equivalent to _client.reset()."""
        self._client.reset()

    def add(self, unit: MemoryUnit) -> str:
        """Store an episodic memory. Computes embedding if not already set.

        Args:
            unit: The memory unit to store.

        Returns:
            The UUID string of the stored record.
        """
        if unit.embedding is None:
            unit.embedding = self._encoder.embed_query(unit.content)
        return self._engine.episodic_add(_unit_to_json(unit))

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
        embedding = query_embedding or self._encoder.embed_query(query)
        if embedding is None:
            return []

        def _to_ms(dt: datetime) -> int:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

        result_json = self._engine.episodic_search(
            json.dumps(embedding),
            agent_id,
            top_k,
            scope_filter.value if scope_filter else None,
            float(min_importance),
            _to_ms(time_range_start) if time_range_start else None,
            _to_ms(time_range_end) if time_range_end else None,
        )
        return [_json_to_unit(r) for r in json.loads(result_json)]

    def get(self, memory_id: str) -> Optional[MemoryUnit]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory's UUID string.

        Returns:
            The MemoryUnit if found, None otherwise.
        """
        try:
            result_json = self._engine.episodic_get(memory_id)
        except RuntimeError:
            return None  # invalid UUID format
        data = json.loads(result_json)
        return _json_to_unit(data) if data is not None else None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory's UUID string.

        Returns:
            True if deleted, False if not found.
        """
        try:
            return self._engine.episodic_delete(memory_id)
        except RuntimeError:
            return False  # invalid UUID format

    def update_metadata(self, memory_id: str, updates: dict) -> bool:
        """Update metadata fields on a stored memory.

        Args:
            memory_id: The memory's UUID string.
            updates: Dict of fields to update (e.g. decay_score, access_count).

        Returns:
            True if updated, False if not found.
        """
        serializable = {}
        for k, v in updates.items():
            if isinstance(v, datetime):
                serializable[k] = _dt_iso(v)
            else:
                serializable[k] = v
        return self._engine.episodic_update_metadata(memory_id, json.dumps(serializable))

    def count(self, agent_id: Optional[str] = None) -> int:
        """Count stored episodic memories.

        Args:
            agent_id: If provided, count only this agent's memories.

        Returns:
            Number of stored memories.
        """
        return self._engine.episodic_count(agent_id)

    # ── Methods used by DecayEngine (replace ChromaDB _collection access) ──────

    def scan_batch(
        self,
        agent_id: Optional[str],
        limit: int,
        offset: int,
    ) -> list[dict]:
        """Paginated scan for decay processing.

        Returns list of dicts with: id, accessed_at (ISO string), decay_score.
        Replaces direct _collection.get() calls in DecayEngine.
        """
        return json.loads(self._engine.episodic_scan_batch(agent_id, limit, offset))

    def bulk_update_decay(self, updates: list[tuple[str, float]]) -> None:
        """Batch update decay scores.

        Args:
            updates: List of (uuid_str, new_decay_score) pairs.
        """
        self._engine.episodic_bulk_update_decay(json.dumps(updates))

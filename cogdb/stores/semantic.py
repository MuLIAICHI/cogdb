"""Semantic memory store — Rust-backed via cogdb_engine PyO3 bindings.

Public API is identical to the previous NetworkX+SQLite implementation.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from cogdb.models import SemanticTriple
from cogdb.utils.config import CogDBConfig
from cogdb._engine_cache import get_engine, release_engine


def _dt_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.isoformat() + "+00:00"
    return dt.isoformat()


def _triple_to_json(triple: SemanticTriple) -> str:
    d = triple.to_dict()
    d["valid_from"] = _dt_iso(triple.valid_from)
    if triple.valid_until:
        d["valid_until"] = _dt_iso(triple.valid_until)
    return json.dumps(d)


def _json_to_triple(data: dict) -> SemanticTriple:
    return SemanticTriple.from_dict(data)


class SemanticStore:
    """Rust-backed semantic triple store. Thread-safe; API matches NetworkX version.

    Args:
        config: CogDB configuration.

    Example:
        >>> store = SemanticStore(config)
        >>> store.add_triple(SemanticTriple(
        ...     subject="user", predicate="prefers", object="dark_mode",
        ...     agent_id="ui-agent", confidence=0.95
        ... ))
        >>> facts = store.query_subject("user")
    """

    def __init__(self, config: CogDBConfig) -> None:
        self._config = config
        self._db_path = config.db_path
        self._engine = get_engine(
            db_path=config.db_path,
            contradiction_check=config.contradiction_check,
        )

    def __del__(self) -> None:
        """Release the engine reference so SQLite connections close on GC."""
        try:
            db_path = self._db_path
            self._engine = None
            release_engine(db_path)
        except Exception:
            pass

    def add_triple(self, triple: SemanticTriple) -> str:
        """Add a semantic fact to the knowledge graph.

        Args:
            triple: The semantic triple to store.

        Returns:
            The UUID string of the stored triple.
        """
        return self._engine.semantic_add_triple(_triple_to_json(triple))

    def query_subject(
        self,
        subject: str,
        active_only: bool = True,
        agent_id: Optional[str] = None,
    ) -> list[SemanticTriple]:
        """Get all facts about a subject.

        Args:
            subject: The entity to query facts about.
            active_only: If True, only return currently valid facts.
            agent_id: Filter by agent ownership.

        Returns:
            List of matching triples.
        """
        result_json = self._engine.semantic_query_subject(subject, active_only, agent_id)
        return [_json_to_triple(d) for d in json.loads(result_json)]

    def query_entity(
        self,
        entity: str,
        depth: int = 1,
        active_only: bool = True,
    ) -> list[SemanticTriple]:
        """Get all facts connected to an entity (BFS up to depth hops).

        Args:
            entity: The entity to explore.
            depth: How many hops in the graph to traverse.
            active_only: If True, only return currently valid facts.

        Returns:
            All triples within the specified depth from the entity.
        """
        result_json = self._engine.semantic_query_entity(entity, depth, active_only)
        return [_json_to_triple(d) for d in json.loads(result_json)]

    def search_text(self, query: str, active_only: bool = True) -> list[SemanticTriple]:
        """Full-text LIKE search across subject, predicate, and object fields.

        Args:
            query: Search text.
            active_only: If True, only return currently valid facts.

        Returns:
            Matching triples.
        """
        result_json = self._engine.semantic_search_text(query, active_only)
        return [_json_to_triple(d) for d in json.loads(result_json)]

    def get_entities(self) -> list[str]:
        """List all entities in the active knowledge graph.

        Returns:
            List of entity names.
        """
        return self._engine.semantic_get_entities()

    def get_neighbors(self, entity: str) -> list[str]:
        """Get directly connected entities (outgoing + incoming).

        Args:
            entity: The entity to find neighbors for.

        Returns:
            List of neighboring entity names.
        """
        return self._engine.semantic_get_neighbors(entity)

    def delete_triple(self, triple_id: str) -> bool:
        """Delete a triple by ID.

        Args:
            triple_id: The triple's UUID string.

        Returns:
            True if deleted, False if not found.
        """
        try:
            return self._engine.semantic_delete_triple(triple_id)
        except RuntimeError:
            return False  # invalid UUID format

    def count(self, active_only: bool = True) -> int:
        """Count stored triples.

        Args:
            active_only: If True, count only currently valid facts.

        Returns:
            Number of triples.
        """
        return self._engine.semantic_count(active_only)

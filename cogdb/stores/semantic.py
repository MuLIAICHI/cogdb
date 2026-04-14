"""Semantic memory store — temporal knowledge graph.

Uses NetworkX for in-memory graph operations and SQLite for
persistence. Entities and relationships carry validity windows,
confidence scores, and provenance links.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from cogdb.models import SemanticTriple
from cogdb.utils.config import CogDBConfig


class SemanticStore:
    """Temporal knowledge graph for semantic memory.

    Facts are stored as RDF-style triples (subject, predicate, object)
    with validity windows and confidence scores. Supports temporal
    queries, contradiction detection, and graph traversal.

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
        self._lock = threading.Lock()
        self._graph = nx.DiGraph()
        self._db_path = Path(config.db_path) / "semantic" / "knowledge.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_graph()

    def _init_db(self) -> None:
        """Initialize SQLite tables for persistent triple storage."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                valid_from TEXT NOT NULL,
                valid_until TEXT,
                source_episodes TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent ON triples(agent_id)
        """)
        conn.commit()
        conn.close()

    def _load_graph(self) -> None:
        """Load all active triples into the NetworkX graph."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM triples WHERE valid_until IS NULL").fetchall()
        conn.close()

        for row in rows:
            triple = self._row_to_triple(row)
            self._add_to_graph(triple)

    def _add_to_graph(self, triple: SemanticTriple) -> None:
        """Add a triple to the in-memory graph."""
        self._graph.add_node(triple.subject, node_type="entity")
        self._graph.add_node(triple.object, node_type="entity")
        self._graph.add_edge(
            triple.subject,
            triple.object,
            triple_id=triple.id,
            predicate=triple.predicate,
            confidence=triple.confidence,
            valid_from=triple.valid_from.isoformat(),
        )

    def add_triple(self, triple: SemanticTriple) -> str:
        """Add a semantic fact to the knowledge graph.

        If contradiction_check is enabled in config, checks for
        existing triples with the same subject+predicate and
        supersedes them if the new triple has higher confidence.

        Args:
            triple: The semantic triple to store.

        Returns:
            The ID of the stored triple.

        Example:
            >>> triple = SemanticTriple(
            ...     subject="project", predicate="status", object="active",
            ...     agent_id="pm-agent"
            ... )
            >>> store.add_triple(triple)
        """
        with self._lock:
            if self._config.contradiction_check:
                self._handle_contradictions(triple)

            self._persist_triple(triple)
            self._add_to_graph(triple)

        return triple.id

    def _handle_contradictions(self, new_triple: SemanticTriple) -> list[SemanticTriple]:
        """Detect and handle contradicting triples.

        When a new triple shares the same subject+predicate as an existing
        active triple but has a different object, the older triple is
        superseded (its valid_until is set to now).

        Returns:
            List of superseded triples.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM triples
               WHERE subject = ? AND predicate = ? AND valid_until IS NULL
               AND object != ?""",
            (new_triple.subject, new_triple.predicate, new_triple.object),
        ).fetchall()
        conn.close()

        superseded = []
        for row in rows:
            old_triple = self._row_to_triple(row)
            old_triple.supersede(new_triple)
            self._update_triple(old_triple)

            # Remove old edge from graph
            if self._graph.has_edge(old_triple.subject, old_triple.object):
                edge_data = self._graph[old_triple.subject][old_triple.object]
                if edge_data.get("triple_id") == old_triple.id:
                    self._graph.remove_edge(old_triple.subject, old_triple.object)

            superseded.append(old_triple)

        return superseded

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

        Example:
            >>> facts = store.query_subject("user")
            >>> for f in facts:
            ...     print(f"{f.predicate}: {f.object} (confidence: {f.confidence})")
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM triples WHERE subject = ?"
            params: list = [subject]

            if active_only:
                query += " AND valid_until IS NULL"
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)

            rows = conn.execute(query, params).fetchall()
            conn.close()

        return [self._row_to_triple(r) for r in rows]

    def query_entity(
        self,
        entity: str,
        depth: int = 1,
        active_only: bool = True,
    ) -> list[SemanticTriple]:
        """Get all facts connected to an entity (as subject or object).

        Args:
            entity: The entity to explore.
            depth: How many hops in the graph to traverse.
            active_only: If True, only return currently valid facts.

        Returns:
            All triples within the specified depth from the entity.
        """
        with self._lock:
            if entity not in self._graph:
                return []

            # BFS to find all connected nodes within depth
            visited: set[str] = set()
            frontier = {entity}

            for _ in range(depth):
                next_frontier: set[str] = set()
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    next_frontier.update(self._graph.successors(node))
                    next_frontier.update(self._graph.predecessors(node))
                frontier = next_frontier - visited

            visited.update(frontier)

        # Fetch all triples involving visited nodes
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in visited)
        query = f"""SELECT * FROM triples
                    WHERE (subject IN ({placeholders}) OR object IN ({placeholders}))"""
        params = list(visited) + list(visited)

        if active_only:
            query += " AND valid_until IS NULL"

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [self._row_to_triple(r) for r in rows]

    def search_text(self, query: str, active_only: bool = True) -> list[SemanticTriple]:
        """Full-text search across triples (subject, predicate, object fields).

        Args:
            query: Search text.
            active_only: If True, only return currently valid facts.

        Returns:
            Matching triples.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            sql = """SELECT * FROM triples
                     WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)"""
            pattern = f"%{query}%"
            params: list = [pattern, pattern, pattern]

            if active_only:
                sql += " AND valid_until IS NULL"

            rows = conn.execute(sql, params).fetchall()
            conn.close()

        return [self._row_to_triple(r) for r in rows]

    def get_entities(self) -> list[str]:
        """List all entities in the active knowledge graph.

        Returns:
            List of entity names.
        """
        with self._lock:
            return list(self._graph.nodes())

    def get_neighbors(self, entity: str) -> list[str]:
        """Get directly connected entities.

        Args:
            entity: The entity to find neighbors for.

        Returns:
            List of neighboring entity names.
        """
        with self._lock:
            if entity not in self._graph:
                return []
            neighbors = set(self._graph.successors(entity))
            neighbors.update(self._graph.predecessors(entity))
            return list(neighbors)

    def delete_triple(self, triple_id: str) -> bool:
        """Delete a triple by ID.

        Args:
            triple_id: The triple's UUID.

        Returns:
            True if deleted.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("DELETE FROM triples WHERE id = ?", (triple_id,))
            conn.commit()
            conn.close()

            # Rebuild graph (simple approach for PoC)
            self._graph.clear()
            self._load_graph()

        return True

    def count(self, active_only: bool = True) -> int:
        """Count stored triples.

        Args:
            active_only: If True, count only currently valid facts.

        Returns:
            Number of triples.
        """
        conn = sqlite3.connect(str(self._db_path))
        query = "SELECT COUNT(*) FROM triples"
        if active_only:
            query += " WHERE valid_until IS NULL"
        count = conn.execute(query).fetchone()[0]
        conn.close()
        return count

    def _persist_triple(self, triple: SemanticTriple) -> None:
        """Write a triple to SQLite."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """INSERT OR REPLACE INTO triples
               (id, subject, predicate, object, agent_id, confidence,
                valid_from, valid_until, source_episodes, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                triple.id,
                triple.subject,
                triple.predicate,
                triple.object,
                triple.agent_id,
                triple.confidence,
                triple.valid_from.isoformat(),
                triple.valid_until.isoformat() if triple.valid_until else None,
                json.dumps(triple.source_episodes),
                json.dumps(triple.metadata),
            ),
        )
        conn.commit()
        conn.close()

    def _update_triple(self, triple: SemanticTriple) -> None:
        """Update an existing triple in SQLite."""
        self._persist_triple(triple)

    @staticmethod
    def _row_to_triple(row: sqlite3.Row) -> SemanticTriple:
        """Convert a SQLite row to a SemanticTriple."""
        return SemanticTriple(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            agent_id=row["agent_id"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=(
                datetime.fromisoformat(row["valid_until"])
                if row["valid_until"]
                else None
            ),
            source_episodes=json.loads(row["source_episodes"]),
            metadata=json.loads(row["metadata"]),
        )

"""Procedural memory store — Rust-backed via cogdb_engine PyO3 bindings.

Public API is identical to the previous SQLite-backed implementation.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

# Common English function words that carry no procedure-matching signal.
_STOP_WORDS: frozenset[str] = frozenset({
    "what", "when", "where", "have", "been", "that", "this", "with",
    "from", "they", "their", "does", "done", "will", "would", "could",
    "should", "there", "these", "those", "also", "just", "each", "some",
    "more", "than", "into", "over", "then", "them", "most", "your",
    "come", "after", "before", "used", "look", "like", "issue", "issues",
    "past", "last", "next", "first", "since", "actually", "specific",
    "which", "upon", "about", "such", "being", "given", "while",
})

from cogdb.models import ProcedureStep, ProcedureTemplate
from cogdb.utils.config import CogDBConfig
from cogdb._engine_cache import get_engine, release_engine


def _dt_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.isoformat() + "+00:00"
    return dt.isoformat()


def _proc_to_json(proc: ProcedureTemplate) -> str:
    d = proc.to_dict()
    d["created_at"] = _dt_iso(proc.created_at)
    d["updated_at"] = _dt_iso(proc.updated_at)
    return json.dumps(d)


def _json_to_proc(data: dict) -> ProcedureTemplate:
    return ProcedureTemplate.from_dict(data)


class ProceduralStore:
    """Rust-backed procedural store. Thread-safe; API matches SQLite version.

    Args:
        config: CogDB configuration.

    Example:
        >>> store = ProceduralStore(config)
        >>> proc = ProcedureTemplate(
        ...     name="deploy_app",
        ...     description="Deploy a web app to production",
        ...     steps=[ProcedureStep(action="test", tool="pytest")],
        ...     agent_id="devops-agent",
        ...     applicable_contexts=["deployment", "release"],
        ... )
        >>> store.add(proc)
        >>> matches = store.search_by_context("deploying to production")
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

    def add(self, procedure: ProcedureTemplate) -> str:
        """Store a learned procedure.

        Args:
            procedure: The procedure template to store.

        Returns:
            The UUID string of the stored record.
        """
        return self._engine.procedural_add(_proc_to_json(procedure))

    def get(self, procedure_id: str) -> Optional[ProcedureTemplate]:
        """Retrieve a procedure by ID.

        Args:
            procedure_id: The procedure's UUID string.

        Returns:
            The ProcedureTemplate if found, None otherwise.
        """
        try:
            result_json = self._engine.procedural_get(procedure_id)
        except RuntimeError:
            return None  # invalid UUID format
        data = json.loads(result_json)
        return _json_to_proc(data) if data is not None else None

    def search_by_context(
        self,
        context: str,
        agent_id: Optional[str] = None,
        min_success_rate: float = 0.0,
    ) -> list[ProcedureTemplate]:
        """Find procedures applicable to a given context.

        The context string is tokenized into individual keywords so that natural-
        language queries (e.g. "How do we fix CORS errors?") correctly match
        procedures whose ``applicable_contexts`` lists contain those terms.
        The Rust store uses a substring LIKE search, so we search word-by-word
        and merge results by ID, re-sorting by success_rate.

        Args:
            context: Description of the current task context (can be a full
                natural-language question or a short technical keyword).
            agent_id: Filter by agent ownership.
            min_success_rate: Minimum success rate threshold.

        Returns:
            Matching procedures, sorted by success rate descending.

        Example:
            >>> matches = store.search_by_context("fix CORS nginx error")
            >>> matches = store.search_by_context("cors")  # single keyword also works
        """
        keywords: list[str] = []
        seen_kw: set[str] = set()
        for raw in context.split():
            kw = raw.strip("?.!,;:'\"()").lower()
            if len(kw) >= 4 and kw not in _STOP_WORDS and kw not in seen_kw:
                keywords.append(kw)
                seen_kw.add(kw)
            if len(keywords) >= 8:
                break

        if not keywords:
            # All tokens were too short or stop words — fall back to full string.
            result_json = self._engine.procedural_search_by_context(
                context, agent_id, float(min_success_rate)
            )
            return [_json_to_proc(d) for d in json.loads(result_json)]

        seen: dict[str, dict] = {}
        match_count: dict[str, int] = {}  # procedure ID → number of distinct keywords that matched
        for kw in keywords:
            result_json = self._engine.procedural_search_by_context(
                kw, agent_id, float(min_success_rate)
            )
            for d in json.loads(result_json):
                pid = d["id"]
                if pid not in seen:
                    seen[pid] = d
                    match_count[pid] = 0
                match_count[pid] += 1

        procs = [_json_to_proc(d) for d in seen.values()]
        # Sort by relevance (keyword-match count) first, then by success_rate.
        # This ensures the most query-relevant procedure appears first rather
        # than the one with the highest success_rate across all contexts.
        procs.sort(key=lambda p: (match_count[p.id], p.success_rate), reverse=True)
        return procs

    def search_by_name(self, name: str) -> list[ProcedureTemplate]:
        """Find procedures by name (exact or partial match).

        Args:
            name: Procedure name to search for.

        Returns:
            Matching procedures.
        """
        result_json = self._engine.procedural_search_by_name(name)
        return [_json_to_proc(d) for d in json.loads(result_json)]

    def record_execution(self, procedure_id: str, success: bool) -> bool:
        """Record the outcome of a procedure execution.

        Updates the procedure's EMA success rate (α=0.3) and execution count.

        Args:
            procedure_id: The procedure's UUID string.
            success: Whether the execution was successful.

        Returns:
            True if the procedure was found and updated.
        """
        try:
            return self._engine.procedural_record_execution(procedure_id, success)
        except RuntimeError:
            return False  # invalid UUID format

    def list_all(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[ProcedureTemplate]:
        """List all stored procedures.

        Args:
            agent_id: Filter by agent ownership.
            limit: Maximum results.

        Returns:
            List of procedures.
        """
        result_json = self._engine.procedural_list_all(agent_id, limit)
        return [_json_to_proc(d) for d in json.loads(result_json)]

    def delete(self, procedure_id: str) -> bool:
        """Delete a procedure by ID.

        Args:
            procedure_id: The procedure's UUID string.

        Returns:
            True if deleted, False if not found.
        """
        try:
            return self._engine.procedural_delete(procedure_id)
        except RuntimeError:
            return False  # invalid UUID format

    def count(self, agent_id: Optional[str] = None) -> int:
        """Count stored procedures.

        Args:
            agent_id: If provided, count only this agent's procedures.

        Returns:
            Number of stored procedures.
        """
        return self._engine.procedural_count(agent_id)

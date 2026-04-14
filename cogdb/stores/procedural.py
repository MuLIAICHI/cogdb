"""Procedural memory store — learned workflow templates.

Stores reusable action sequences extracted from successful
agent task completions. The least-addressed memory type
in the current AI agent memory landscape.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from cogdb.models import ProcedureStep, ProcedureTemplate
from cogdb.utils.config import CogDBConfig


class ProceduralStore:
    """SQLite-backed store for procedural memories (learned workflows).

    When an agent successfully completes a multi-step task, the
    solution pattern is captured as a ProcedureTemplate. On future
    similar tasks, the agent can recall and reuse the procedure.

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
        self._lock = threading.Lock()
        self._db_path = Path(config.db_path) / "procedural" / "procedures.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                steps TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                success_rate REAL DEFAULT 1.0,
                execution_count INTEGER DEFAULT 0,
                source_episodes TEXT DEFAULT '[]',
                applicable_contexts TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_proc_agent ON procedures(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_proc_name ON procedures(name)
        """)
        conn.commit()
        conn.close()

    def add(self, procedure: ProcedureTemplate) -> str:
        """Store a learned procedure.

        Args:
            procedure: The procedure template to store.

        Returns:
            The ID of the stored procedure.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                """INSERT OR REPLACE INTO procedures
                   (id, name, description, steps, agent_id, success_rate,
                    execution_count, source_episodes, applicable_contexts,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    procedure.id,
                    procedure.name,
                    procedure.description,
                    json.dumps([self._step_to_dict(s) for s in procedure.steps]),
                    procedure.agent_id,
                    procedure.success_rate,
                    procedure.execution_count,
                    json.dumps(procedure.source_episodes),
                    json.dumps(procedure.applicable_contexts),
                    procedure.created_at.isoformat(),
                    procedure.updated_at.isoformat(),
                ),
            )
            conn.commit()
            conn.close()

        return procedure.id

    def get(self, procedure_id: str) -> Optional[ProcedureTemplate]:
        """Retrieve a procedure by ID.

        Args:
            procedure_id: The procedure's UUID.

        Returns:
            The ProcedureTemplate if found, None otherwise.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM procedures WHERE id = ?", (procedure_id,)
            ).fetchone()
            conn.close()

        if row is None:
            return None
        return self._row_to_procedure(row)

    def search_by_context(
        self,
        context: str,
        agent_id: Optional[str] = None,
        min_success_rate: float = 0.0,
    ) -> list[ProcedureTemplate]:
        """Find procedures applicable to a given context.

        Searches across procedure names, descriptions, and
        applicable_contexts fields using text matching.

        Args:
            context: Description of the current task context.
            agent_id: Filter by agent ownership.
            min_success_rate: Minimum success rate threshold.

        Returns:
            Matching procedures, sorted by success rate descending.

        Example:
            >>> procs = store.search_by_context("deploy frontend app")
            >>> for p in procs:
            ...     print(f"{p.name}: {p.success_rate:.0%} success")
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            # Search across name, description, and applicable_contexts
            query = """SELECT * FROM procedures
                       WHERE (name LIKE ? OR description LIKE ? OR applicable_contexts LIKE ?)
                       AND success_rate >= ?"""
            pattern = f"%{context}%"
            params: list = [pattern, pattern, pattern, min_success_rate]

            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)

            query += " ORDER BY success_rate DESC, execution_count DESC"

            rows = conn.execute(query, params).fetchall()
            conn.close()

        return [self._row_to_procedure(r) for r in rows]

    def search_by_name(self, name: str) -> list[ProcedureTemplate]:
        """Find procedures by name (exact or partial match).

        Args:
            name: Procedure name to search for.

        Returns:
            Matching procedures.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM procedures WHERE name LIKE ?", (f"%{name}%",)
            ).fetchall()
            conn.close()

        return [self._row_to_procedure(r) for r in rows]

    def record_execution(self, procedure_id: str, success: bool) -> bool:
        """Record the outcome of a procedure execution.

        Updates the procedure's success rate and execution count.

        Args:
            procedure_id: The procedure's UUID.
            success: Whether the execution was successful.

        Returns:
            True if the procedure was found and updated.
        """
        proc = self.get(procedure_id)
        if proc is None:
            return False

        proc.record_execution(success)
        self.add(proc)  # Upsert with updated stats
        return True

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
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM procedures"
            params: list = []

            if agent_id:
                query += " WHERE agent_id = ?"
                params.append(agent_id)

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            conn.close()

        return [self._row_to_procedure(r) for r in rows]

    def delete(self, procedure_id: str) -> bool:
        """Delete a procedure by ID.

        Args:
            procedure_id: The procedure's UUID.

        Returns:
            True if deleted.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                "DELETE FROM procedures WHERE id = ?", (procedure_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
        return deleted

    def count(self, agent_id: Optional[str] = None) -> int:
        """Count stored procedures.

        Args:
            agent_id: If provided, count only this agent's procedures.

        Returns:
            Number of stored procedures.
        """
        conn = sqlite3.connect(str(self._db_path))
        if agent_id:
            count = conn.execute(
                "SELECT COUNT(*) FROM procedures WHERE agent_id = ?", (agent_id,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        conn.close()
        return count

    @staticmethod
    def _step_to_dict(step: ProcedureStep) -> dict:
        return {
            "action": step.action,
            "tool": step.tool,
            "parameters": step.parameters,
            "expected_output": step.expected_output,
            "fallback_action": step.fallback_action,
        }

    @staticmethod
    def _row_to_procedure(row: sqlite3.Row) -> ProcedureTemplate:
        steps_data = json.loads(row["steps"])
        return ProcedureTemplate(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            steps=[
                ProcedureStep(
                    action=s["action"],
                    tool=s.get("tool"),
                    parameters=s.get("parameters", {}),
                    expected_output=s.get("expected_output"),
                    fallback_action=s.get("fallback_action"),
                )
                for s in steps_data
            ],
            agent_id=row["agent_id"],
            success_rate=row["success_rate"],
            execution_count=row["execution_count"],
            source_episodes=json.loads(row["source_episodes"]),
            applicable_contexts=json.loads(row["applicable_contexts"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

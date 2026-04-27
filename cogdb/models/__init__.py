"""Core data models for CogDB.

All memory records, knowledge graph triples, procedure templates,
and query/response types are defined here. These are the shared
vocabulary across all stores, pipeline stages, and adapters.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MemoryType(str, Enum):
    """The three cognitive memory types."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(str, Enum):
    """Multi-agent memory visibility scopes."""

    PRIVATE = "private"        # Single agent, fully isolated
    TEAM = "team"              # Defined agent group, read-write
    ORGANIZATION = "org"       # All agents, read + permissioned write
    SESSION = "session"        # Ephemeral, auto-deleted after conversation


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryUnit:
    """Universal memory record — the atom of CogDB.

    Every memory (episodic event, semantic fact, procedural step)
    is stored as a MemoryUnit with type-specific metadata.

    Example:
        >>> unit = MemoryUnit(
        ...     content="User prefers dark mode",
        ...     memory_type=MemoryType.EPISODIC,
        ...     agent_id="ui-agent",
        ...     importance=0.8,
        ... )
        >>> unit.id  # auto-generated UUID
        'a1b2c3d4-...'
    """

    content: str
    memory_type: MemoryType
    agent_id: str
    importance: float = 0.5
    scope: MemoryScope = MemoryScope.PRIVATE
    id: str = field(default_factory=_uuid)
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    accessed_at: datetime = field(default_factory=_now)
    access_count: int = 0
    decay_score: float = 1.0
    team_id: Optional[str] = None

    def touch(self) -> None:
        """Update access tracking — called on every retrieval."""
        self.accessed_at = _now()
        self.access_count += 1

    def effective_importance(self) -> float:
        """Importance adjusted by decay. Used for retrieval ranking."""
        return self.importance * self.decay_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/transport."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "agent_id": self.agent_id,
            "scope": self.scope.value,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "decay_score": self.decay_score,
            "team_id": self.team_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryUnit:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            agent_id=data["agent_id"],
            scope=MemoryScope(data.get("scope", "private")),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            decay_score=data.get("decay_score", 1.0),
            team_id=data.get("team_id"),
        )


@dataclass
class SemanticTriple:
    """A fact in the temporal knowledge graph.

    Triples carry validity windows — facts have lifecycles.
    They can be confirmed, contradicted, superseded, or expired.

    Example:
        >>> triple = SemanticTriple(
        ...     subject="user_settings",
        ...     predicate="theme",
        ...     object="dark_mode",
        ...     confidence=0.95,
        ...     agent_id="ui-agent",
        ... )
    """

    subject: str
    predicate: str
    object: str
    agent_id: str
    confidence: float = 1.0
    id: str = field(default_factory=_uuid)
    valid_from: datetime = field(default_factory=_now)
    valid_until: Optional[datetime] = None
    source_episodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Whether this fact is currently valid."""
        now = _now()
        if self.valid_until and now > self.valid_until:
            return False
        return now >= self.valid_from

    def supersede(self, new_triple: SemanticTriple) -> None:
        """Mark this triple as superseded by a newer fact."""
        self.valid_until = _now()
        self.metadata["superseded_by"] = new_triple.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "agent_id": self.agent_id,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "source_episodes": self.source_episodes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticTriple:
        return cls(
            id=data["id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            agent_id=data["agent_id"],
            confidence=data.get("confidence", 1.0),
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_until=(
                datetime.fromisoformat(data["valid_until"])
                if data.get("valid_until")
                else None
            ),
            source_episodes=data.get("source_episodes", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProcedureStep:
    """A single step within a learned procedure."""

    action: str
    tool: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None
    fallback_action: Optional[str] = None


@dataclass
class ProcedureTemplate:
    """A learned workflow extracted from successful agent task completions.

    When an agent solves a multi-step problem, the solution pattern
    is captured as a reusable template with success tracking.

    Example:
        >>> proc = ProcedureTemplate(
        ...     name="deploy_frontend",
        ...     description="Standard frontend deployment pipeline",
        ...     steps=[
        ...         ProcedureStep(action="run_tests", tool="pytest"),
        ...         ProcedureStep(action="build", tool="npm"),
        ...         ProcedureStep(action="deploy", tool="vercel"),
        ...     ],
        ...     agent_id="devops-agent",
        ... )
    """

    name: str
    description: str
    steps: list[ProcedureStep]
    agent_id: str
    id: str = field(default_factory=_uuid)
    success_rate: float = 1.0
    execution_count: int = 0
    source_episodes: list[str] = field(default_factory=list)
    applicable_contexts: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def record_execution(self, success: bool) -> None:
        """Update success rate after an execution."""
        self.execution_count += 1
        # Exponential moving average
        alpha = 0.3
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "action": s.action,
                    "tool": s.tool,
                    "parameters": s.parameters,
                    "expected_output": s.expected_output,
                    "fallback_action": s.fallback_action,
                }
                for s in self.steps
            ],
            "agent_id": self.agent_id,
            "success_rate": self.success_rate,
            "execution_count": self.execution_count,
            "source_episodes": self.source_episodes,
            "applicable_contexts": self.applicable_contexts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcedureTemplate:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=[
                ProcedureStep(
                    action=s["action"],
                    tool=s.get("tool"),
                    parameters=s.get("parameters", {}),
                    expected_output=s.get("expected_output"),
                    fallback_action=s.get("fallback_action"),
                )
                for s in data["steps"]
            ],
            agent_id=data["agent_id"],
            success_rate=data.get("success_rate", 1.0),
            execution_count=data.get("execution_count", 0),
            source_episodes=data.get("source_episodes", []),
            applicable_contexts=data.get("applicable_contexts", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class ContextResponse:
    """The structured response from progressive memory loading.

    Returned by CognitiveDB.get_context() — gives agents exactly
    the context they need within their token budget.
    """

    level: int
    token_count: int
    token_budget: int
    identity: str
    critical_facts: list[str] = field(default_factory=list)
    relevant_memories: list[MemoryUnit] = field(default_factory=list)
    deep_results: list[MemoryUnit] = field(default_factory=list)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.token_count)

    @property
    def utilization(self) -> float:
        """Fraction of token budget used (0.0 to 1.0)."""
        if self.token_budget == 0:
            return 0.0
        return min(1.0, self.token_count / self.token_budget)


@dataclass
class RecallQuery:
    """A structured query for memory retrieval."""

    query: str
    agent_id: str
    token_budget: int = 1000
    memory_types: list[MemoryType] = field(
        default_factory=lambda: [MemoryType.EPISODIC, MemoryType.SEMANTIC]
    )
    scope_filter: Optional[MemoryScope] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    min_importance: float = 0.0
    max_results: int = 20

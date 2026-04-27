"""Heuristic importance scoring for memory units.

Phase 0: rule-based scoring using content signals and access patterns.
Phase 2 will replace this with a lightweight ML model.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from cogdb.models import MemoryType, MemoryUnit


# Keywords that raise importance
_HIGH_IMPORTANCE_PATTERNS = [
    r"\b(error|fail|crash|critical|urgent|important|always|never)\b",
    r"\b(prefer|must|require|need|expect)\b",
    r"\b(password|secret|token|key|credential)\b",
    r"\b(deadline|due|asap|priority)\b",
]

_LOW_IMPORTANCE_PATTERNS = [
    r"\b(maybe|perhaps|might|could|possibly)\b",
    r"\b(test|example|sample|dummy|placeholder)\b",
]

_HIGH_RE = re.compile("|".join(_HIGH_IMPORTANCE_PATTERNS), re.IGNORECASE)
_LOW_RE = re.compile("|".join(_LOW_IMPORTANCE_PATTERNS), re.IGNORECASE)


def score_importance(
    content: str,
    memory_type: MemoryType,
    access_count: int = 0,
    recency_hours: float = 0.0,
    explicit_importance: float | None = None,
) -> float:
    """Compute a heuristic importance score for a memory unit.

    Combines content signals, memory type weights, access frequency,
    and recency into a single 0.0–1.0 score.

    Args:
        content: Raw text content of the memory.
        memory_type: Episodic, semantic, or procedural.
        access_count: How many times this memory has been retrieved.
        recency_hours: Hours since creation (0 = just created).
        explicit_importance: If provided, blend it with heuristic score.

    Returns:
        Importance score between 0.0 and 1.0.

    Example:
        >>> score_importance("User prefers dark mode", MemoryType.EPISODIC)
        0.55
        >>> score_importance("CRITICAL: API key expired", MemoryType.EPISODIC)
        0.82
    """
    score = _base_score(memory_type)
    score += _content_signal(content)
    score += _access_boost(access_count)
    score += _recency_boost(recency_hours)

    score = max(0.0, min(1.0, score))

    if explicit_importance is not None:
        # Weighted blend: explicit has 60% weight
        score = 0.6 * explicit_importance + 0.4 * score

    return round(score, 4)


def score_memory_unit(unit: MemoryUnit) -> float:
    """Convenience wrapper to score a MemoryUnit directly.

    Args:
        unit: The memory unit to score.

    Returns:
        Importance score between 0.0 and 1.0.

    Example:
        >>> unit = MemoryUnit(content="deploy failed", memory_type=MemoryType.EPISODIC, agent_id="a1")
        >>> score_memory_unit(unit)
        0.71
    """
    now = datetime.now(timezone.utc)
    recency_hours = (now - unit.created_at).total_seconds() / 3600.0

    return score_importance(
        content=unit.content,
        memory_type=unit.memory_type,
        access_count=unit.access_count,
        recency_hours=recency_hours,
    )


def _base_score(memory_type: MemoryType) -> float:
    """Base score by memory type — procedural and semantic start higher."""
    return {
        MemoryType.PROCEDURAL: 0.6,
        MemoryType.SEMANTIC: 0.55,
        MemoryType.EPISODIC: 0.4,
    }[memory_type]


def _content_signal(content: str) -> float:
    """Scan content for high/low importance keywords."""
    high_hits = len(_HIGH_RE.findall(content))
    low_hits = len(_LOW_RE.findall(content))

    # Each high-signal keyword adds 0.05, capped at +0.25
    # Each low-signal keyword subtracts 0.03, capped at -0.15
    boost = min(0.25, high_hits * 0.05)
    penalty = min(0.15, low_hits * 0.03)
    return boost - penalty


def _access_boost(access_count: int) -> float:
    """Frequently accessed memories are likely more important."""
    if access_count <= 0:
        return 0.0
    # Logarithmic boost, max +0.15
    import math
    return min(0.15, 0.05 * math.log1p(access_count))


def _recency_boost(recency_hours: float) -> float:
    """Recent memories get a small boost that decays over 48 hours."""
    if recency_hours <= 0:
        return 0.05
    if recency_hours >= 48:
        return 0.0
    # Linear decay from +0.05 to 0 over 48 hours
    return 0.05 * (1.0 - recency_hours / 48.0)

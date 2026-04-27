"""Episodic → semantic consolidation pipeline.

When enough episodic memories accumulate, the consolidator distills
recurring patterns into semantic triples (knowledge graph facts).
No external LLM calls — uses heuristic pattern extraction.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from cogdb.models import MemoryType, SemanticTriple
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig


# Simple (subject, predicate, object) extraction patterns.
# Matches phrases like "X prefers Y", "X uses Y", "X is Y", etc.
_SPO_PATTERNS = [
    (r"(\w[\w\s]{1,30}?)\s+(prefers?|likes?)\s+(.+)", "prefers"),
    (r"(\w[\w\s]{1,30}?)\s+(uses?|utilizes?)\s+(.+)", "uses"),
    (r"(\w[\w\s]{1,30}?)\s+(is|are|was)\s+(.+)", "is"),
    (r"(\w[\w\s]{1,30}?)\s+(has|have|had)\s+(.+)", "has"),
    (r"(\w[\w\s]{1,30}?)\s+(requires?|needs?)\s+(.+)", "requires"),
    (r"(\w[\w\s]{1,30}?)\s+(avoids?|never)\s+(.+)", "avoids"),
]

_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), predicate)
    for pattern, predicate in _SPO_PATTERNS
]


class Consolidator:
    """Distills high-frequency episodic patterns into semantic facts.

    Triggered when episodic memory count exceeds config.consolidation_threshold.
    Extracts recurring (subject, predicate, object) triples and writes
    them to the semantic store with confidence proportional to repetition.

    Args:
        episodic: Episodic memory store to read from.
        semantic: Semantic store to write distilled facts to.
        config: CogDB configuration.

    Example:
        >>> consolidator = Consolidator(episodic, semantic, config)
        >>> new_triples = consolidator.run(agent_id="ui-agent")
        >>> print(f"Distilled {new_triples} new facts")
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        config: CogDBConfig,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._config = config
        self._lock = threading.Lock()

    def should_consolidate(self, agent_id: str) -> bool:
        """Check if consolidation threshold has been reached for an agent.

        Args:
            agent_id: The agent to check.

        Returns:
            True if episodic count >= config.consolidation_threshold.

        Example:
            >>> if consolidator.should_consolidate("ui-agent"):
            ...     consolidator.run("ui-agent")
        """
        count = self._episodic.count(agent_id=agent_id)
        return count >= self._config.consolidation_threshold

    def run(self, agent_id: str, max_episodes: int = 100) -> int:
        """Run a consolidation pass for one agent.

        Scans recent episodic memories, extracts SPO patterns, groups
        by (subject, predicate, object) tuple, and writes the most
        frequent ones as semantic triples with confidence scores.

        Args:
            agent_id: The agent whose memories to consolidate.
            max_episodes: Maximum number of recent episodes to scan.

        Returns:
            Number of new semantic triples written.

        Example:
            >>> count = consolidator.run("ui-agent", max_episodes=50)
        """
        with self._lock:
            # Pull recent episodic memories
            episodes = self._episodic.search(
                query="",  # broad scan — no semantic query
                agent_id=agent_id,
                top_k=max_episodes,
            )

            if not episodes:
                return 0

            # Extract all SPO triples from episode texts
            raw_triples: list[tuple[str, str, str, str]] = []  # (subj, pred, obj, episode_id)
            for ep in episodes:
                extracted = _extract_triples(ep.content)
                for subj, pred, obj in extracted:
                    raw_triples.append((subj, pred, obj, ep.id))

            if not raw_triples:
                return 0

            # Count (subj, pred, obj) occurrences and collect source episode IDs
            spo_counter: Counter = Counter()
            spo_sources: dict[tuple, list[str]] = {}

            for subj, pred, obj, ep_id in raw_triples:
                key = (subj.lower().strip(), pred, obj.lower().strip())
                spo_counter[key] += 1
                spo_sources.setdefault(key, []).append(ep_id)

            # Write triples with count >= 2 (appeared in at least 2 memories)
            written = 0
            total_episodes = len(episodes)

            for (subj, pred, obj), count in spo_counter.items():
                if count < 2:
                    continue

                confidence = min(1.0, count / max(1, total_episodes * 0.3))

                # Skip if a similar active triple already exists
                existing = self._semantic.query_subject(
                    subject=subj, active_only=True, agent_id=agent_id
                )
                already_known = any(
                    t.predicate == pred and t.object.lower() == obj
                    for t in existing
                )
                if already_known:
                    continue

                triple = SemanticTriple(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    agent_id=agent_id,
                    confidence=confidence,
                    source_episodes=list(set(spo_sources[(subj, pred, obj)])),
                    metadata={"source": "consolidation", "episode_count": count},
                )
                self._semantic.add(triple)
                written += 1

        return written


def _extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) tuples from text using regex patterns."""
    results = []
    # Split into sentences for cleaner matching
    sentences = re.split(r"[.!?\n]+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue

        for pattern, predicate in _COMPILED:
            match = pattern.search(sentence)
            if match:
                subj = match.group(1).strip()
                obj = match.group(3).strip()

                # Clean up object — stop at punctuation or connectors
                obj = re.split(r"[,;]|\band\b|\bbut\b", obj)[0].strip()

                # Filter noise: skip if subject/object is too short or too long
                if 2 <= len(subj) <= 50 and 2 <= len(obj) <= 80:
                    results.append((subj, predicate, obj))
                break  # one pattern per sentence

    return results

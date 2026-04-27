"""Memory decay and eviction — importance degrades over time.

Implements exponential decay based on time since last access.
Memories below the eviction threshold are candidates for removal.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from typing import Optional

from cogdb.stores.episodic import EpisodicStore
from cogdb.utils.config import CogDBConfig


class DecayEngine:
    """Applies time-based importance decay to episodic memories.

    Uses an exponential decay model: decay_score = exp(-λ * hours_since_access)
    where λ is derived from config.decay_half_life_hours.

    Args:
        episodic: The episodic store to operate on.
        config: CogDB configuration.

    Example:
        >>> engine = DecayEngine(episodic_store, config)
        >>> evicted = engine.run_decay_pass(eviction_threshold=0.05)
        >>> print(f"Evicted {evicted} stale memories")
    """

    def __init__(self, episodic: EpisodicStore, config: CogDBConfig) -> None:
        self._episodic = episodic
        self._config = config
        self._lock = threading.Lock()
        # λ = ln(2) / half_life  so that decay(half_life) = 0.5
        self._lambda = math.log(2) / max(1.0, config.decay_half_life_hours)

    def compute_decay(self, last_accessed_at: datetime) -> float:
        """Compute the current decay score for a memory.

        Args:
            last_accessed_at: Timestamp of the last access.

        Returns:
            Decay score between 0.0 (fully decayed) and 1.0 (fresh).

        Example:
            >>> score = engine.compute_decay(datetime.now(timezone.utc))
            >>> score  # just accessed → close to 1.0
            0.9999...
        """
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - last_accessed_at).total_seconds() / 3600.0
        hours_elapsed = max(0.0, hours_elapsed)
        return math.exp(-self._lambda * hours_elapsed)

    def run_decay_pass(
        self,
        agent_id: Optional[str] = None,
        eviction_threshold: float = 0.05,
        batch_size: int = 200,
    ) -> int:
        """Apply decay to all stored memories and evict those below threshold.

        Iterates over stored memories in batches, recomputes their decay
        score, updates metadata, and deletes those below eviction_threshold.

        Args:
            agent_id: Limit decay pass to a specific agent. None = all agents.
            eviction_threshold: Memories with decay_score below this are deleted.
            batch_size: Number of memories to process per batch (RAM guard).

        Returns:
            Number of memories evicted during this pass.

        Example:
            >>> evicted = engine.run_decay_pass(eviction_threshold=0.1)
        """
        with self._lock:
            evicted = 0
            offset = 0

            while True:
                # Fetch a batch via the ChromaDB collection directly
                collection = self._episodic._collection
                where = {"agent_id": agent_id} if agent_id else None

                try:
                    if where:
                        batch = collection.get(
                            where=where,
                            limit=batch_size,
                            offset=offset,
                            include=["metadatas"],
                        )
                    else:
                        batch = collection.get(
                            limit=batch_size,
                            offset=offset,
                            include=["metadatas"],
                        )
                except Exception:
                    break

                ids = batch.get("ids", [])
                metadatas = batch.get("metadatas", [])

                if not ids:
                    break

                to_delete = []
                to_update_ids = []
                to_update_metas = []

                for memory_id, meta in zip(ids, metadatas):
                    accessed_at_str = meta.get("accessed_at", meta.get("created_at", ""))
                    if not accessed_at_str:
                        continue

                    try:
                        accessed_at = datetime.fromisoformat(accessed_at_str)
                    except ValueError:
                        continue

                    new_decay = self.compute_decay(accessed_at)

                    if new_decay < eviction_threshold:
                        to_delete.append(memory_id)
                    else:
                        to_update_ids.append(memory_id)
                        to_update_metas.append({**meta, "decay_score": new_decay})

                if to_delete:
                    collection.delete(ids=to_delete)
                    evicted += len(to_delete)

                if to_update_ids:
                    collection.update(ids=to_update_ids, metadatas=to_update_metas)

                if len(ids) < batch_size:
                    break

                offset += batch_size

        return evicted

    def refresh_decay(self, memory_id: str) -> Optional[float]:
        """Recompute and persist decay score for a single memory.

        Args:
            memory_id: The memory's UUID.

        Returns:
            New decay score, or None if memory not found.
        """
        unit = self._episodic.get(memory_id)
        if unit is None:
            return None

        new_decay = self.compute_decay(unit.accessed_at)
        self._episodic.update_metadata(memory_id, {"decay_score": new_decay})
        return new_decay

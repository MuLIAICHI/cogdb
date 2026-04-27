"""Memory encoder — converts raw text into MemoryUnits with embeddings.

Handles text → embedding computation via sentence-transformers and
automatic metadata extraction (keywords, entities, length signals).
Embeddings are computed lazily at encode time (not at query time).
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from cogdb.models import MemoryScope, MemoryType, MemoryUnit
from cogdb.models.importance import score_importance
from cogdb.utils.config import CogDBConfig
from cogdb.utils.tokenizer import count_tokens

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class Encoder:
    """Converts raw text into fully populated MemoryUnits.

    Computes embeddings via sentence-transformers and extracts
    lightweight metadata (token count, keywords, content signals)
    without any external LLM calls.

    Args:
        config: CogDB configuration (uses config.embedding_model).

    Example:
        >>> encoder = Encoder(config)
        >>> unit = encoder.encode(
        ...     text="User prefers dark mode",
        ...     memory_type=MemoryType.EPISODIC,
        ...     agent_id="ui-agent",
        ... )
        >>> unit.embedding is not None
        True
    """

    def __init__(self, config: CogDBConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._model: Optional[Any] = None  # lazy load

    def _get_model(self) -> Optional[Any]:
        """Load the sentence-transformers model on first use."""
        if self._model is not None:
            return self._model
        if not _ST_AVAILABLE:
            return None
        with self._lock:
            if self._model is None:
                self._model = SentenceTransformer(self._config.embedding_model)
        return self._model

    def encode(
        self,
        text: str,
        memory_type: MemoryType,
        agent_id: str,
        scope: MemoryScope = MemoryScope.PRIVATE,
        importance: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        team_id: Optional[str] = None,
    ) -> MemoryUnit:
        """Encode raw text into a MemoryUnit with embedding and metadata.

        Args:
            text: Raw text content to encode.
            memory_type: Episodic, semantic, or procedural.
            agent_id: The owning agent's ID.
            scope: Memory visibility scope.
            importance: Explicit importance (0.0–1.0). If None, auto-scored.
            metadata: Extra key-value metadata to attach.
            team_id: Team ID for team-scoped memories.

        Returns:
            A fully populated MemoryUnit with embedding and scored importance.

        Example:
            >>> unit = encoder.encode("API key expired", MemoryType.EPISODIC, "ops-agent")
            >>> 0.0 <= unit.importance <= 1.0
            True
        """
        embedding = self._compute_embedding(text)
        extracted_meta = _extract_metadata(text)

        if metadata:
            extracted_meta.update(metadata)

        scored_importance = score_importance(
            content=text,
            memory_type=memory_type,
            explicit_importance=importance,
        )

        return MemoryUnit(
            content=text,
            memory_type=memory_type,
            agent_id=agent_id,
            scope=scope,
            importance=scored_importance,
            embedding=embedding,
            metadata=extracted_meta,
            team_id=team_id,
        )

    def encode_batch(
        self,
        texts: list[str],
        memory_type: MemoryType,
        agent_id: str,
        scope: MemoryScope = MemoryScope.PRIVATE,
    ) -> list[MemoryUnit]:
        """Encode multiple texts efficiently using batched embedding.

        Args:
            texts: List of raw text strings.
            memory_type: Memory type for all units.
            agent_id: Owning agent for all units.
            scope: Memory scope for all units.

        Returns:
            List of MemoryUnits in the same order as input texts.

        Example:
            >>> units = encoder.encode_batch(["fact A", "fact B"], MemoryType.SEMANTIC, "a1")
            >>> len(units) == 2
            True
        """
        embeddings = self._compute_embeddings_batch(texts)

        units = []
        for i, text in enumerate(texts):
            extracted_meta = _extract_metadata(text)
            scored_importance = score_importance(
                content=text,
                memory_type=memory_type,
            )
            unit = MemoryUnit(
                content=text,
                memory_type=memory_type,
                agent_id=agent_id,
                scope=scope,
                importance=scored_importance,
                embedding=embeddings[i] if embeddings else None,
                metadata=extracted_meta,
            )
            units.append(unit)

        return units

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Compute an embedding for a query string.

        Used to pass pre-computed query embeddings to the episodic store,
        avoiding redundant embedding computation during retrieval.

        Args:
            text: Query text.

        Returns:
            Embedding vector, or None if sentence-transformers unavailable.

        Example:
            >>> vec = encoder.embed_query("dark mode preferences")
            >>> len(vec) == 384
            True
        """
        return self._compute_embedding(text)

    def _compute_embedding(self, text: str) -> Optional[list[float]]:
        model = self._get_model()
        if model is None:
            return None
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _compute_embeddings_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        model = self._get_model()
        if model is None:
            return None
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


def _extract_metadata(text: str) -> dict[str, Any]:
    """Extract lightweight metadata from text without an LLM.

    Extracts token count, word count, capitalized terms (likely entities),
    and whether the text contains question or negation signals.
    """
    token_count = count_tokens(text)
    words = text.split()
    word_count = len(words)

    # Capitalized words (skip first word and all-caps abbreviations)
    capitalized = [
        w.strip(".,!?;:\"'")
        for w in words[1:]
        if w and w[0].isupper() and not w.isupper() and len(w) > 1
    ]

    has_question = text.rstrip().endswith("?") or bool(
        re.search(r"\b(what|how|why|when|where|who|which)\b", text, re.IGNORECASE)
    )
    has_negation = bool(
        re.search(r"\b(not|never|no|cannot|can't|don't|doesn't|isn't|aren't)\b", text, re.IGNORECASE)
    )

    return {
        "token_count": token_count,
        "word_count": word_count,
        "entities": capitalized[:10],  # cap at 10
        "has_question": has_question,
        "has_negation": has_negation,
    }

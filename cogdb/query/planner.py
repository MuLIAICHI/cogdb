"""Query planner — routes queries to the right store(s) and merges results.

Analyzes the query intent, decides which stores to involve, enforces
scope access control, and merges results into a unified ranked list.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import Optional

from cogdb.models import (
    ContextResponse,
    MemoryScope,
    MemoryType,
    MemoryUnit,
    RecallQuery,
    SemanticTriple,
)
from cogdb.pipeline.retriever import Retriever
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.procedural import ProceduralStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig
from cogdb.utils.tokenizer import count_tokens


# Keywords that hint at which store to prioritize
_EPISODIC_HINTS = re.compile(
    r"\b(remember|recall|last time|yesterday|happened|did|said|told|saw|ago|recent|history|log)\b",
    re.IGNORECASE,
)
_SEMANTIC_HINTS = re.compile(
    r"\b(know|fact|true|is|are|what is|define|relationship|between|how does|who is)\b",
    re.IGNORECASE,
)
_PROCEDURAL_HINTS = re.compile(
    r"\b(how to|steps|procedure|workflow|process|deploy|run|execute|do I|should I)\b",
    re.IGNORECASE,
)


class QueryPlanner:
    """Routes recall queries to the optimal store(s) and merges results.

    Applies scope-based access control and intent-based store routing
    to return the most relevant memories within the token budget.

    Args:
        episodic: Episodic memory store.
        semantic: Semantic memory store.
        procedural: Procedural memory store.
        retriever: Token-budget-aware retriever.
        config: CogDB configuration.

    Example:
        >>> planner = QueryPlanner(episodic, semantic, procedural, retriever, config)
        >>> results = planner.execute(RecallQuery(
        ...     query="how do we deploy the frontend?",
        ...     agent_id="dev-agent",
        ...     token_budget=500,
        ... ))
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        procedural: ProceduralStore,
        retriever: Retriever,
        config: CogDBConfig,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural
        self._retriever = retriever
        self._config = config
        self._lock = threading.Lock()

    def execute(self, query: RecallQuery) -> list[MemoryUnit]:
        """Execute a recall query with intent-based store routing.

        Analyzes the query text to determine which store(s) to prioritize,
        enforces scope access control, then delegates to the retriever
        for token-budget-aware result merging.

        Args:
            query: Structured recall query.

        Returns:
            List of MemoryUnits within the token budget, ranked by relevance.

        Example:
            >>> results = planner.execute(RecallQuery(
            ...     query="what did the user say about dark mode?",
            ...     agent_id="ui-agent",
            ...     token_budget=300,
            ... ))
        """
        routed_query = self._route(query)
        return self._retriever.recall(routed_query)

    def get_context(
        self,
        agent_id: str,
        level: int = 2,
        task_hint: Optional[str] = None,
        token_budget: Optional[int] = None,
        identity: Optional[str] = None,
    ) -> ContextResponse:
        """Build progressive context via the retriever.

        Thin pass-through that applies access control before delegating
        to retriever.get_context().

        Args:
            agent_id: The agent requesting context.
            level: Max context level (0–3).
            task_hint: Current task description for L2/L3 relevance.
            token_budget: Override default token budget.
            identity: Agent identity string for L0.

        Returns:
            ContextResponse with tiered memory contents.

        Example:
            >>> ctx = planner.get_context("dev-agent", level=2, task_hint="deploy pipeline")
        """
        return self._retriever.get_context(
            agent_id=agent_id,
            level=level,
            task_hint=task_hint,
            token_budget=token_budget,
            identity=identity,
        )

    def explain(self, query: RecallQuery) -> dict:
        """Return a human-readable routing plan without executing the query.

        Useful for debugging — shows which stores would be queried and why.

        Args:
            query: The recall query to explain.

        Returns:
            Dict with keys: stores, reasoning, estimated_cost_tokens.

        Example:
            >>> plan = planner.explain(RecallQuery(query="how to deploy", agent_id="a1"))
            >>> plan["stores"]
            ['procedural', 'episodic']
        """
        routed = self._route(query)
        store_names = [t.value for t in routed.memory_types]
        reasoning = _explain_routing(query.query, routed.memory_types)

        return {
            "stores": store_names,
            "reasoning": reasoning,
            "token_budget": routed.token_budget,
            "scope_filter": routed.scope_filter.value if routed.scope_filter else None,
        }

    def _route(self, query: RecallQuery) -> RecallQuery:
        """Rewrite query.memory_types based on intent analysis."""
        text = query.query

        # If the caller explicitly restricted to a single store, honour it
        if len(query.memory_types) == 1:
            return query

        episodic_score = len(_EPISODIC_HINTS.findall(text))
        semantic_score = len(_SEMANTIC_HINTS.findall(text))
        procedural_score = len(_PROCEDURAL_HINTS.findall(text))

        total = episodic_score + semantic_score + procedural_score

        if total == 0:
            # No strong signal — return all requested stores unchanged
            return query

        # Build a prioritized type list (highest score first, include all with score > 0)
        scored = [
            (MemoryType.EPISODIC, episodic_score),
            (MemoryType.SEMANTIC, semantic_score),
            (MemoryType.PROCEDURAL, procedural_score),
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Include stores with at least 1 signal hit; always include top scorer
        selected = [t for t, s in scored if s > 0]
        if not selected:
            selected = [t for t, _ in scored]

        import dataclasses
        return dataclasses.replace(query, memory_types=selected)


def _explain_routing(query_text: str, types: list[MemoryType]) -> str:
    """Generate a human-readable explanation of why stores were selected."""
    reasons = []
    if MemoryType.EPISODIC in types and _EPISODIC_HINTS.search(query_text):
        reasons.append("episodic: temporal/recall keywords detected")
    if MemoryType.SEMANTIC in types and _SEMANTIC_HINTS.search(query_text):
        reasons.append("semantic: factual/relational keywords detected")
    if MemoryType.PROCEDURAL in types and _PROCEDURAL_HINTS.search(query_text):
        reasons.append("procedural: workflow/how-to keywords detected")
    if not reasons:
        reasons.append("no strong signal — querying all stores")
    return "; ".join(reasons)

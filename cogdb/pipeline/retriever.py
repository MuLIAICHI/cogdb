"""Token-budget-aware retriever — the core innovation of CogDB.

Retrieves memories across all three stores and returns them
in the most token-efficient format for LLM consumption.
Implements progressive loading (L0 → L3).
"""

from __future__ import annotations

from typing import Optional

from cogdb.models import (
    ContextResponse,
    MemoryScope,
    MemoryType,
    MemoryUnit,
    RecallQuery,
    SemanticTriple,
)
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.procedural import ProceduralStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig
from cogdb.utils.tokenizer import count_tokens, truncate_to_budget


class Retriever:
    """Token-budget-aware retriever across all memory stores.

    Given a query and token budget, retrieves the most relevant
    memories from episodic, semantic, and procedural stores,
    fitting them within the budget using progressive loading.

    Args:
        episodic: Episodic memory store.
        semantic: Semantic memory store.
        procedural: Procedural memory store.
        config: CogDB configuration.

    Example:
        >>> retriever = Retriever(episodic, semantic, procedural, config)
        >>> results = retriever.recall(RecallQuery(
        ...     query="How do we deploy?",
        ...     agent_id="dev-agent",
        ...     token_budget=500,
        ... ))
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        procedural: ProceduralStore,
        config: CogDBConfig,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural
        self._config = config

    def recall(self, query: RecallQuery) -> list[MemoryUnit]:
        """Retrieve memories matching a query within a token budget.

        Queries all requested memory stores, ranks results by
        effective importance, and greedily fills the token budget
        with the most important memories that fit.

        Args:
            query: Structured recall query with budget and filters.

        Returns:
            List of MemoryUnits fitting within the token budget,
            sorted by effective importance (highest first).
        """
        candidates: list[MemoryUnit] = []

        # Collect from episodic store
        if MemoryType.EPISODIC in query.memory_types:
            episodic_results = self._episodic.search(
                query=query.query,
                agent_id=query.agent_id,
                top_k=query.max_results,
                scope_filter=query.scope_filter,
                min_importance=query.min_importance,
                time_range_start=query.time_range_start,
                time_range_end=query.time_range_end,
            )
            candidates.extend(episodic_results)

        # Collect from semantic store
        if MemoryType.SEMANTIC in query.memory_types:
            semantic_triples = self._semantic.search_text(query.query)
            for triple in semantic_triples:
                unit = self._triple_to_unit(triple)
                candidates.append(unit)

        # Collect from procedural store
        if MemoryType.PROCEDURAL in query.memory_types:
            procedures = self._procedural.search_by_context(
                context=query.query,
                agent_id=query.agent_id,
            )
            for proc in procedures:
                unit = MemoryUnit(
                    id=proc.id,
                    content=self._procedure_to_text(proc),
                    memory_type=MemoryType.PROCEDURAL,
                    agent_id=proc.agent_id,
                    importance=proc.success_rate,
                    created_at=proc.created_at,
                    accessed_at=proc.updated_at,
                    metadata={"procedure_name": proc.name},
                )
                candidates.append(unit)

        # Rank by effective importance
        candidates.sort(key=lambda m: m.effective_importance(), reverse=True)

        # Greedy token-budget filling
        selected: list[MemoryUnit] = []
        tokens_used = 0

        for memory in candidates:
            memory_tokens = count_tokens(memory.content)
            if tokens_used + memory_tokens <= query.token_budget:
                memory.touch()
                selected.append(memory)
                tokens_used += memory_tokens
            elif tokens_used < query.token_budget:
                # Truncate last memory to fit remaining budget
                remaining = query.token_budget - tokens_used
                memory.content = truncate_to_budget(memory.content, remaining)
                memory.touch()
                selected.append(memory)
                break

        return selected

    def get_context(
        self,
        agent_id: str,
        level: int = 2,
        task_hint: Optional[str] = None,
        token_budget: Optional[int] = None,
        identity: Optional[str] = None,
    ) -> ContextResponse:
        """Build progressive context for an agent.

        Fills context in tiers:
        - L0: Agent identity (~50 tokens)
        - L1: Critical facts from knowledge graph (~200 tokens)
        - L2: Task-relevant memories (~500 tokens)
        - L3: Deep similarity search (remaining budget)

        Args:
            agent_id: The agent requesting context.
            level: Maximum context level to load (0-3).
            task_hint: Description of current task (improves L2/L3).
            token_budget: Override default budget.
            identity: Agent identity string for L0.

        Returns:
            ContextResponse with tiered memory contents.

        Example:
            >>> ctx = retriever.get_context(
            ...     agent_id="ui-agent",
            ...     level=2,
            ...     task_hint="redesigning the settings page",
            ... )
            >>> print(f"Used {ctx.token_count}/{ctx.token_budget} tokens")
        """
        budget = token_budget or self._config.default_token_budget
        response = ContextResponse(
            level=level,
            token_count=0,
            token_budget=budget,
            identity=identity or f"Agent: {agent_id}",
        )

        # L0: Identity
        l0_text = response.identity
        l0_tokens = count_tokens(l0_text)
        response.token_count += l0_tokens

        if level < 1:
            return response

        # L1: Critical facts from knowledge graph
        l1_budget = min(
            self._config.l1_token_budget,
            budget - response.token_count,
        )
        if l1_budget > 0:
            # Get high-confidence facts for this agent
            all_entities = self._semantic.get_entities()
            critical_facts: list[str] = []
            l1_tokens_used = 0

            for entity in all_entities[:20]:  # Cap entity scan
                triples = self._semantic.query_subject(
                    entity, active_only=True, agent_id=agent_id
                )
                # Also include org-level facts (no agent filter)
                if not triples:
                    triples = self._semantic.query_subject(entity, active_only=True)

                for triple in sorted(
                    triples, key=lambda t: t.confidence, reverse=True
                ):
                    fact_text = f"{triple.subject} {triple.predicate} {triple.object}"
                    fact_tokens = count_tokens(fact_text)
                    if l1_tokens_used + fact_tokens <= l1_budget:
                        critical_facts.append(fact_text)
                        l1_tokens_used += fact_tokens

            response.critical_facts = critical_facts
            response.token_count += l1_tokens_used

        if level < 2:
            return response

        # L2: Task-relevant memories
        l2_budget = min(
            self._config.l2_token_budget,
            budget - response.token_count,
        )
        if l2_budget > 0 and task_hint:
            relevant = self.recall(
                RecallQuery(
                    query=task_hint,
                    agent_id=agent_id,
                    token_budget=l2_budget,
                    memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
                )
            )
            response.relevant_memories = relevant
            response.token_count += sum(
                count_tokens(m.content) for m in relevant
            )

        if level < 3:
            return response

        # L3: Deep search with remaining budget
        l3_budget = budget - response.token_count
        if l3_budget > 50 and task_hint:  # Only if meaningful budget remains
            deep = self.recall(
                RecallQuery(
                    query=task_hint,
                    agent_id=agent_id,
                    token_budget=l3_budget,
                    memory_types=[
                        MemoryType.EPISODIC,
                        MemoryType.SEMANTIC,
                        MemoryType.PROCEDURAL,
                    ],
                    max_results=50,
                )
            )
            # Filter out memories already in L2
            l2_ids = {m.id for m in response.relevant_memories}
            deep = [m for m in deep if m.id not in l2_ids]
            response.deep_results = deep
            response.token_count += sum(
                count_tokens(m.content) for m in deep
            )

        return response

    @staticmethod
    def _triple_to_unit(triple: SemanticTriple) -> MemoryUnit:
        """Convert a semantic triple to a MemoryUnit for unified ranking."""
        content = f"{triple.subject} {triple.predicate} {triple.object}"
        return MemoryUnit(
            id=triple.id,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            agent_id=triple.agent_id,
            importance=triple.confidence,
            created_at=triple.valid_from,
            metadata={
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
            },
        )

    @staticmethod
    def _procedure_to_text(proc) -> str:
        """Convert a procedure to a compact text representation."""
        steps_text = " → ".join(
            f"{s.action}" + (f" ({s.tool})" if s.tool else "")
            for s in proc.steps
        )
        return f"Procedure '{proc.name}': {proc.description}. Steps: {steps_text}"

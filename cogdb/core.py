"""CognitiveDB — the single entry point for agent memory.

Composes all three stores and the retrieval pipeline into
a clean API that agents interact with directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from cogdb.models import (
    ContextResponse,
    MemoryScope,
    MemoryType,
    MemoryUnit,
    ProcedureStep,
    ProcedureTemplate,
    RecallQuery,
    SemanticTriple,
)
from cogdb.pipeline.retriever import Retriever
from cogdb.schema import MetadataSchema
from cogdb.schema.registry import SchemaRegistry
from cogdb.stores.episodic import EpisodicStore
from cogdb.stores.procedural import ProceduralStore
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig


class CognitiveDB:
    """A cognitive database engine for AI agents.

    Unifies episodic, semantic, and procedural memory into a
    single interface with token-cost-aware retrieval and
    multi-agent memory scopes.

    Args:
        config: Configuration object. If None, uses defaults.
        db_path: Shortcut to set config.db_path.

    Example:
        >>> db = CognitiveDB(db_path="./my_agent_memory")
        >>>
        >>> # Store memories
        >>> db.remember("User prefers dark mode", agent_id="ui-agent")
        >>> db.learn("user", "prefers", "dark_mode", agent_id="ui-agent")
        >>>
        >>> # Recall with token budget
        >>> memories = db.recall("UI preferences", agent_id="ui-agent", token_budget=500)
        >>>
        >>> # Get progressive context
        >>> ctx = db.get_context(agent_id="ui-agent", task_hint="settings page")
    """

    def __init__(
        self,
        config: Optional[CogDBConfig] = None,
        db_path: Optional[str] = None,
    ) -> None:
        if config is None:
            config = CogDBConfig()
        if db_path is not None:
            config.db_path = db_path

        self._config = config
        config.ensure_dirs()

        # Initialize stores
        self._episodic = EpisodicStore(config)
        self._semantic = SemanticStore(config)
        self._procedural = ProceduralStore(config)

        # Initialize retriever
        self._retriever = Retriever(
            self._episodic, self._semantic, self._procedural, config
        )

        # Initialize schema registry
        self._schema_registry = SchemaRegistry(
            db_path=config.db_path,
            strict=config.strict_metadata_validation,
        )

    # ── Episodic Memory ─────────────────────────────────────────

    def remember(
        self,
        content: str,
        agent_id: Optional[str] = None,
        importance: float = 0.5,
        scope: MemoryScope = MemoryScope.PRIVATE,
        metadata: Optional[dict[str, Any]] = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        team_id: Optional[str] = None,
    ) -> str:
        """Store an episodic memory.

        Args:
            content: The text content to remember.
            agent_id: The agent storing this memory.
            importance: Importance score (0.0 to 1.0).
            scope: Visibility scope for multi-agent access.
            metadata: Additional key-value metadata.
            memory_type: Override memory type (default: episodic).
            team_id: Team identifier for team-scoped memories.

        Returns:
            The UUID of the stored memory.

        Example:
            >>> db.remember(
            ...     "Deployment failed due to missing env var DB_HOST",
            ...     agent_id="devops-agent",
            ...     importance=0.9,
            ...     metadata={"error_type": "config", "service": "api"},
            ... )
        """
        resolved_agent = agent_id or self._config.default_agent_id
        resolved_metadata = metadata or {}

        self._schema_registry.validate_and_raise(resolved_metadata, resolved_agent)

        unit = MemoryUnit(
            content=content,
            memory_type=memory_type,
            agent_id=resolved_agent,
            importance=importance,
            scope=scope,
            metadata=resolved_metadata,
            team_id=team_id,
        )
        return self._episodic.add(unit)

    # ── Semantic Memory ─────────────────────────────────────────

    def learn(
        self,
        subject: str,
        predicate: str,
        object: str,
        agent_id: Optional[str] = None,
        confidence: float = 1.0,
        valid_from: Optional[datetime] = None,
        source_episodes: Optional[list[str]] = None,
    ) -> str:
        """Store a semantic fact in the knowledge graph.

        If an existing fact with the same subject+predicate exists,
        it will be superseded (marked as no longer valid).

        Args:
            subject: The entity this fact is about.
            predicate: The relationship or property.
            object: The value or target entity.
            agent_id: The agent asserting this fact.
            confidence: Confidence score (0.0 to 1.0).
            valid_from: When this fact becomes valid (default: now).
            source_episodes: IDs of episodic memories that support this fact.

        Returns:
            The UUID of the stored triple.

        Example:
            >>> db.learn(
            ...     subject="api_service",
            ...     predicate="deployed_version",
            ...     object="v2.3.1",
            ...     agent_id="devops-agent",
            ...     confidence=1.0,
            ... )
        """
        triple = SemanticTriple(
            subject=subject,
            predicate=predicate,
            object=object,
            agent_id=agent_id or self._config.default_agent_id,
            confidence=confidence,
            source_episodes=source_episodes or [],
        )
        if valid_from:
            triple.valid_from = valid_from

        return self._semantic.add_triple(triple)

    def query_knowledge(
        self,
        entity: str,
        depth: int = 1,
        active_only: bool = True,
    ) -> list[SemanticTriple]:
        """Query the knowledge graph around an entity.

        Args:
            entity: The entity to explore.
            depth: How many relationship hops to traverse.
            active_only: Only return currently valid facts.

        Returns:
            List of semantic triples connected to the entity.

        Example:
            >>> facts = db.query_knowledge("api_service", depth=2)
            >>> for f in facts:
            ...     print(f"{f.subject} → {f.predicate} → {f.object}")
        """
        return self._semantic.query_entity(entity, depth=depth, active_only=active_only)

    # ── Procedural Memory ───────────────────────────────────────

    def learn_procedure(
        self,
        name: str,
        steps: list[dict[str, Any]],
        agent_id: Optional[str] = None,
        description: str = "",
        success_rate: float = 1.0,
        source_episodes: Optional[list[str]] = None,
        applicable_contexts: Optional[list[str]] = None,
    ) -> str:
        """Store a learned procedure (workflow template).

        Args:
            name: Name of the procedure.
            steps: List of step dicts with 'action', 'tool', 'parameters'.
            agent_id: The agent that learned this procedure.
            description: Human-readable description.
            success_rate: Initial success rate (0.0 to 1.0).
            source_episodes: IDs of episodes this was extracted from.
            applicable_contexts: Keywords describing when to use this.

        Returns:
            The UUID of the stored procedure.

        Example:
            >>> db.learn_procedure(
            ...     name="fix_cors_error",
            ...     description="Fix CORS errors in the API gateway",
            ...     steps=[
            ...         {"action": "check_config", "tool": "cat nginx.conf"},
            ...         {"action": "add_headers", "tool": "sed"},
            ...         {"action": "restart", "tool": "systemctl restart nginx"},
            ...         {"action": "verify", "tool": "curl -I"},
            ...     ],
            ...     agent_id="devops-agent",
            ...     applicable_contexts=["cors", "api", "nginx", "headers"],
            ... )
        """
        proc_steps = [
            ProcedureStep(
                action=s.get("action", ""),
                tool=s.get("tool"),
                parameters=s.get("parameters", {}),
                expected_output=s.get("expected_output"),
                fallback_action=s.get("fallback_action"),
            )
            for s in steps
        ]

        procedure = ProcedureTemplate(
            name=name,
            description=description,
            steps=proc_steps,
            agent_id=agent_id or self._config.default_agent_id,
            success_rate=success_rate,
            source_episodes=source_episodes or [],
            applicable_contexts=applicable_contexts or [],
        )

        return self._procedural.add(procedure)

    # ── Retrieval ───────────────────────────────────────────────

    def recall(
        self,
        query: str,
        agent_id: Optional[str] = None,
        token_budget: Optional[int] = None,
        memory_types: Optional[list[MemoryType]] = None,
        scope_filter: Optional[MemoryScope] = None,
        min_importance: float = 0.0,
        max_results: int = 20,
    ) -> list[MemoryUnit]:
        """Recall memories matching a query within a token budget.

        Searches across all requested memory types, ranks by
        importance, and returns results that fit within the budget.

        Args:
            query: Natural language query.
            agent_id: The querying agent.
            token_budget: Max tokens in the response.
            memory_types: Which memory types to search.
            scope_filter: Filter by scope.
            min_importance: Minimum importance threshold.
            max_results: Max candidate results per store.

        Returns:
            List of MemoryUnits within budget, sorted by importance.

        Example:
            >>> memories = db.recall(
            ...     "deployment errors this week",
            ...     agent_id="devops-agent",
            ...     token_budget=500,
            ... )
        """
        recall_query = RecallQuery(
            query=query,
            agent_id=agent_id or self._config.default_agent_id,
            token_budget=token_budget or self._config.default_token_budget,
            memory_types=memory_types
            or [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            scope_filter=scope_filter,
            min_importance=min_importance,
            max_results=max_results,
        )
        return self._retriever.recall(recall_query)

    def get_context(
        self,
        agent_id: Optional[str] = None,
        level: int = 2,
        task_hint: Optional[str] = None,
        token_budget: Optional[int] = None,
        identity: Optional[str] = None,
    ) -> ContextResponse:
        """Build progressive context for an agent.

        Returns tiered context from L0 (identity) through L3 (deep search).

        Args:
            agent_id: The agent requesting context.
            level: Max context level (0-3).
            task_hint: Current task description for relevance.
            token_budget: Override default budget.
            identity: Agent identity string.

        Returns:
            ContextResponse with tiered memory contents.

        Example:
            >>> ctx = db.get_context(
            ...     agent_id="ui-agent",
            ...     level=2,
            ...     task_hint="redesigning the settings page",
            ...     token_budget=800,
            ... )
            >>> print(f"Loaded {len(ctx.critical_facts)} facts, "
            ...       f"{len(ctx.relevant_memories)} relevant memories")
        """
        return self._retriever.get_context(
            agent_id=agent_id or self._config.default_agent_id,
            level=level,
            task_hint=task_hint,
            token_budget=token_budget,
            identity=identity,
        )

    # ── Schema Registry ─────────────────────────────────────────

    def register_schema(self, schema: MetadataSchema) -> None:
        """Register or update the typed metadata schema for an agent.

        Once registered, every call to ``remember()`` for this agent will
        validate the provided metadata against the schema. Re-registering
        overwrites the existing schema and increments the version counter.

        Args:
            schema: The MetadataSchema to register.

        Returns:
            None

        Example:
            >>> from cogdb.schema import MetadataSchema, FieldSchema
            >>> db.register_schema(MetadataSchema(
            ...     agent_id="devops-agent",
            ...     fields={
            ...         "tool":      FieldSchema(type="str", required=True),
            ...         "exit_code": FieldSchema(type="int", required=False, default=0),
            ...     },
            ... ))
        """
        self._schema_registry.register(schema)

    def get_schema(self, agent_id: str) -> Optional[MetadataSchema]:
        """Retrieve the registered metadata schema for an agent.

        Args:
            agent_id: The agent whose schema to retrieve.

        Returns:
            MetadataSchema if registered, None otherwise.

        Example:
            >>> schema = db.get_schema("devops-agent")
            >>> if schema:
            ...     print(f"v{schema.version}: {list(schema.fields)}")
        """
        return self._schema_registry.get(agent_id)

    def list_schemas(self) -> list[MetadataSchema]:
        """Return all registered metadata schemas, sorted by agent_id.

        Returns:
            List of MetadataSchema instances.

        Example:
            >>> for s in db.list_schemas():
            ...     print(s.agent_id, "v" + str(s.version))
        """
        return self._schema_registry.list_schemas()

    # ── Utilities ───────────────────────────────────────────────

    def forget(self, memory_id: str, memory_type: MemoryType) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: The memory's UUID.
            memory_type: Which store to delete from.

        Returns:
            True if deleted.
        """
        if memory_type == MemoryType.EPISODIC:
            return self._episodic.delete(memory_id)
        elif memory_type == MemoryType.SEMANTIC:
            return self._semantic.delete_triple(memory_id)
        elif memory_type == MemoryType.PROCEDURAL:
            return self._procedural.delete(memory_id)
        return False

    def stats(self) -> dict[str, int]:
        """Get memory statistics across all stores.

        Returns:
            Dict with counts per memory type.

        Example:
            >>> db.stats()
            {'episodic': 142, 'semantic': 58, 'procedural': 12, 'total': 212}
        """
        episodic_count = self._episodic.count()
        semantic_count = self._semantic.count()
        procedural_count = self._procedural.count()

        return {
            "episodic": episodic_count,
            "semantic": semantic_count,
            "procedural": procedural_count,
            "total": episodic_count + semantic_count + procedural_count,
        }

    @property
    def episodic(self) -> EpisodicStore:
        """Direct access to the episodic store."""
        return self._episodic

    @property
    def semantic(self) -> SemanticStore:
        """Direct access to the semantic store."""
        return self._semantic

    @property
    def procedural(self) -> ProceduralStore:
        """Direct access to the procedural store."""
        return self._procedural

"""Typed metadata schema definitions for episodic memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "list", "dict", "any"}
)


@dataclass
class FieldSchema:
    """Definition of a single metadata field.

    Args:
        type: Expected Python type. One of: str, int, float, bool, list, dict, any.
        required: If True the field must be present on every episodic write.
        default: Documented default when the field is absent and not required.
        description: Human-readable description for tooling and documentation.

    Example:
        >>> FieldSchema(type="str", required=True, description="Tool that produced this memory")
        >>> FieldSchema(type="int", required=False, default=0, description="Process exit code")
    """

    type: str
    required: bool = False
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported field type '{self.type}'. "
                f"Supported: {sorted(SUPPORTED_TYPES)}"
            )


@dataclass
class MetadataSchema:
    """Typed schema for episodic memory metadata for a specific agent.

    Re-registering a schema for the same agent_id overwrites the existing
    schema and increments the version counter — providing a hook for
    future migration tooling (Phase 3C).

    Args:
        agent_id: The agent this schema applies to.
        fields: Mapping of field name to FieldSchema definition.
        version: Schema version. Auto-incremented on re-registration.
        created_at: ISO-8601 timestamp set by SchemaRegistry on registration.

    Example:
        >>> from cogdb.schema import MetadataSchema, FieldSchema
        >>> schema = MetadataSchema(
        ...     agent_id="devops-agent",
        ...     fields={
        ...         "tool":      FieldSchema(type="str", required=True,
        ...                                  description="CLI tool used"),
        ...         "exit_code": FieldSchema(type="int", required=False, default=0),
        ...         "service":   FieldSchema(type="str", required=False),
        ...     },
        ... )
        >>> db.register_schema(schema)
    """

    agent_id: str
    fields: dict[str, FieldSchema] = field(default_factory=dict)
    version: int = 1
    created_at: str = ""  # ISO string; populated by SchemaRegistry.register()


class SchemaValidationError(ValueError):
    """Raised when episodic metadata does not conform to a registered schema.

    Attributes:
        errors: Field-level error messages describing each violation.

    Example:
        >>> try:
        ...     db.remember("task failed", agent_id="devops-agent",
        ...                 metadata={"exit_code": "oops"})
        ... except SchemaValidationError as e:
        ...     print(e.errors)
        ['metadata.exit_code: expected int, got str']
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"Metadata schema violation ({len(errors)} error(s)): "
            + "; ".join(errors)
        )

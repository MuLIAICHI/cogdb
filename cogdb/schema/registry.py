"""Schema registry — persists and validates metadata schemas per agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cogdb.schema import FieldSchema, MetadataSchema, SchemaValidationError


class SchemaRegistry:
    """Persists and validates typed metadata schemas for episodic memories.

    Schemas are stored as JSON at ``{db_path}/schemas.json``.
    Each schema is keyed by ``agent_id``. Validation applies only to
    episodic memories; other memory types are unaffected.

    Args:
        db_path: Root storage directory (matches CogDBConfig.db_path).
        strict: If True, raise SchemaValidationError on violation.
                If False, validation runs but violations are silently ignored.

    Example:
        >>> registry = SchemaRegistry("./my_db", strict=True)
        >>> registry.register(MetadataSchema(
        ...     agent_id="devops-agent",
        ...     fields={"tool": FieldSchema(type="str", required=True)},
        ... ))
        >>> errors = registry.validate({"exit_code": 0}, "devops-agent")
        >>> # ["metadata.tool: required field missing"]
    """

    _FILENAME = "schemas.json"

    def __init__(self, db_path: "str | Any", strict: bool = True) -> None:
        # Accept a CogDBConfig object or a plain path string
        if hasattr(db_path, "db_path"):
            db_path = db_path.db_path
        self._path = Path(db_path) / self._FILENAME
        self._strict = strict
        self._schemas: dict[str, MetadataSchema] = {}
        self._load()

    # ── Public API ───────────────────────────────────────────────

    def register(self, schema: MetadataSchema) -> None:
        """Register or update the metadata schema for an agent.

        If a schema already exists for this agent_id the version is
        incremented and the field definitions are replaced.

        Args:
            schema: The MetadataSchema to register.

        Returns:
            None

        Example:
            >>> registry.register(MetadataSchema(
            ...     agent_id="devops-agent",
            ...     fields={"tool": FieldSchema(type="str", required=True)},
            ... ))
        """
        existing = self._schemas.get(schema.agent_id)
        if existing is not None:
            schema.version = existing.version + 1
        schema.created_at = datetime.now(timezone.utc).isoformat()
        self._schemas[schema.agent_id] = schema
        self._save()

    def get(self, agent_id: str) -> Optional[MetadataSchema]:
        """Retrieve the registered schema for an agent.

        Args:
            agent_id: The agent whose schema to retrieve.

        Returns:
            MetadataSchema if registered, None otherwise.

        Example:
            >>> schema = registry.get("devops-agent")
            >>> if schema:
            ...     print(schema.version)
        """
        return self._schemas.get(agent_id)

    def list_schemas(self) -> list[MetadataSchema]:
        """Return all registered schemas, sorted by agent_id.

        Returns:
            List of MetadataSchema instances.

        Example:
            >>> for s in registry.list_schemas():
            ...     print(s.agent_id, "v" + str(s.version))
        """
        return sorted(self._schemas.values(), key=lambda s: s.agent_id)

    def validate(self, metadata: dict, agent_id: str) -> list[str]:
        """Validate metadata against the registered schema for an agent.

        Returns a list of field-level error strings. An empty list means
        the metadata is valid. If no schema is registered for this agent,
        always returns [].

        Args:
            metadata: The metadata dict to validate.
            agent_id: The agent whose schema to validate against.

        Returns:
            List of error strings, e.g.
            ``["metadata.tool: required field missing",
               "metadata.exit_code: expected int, got str"]``.

        Example:
            >>> errors = registry.validate({"exit_code": "bad"}, "devops-agent")
        """
        schema = self._schemas.get(agent_id)
        if schema is None:
            return []

        errors: list[str] = []
        for field_name, field_def in schema.fields.items():
            if field_name not in metadata:
                if field_def.required and field_def.default is None:
                    errors.append(f"metadata.{field_name}: required field missing")
            else:
                value = metadata[field_name]
                if not self._check_type(value, field_def.type):
                    actual = type(value).__name__
                    errors.append(
                        f"metadata.{field_name}: expected {field_def.type}, got {actual}"
                    )
        return errors

    def validate_and_raise(self, metadata: dict, agent_id: str) -> None:
        """Validate metadata and raise SchemaValidationError if invalid.

        No-op when no schema is registered for this agent. When
        ``strict=False``, validation runs but errors are silently ignored.

        Args:
            metadata: The metadata dict to validate.
            agent_id: The agent whose schema to validate against.

        Returns:
            None

        Raises:
            SchemaValidationError: When strict=True and validation fails.

        Example:
            >>> registry.validate_and_raise({"tool": "bash"}, "devops-agent")
        """
        if not self._strict:
            return
        errors = self.validate(metadata, agent_id)
        if errors:
            raise SchemaValidationError(errors)

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in raw.get("schemas", []):
                fields = {
                    name: FieldSchema(
                        type=fdef["type"],
                        required=fdef.get("required", False),
                        default=fdef.get("default"),
                        description=fdef.get("description", ""),
                    )
                    for name, fdef in entry.get("fields", {}).items()
                }
                schema = MetadataSchema(
                    agent_id=entry["agent_id"],
                    name=entry.get("name", ""),
                    fields=fields,
                    version=entry.get("version", 1),
                    created_at=entry.get("created_at", ""),
                )
                self._schemas[schema.agent_id] = schema
        except Exception:
            pass  # Corrupt file — start fresh; existing memories are unaffected

    def _save(self) -> None:
        entries = [
            {
                "agent_id": schema.agent_id,
                "name": getattr(schema, "name", ""),
                "fields": {
                    name: {
                        "type": fd.type,
                        "required": fd.required,
                        "default": fd.default,
                        "description": fd.description,
                    }
                    for name, fd in schema.fields.items()
                },
                "version": schema.version,
                "created_at": schema.created_at,
            }
            for schema in self._schemas.values()
        ]
        self._path.write_text(
            json.dumps({"schemas": entries}, indent=2),
            encoding="utf-8",
        )

    # ── Type checking ────────────────────────────────────────────

    @staticmethod
    def _check_type(value: object, type_name: str) -> bool:
        if type_name == "any":
            return True
        # bool must be checked before int because bool is a subclass of int
        if type_name == "bool":
            return isinstance(value, bool)
        if type_name == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if type_name == "str":
            return isinstance(value, str)
        if type_name == "list":
            return isinstance(value, list)
        if type_name == "dict":
            return isinstance(value, dict)
        return True

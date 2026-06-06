"""Phase 3C: Schema migration — safe field add, rename, drop with version tracking."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from cogdb.schema import FieldSchema, MetadataSchema


@dataclass
class FieldChange:
    """Describes a single field change in a migration.

    Args:
        operation: One of "add", "rename", "drop", "change_type", "change_required".
        field_name: The field this change targets.
        new_name: Target name (rename only).
        new_schema: Replacement FieldSchema (add / change_type).
        default_value: Fill value for existing records when adding a field.

    Example:
        >>> FieldChange("add", "priority", new_schema=FieldSchema(type="int"), default_value=0)
    """

    operation: str
    field_name: str
    new_name: Optional[str] = None
    new_schema: Optional[FieldSchema] = None
    default_value: Any = None


@dataclass
class SchemaMigration:
    """A versioned schema migration for a single agent's metadata schema.

    Args:
        agent_id: The agent whose schema is being migrated.
        from_version: The schema version this migration starts from.
        to_version: The schema version this migration produces.
        changes: Ordered list of FieldChange operations.
        description: Human-readable summary of the migration.
        created_at: Timestamp of migration creation.

    Example:
        >>> migration = (
        ...     SchemaMigration(agent_id="my-agent", from_version=1, to_version=2)
        ...     .add_field("priority", FieldSchema(type="int", default=0), default=0)
        ...     .rename_field("tags", "labels")
        ... )
    """

    agent_id: str
    from_version: int
    to_version: int
    changes: list[FieldChange] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_field(self, name: str, schema: FieldSchema, default: Any = None) -> "SchemaMigration":
        """Add a new field to the schema.

        Args:
            name: Name of the new field.
            schema: FieldSchema definition for the new field.
            default: Value injected into existing records that lack this field.

        Returns:
            self, for method chaining.
        """
        self.changes.append(FieldChange("add", name, new_schema=schema, default_value=default))
        return self

    def rename_field(self, old_name: str, new_name: str) -> "SchemaMigration":
        """Rename an existing field.

        Args:
            old_name: Current field name.
            new_name: Desired field name.

        Returns:
            self, for method chaining.
        """
        self.changes.append(FieldChange("rename", old_name, new_name=new_name))
        return self

    def drop_field(self, name: str) -> "SchemaMigration":
        """Remove an optional field from the schema.

        Args:
            name: Name of the field to drop. Raises ValueError if required.

        Returns:
            self, for method chaining.
        """
        self.changes.append(FieldChange("drop", name))
        return self

    def change_type(self, name: str, new_schema: FieldSchema) -> "SchemaMigration":
        """Replace the FieldSchema for an existing field.

        Args:
            name: Name of the field to change.
            new_schema: New FieldSchema definition.

        Returns:
            self, for method chaining.
        """
        self.changes.append(FieldChange("change_type", name, new_schema=new_schema))
        return self


class SchemaMigrator:
    """Applies SchemaMigrations to a MetadataSchema and backfills existing records.

    Args:
        registry: The SchemaRegistry to read/write schemas.

    Example:
        >>> migrator = SchemaMigrator(registry)
        >>> migration = (
        ...     SchemaMigration(agent_id="my-agent", from_version=1, to_version=2)
        ...     .add_field("priority", FieldSchema(type="int", default=0), default=0)
        ...     .rename_field("tags", "labels")
        ... )
        >>> new_schema = migrator.apply(migration)
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def apply(self, migration: SchemaMigration) -> MetadataSchema:
        """Apply a migration to the stored schema and register the result.

        Args:
            migration: The SchemaMigration to apply.

        Returns:
            The updated MetadataSchema (already persisted in the registry).

        Raises:
            ValueError: If from_version doesn't match the current schema version,
                        the agent has no registered schema, a field constraint is
                        violated (e.g. dropping a required field), or a duplicate
                        field name is added.

        Example:
            >>> new_schema = migrator.apply(migration)
            >>> print(new_schema.version)
            2
        """
        current = self._registry.get(migration.agent_id)
        if current is None:
            raise ValueError(f"No schema registered for agent_id={migration.agent_id!r}")
        if current.version != migration.from_version:
            raise ValueError(
                f"Schema version mismatch: expected {migration.from_version}, "
                f"got {current.version}"
            )

        new_fields: dict[str, FieldSchema] = copy.deepcopy(current.fields)

        for change in migration.changes:
            if change.operation == "add":
                if change.field_name in new_fields:
                    raise ValueError(f"Field {change.field_name!r} already exists")
                new_fields[change.field_name] = change.new_schema

            elif change.operation == "rename":
                if change.field_name not in new_fields:
                    raise ValueError(f"Field {change.field_name!r} not found")
                new_fields[change.new_name] = new_fields.pop(change.field_name)

            elif change.operation == "drop":
                if change.field_name not in new_fields:
                    raise ValueError(f"Field {change.field_name!r} not found")
                dropped = new_fields[change.field_name]
                if dropped.required:
                    raise ValueError(
                        f"Cannot drop required field {change.field_name!r} — "
                        "make it optional first"
                    )
                new_fields.pop(change.field_name)

            elif change.operation == "change_type":
                if change.field_name not in new_fields:
                    raise ValueError(f"Field {change.field_name!r} not found")
                new_fields[change.field_name] = change.new_schema

        new_schema = MetadataSchema(
            name=getattr(current, "name", ""),
            agent_id=migration.agent_id,
            fields=new_fields,
            version=migration.to_version,
        )
        self._registry.register(new_schema)
        return new_schema

    def migrate_metadata(
        self,
        migration: SchemaMigration,
        records: list[dict],
    ) -> list[dict]:
        """Backfill a list of metadata dicts according to the migration's changes.

        Returns a new list of updated dicts; originals are not mutated.
        Use this after calling apply() to bring existing memory metadata
        in line with the new schema.

        Args:
            migration: The migration whose changes to replay.
            records: List of metadata dicts to update.

        Returns:
            New list of updated metadata dicts.

        Example:
            >>> updated = migrator.migrate_metadata(migration, existing_records)
        """
        result = []
        for record in records:
            updated = copy.deepcopy(record)
            for change in migration.changes:
                if change.operation == "add":
                    if change.field_name not in updated:
                        updated[change.field_name] = change.default_value
                elif change.operation == "rename":
                    if change.field_name in updated:
                        updated[change.new_name] = updated.pop(change.field_name)
                elif change.operation == "drop":
                    updated.pop(change.field_name, None)
                elif change.operation == "change_type":
                    pass  # type coercion is left to the caller
            result.append(updated)
        return result

    def plan(self, migration: SchemaMigration) -> list[str]:
        """Return a human-readable description of what this migration will do.

        Useful for dry-run inspection before calling apply().

        Args:
            migration: The migration to describe.

        Returns:
            List of strings, one per line of the plan.

        Example:
            >>> for line in migrator.plan(migration):
            ...     print(line)
        """
        lines = [
            f"Migration: {migration.agent_id} v{migration.from_version} → v{migration.to_version}",
            f"Description: {migration.description or '(none)'}",
            "Changes:",
        ]
        op_labels = {
            "add": "ADD",
            "rename": "RENAME",
            "drop": "DROP",
            "change_type": "CHANGE TYPE",
            "change_required": "CHANGE REQUIRED",
        }
        for change in migration.changes:
            label = op_labels.get(change.operation, change.operation.upper())
            if change.operation == "rename":
                lines.append(f"  {label}: {change.field_name!r} → {change.new_name!r}")
            elif change.operation == "add":
                t = change.new_schema.type if change.new_schema else "?"
                lines.append(
                    f"  {label}: {change.field_name!r} "
                    f"(type={t}, default={change.default_value!r})"
                )
            else:
                lines.append(f"  {label}: {change.field_name!r}")
        return lines

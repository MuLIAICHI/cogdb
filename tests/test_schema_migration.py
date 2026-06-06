"""Tests for Phase 3C: schema migration."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from cogdb.schema import FieldSchema, MetadataSchema
from cogdb.schema.migration import FieldChange, SchemaMigration, SchemaMigrator
from cogdb.schema.registry import SchemaRegistry


def make_registry(tmp_path):
    from cogdb.utils.config import CogDBConfig
    cfg = CogDBConfig(db_path=tmp_path)
    return SchemaRegistry(cfg)


class TestSchemaMigration:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.registry = make_registry(self.tmp)
        self.registry.register(MetadataSchema(
            name="test", agent_id="agent1",
            fields={
                "session_id": FieldSchema(type="str", required=True),
                "tags": FieldSchema(type="list", required=False, default=[]),
            }
        ))

    def test_add_field(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .add_field("priority", FieldSchema(type="int", default=0), default=0)
        )
        migrator = SchemaMigrator(self.registry)
        new_schema = migrator.apply(migration)
        assert new_schema.version == 2
        assert "priority" in new_schema.fields

    def test_rename_field(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .rename_field("tags", "labels")
        )
        migrator = SchemaMigrator(self.registry)
        new_schema = migrator.apply(migration)
        assert "labels" in new_schema.fields
        assert "tags" not in new_schema.fields

    def test_drop_optional_field(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .drop_field("tags")
        )
        migrator = SchemaMigrator(self.registry)
        new_schema = migrator.apply(migration)
        assert "tags" not in new_schema.fields

    def test_drop_required_field_raises(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .drop_field("session_id")  # required field
        )
        migrator = SchemaMigrator(self.registry)
        with pytest.raises(ValueError, match="required"):
            migrator.apply(migration)

    def test_version_mismatch_raises(self):
        migration = SchemaMigration(agent_id="agent1", from_version=99, to_version=100)
        migrator = SchemaMigrator(self.registry)
        with pytest.raises(ValueError, match="version mismatch"):
            migrator.apply(migration)

    def test_unknown_agent_raises(self):
        migration = SchemaMigration(agent_id="nobody", from_version=1, to_version=2)
        migrator = SchemaMigrator(self.registry)
        with pytest.raises(ValueError, match="No schema"):
            migrator.apply(migration)

    def test_migrate_metadata_backfill(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .add_field("priority", FieldSchema(type="int", default=0), default=0)
        )
        migrator = SchemaMigrator(self.registry)
        records = [{"session_id": "abc"}, {"session_id": "def", "priority": 5}]
        updated = migrator.migrate_metadata(migration, records)
        assert updated[0]["priority"] == 0
        assert updated[1]["priority"] == 5

    def test_migrate_metadata_rename(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
            .rename_field("tags", "labels")
        )
        migrator = SchemaMigrator(self.registry)
        records = [{"session_id": "x", "tags": ["a", "b"]}]
        updated = migrator.migrate_metadata(migration, records)
        assert "labels" in updated[0]
        assert "tags" not in updated[0]

    def test_plan_output(self):
        migration = (
            SchemaMigration(agent_id="agent1", from_version=1, to_version=2,
                            description="Add priority field")
            .add_field("priority", FieldSchema(type="int", default=0))
            .drop_field("tags")
        )
        migrator = SchemaMigrator(self.registry)
        plan = migrator.plan(migration)
        assert isinstance(plan, list)
        assert any("ADD" in line for line in plan)
        assert any("DROP" in line for line in plan)

    def test_chained_migration(self):
        m1 = (SchemaMigration(agent_id="agent1", from_version=1, to_version=2)
              .add_field("priority", FieldSchema(type="int", default=0)))
        migrator = SchemaMigrator(self.registry)
        migrator.apply(m1)
        m2 = (SchemaMigration(agent_id="agent1", from_version=2, to_version=3)
              .rename_field("priority", "urgency"))
        new = migrator.apply(m2)
        assert new.version == 3
        assert "urgency" in new.fields

    def test_cognitivedb_migrate_schema(self):
        from cogdb.core import CognitiveDB
        db = CognitiveDB(db_path=self.tmp)
        db.register_schema(MetadataSchema(
            name="myschema", agent_id="db-agent",
            fields={"note": FieldSchema(type="str", required=False)}
        ))
        from cogdb.schema.migration import SchemaMigration
        m = (SchemaMigration(agent_id="db-agent", from_version=1, to_version=2)
             .add_field("score", FieldSchema(type="float", default=0.0)))
        result = db.migrate_schema(m)
        assert result.version == 2
        assert "score" in result.fields

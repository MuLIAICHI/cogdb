"""Tests for Phase 3A dynamic typed metadata schemas."""

from __future__ import annotations

import json
import tempfile

import pytest

from cogdb.core import CognitiveDB
from cogdb.schema import FieldSchema, MetadataSchema, SchemaValidationError
from cogdb.schema.registry import SchemaRegistry
from cogdb.utils.config import CogDBConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def registry(tmpdir):
    return SchemaRegistry(tmpdir, strict=True)


@pytest.fixture
def devops_schema():
    return MetadataSchema(
        agent_id="devops-agent",
        fields={
            "tool":      FieldSchema(type="str", required=True, description="CLI tool"),
            "exit_code": FieldSchema(type="int", required=False, default=0),
            "service":   FieldSchema(type="str", required=False),
            "success":   FieldSchema(type="bool", required=False),
        },
    )


@pytest.fixture
def db(tmpdir):
    return CognitiveDB(db_path=tmpdir)


# ── FieldSchema ───────────────────────────────────────────────────────────────


class TestFieldSchema:
    def test_valid_types_accepted(self):
        for t in ("str", "int", "float", "bool", "list", "dict", "any"):
            fs = FieldSchema(type=t)
            assert fs.type == t

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported field type"):
            FieldSchema(type="uuid")

    def test_defaults(self):
        fs = FieldSchema(type="str")
        assert fs.required is False
        assert fs.default is None
        assert fs.description == ""


# ── MetadataSchema ────────────────────────────────────────────────────────────


class TestMetadataSchema:
    def test_construction(self, devops_schema):
        assert devops_schema.agent_id == "devops-agent"
        assert devops_schema.version == 1
        assert "tool" in devops_schema.fields
        assert devops_schema.fields["tool"].required is True

    def test_default_version_is_one(self):
        s = MetadataSchema(agent_id="a")
        assert s.version == 1

    def test_empty_fields(self):
        s = MetadataSchema(agent_id="a")
        assert s.fields == {}


# ── SchemaValidationError ─────────────────────────────────────────────────────


class TestSchemaValidationError:
    def test_errors_attribute(self):
        errs = ["metadata.tool: required field missing"]
        exc = SchemaValidationError(errs)
        assert exc.errors == errs

    def test_str_includes_count(self):
        exc = SchemaValidationError(["e1", "e2"])
        assert "2 error(s)" in str(exc)

    def test_str_includes_messages(self):
        exc = SchemaValidationError(["metadata.x: required field missing"])
        assert "metadata.x" in str(exc)


# ── SchemaRegistry.register ───────────────────────────────────────────────────


class TestSchemaRegistryRegister:
    def test_register_new(self, registry, devops_schema):
        registry.register(devops_schema)
        assert registry.get("devops-agent") is not None

    def test_register_sets_created_at(self, registry, devops_schema):
        registry.register(devops_schema)
        assert registry.get("devops-agent").created_at != ""

    def test_reregister_bumps_version(self, registry, devops_schema):
        registry.register(devops_schema)
        assert registry.get("devops-agent").version == 1
        registry.register(MetadataSchema(agent_id="devops-agent", fields={}))
        assert registry.get("devops-agent").version == 2

    def test_reregister_replaces_fields(self, registry, devops_schema):
        registry.register(devops_schema)
        updated = MetadataSchema(
            agent_id="devops-agent",
            fields={"new_field": FieldSchema(type="str")},
        )
        registry.register(updated)
        schema = registry.get("devops-agent")
        assert "new_field" in schema.fields
        assert "tool" not in schema.fields


# ── SchemaRegistry.validate ───────────────────────────────────────────────────


class TestSchemaRegistryValidate:
    def test_no_schema_always_passes(self, registry):
        assert registry.validate({"anything": 123}, "unknown-agent") == []

    def test_valid_metadata_no_errors(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"tool": "bash", "exit_code": 0}, "devops-agent")
        assert errors == []

    def test_required_field_missing(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({}, "devops-agent")
        assert any("tool" in e and "required" in e for e in errors)

    def test_wrong_type_reported(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"tool": "bash", "exit_code": "oops"}, "devops-agent")
        assert any("exit_code" in e and "int" in e for e in errors)

    def test_error_message_format(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"tool": 99}, "devops-agent")
        assert errors[0].startswith("metadata.tool:")
        assert "expected str" in errors[0]
        assert "got int" in errors[0]

    def test_optional_field_absent_is_ok(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"tool": "curl"}, "devops-agent")
        assert errors == []

    def test_unknown_extra_fields_ignored(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"tool": "curl", "unknown": True}, "devops-agent")
        assert errors == []

    def test_bool_not_accepted_as_int(self, registry):
        schema = MetadataSchema(
            agent_id="a",
            fields={"count": FieldSchema(type="int", required=True)},
        )
        registry.register(schema)
        errors = registry.validate({"count": True}, "a")
        assert any("count" in e for e in errors)

    def test_int_accepted_as_float(self, registry):
        schema = MetadataSchema(
            agent_id="a",
            fields={"score": FieldSchema(type="float", required=True)},
        )
        registry.register(schema)
        errors = registry.validate({"score": 3}, "a")
        assert errors == []

    def test_multiple_errors_returned(self, registry, devops_schema):
        registry.register(devops_schema)
        errors = registry.validate({"exit_code": "bad", "success": 1}, "devops-agent")
        # tool required missing + exit_code wrong type + success wrong type
        assert len(errors) >= 2


# ── SchemaRegistry.validate_and_raise ────────────────────────────────────────


class TestSchemaRegistryValidateAndRaise:
    def test_strict_raises_on_error(self, registry, devops_schema):
        registry.register(devops_schema)
        with pytest.raises(SchemaValidationError) as exc_info:
            registry.validate_and_raise({}, "devops-agent")
        assert exc_info.value.errors

    def test_non_strict_does_not_raise(self, tmpdir, devops_schema):
        lax = SchemaRegistry(tmpdir, strict=False)
        lax.register(devops_schema)
        lax.validate_and_raise({}, "devops-agent")  # must not raise


# ── Persistence ───────────────────────────────────────────────────────────────


class TestSchemaRegistryPersistence:
    def test_roundtrip(self, tmpdir, devops_schema):
        r1 = SchemaRegistry(tmpdir, strict=True)
        r1.register(devops_schema)

        r2 = SchemaRegistry(tmpdir, strict=True)
        schema = r2.get("devops-agent")
        assert schema is not None
        assert schema.version == 1
        assert schema.fields["tool"].required is True
        assert schema.fields["exit_code"].default == 0

    def test_file_written(self, tmpdir, devops_schema):
        r = SchemaRegistry(tmpdir, strict=True)
        r.register(devops_schema)
        path = __import__("pathlib").Path(tmpdir) / "schemas.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert any(s["agent_id"] == "devops-agent" for s in data["schemas"])

    def test_corrupt_file_starts_fresh(self, tmpdir):
        path = __import__("pathlib").Path(tmpdir) / "schemas.json"
        path.write_text("not valid json", encoding="utf-8")
        r = SchemaRegistry(tmpdir, strict=True)
        assert r.list_schemas() == []


# ── CognitiveDB integration ───────────────────────────────────────────────────


class TestCognitiveDBSchema:
    def test_register_and_get(self, db, devops_schema):
        db.register_schema(devops_schema)
        schema = db.get_schema("devops-agent")
        assert schema is not None
        assert schema.agent_id == "devops-agent"

    def test_get_unregistered_returns_none(self, db):
        assert db.get_schema("nobody") is None

    def test_list_schemas(self, db, devops_schema):
        db.register_schema(devops_schema)
        db.register_schema(MetadataSchema(agent_id="other-agent", fields={}))
        schemas = db.list_schemas()
        ids = [s.agent_id for s in schemas]
        assert "devops-agent" in ids
        assert "other-agent" in ids

    def test_remember_valid_metadata_succeeds(self, db, devops_schema):
        db.register_schema(devops_schema)
        uid = db.remember(
            "Deployment succeeded",
            agent_id="devops-agent",
            metadata={"tool": "kubectl"},
        )
        assert uid is not None

    def test_remember_invalid_metadata_raises(self, db, devops_schema):
        db.register_schema(devops_schema)
        with pytest.raises(SchemaValidationError) as exc_info:
            db.remember(
                "Deployment failed",
                agent_id="devops-agent",
                metadata={"tool": 99, "exit_code": "oops"},
            )
        assert len(exc_info.value.errors) >= 1

    def test_remember_required_field_missing_raises(self, db, devops_schema):
        db.register_schema(devops_schema)
        with pytest.raises(SchemaValidationError) as exc_info:
            db.remember("task done", agent_id="devops-agent", metadata={})
        assert any("tool" in e for e in exc_info.value.errors)

    def test_remember_no_schema_always_passes(self, db):
        uid = db.remember(
            "Unregistered agent memory",
            agent_id="unregistered",
            metadata={"anything": True, "random": 42},
        )
        assert uid is not None

    def test_strict_false_allows_violations(self, tmpdir, devops_schema):
        config = CogDBConfig(db_path=tmpdir, strict_metadata_validation=False)
        lax_db = CognitiveDB(config=config)
        lax_db.register_schema(devops_schema)
        uid = lax_db.remember(
            "bad metadata but not strict",
            agent_id="devops-agent",
            metadata={},  # missing required "tool"
        )
        assert uid is not None

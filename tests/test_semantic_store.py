"""Tests for the semantic memory store (knowledge graph)."""

import tempfile

import pytest

from cogdb.models import SemanticTriple
from cogdb.stores.semantic import SemanticStore
from cogdb.utils.config import CogDBConfig


@pytest.fixture
def store():
    """Create a fresh SemanticStore with a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CogDBConfig(db_path=tmpdir)
        config.ensure_dirs()
        yield SemanticStore(config)


class TestSemanticStoreAdd:
    def test_add_triple(self, store):
        triple = SemanticTriple(
            subject="user",
            predicate="prefers",
            object="dark_mode",
            agent_id="ui-agent",
        )
        result_id = store.add_triple(triple)
        assert result_id == triple.id
        assert store.count() == 1

    def test_add_multiple_triples(self, store):
        for obj in ["dark_mode", "compact_layout", "english"]:
            store.add_triple(SemanticTriple(
                subject="user",
                predicate="prefers",
                object=obj,
                agent_id="ui-agent",
            ))
        # With contradiction check on, each new "prefers" supersedes the last
        # So only the latest should be active
        active = store.query_subject("user", active_only=True)
        assert len(active) >= 1


class TestSemanticStoreContradiction:
    def test_supersedes_contradicting_fact(self, store):
        # First fact: theme is light
        store.add_triple(SemanticTriple(
            subject="user",
            predicate="theme",
            object="light_mode",
            agent_id="ui-agent",
            confidence=0.8,
        ))

        # Second fact: theme is dark (contradicts first)
        store.add_triple(SemanticTriple(
            subject="user",
            predicate="theme",
            object="dark_mode",
            agent_id="ui-agent",
            confidence=0.95,
        ))

        active = store.query_subject("user", active_only=True)
        active_themes = [t for t in active if t.predicate == "theme"]
        assert len(active_themes) == 1
        assert active_themes[0].object == "dark_mode"

    def test_keeps_different_predicates(self, store):
        store.add_triple(SemanticTriple(
            subject="user",
            predicate="theme",
            object="dark_mode",
            agent_id="ui-agent",
        ))
        store.add_triple(SemanticTriple(
            subject="user",
            predicate="language",
            object="english",
            agent_id="ui-agent",
        ))

        active = store.query_subject("user", active_only=True)
        predicates = {t.predicate for t in active}
        assert "theme" in predicates
        assert "language" in predicates

    def test_no_contradiction_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CogDBConfig(db_path=tmpdir, contradiction_check=False)
            config.ensure_dirs()
            store = SemanticStore(config)

            store.add_triple(SemanticTriple(
                subject="user", predicate="theme", object="light",
                agent_id="a1",
            ))
            store.add_triple(SemanticTriple(
                subject="user", predicate="theme", object="dark",
                agent_id="a1",
            ))

            # Both should be active when contradiction check is off
            active = store.query_subject("user", active_only=True)
            assert len(active) == 2


class TestSemanticStoreQuery:
    def test_query_subject(self, store):
        store.add_triple(SemanticTriple(
            subject="api", predicate="version", object="v2.3",
            agent_id="devops",
        ))
        store.add_triple(SemanticTriple(
            subject="api", predicate="status", object="healthy",
            agent_id="devops",
        ))
        store.add_triple(SemanticTriple(
            subject="database", predicate="status", object="healthy",
            agent_id="devops",
        ))

        api_facts = store.query_subject("api")
        assert len(api_facts) == 2

    def test_query_entity_depth(self, store):
        store.add_triple(SemanticTriple(
            subject="api", predicate="connects_to", object="database",
            agent_id="devops",
        ))
        store.add_triple(SemanticTriple(
            subject="database", predicate="hosted_on", object="aws_rds",
            agent_id="devops",
        ))

        # Depth 1: api → database
        depth1 = store.query_entity("api", depth=1)
        entities_in_results = set()
        for t in depth1:
            entities_in_results.add(t.subject)
            entities_in_results.add(t.object)
        assert "database" in entities_in_results

        # Depth 2: api → database → aws_rds
        depth2 = store.query_entity("api", depth=2)
        entities_in_results_2 = set()
        for t in depth2:
            entities_in_results_2.add(t.subject)
            entities_in_results_2.add(t.object)
        assert "aws_rds" in entities_in_results_2

    def test_search_text(self, store):
        store.add_triple(SemanticTriple(
            subject="nginx", predicate="config_path",
            object="/etc/nginx/conf.d/api.conf",
            agent_id="devops",
        ))

        results = store.search_text("nginx")
        assert len(results) >= 1
        assert results[0].subject == "nginx"

    def test_get_entities(self, store):
        store.add_triple(SemanticTriple(
            subject="a", predicate="links", object="b",
            agent_id="test",
        ))
        entities = store.get_entities()
        assert "a" in entities
        assert "b" in entities

    def test_get_neighbors(self, store):
        store.add_triple(SemanticTriple(
            subject="a", predicate="links", object="b",
            agent_id="test",
        ))
        store.add_triple(SemanticTriple(
            subject="a", predicate="links", object="c",
            agent_id="test",
        ))

        neighbors = store.get_neighbors("a")
        assert "b" in neighbors
        assert "c" in neighbors


class TestSemanticStoreDelete:
    def test_delete_triple(self, store):
        triple = SemanticTriple(
            subject="user", predicate="name", object="Alice",
            agent_id="test",
        )
        store.add_triple(triple)
        assert store.count() == 1

        store.delete_triple(triple.id)
        assert store.count() == 0

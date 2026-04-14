"""Example: Multi-agent memory with CogDB.

Demonstrates a DevOps agent and a UI agent sharing a CogDB instance
with different memory scopes and progressive context loading.
"""

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType


def main():
    # Initialize CogDB
    db = CognitiveDB(db_path="./example_memory")

    print("=== CogDB Multi-Agent Memory Example ===\n")

    # ── DevOps Agent stores memories ────────────────────────

    # Episodic: what happened
    db.remember(
        "Deployed API v2.3.1 to production. All health checks passed.",
        agent_id="devops-agent",
        importance=0.7,
        scope=MemoryScope.ORGANIZATION,  # All agents can see this
    )

    db.remember(
        "CORS error on /api/v2/users endpoint. Fixed by adding Access-Control-Allow-Origin header to nginx config.",
        agent_id="devops-agent",
        importance=0.9,
        metadata={"error_type": "cors", "service": "api-gateway"},
    )

    db.remember(
        "Database migration took 45 minutes due to large users table. Next time, consider batched migration.",
        agent_id="devops-agent",
        importance=0.8,
    )

    # Semantic: what is known
    db.learn(
        subject="api_service",
        predicate="current_version",
        object="v2.3.1",
        agent_id="devops-agent",
        confidence=1.0,
    )

    db.learn(
        subject="api_service",
        predicate="deployed_at",
        object="2026-04-14T10:30:00Z",
        agent_id="devops-agent",
    )

    db.learn(
        subject="nginx",
        predicate="cors_config",
        object="Access-Control-Allow-Origin: * in /etc/nginx/conf.d/api.conf",
        agent_id="devops-agent",
        confidence=0.95,
    )

    # Procedural: how to do things
    db.learn_procedure(
        name="fix_cors_error",
        description="Fix CORS errors in the API gateway nginx config",
        steps=[
            {"action": "check_nginx_config", "tool": "cat /etc/nginx/conf.d/api.conf"},
            {"action": "add_cors_headers", "tool": "sed -i 's/location \\//location \\/ {\\n    add_header Access-Control-Allow-Origin *;/'"},
            {"action": "test_config", "tool": "nginx -t"},
            {"action": "reload_nginx", "tool": "systemctl reload nginx"},
            {"action": "verify_fix", "tool": "curl -I https://api.example.com/v2/users"},
        ],
        agent_id="devops-agent",
        success_rate=0.95,
        applicable_contexts=["cors", "nginx", "api", "headers", "403", "preflight"],
    )

    # ── UI Agent stores memories ────────────────────────────

    db.remember(
        "User complained about settings page loading slowly. Identified 3 unnecessary API calls.",
        agent_id="ui-agent",
        importance=0.8,
    )

    db.learn(
        subject="user_preference",
        predicate="theme",
        object="dark_mode",
        agent_id="ui-agent",
        confidence=0.95,
    )

    db.learn(
        subject="user_preference",
        predicate="layout",
        object="compact",
        agent_id="ui-agent",
        confidence=0.9,
    )

    # ── Query examples ──────────────────────────────────────

    print("--- Memory Stats ---")
    stats = db.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n--- DevOps Agent: Recall CORS issues (budget: 300 tokens) ---")
    cors_memories = db.recall(
        "CORS error fix",
        agent_id="devops-agent",
        token_budget=300,
        memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
    )
    for m in cors_memories:
        print(f"  [{m.memory_type.value}] (importance: {m.importance:.1f}) {m.content[:100]}...")

    print("\n--- UI Agent: Knowledge graph query ---")
    prefs = db.query_knowledge("user_preference", depth=1)
    for fact in prefs:
        print(f"  {fact.subject} → {fact.predicate} → {fact.object} (confidence: {fact.confidence})")

    print("\n--- UI Agent: Progressive context (L2, budget: 500) ---")
    ctx = db.get_context(
        agent_id="ui-agent",
        level=2,
        task_hint="redesigning the settings page",
        token_budget=500,
    )
    print(f"  Identity: {ctx.identity}")
    print(f"  Critical facts: {len(ctx.critical_facts)}")
    for fact in ctx.critical_facts:
        print(f"    - {fact}")
    print(f"  Relevant memories: {len(ctx.relevant_memories)}")
    for m in ctx.relevant_memories:
        print(f"    - [{m.memory_type.value}] {m.content[:80]}...")
    print(f"  Token usage: {ctx.token_count}/{ctx.token_budget} ({ctx.utilization:.0%})")

    print("\n--- Cross-agent: DevOps agent sees org-scoped deployment info ---")
    deployment_info = db.recall(
        "deployment version",
        agent_id="ui-agent",  # UI agent querying
        token_budget=200,
    )
    for m in deployment_info:
        print(f"  [{m.scope.value}] {m.content[:100]}...")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

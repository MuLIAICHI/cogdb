"""CogDB Benchmark Suite — tests CogDB's three core differentiators.

Suite 1 — Tri-Memory Retrieval:
    Synthetic DevOps team scenario (3 agents, 20 sessions).
    90 stored memories (50 episodic + 30 semantic + 10 procedural).
    30 questions that require combining memory types to answer.
    LLM judge scores answer quality 0-100.

Suite 2 — Token Efficiency:
    Same data, same questions.
    Compares three retrieval approaches per question:
      progressive  — CogDB L0-L3 get_context (token-budgeted)
      dump-all     — CogDB recall with uncapped budget
      baseline     — raw ChromaDB top-k, no budget management
    Reports information density = quality / tokens_used.

Suite 3 — Multi-Agent Consistency:
    3 concurrent threads (planner, coder, reviewer), 50 ops each.
    Measures: consistency_score, data_loss_score, conflict_resolution_accuracy.

Suite 4 — Throughput:
    Measures raw write/search/scan latency (ms/op) and ops/sec.
    Two modes per operation:
      full-pipeline — includes Python sentence-transformers encoding step.
      raw-storage   — pre-computed embedding, isolates pure Rust I/O cost.

Usage:
    python -m benchmarks.cogdb_bench --suite all
    python -m benchmarks.cogdb_bench --suite tri-memory
    python -m benchmarks.cogdb_bench --suite token-efficiency
    python -m benchmarks.cogdb_bench --suite consistency
    python -m benchmarks.cogdb_bench --suite throughput
    python -m benchmarks.cogdb_bench --suite all --no-llm   # skip OpenAI, use keyword fallback
    python -m benchmarks.cogdb_bench --suite all --out results/my_run.json

Requires OPENAI_API_KEY for LLM judge. Falls back to keyword scoring automatically
if the key is absent or --no-llm is passed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cogdb.core import CognitiveDB
from cogdb.models import MemoryScope, MemoryType
from cogdb.utils.tokenizer import count_tokens

try:
    from openai import OpenAI as _OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ── Synthetic DevOps scenario data ────────────────────────────────────────────

EPISODIC_MEMORIES: list[dict[str, Any]] = [
    # Sessions 1-2: project bootstrap + first CORS bug
    {"content": "Session 1: Planner initialised project — API gateway (Python/FastAPI), React frontend, PostgreSQL database.", "agent_id": "planner", "importance": 0.6},
    {"content": "Session 1: Coder scaffolded FastAPI app with /api/users and /api/tasks endpoints.", "agent_id": "coder", "importance": 0.55},
    {"content": "Session 1: Reviewer approved initial project layout PR #1 with no blocking comments.", "agent_id": "reviewer", "importance": 0.5},
    {"content": "Session 2: Frontend call to /api/users returned CORS error: No Access-Control-Allow-Origin header present.", "agent_id": "coder", "importance": 0.9},
    {"content": "Session 2: Coder diagnosed CORS error — nginx was stripping response headers before forwarding to the browser.", "agent_id": "coder", "importance": 0.85},
    # Sessions 3-4: CORS fix + staging deploy
    {"content": "Session 3: Coder added Access-Control-Allow-Origin: * and Access-Control-Allow-Methods headers to nginx.conf.", "agent_id": "coder", "importance": 0.88},
    {"content": "Session 3: Staging deploy succeeded after CORS fix; all frontend-to-API calls verified working.", "agent_id": "coder", "importance": 0.75},
    {"content": "Session 3: Reviewer confirmed CORS fix in PR #4 and noted we should restrict Allow-Origin to our domain in production.", "agent_id": "reviewer", "importance": 0.8},
    {"content": "Session 4: Production deploy of v1.0.0 completed using standard deploy flow — no rollback needed.", "agent_id": "planner", "importance": 0.7},
    {"content": "Session 4: Planner updated deployment runbook after CORS incident to include nginx header verification step.", "agent_id": "planner", "importance": 0.82},
    # Sessions 5-6: rate limiting + performance
    {"content": "Session 5: Reviewer flagged missing rate limiting in PR #12 — /api/tasks endpoint had no throttle.", "agent_id": "reviewer", "importance": 0.85},
    {"content": "Session 5: Coder added token-bucket rate limiter: 100 req/min per IP on all API endpoints.", "agent_id": "coder", "importance": 0.8},
    {"content": "Session 6: Coder profiled API — /api/tasks was doing N+1 SQL queries, response time 800 ms average.", "agent_id": "coder", "importance": 0.88},
    {"content": "Session 6: Coder fixed N+1 query with SQLAlchemy eager loading; response time dropped to 45 ms.", "agent_id": "coder", "importance": 0.9},
    {"content": "Session 6: Reviewer approved performance PR #15 after load test showed 10x throughput improvement.", "agent_id": "reviewer", "importance": 0.75},
    # Sessions 7-8: database migration
    {"content": "Session 7: Planner planned migration of users table — adding email_verified boolean column.", "agent_id": "planner", "importance": 0.7},
    {"content": "Session 7: Coder ran Alembic migration script on staging — migration completed successfully, zero downtime.", "agent_id": "coder", "importance": 0.78},
    {"content": "Session 8: Production database migration of users table executed during low-traffic window at 02:00 UTC.", "agent_id": "planner", "importance": 0.85},
    {"content": "Session 8: Post-migration: row count check passed, no data loss detected, email_verified column backfilled to False.", "agent_id": "coder", "importance": 0.8},
    {"content": "Session 8: Reviewer signed off on migration PR #18 after reviewing rollback plan.", "agent_id": "reviewer", "importance": 0.72},
    # Sessions 9-10: security audit
    {"content": "Session 9: Security audit found SQL injection risk in raw query in /api/search endpoint.", "agent_id": "reviewer", "importance": 0.95},
    {"content": "Session 9: Coder replaced raw SQL in /api/search with parameterised SQLAlchemy query to fix injection risk.", "agent_id": "coder", "importance": 0.92},
    {"content": "Session 10: Security re-audit confirmed SQL injection fix; no further critical vulnerabilities found.", "agent_id": "reviewer", "importance": 0.88},
    {"content": "Session 10: Planner added security audit to quarterly release checklist after SQL injection finding.", "agent_id": "planner", "importance": 0.8},
    {"content": "Session 10: API rate-limiting also helped mitigate brute-force risk found during security audit.", "agent_id": "coder", "importance": 0.75},
    # Sessions 11-12: production outage
    {"content": "Session 11: Production outage at 14:32 UTC — API gateway returned 502 for all requests for 18 minutes.", "agent_id": "planner", "importance": 0.98},
    {"content": "Session 11: Root cause of outage: Kubernetes pod OOM-killed after memory leak in background task worker.", "agent_id": "coder", "importance": 0.95},
    {"content": "Session 12: Post-mortem completed — added memory limit to worker pod and set up Grafana alert for RSS > 400 MB.", "agent_id": "planner", "importance": 0.9},
    {"content": "Session 12: Coder patched memory leak in task worker — queue processor was not releasing completed job objects.", "agent_id": "coder", "importance": 0.93},
    {"content": "Session 12: Reviewer led post-mortem retro; incident response checklist updated with OOM detection step.", "agent_id": "reviewer", "importance": 0.85},
    # Sessions 13-14: hotfix flow introduced
    {"content": "Session 13: Hotfix v1.1.1 deployed for memory leak — followed emergency hotfix branch process.", "agent_id": "coder", "importance": 0.87},
    {"content": "Session 13: Hotfix deploy completed in 12 minutes from patch to production using hotfix deploy procedure.", "agent_id": "planner", "importance": 0.82},
    {"content": "Session 14: Coder deployed v1.2.0 with rate limiting, performance fixes, and memory leak patch bundled.", "agent_id": "coder", "importance": 0.78},
    {"content": "Session 14: Grafana dashboard now shows API latency p95, error rate, and pod memory — set up post-outage.", "agent_id": "planner", "importance": 0.8},
    {"content": "Session 14: Reviewer approved v1.2.0 release PR after reviewing all three bundled fixes.", "agent_id": "reviewer", "importance": 0.7},
    # Sessions 15-16: auth work
    {"content": "Session 15: Coder implemented JWT authentication for all API endpoints — replaced session cookies.", "agent_id": "coder", "importance": 0.82},
    {"content": "Session 15: Planner decided to use short-lived JWT (15 min) with refresh tokens stored in HttpOnly cookies.", "agent_id": "planner", "importance": 0.85},
    {"content": "Session 16: Reviewer flagged JWT secret must rotate every 90 days — added to ops runbook.", "agent_id": "reviewer", "importance": 0.83},
    {"content": "Session 16: v1.2.3 deployed with JWT auth — all existing sessions invalidated and users prompted to log in.", "agent_id": "coder", "importance": 0.75},
    {"content": "Session 16: Post-deploy monitoring showed 0 authentication errors; 98 % of users re-authenticated within 1 hour.", "agent_id": "planner", "importance": 0.7},
    # Sessions 17-18: CI/CD improvements
    {"content": "Session 17: Planner set up GitHub Actions pipeline — lint, type-check, test, build, deploy stages.", "agent_id": "planner", "importance": 0.75},
    {"content": "Session 17: Coder added pre-deploy integration tests to CI; deploy blocked if any test fails.", "agent_id": "coder", "importance": 0.8},
    {"content": "Session 18: First automated deploy via CI triggered by merge to main — zero manual steps required.", "agent_id": "planner", "importance": 0.72},
    {"content": "Session 18: Reviewer added required PR approval count (2) to branch protection rules after CI was set up.", "agent_id": "reviewer", "importance": 0.7},
    {"content": "Session 18: Deploy time reduced from 25 min manual to 8 min automated after CI/CD pipeline introduction.", "agent_id": "coder", "importance": 0.78},
    # Sessions 19-20: current state
    {"content": "Session 19: Planner ran sprint review — API v1.2.3 stable, all critical issues resolved, next sprint is new features.", "agent_id": "planner", "importance": 0.68},
    {"content": "Session 19: Coder started work on /api/notifications endpoint — will require WebSocket support.", "agent_id": "coder", "importance": 0.6},
    {"content": "Session 20: Team retrospective: agreed to run security audit before every major release going forward.", "agent_id": "reviewer", "importance": 0.75},
    {"content": "Session 20: Coder noted next performance challenge: WebSocket connections may need Redis pub/sub backing.", "agent_id": "coder", "importance": 0.65},
    {"content": "Session 20: Planner scheduled v1.3.0 milestone: WebSocket notifications + Redis integration.", "agent_id": "planner", "importance": 0.7},
]

SEMANTIC_FACTS: list[dict[str, Any]] = [
    # Team language preferences
    {"subject": "coder", "predicate": "prefers_language", "object": "Python", "agent_id": "planner", "confidence": 1.0},
    {"subject": "reviewer", "predicate": "prefers_language", "object": "JavaScript", "agent_id": "planner", "confidence": 1.0},
    {"subject": "planner", "predicate": "prefers_language", "object": "Python", "agent_id": "planner", "confidence": 0.9},
    # Team tool preferences
    {"subject": "coder", "predicate": "prefers_ide", "object": "VSCode", "agent_id": "planner", "confidence": 0.95},
    {"subject": "reviewer", "predicate": "prefers_ide", "object": "IntelliJ", "agent_id": "planner", "confidence": 0.9},
    {"subject": "planner", "predicate": "prefers_ide", "object": "VSCode", "agent_id": "planner", "confidence": 0.85},
    # Team ownership
    {"subject": "coder", "predicate": "owns_service", "object": "api_gateway", "agent_id": "planner", "confidence": 1.0},
    {"subject": "coder", "predicate": "owns_service", "object": "task_worker", "agent_id": "planner", "confidence": 1.0},
    {"subject": "reviewer", "predicate": "owns_service", "object": "code_quality", "agent_id": "planner", "confidence": 1.0},
    {"subject": "planner", "predicate": "owns_service", "object": "infrastructure", "agent_id": "planner", "confidence": 1.0},
    # Service architecture
    {"subject": "api_gateway", "predicate": "language", "object": "Python", "agent_id": "planner", "confidence": 1.0},
    {"subject": "api_gateway", "predicate": "framework", "object": "FastAPI", "agent_id": "planner", "confidence": 1.0},
    {"subject": "api_gateway", "predicate": "database", "object": "PostgreSQL", "agent_id": "planner", "confidence": 1.0},
    {"subject": "api_gateway", "predicate": "deployed_version", "object": "v1.2.3", "agent_id": "planner", "confidence": 1.0},
    {"subject": "api_gateway", "predicate": "deployed_on", "object": "Kubernetes", "agent_id": "planner", "confidence": 1.0},
    {"subject": "frontend", "predicate": "framework", "object": "React", "agent_id": "planner", "confidence": 1.0},
    {"subject": "frontend", "predicate": "language", "object": "TypeScript", "agent_id": "planner", "confidence": 1.0},
    # Infrastructure
    {"subject": "monitoring", "predicate": "tool", "object": "Grafana", "agent_id": "planner", "confidence": 1.0},
    {"subject": "ci_cd", "predicate": "tool", "object": "GitHub Actions", "agent_id": "planner", "confidence": 1.0},
    {"subject": "container_runtime", "predicate": "tool", "object": "Kubernetes", "agent_id": "planner", "confidence": 1.0},
    # Architecture decisions
    {"subject": "authentication", "predicate": "method", "object": "JWT with refresh tokens", "agent_id": "planner", "confidence": 1.0},
    {"subject": "rate_limiting", "predicate": "strategy", "object": "token-bucket 100 req/min per IP", "agent_id": "planner", "confidence": 1.0},
    {"subject": "database_migrations", "predicate": "tool", "object": "Alembic", "agent_id": "planner", "confidence": 1.0},
    # Team roles
    {"subject": "planner", "predicate": "role", "object": "tech_lead", "agent_id": "planner", "confidence": 1.0},
    {"subject": "coder", "predicate": "role", "object": "backend_engineer", "agent_id": "planner", "confidence": 1.0},
    {"subject": "reviewer", "predicate": "role", "object": "senior_engineer", "agent_id": "planner", "confidence": 1.0},
    # Current state
    {"subject": "api_gateway", "predicate": "status", "object": "stable", "agent_id": "planner", "confidence": 1.0},
    {"subject": "next_milestone", "predicate": "version", "object": "v1.3.0", "agent_id": "planner", "confidence": 0.95},
    {"subject": "next_milestone", "predicate": "features", "object": "WebSocket notifications and Redis pub/sub", "agent_id": "planner", "confidence": 0.95},
    {"subject": "coder", "predicate": "expertise", "object": "Python backend PostgreSQL performance", "agent_id": "planner", "confidence": 0.9},
]

PROCEDURAL_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "standard_deploy_flow",
        "description": "Standard procedure to deploy a new version to production",
        "applicable_contexts": ["deploy", "release", "production", "version"],
        "steps": [
            {"action": "run_tests", "tool": "pytest + GitHub Actions CI"},
            {"action": "merge_to_main", "tool": "GitHub PR with 2 approvals"},
            {"action": "build_docker_image", "tool": "docker build -t api:{version}"},
            {"action": "push_image", "tool": "docker push registry.internal/api:{version}"},
            {"action": "update_kubernetes", "tool": "kubectl set image deployment/api api=registry.internal/api:{version}"},
            {"action": "verify_nginx_headers", "tool": "curl -I https://api.domain.com/health — check CORS and security headers"},
            {"action": "smoke_test", "tool": "run post-deploy integration test suite"},
            {"action": "monitor_30min", "tool": "watch Grafana dashboard for error rate and p95 latency"},
        ],
        "success_rate": 0.94,
    },
    {
        "name": "cors_fix_procedure",
        "description": "Steps to diagnose and fix CORS errors in the nginx/API gateway",
        "applicable_contexts": ["CORS", "Access-Control", "nginx", "headers", "browser error"],
        "steps": [
            {"action": "reproduce_in_browser", "tool": "open browser DevTools Network tab, note exact error"},
            {"action": "check_nginx_config", "tool": "cat /etc/nginx/nginx.conf | grep -i cors"},
            {"action": "add_cors_headers", "tool": "add Access-Control-Allow-Origin, Access-Control-Allow-Methods, Access-Control-Allow-Headers to nginx.conf"},
            {"action": "restrict_origin", "tool": "set Allow-Origin to specific domain, not wildcard, for production"},
            {"action": "reload_nginx", "tool": "nginx -t && systemctl reload nginx"},
            {"action": "verify", "tool": "curl -I -H 'Origin: https://frontend.domain.com' https://api.domain.com/api/users"},
        ],
        "success_rate": 0.98,
    },
    {
        "name": "hotfix_deploy",
        "description": "Emergency hotfix process for critical production bugs requiring immediate patch",
        "applicable_contexts": ["hotfix", "emergency", "critical", "production bug", "patch"],
        "steps": [
            {"action": "create_hotfix_branch", "tool": "git checkout -b hotfix/{ticket} main"},
            {"action": "implement_minimal_fix", "tool": "code change — keep diff small, no refactoring"},
            {"action": "fast_review", "tool": "1 approver minimum (tech lead), async review acceptable"},
            {"action": "run_targeted_tests", "tool": "pytest tests/ -k {affected_area}"},
            {"action": "deploy_to_staging", "tool": "follow standard_deploy_flow steps 3-6"},
            {"action": "verify_staging", "tool": "confirm fix resolves the incident in staging"},
            {"action": "deploy_to_production", "tool": "follow standard_deploy_flow steps 3-8"},
            {"action": "merge_back", "tool": "git merge hotfix/{ticket} into main and develop"},
        ],
        "success_rate": 0.91,
    },
    {
        "name": "incident_response",
        "description": "Production incident response from detection to resolution and post-mortem",
        "applicable_contexts": ["incident", "outage", "502", "down", "production error", "alert"],
        "steps": [
            {"action": "acknowledge_alert", "tool": "respond to Grafana alert within 5 minutes"},
            {"action": "check_error_rate", "tool": "Grafana dashboard: API error rate and pod status"},
            {"action": "check_pod_logs", "tool": "kubectl logs deployment/api --tail=200"},
            {"action": "check_pod_memory", "tool": "kubectl top pods — watch for OOM or RSS > 400 MB"},
            {"action": "identify_root_cause", "tool": "correlate logs, metrics, recent deploys"},
            {"action": "decide_rollback_or_fix", "tool": "if deploy-related: kubectl rollout undo; else: hotfix_deploy"},
            {"action": "verify_recovery", "tool": "confirm error rate returns to baseline on Grafana"},
            {"action": "write_post_mortem", "tool": "document timeline, root cause, fix, and prevention steps"},
        ],
        "success_rate": 0.89,
    },
    {
        "name": "database_migration",
        "description": "Safe zero-downtime database migration using Alembic",
        "applicable_contexts": ["migration", "schema", "database", "PostgreSQL", "column", "Alembic"],
        "steps": [
            {"action": "write_migration_script", "tool": "alembic revision --autogenerate -m 'description'"},
            {"action": "review_migration", "tool": "inspect generated script — verify up() and down() are correct"},
            {"action": "backup_staging_db", "tool": "pg_dump staging_db > backup_$(date +%Y%m%d).sql"},
            {"action": "run_on_staging", "tool": "alembic upgrade head (staging environment)"},
            {"action": "verify_staging", "tool": "run integration tests, check row counts, spot-check data"},
            {"action": "schedule_production_window", "tool": "pick low-traffic window, notify team in Slack"},
            {"action": "backup_production_db", "tool": "pg_dump production_db > prod_backup_$(date +%Y%m%d).sql"},
            {"action": "run_on_production", "tool": "alembic upgrade head (production environment)"},
            {"action": "verify_production", "tool": "row count check, application smoke test, monitor 30 min"},
        ],
        "success_rate": 0.96,
    },
    {
        "name": "code_review_process",
        "description": "Standard code review procedure for all PRs",
        "applicable_contexts": ["code review", "PR", "pull request", "review", "approve"],
        "steps": [
            {"action": "check_pr_description", "tool": "verify PR describes what changed and why"},
            {"action": "run_ci_checks", "tool": "confirm GitHub Actions lint, type-check, test all pass"},
            {"action": "review_diff", "tool": "check for security issues, N+1 queries, rate limiting gaps, missing error handling"},
            {"action": "check_test_coverage", "tool": "new code must have tests; coverage must not drop"},
            {"action": "leave_comments", "tool": "GitHub PR comments — distinguish blocking vs non-blocking"},
            {"action": "approve_or_request_changes", "tool": "2 approvals required to merge per branch protection rules"},
        ],
        "success_rate": 1.0,
    },
    {
        "name": "performance_debugging",
        "description": "Systematic approach to diagnosing and fixing API performance bottlenecks",
        "applicable_contexts": ["performance", "slow", "latency", "bottleneck", "N+1", "profiling"],
        "steps": [
            {"action": "measure_baseline", "tool": "locust or k6 load test — capture p50/p95/p99 latency"},
            {"action": "profile_endpoint", "tool": "py-spy top -- python app.py or cProfile on slow route"},
            {"action": "check_query_plans", "tool": "EXPLAIN ANALYZE on slow PostgreSQL queries"},
            {"action": "look_for_n_plus_one", "tool": "SQLAlchemy debug logging — count queries per request"},
            {"action": "add_eager_loading", "tool": "SQLAlchemy joinedload() or selectinload() where applicable"},
            {"action": "add_caching", "tool": "Redis cache for hot read paths with appropriate TTL"},
            {"action": "retest", "tool": "run load test again, compare p95 before and after"},
        ],
        "success_rate": 0.92,
    },
    {
        "name": "security_audit",
        "description": "Security review procedure run before each major release",
        "applicable_contexts": ["security", "audit", "vulnerability", "injection", "auth", "release"],
        "steps": [
            {"action": "static_analysis", "tool": "bandit -r . for Python security issues"},
            {"action": "dependency_scan", "tool": "pip-audit or safety check for known CVEs"},
            {"action": "sql_injection_check", "tool": "grep for raw SQL strings, verify all queries use parameterisation"},
            {"action": "auth_review", "tool": "verify JWT validation, token expiry, refresh token rotation"},
            {"action": "rate_limiting_check", "tool": "confirm all public endpoints have rate limits applied"},
            {"action": "secrets_scan", "tool": "trufflehog or git-secrets to check for hardcoded credentials"},
            {"action": "fix_findings", "tool": "address all critical and high findings before release"},
        ],
        "success_rate": 0.95,
    },
    {
        "name": "release_tagging",
        "description": "Procedure to tag and publish a versioned release",
        "applicable_contexts": ["release", "tag", "version", "changelog", "publish"],
        "steps": [
            {"action": "update_changelog", "tool": "edit CHANGELOG.md — list all changes since last tag"},
            {"action": "bump_version", "tool": "update version in pyproject.toml / package.json"},
            {"action": "commit_version_bump", "tool": "git commit -m 'chore: bump version to vX.Y.Z'"},
            {"action": "create_tag", "tool": "git tag -a vX.Y.Z -m 'Release vX.Y.Z'"},
            {"action": "push_tag", "tool": "git push origin vX.Y.Z"},
            {"action": "create_github_release", "tool": "gh release create vX.Y.Z --notes-from-tag"},
        ],
        "success_rate": 1.0,
    },
    {
        "name": "pr_review_checklist",
        "description": "Quick checklist reviewers run on every pull request before approving",
        "applicable_contexts": ["PR", "checklist", "review", "merge", "approval"],
        "steps": [
            {"action": "check_tests_pass", "tool": "GitHub Actions CI green"},
            {"action": "check_no_secrets", "tool": "scan diff for API keys, passwords, tokens"},
            {"action": "check_migration_safety", "tool": "if schema change: verify rollback is possible"},
            {"action": "check_rate_limits", "tool": "new endpoints must have rate limiting"},
            {"action": "check_security", "tool": "no raw SQL, no eval(), no os.system() with user input"},
            {"action": "approve", "tool": "GitHub 'Approve' — 2 approvals required before merge"},
        ],
        "success_rate": 1.0,
    },
]

QUESTIONS: list[dict[str, Any]] = [
    # ── Category A: episodic + procedural (10) ─────────────────────────────────
    {
        "question": "What is our standard deploy process and what errors have we encountered during past deploys?",
        "category": "episodic+procedural",
        "expected_keywords": ["deploy", "nginx", "CORS", "procedure", "verify", "smoke_test"],
    },
    {
        "question": "How do we fix CORS errors and has this specific issue come up before in our project?",
        "category": "episodic+procedural",
        "expected_keywords": ["CORS", "nginx.conf", "Access-Control", "Session 2", "headers"],
    },
    {
        "question": "What is our hotfix deploy procedure and when have we actually used it?",
        "category": "episodic+procedural",
        "expected_keywords": ["hotfix", "emergency", "v1.1.1", "memory leak", "branch"],
    },
    {
        "question": "How do we respond to production incidents and what incidents have we experienced?",
        "category": "episodic+procedural",
        "expected_keywords": ["incident", "outage", "502", "OOM", "post-mortem", "Grafana"],
    },
    {
        "question": "What does our code review process look like and what issues have reviewers flagged in past PRs?",
        "category": "episodic+procedural",
        "expected_keywords": ["review", "PR", "rate limiting", "SQL injection", "checklist", "2 approvals"],
    },
    {
        "question": "How do we perform a database migration safely and have we run one before?",
        "category": "episodic+procedural",
        "expected_keywords": ["migration", "Alembic", "PostgreSQL", "backup", "Session 7", "staging"],
    },
    {
        "question": "What is our process for debugging API performance problems and what performance issues have we had?",
        "category": "episodic+procedural",
        "expected_keywords": ["performance", "N+1", "profiling", "SQLAlchemy", "800 ms", "45 ms"],
    },
    {
        "question": "What is the release tagging process and what is the current deployed version?",
        "category": "episodic+procedural",
        "expected_keywords": ["tag", "release", "CHANGELOG", "v1.2.3", "bump version"],
    },
    {
        "question": "What security checks do we run and what vulnerabilities have been found?",
        "category": "episodic+procedural",
        "expected_keywords": ["security", "SQL injection", "bandit", "audit", "parameterised", "rate"],
    },
    {
        "question": "What does a PR review checklist look like and what has been caught in reviews historically?",
        "category": "episodic+procedural",
        "expected_keywords": ["checklist", "review", "rate limiting", "secrets", "migration", "approval"],
    },
    # ── Category B: semantic + episodic (10) ───────────────────────────────────
    {
        "question": "Who on the team prefers Python and have they worked on CORS-related issues before?",
        "category": "semantic+episodic",
        "expected_keywords": ["Python", "coder", "planner", "CORS", "Session 2", "nginx"],
    },
    {
        "question": "What database does the API gateway use and have we had any database problems?",
        "category": "semantic+episodic",
        "expected_keywords": ["PostgreSQL", "database", "migration", "connection", "Alembic"],
    },
    {
        "question": "Who owns the API gateway service and what incidents have they handled?",
        "category": "semantic+episodic",
        "expected_keywords": ["coder", "owns", "api_gateway", "outage", "memory leak", "CORS"],
    },
    {
        "question": "What frontend framework are we using and what frontend-related bugs have we seen?",
        "category": "semantic+episodic",
        "expected_keywords": ["React", "TypeScript", "CORS", "frontend", "browser", "Access-Control"],
    },
    {
        "question": "What is the currently deployed API version and what was in that release?",
        "category": "semantic+episodic",
        "expected_keywords": ["v1.2.3", "deployed_version", "JWT", "rate limiting", "Session 16"],
    },
    {
        "question": "What language does the reviewer prefer and what code issues have they caught?",
        "category": "semantic+episodic",
        "expected_keywords": ["JavaScript", "reviewer", "rate limiting", "SQL injection", "JWT", "PR"],
    },
    {
        "question": "What monitoring tooling do we use and what alerts or incidents has it surfaced?",
        "category": "semantic+episodic",
        "expected_keywords": ["Grafana", "monitoring", "alert", "RSS", "outage", "p95"],
    },
    {
        "question": "What CI/CD system do we use and how has automation changed our deploy process?",
        "category": "semantic+episodic",
        "expected_keywords": ["GitHub Actions", "CI", "automated", "8 min", "25 min", "pipeline"],
    },
    {
        "question": "What are the team members' language preferences and areas of expertise?",
        "category": "semantic+episodic",
        "expected_keywords": ["Python", "JavaScript", "coder", "reviewer", "planner", "expertise"],
    },
    {
        "question": "What authentication method does the project use and what authentication work has been done?",
        "category": "semantic+episodic",
        "expected_keywords": ["JWT", "refresh tokens", "HttpOnly", "Session 15", "v1.2.3", "cookies"],
    },
    # ── Category C: all three memory types (10) ────────────────────────────────
    {
        "question": "What is our complete deploy process and how has it evolved since the production outage?",
        "category": "all",
        "expected_keywords": ["deploy", "outage", "procedure", "OOM", "Grafana", "nginx", "post-mortem"],
    },
    {
        "question": "Who should be assigned to fix a Python backend CORS error and what procedure should they follow?",
        "category": "all",
        "expected_keywords": ["Python", "coder", "CORS", "nginx.conf", "procedure", "Access-Control"],
    },
    {
        "question": "We need a PostgreSQL schema migration — who handles this, what is the procedure, and have we done one before?",
        "category": "all",
        "expected_keywords": ["PostgreSQL", "coder", "Alembic", "migration", "backup", "Session 7"],
    },
    {
        "question": "Given our monitoring setup and past incidents, what alerts should we prioritise for the next sprint?",
        "category": "all",
        "expected_keywords": ["Grafana", "OOM", "RSS", "outage", "alert", "monitoring", "procedure"],
    },
    {
        "question": "What is the full picture of the API gateway service: team ownership, current state, and key processes?",
        "category": "all",
        "expected_keywords": ["coder", "api_gateway", "v1.2.3", "FastAPI", "deploy", "incident"],
    },
    {
        "question": "Who leads incident response and what is the full escalation and resolution procedure?",
        "category": "all",
        "expected_keywords": ["incident", "planner", "post-mortem", "rollback", "Grafana", "procedure"],
    },
    {
        "question": "What performance issues have we experienced and what is our complete performance debugging procedure?",
        "category": "all",
        "expected_keywords": ["N+1", "800 ms", "SQLAlchemy", "profiling", "performance", "procedure"],
    },
    {
        "question": "What has the security audit found so far, who runs it, and what is the full security procedure?",
        "category": "all",
        "expected_keywords": ["SQL injection", "reviewer", "security", "bandit", "parameterised", "audit"],
    },
    {
        "question": "How do we tag a release, what version are we on, and what was included in the last release?",
        "category": "all",
        "expected_keywords": ["tag", "v1.2.3", "JWT", "CHANGELOG", "release", "v1.3.0"],
    },
    {
        "question": "What is the team composition, their individual preferences and expertise, and which standard workflows do we have?",
        "category": "all",
        "expected_keywords": ["Python", "JavaScript", "coder", "reviewer", "planner", "procedure", "deploy"],
    },
]


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question: str
    category: str
    score: int
    tokens_used: int
    context_snippet: str


@dataclass
class TriMemoryResult:
    scores: list[QuestionResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return sum(r.score for r in self.scores) / len(self.scores) if self.scores else 0.0

    def avg_by_category(self) -> dict[str, float]:
        cats: dict[str, list[int]] = {}
        for r in self.scores:
            cats.setdefault(r.category, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in cats.items()}


@dataclass
class ApproachMetrics:
    name: str
    tokens_used: int
    quality_score: int

    @property
    def information_density(self) -> float:
        return self.quality_score / max(1, self.tokens_used)


@dataclass
class TokenEfficiencyResult:
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def avg_density(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for row in self.per_question:
            for approach in ("progressive", "dump_all", "baseline"):
                d = row[approach]["information_density"]
                totals.setdefault(approach, []).append(d)
        return {k: sum(v) / len(v) for k, v in totals.items()}

    def avg_tokens(self) -> dict[str, float]:
        totals: dict[str, list[int]] = {}
        for row in self.per_question:
            for approach in ("progressive", "dump_all", "baseline"):
                t = row[approach]["tokens_used"]
                totals.setdefault(approach, []).append(t)
        return {k: sum(v) / len(v) for k, v in totals.items()}


@dataclass
class ThroughputResult:
    """Raw latency and throughput numbers for Suite 4."""
    n_episodic: int = 0
    n_semantic: int = 0
    n_procedural: int = 0
    n_search: int = 0
    n_scan: int = 0

    # Full-pipeline timings (includes Python embedding step)
    write_episodic_total_ms: float = 0.0   # db.remember() × n_episodic
    search_total_ms: float = 0.0           # db.recall()   × n_search

    # Raw storage timings (pre-computed embeddings, pure Rust I/O)
    write_episodic_raw_total_ms: float = 0.0  # _episodic.add() × n_episodic
    write_semantic_total_ms: float = 0.0      # db.learn()      × n_semantic
    write_procedural_total_ms: float = 0.0    # db.learn_procedure() × n_procedural
    search_raw_total_ms: float = 0.0          # _episodic.search() with pre-computed emb
    scan_batch_total_ms: float = 0.0          # scan_batch(limit=100) × n_scan

    @property
    def write_episodic_ms(self) -> float:
        return self.write_episodic_total_ms / max(1, self.n_episodic)

    @property
    def write_episodic_raw_ms(self) -> float:
        return self.write_episodic_raw_total_ms / max(1, self.n_episodic)

    @property
    def write_semantic_ms(self) -> float:
        return self.write_semantic_total_ms / max(1, self.n_semantic)

    @property
    def write_procedural_ms(self) -> float:
        return self.write_procedural_total_ms / max(1, self.n_procedural)

    @property
    def search_ms(self) -> float:
        return self.search_total_ms / max(1, self.n_search)

    @property
    def search_raw_ms(self) -> float:
        return self.search_raw_total_ms / max(1, self.n_search)

    @property
    def scan_ops_per_sec(self) -> float:
        total_s = self.scan_batch_total_ms / 1000.0
        return round(self.n_scan * 100 / max(0.001, total_s), 1)  # 100 rows/scan


@dataclass
class ConsistencyResult:
    total_reads: int = 0
    correct_reads: int = 0
    supersede_reads: int = 0
    correct_supersede: int = 0
    conflict_opportunities: int = 0
    conflicts_detected: int = 0
    data_loss_count: int = 0
    ops_completed: int = 0
    duration_seconds: float = 0.0

    @property
    def consistency_score(self) -> float:
        return self.correct_reads / max(1, self.total_reads)

    @property
    def supersede_accuracy(self) -> float:
        return self.correct_supersede / max(1, self.supersede_reads)

    @property
    def conflict_resolution_accuracy(self) -> float:
        return self.conflicts_detected / max(1, self.conflict_opportunities)


# ── Database population ────────────────────────────────────────────────────────

def populate_db(db: CognitiveDB) -> None:
    """Write all synthetic fixture data into db."""
    for ep in EPISODIC_MEMORIES:
        db.remember(
            content=ep["content"],
            agent_id=ep["agent_id"],
            importance=ep["importance"],
            scope=MemoryScope.ORGANIZATION,
        )

    for sf in SEMANTIC_FACTS:
        db.learn(
            subject=sf["subject"],
            predicate=sf["predicate"],
            object=sf["object"],
            agent_id=sf["agent_id"],
            confidence=sf["confidence"],
        )

    for pt in PROCEDURAL_TEMPLATES:
        db.learn_procedure(
            name=pt["name"],
            description=pt["description"],
            steps=pt["steps"],
            agent_id="planner",
            success_rate=pt["success_rate"],
            applicable_contexts=pt["applicable_contexts"],
        )


# ── LLM judge ─────────────────────────────────────────────────────────────────

_openai_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None and _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        _openai_client = _OpenAI()
    return _openai_client


def judge_answer(
    question: str,
    context: str,
    expected_keywords: list[str],
    use_llm: bool = True,
) -> int:
    """Score 0-100: does context contain enough to answer question?"""
    client = _get_openai_client() if use_llm else None

    if client:
        try:
            prompt = (
                f"Question: {question}\n\n"
                f"Context retrieved from memory:\n{context}\n\n"
                "Does the context contain enough information to fully answer the question?\n"
                "Score 0-100 where 0=nothing useful, 50=partial, 100=fully answered.\n"
                'Reply ONLY with JSON: {"score": <int>}'
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0,
            )
            raw = resp.choices[0].message.content or '{"score": 0}'
            return int(json.loads(raw)["score"])
        except Exception:
            pass  # fall through to keyword scoring

    # Keyword fallback
    if not context:
        return 0
    context_lower = context.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
    return round(100 * hits / max(1, len(expected_keywords)))


# ── Context builders ───────────────────────────────────────────────────────────

def _memories_to_text(memories: list[Any]) -> str:
    return "\n".join(f"- {m.content}" for m in memories)


def _context_response_to_text(ctx: Any) -> str:
    parts: list[str] = [ctx.identity]
    if ctx.critical_facts:
        parts.append("Facts:\n" + "\n".join(f"  {f}" for f in ctx.critical_facts))
    if ctx.relevant_memories:
        parts.append("Relevant:\n" + _memories_to_text(ctx.relevant_memories))
    if ctx.deep_results:
        parts.append("Deep:\n" + _memories_to_text(ctx.deep_results))
    return "\n".join(parts)


# ── Suite 1: Tri-Memory Retrieval ─────────────────────────────────────────────

def run_tri_memory_suite(db: CognitiveDB, use_llm: bool = True) -> TriMemoryResult:
    """Score answer quality across 30 cross-memory questions."""
    print("\n[Suite 1] Tri-Memory Retrieval — 30 questions")
    result = TriMemoryResult()

    for i, q in enumerate(QUESTIONS, 1):
        ctx = db.get_context(
            agent_id="planner",
            level=3,
            task_hint=q["question"],
            token_budget=800,
        )
        context_text = _context_response_to_text(ctx)
        tokens = ctx.token_count

        score = judge_answer(
            question=q["question"],
            context=context_text,
            expected_keywords=q["expected_keywords"],
            use_llm=use_llm,
        )

        result.scores.append(QuestionResult(
            question=q["question"][:80],
            category=q["category"],
            score=score,
            tokens_used=tokens,
            context_snippet=context_text[:120],
        ))
        print(f"  Q{i:02d} [{q['category']:20s}] score={score:3d}  tokens={tokens}")

    return result


# ── Suite 2: Token Efficiency ──────────────────────────────────────────────────

def _baseline_chromadb(db: CognitiveDB, query: str, top_k: int = 10) -> tuple[str, int]:
    """Raw episodic top-k search, no token budget management."""
    memories = db._episodic.search(
        query=query,
        agent_id="planner",
        top_k=top_k,
    )
    text = _memories_to_text(memories)
    return text, count_tokens(text)


def run_token_efficiency_suite(
    db: CognitiveDB, use_llm: bool = True, sample_size: int = 10
) -> TokenEfficiencyResult:
    """Compare progressive loading vs dump-all vs raw ChromaDB."""
    print(f"\n[Suite 2] Token Efficiency — {sample_size} questions × 3 approaches")
    result = TokenEfficiencyResult()
    questions = QUESTIONS[:sample_size]

    for i, q in enumerate(questions, 1):
        query = q["question"]
        kw = q["expected_keywords"]
        row: dict[str, Any] = {"question": query[:80], "category": q["category"]}

        # Approach 1: CogDB progressive loading (L0-L3, budget=500)
        ctx = db.get_context(agent_id="planner", level=3, task_hint=query, token_budget=500)
        prog_text = _context_response_to_text(ctx)
        prog_tokens = ctx.token_count
        prog_score = judge_answer(query, prog_text, kw, use_llm)

        # Approach 2: CogDB dump-all (recall with near-infinite budget)
        all_memories = db.recall(
            query=query, agent_id="planner", token_budget=8000,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
        )
        dump_text = _memories_to_text(all_memories)
        dump_tokens = count_tokens(dump_text)
        dump_score = judge_answer(query, dump_text, kw, use_llm)

        # Approach 3: Baseline ChromaDB top-10 similarity
        base_text, base_tokens = _baseline_chromadb(db, query, top_k=10)
        base_score = judge_answer(query, base_text, kw, use_llm)

        def _density(score: int, tokens: int) -> float:
            return round(score / max(1, tokens), 4)

        row["progressive"] = {
            "tokens_used": prog_tokens,
            "quality_score": prog_score,
            "information_density": _density(prog_score, prog_tokens),
        }
        row["dump_all"] = {
            "tokens_used": dump_tokens,
            "quality_score": dump_score,
            "information_density": _density(dump_score, dump_tokens),
        }
        row["baseline"] = {
            "tokens_used": base_tokens,
            "quality_score": base_score,
            "information_density": _density(base_score, base_tokens),
        }

        result.per_question.append(row)
        print(
            f"  Q{i:02d}  progressive={prog_tokens}t/{prog_score}q  "
            f"dump={dump_tokens}t/{dump_score}q  base={base_tokens}t/{base_score}q"
        )

    return result


# ── Suite 4: Throughput ───────────────────────────────────────────────────────

def _median_ms(samples: list[float]) -> float:
    s = sorted(samples)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) * 1000


def run_throughput_suite(n_write: int = 100, n_search: int = 50, n_scan: int = 20) -> ThroughputResult:
    """Measure raw write, search, and scan latency against the Rust storage engine.

    Two timing modes per operation:
      full-pipeline — goes through db.remember() / db.recall(), includes the
                      Python sentence-transformers embedding step.
      raw-storage   — bypasses encoding: pre-computed embeddings go directly
                      to the store, isolating pure Rust I/O cost.

    Args:
        n_write:  Number of episodic write operations to time.
        n_search: Number of search queries to time.
        n_scan:   Number of scan_batch() calls (100 rows each) to time.
    """
    import statistics
    from cogdb.models import MemoryUnit, MemoryType, MemoryScope, SemanticTriple, ProcedureStep, ProcedureTemplate

    print(f"\n[Suite 4] Throughput — {n_write} writes · {n_search} searches · {n_scan} scans")
    result = ThroughputResult(
        n_episodic=n_write,
        n_semantic=n_write,
        n_procedural=max(1, n_write // 5),
        n_search=n_search,
        n_scan=n_scan,
    )
    tmpdir = tempfile.mkdtemp(prefix="cogdb_bench_throughput_")

    try:
        db = CognitiveDB(db_path=tmpdir)

        # ── Warm-up: load the embedding model once ────────────────────────────
        print("  [warm-up] loading embedding model…", end=" ", flush=True)
        t_wu = time.perf_counter()
        db.remember(content="warm-up", agent_id="bench", importance=0.5)
        sample_emb = db._episodic._encoder.embed_query("warm-up query")
        print(f"done ({(time.perf_counter() - t_wu) * 1000:.0f} ms)")

        # ── 1. Full-pipeline episodic write (db.remember, includes encoding) ──
        print(f"  [1/5] full-pipeline write × {n_write}…", end=" ", flush=True)
        samples_full_write: list[float] = []
        for i in range(n_write):
            t0 = time.perf_counter()
            db.remember(
                content=f"Throughput bench event {i}: operation completed at step {i * 7}",
                agent_id="bench",
                importance=0.5 + (i % 5) * 0.1,
                scope=MemoryScope.PRIVATE,
            )
            samples_full_write.append(time.perf_counter() - t0)
        result.write_episodic_total_ms = sum(samples_full_write) * 1000
        med = _median_ms(samples_full_write)
        print(f"median {med:.1f} ms/op  ({1000 / med:.0f} ops/s)")

        # ── 2. Raw-storage episodic write (pre-computed embedding) ────────────
        print(f"  [2/5] raw-storage write  × {n_write}…", end=" ", flush=True)
        samples_raw_write: list[float] = []
        for i in range(n_write):
            unit = MemoryUnit(
                content=f"Raw write bench {i}",
                memory_type=MemoryType.EPISODIC,
                agent_id="bench_raw",
                importance=0.5,
                embedding=sample_emb,
            )
            t0 = time.perf_counter()
            db._episodic.add(unit)
            samples_raw_write.append(time.perf_counter() - t0)
        result.write_episodic_raw_total_ms = sum(samples_raw_write) * 1000
        med = _median_ms(samples_raw_write)
        print(f"median {med:.1f} ms/op  ({1000 / med:.0f} ops/s)")

        # ── 3. Semantic write ─────────────────────────────────────────────────
        print(f"  [3/5] semantic write     × {result.n_semantic}…", end=" ", flush=True)
        samples_sem: list[float] = []
        for i in range(result.n_semantic):
            t0 = time.perf_counter()
            db.learn(
                subject=f"entity_{i}",
                predicate="bench_rel",
                object=f"value_{i}",
                agent_id="bench",
                confidence=0.9,
            )
            samples_sem.append(time.perf_counter() - t0)
        result.write_semantic_total_ms = sum(samples_sem) * 1000
        med = _median_ms(samples_sem)
        print(f"median {med:.1f} ms/op  ({1000 / med:.0f} ops/s)")

        # ── 4. Full-pipeline search (includes encoding) ───────────────────────
        print(f"  [4/5] full-pipeline search × {n_search}…", end=" ", flush=True)
        samples_search: list[float] = []
        queries = [f"operation at step {i * 7}" for i in range(n_search)]
        for q in queries:
            t0 = time.perf_counter()
            db.recall(query=q, agent_id="bench", token_budget=500,
                      memory_types=[MemoryType.EPISODIC])
            samples_search.append(time.perf_counter() - t0)
        result.search_total_ms = sum(samples_search) * 1000
        med = _median_ms(samples_search)
        print(f"median {med:.1f} ms/op  ({1000 / med:.0f} ops/s)")

        # ── 5. Raw-storage search (pre-computed embedding) ────────────────────
        print(f"  [5/5] raw-storage search × {n_search}…", end=" ", flush=True)
        samples_raw_search: list[float] = []
        for q in queries:
            emb = db._episodic._encoder.embed_query(q)
            t0 = time.perf_counter()
            db._episodic.search(query=q, agent_id="bench", top_k=10,
                                query_embedding=emb)
            samples_raw_search.append(time.perf_counter() - t0)
        result.search_raw_total_ms = sum(samples_raw_search) * 1000
        med = _median_ms(samples_raw_search)
        print(f"median {med:.1f} ms/op  ({1000 / med:.0f} ops/s)")

        # ── 6. Scan batch ─────────────────────────────────────────────────────
        print(f"  [scan ] scan_batch(100)  × {n_scan}…", end=" ", flush=True)
        samples_scan: list[float] = []
        for _ in range(n_scan):
            t0 = time.perf_counter()
            db._episodic.scan_batch(agent_id=None, limit=100, offset=0)
            samples_scan.append(time.perf_counter() - t0)
        result.scan_batch_total_ms = sum(samples_scan) * 1000
        med = _median_ms(samples_scan)
        print(f"median {med:.1f} ms  ({result.scan_ops_per_sec:.0f} rows/s)")

    finally:
        try:
            db._episodic._client.reset()
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


# ── Suite 3: Multi-Agent Consistency ──────────────────────────────────────────

_GROUND_TRUTH: list[tuple[str, str, str]] = [
    ("service_alpha", "owner", "coder"),
    ("service_beta", "language", "Python"),
    ("service_gamma", "status", "deployed"),
    ("service_delta", "version", "v2.0.0"),
    ("service_epsilon", "database", "PostgreSQL"),
    ("team_config", "deploy_tool", "kubectl"),
    ("team_config", "ci_tool", "GitHub Actions"),
    ("team_config", "monitor_tool", "Grafana"),
    ("project_x", "phase", "alpha"),
    ("project_x", "lead", "planner"),
    ("infra_k8s", "cluster", "prod-eu-west"),
    ("infra_k8s", "namespace", "api-services"),
    ("auth_config", "method", "JWT"),
    ("auth_config", "expiry", "15min"),
    ("db_config", "max_connections", "100"),
    ("db_config", "backup_schedule", "daily"),
    ("cache_config", "backend", "Redis"),
    ("cache_config", "ttl_seconds", "300"),
    ("rate_limit", "strategy", "token-bucket"),
    ("rate_limit", "limit", "100rpm"),
]

_SUPERSEDE_UPDATES: list[tuple[str, str, str]] = [
    ("service_delta", "version", "v2.1.0"),
    ("project_x", "phase", "beta"),
    ("auth_config", "expiry", "30min"),
    ("db_config", "max_connections", "200"),
    ("cache_config", "ttl_seconds", "600"),
]

_CONFLICT_SUBJECTS: list[tuple[str, str]] = [
    ("shared_resource_1", "owner"),
    ("shared_resource_2", "status"),
    ("shared_resource_3", "version"),
]


def run_consistency_suite() -> ConsistencyResult:
    """3 concurrent threads, 50 ops each. Measure read correctness and conflict detection."""
    print("\n[Suite 3] Multi-Agent Consistency — 3 threads × 50 ops")
    result = ConsistencyResult()
    tmpdir = tempfile.mkdtemp(prefix="cogdb_bench_consistency_")

    try:
        db = CognitiveDB(db_path=tmpdir)
        result_lock = threading.Lock()
        barrier = threading.Barrier(3)
        supersede_done = threading.Event()  # agent-A sets this after supersede writes complete
        start_time = [0.0]

        # Pre-populate ground truth facts from agent-a (org scope via semantic store)
        for subject, predicate, obj in _GROUND_TRUTH:
            db.learn(subject=subject, predicate=predicate, object=obj, agent_id="agent-a", confidence=1.0)

        # ── Thread functions ───────────────────────────────────────────────────

        def agent_a_work() -> None:
            """Writes new episodic memories, applies supersede updates, does recalls."""
            barrier.wait()
            ops = 0
            for i in range(20):
                db.remember(
                    content=f"Agent-A episodic event {i}: task completed successfully",
                    agent_id="agent-a",
                    importance=0.6,
                    scope=MemoryScope.ORGANIZATION,
                )
                ops += 1

            # Apply supersede updates, then signal agent-c it's safe to read
            for subject, predicate, new_val in _SUPERSEDE_UPDATES:
                db.learn(subject=subject, predicate=predicate, object=new_val, agent_id="agent-a", confidence=1.0)
                ops += 1
            supersede_done.set()

            # Recall 24 times
            for i in range(24):
                db.recall(query=f"event {i}", agent_id="agent-a", token_budget=500)
                ops += 1

            with result_lock:
                result.ops_completed += ops

        def agent_b_work() -> None:
            """Reads ground truth facts written by agent-a. Measures correctness.
            Also writes conflicting facts on shared subjects simultaneously with agent-c."""
            barrier.wait()
            ops = 0
            correct = 0
            total = 0

            # Read ground truth facts (written by agent-a pre-thread)
            for subject, predicate, expected_val in _GROUND_TRUTH:
                triples = db.query_knowledge(entity=subject, active_only=True)
                matching = [t for t in triples if t.predicate == predicate]
                total += 1
                if matching and matching[0].object == expected_val:
                    correct += 1
                ops += 1

            # Write conflicting facts on shared subjects (concurrent with agent-c)
            for subject, predicate in _CONFLICT_SUBJECTS:
                db.learn(subject=subject, predicate=predicate, object="value-from-B", agent_id="agent-b", confidence=0.8)
                ops += 1

            # Fill remaining ops with recalls
            for i in range(50 - ops):
                db.recall(query=f"service {i}", agent_id="agent-b", token_budget=300)
                ops += 1

            with result_lock:
                result.total_reads += total
                result.correct_reads += correct
                result.ops_completed += ops

        def agent_c_work() -> None:
            """Reads superseded values (expects new values), writes conflicts, measures both."""
            barrier.wait()
            ops = 0
            supersede_total = 0
            supersede_correct = 0

            # Wait until agent-a has finished all supersede writes before reading
            supersede_done.wait()

            # Read superseded facts — should see new values
            for subject, predicate, new_val in _SUPERSEDE_UPDATES:
                triples = db.query_knowledge(entity=subject, active_only=True)
                matching = [t for t in triples if t.predicate == predicate]
                supersede_total += 1
                if matching and matching[0].object == new_val:
                    supersede_correct += 1
                ops += 1

            # Write conflicting facts on shared subjects (concurrent with agent-b)
            for subject, predicate in _CONFLICT_SUBJECTS:
                db.learn(subject=subject, predicate=predicate, object="value-from-C", agent_id="agent-c", confidence=0.8)
                ops += 1

            # Check how many shared subjects have exactly one active triple (conflict resolution)
            conflicts_resolved = 0
            for subject, predicate in _CONFLICT_SUBJECTS:
                triples = db.query_knowledge(entity=subject, active_only=True)
                active = [t for t in triples if t.predicate == predicate]
                if len(active) <= 1:
                    conflicts_resolved += 1
            ops += len(_CONFLICT_SUBJECTS)

            # Fill remaining ops
            for i in range(50 - ops):
                db.recall(query=f"config {i}", agent_id="agent-c", token_budget=300)
                ops += 1

            with result_lock:
                result.supersede_reads += supersede_total
                result.correct_supersede += supersede_correct
                result.conflict_opportunities += len(_CONFLICT_SUBJECTS)
                result.conflicts_detected += conflicts_resolved
                result.ops_completed += ops

        # ── Run threads ────────────────────────────────────────────────────────
        threads = [
            threading.Thread(target=agent_a_work, name="agent-a"),
            threading.Thread(target=agent_b_work, name="agent-b"),
            threading.Thread(target=agent_c_work, name="agent-c"),
        ]

        t0 = time.perf_counter()
        start_time[0] = t0
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result.duration_seconds = round(time.perf_counter() - t0, 3)

        print(f"  ops_completed={result.ops_completed}")
        print(f"  consistency_score={result.consistency_score:.2%}  "
              f"({result.correct_reads}/{result.total_reads} reads correct)")
        print(f"  supersede_accuracy={result.supersede_accuracy:.2%}  "
              f"({result.correct_supersede}/{result.supersede_reads})")
        print(f"  conflict_resolution={result.conflict_resolution_accuracy:.2%}  "
              f"({result.conflicts_detected}/{result.conflict_opportunities})")
        print(f"  duration={result.duration_seconds}s")

    finally:
        try:
            db._episodic._client.reset()
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


# ── Report formatting ──────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    tri: Optional[TriMemoryResult],
    tok: Optional[TokenEfficiencyResult],
    con: Optional[ConsistencyResult],
    thr: Optional[ThroughputResult] = None,
) -> None:
    SEP = "─" * 70
    print(f"\n{'═' * 70}")
    print("  CogDB BENCHMARK REPORT")
    print(f"{'═' * 70}")

    if tri:
        print(f"\n{'─' * 70}")
        print("  SUITE 1 — Tri-Memory Retrieval")
        print(SEP)
        by_cat = tri.avg_by_category()
        for cat, avg in sorted(by_cat.items()):
            bar = _bar(avg / 100)
            print(f"  {cat:22s}  {avg:5.1f}/100  {bar}")
        print(f"  {'OVERALL':22s}  {tri.avg_score:5.1f}/100  {_bar(tri.avg_score / 100)}")

    if tok:
        print(f"\n{'─' * 70}")
        print("  SUITE 2 — Token Efficiency")
        print(SEP)
        avg_tokens = tok.avg_tokens()
        avg_density = tok.avg_density()
        print(f"  {'Approach':15s}  {'Avg tokens':>11}  {'Avg density':>12}  {'(quality/token)':15s}")
        print(f"  {'─'*15}  {'─'*11}  {'─'*12}")
        for approach in ("progressive", "dump_all", "baseline"):
            t = avg_tokens[approach]
            d = avg_density[approach]
            print(f"  {approach:15s}  {t:11.0f}  {d:12.4f}  {_bar(min(d * 200, 1.0))}")

    if con:
        print(f"\n{'─' * 70}")
        print("  SUITE 3 — Multi-Agent Consistency")
        print(SEP)
        metrics = [
            ("consistency_score",   con.consistency_score,            "correct reads / total reads"),
            ("supersede_accuracy",  con.supersede_accuracy,           "new value visible after supersede"),
            ("conflict_resolution", con.conflict_resolution_accuracy, "single winner after concurrent write"),
        ]
        for name, val, desc in metrics:
            bar = _bar(val)
            print(f"  {name:22s}  {val:5.1%}  {bar}  ({desc})")
        print(f"  ops={con.ops_completed}  duration={con.duration_seconds}s")

    if thr:
        print(f"\n{'─' * 70}")
        print("  SUITE 4 — Throughput  (median latency per operation)")
        print(SEP)
        enc_cost = thr.write_episodic_ms - thr.write_episodic_raw_ms
        enc_search_cost = thr.search_ms - thr.search_raw_ms
        rows = [
            ("Operation",             "Full pipeline", "Raw storage", "Encoding cost"),
            ("─" * 26,               "─" * 14,        "─" * 12,     "─" * 13),
            ("episodic write",
             f"{thr.write_episodic_ms:6.1f} ms/op",
             f"{thr.write_episodic_raw_ms:6.1f} ms/op",
             f"~{max(0.0, enc_cost):5.1f} ms"),
            ("semantic write",
             f"{thr.write_semantic_ms:6.1f} ms/op",
             "        —",
             "        —"),
            ("search (recall)",
             f"{thr.search_ms:6.1f} ms/op",
             f"{thr.search_raw_ms:6.1f} ms/op",
             f"~{max(0.0, enc_search_cost):5.1f} ms"),
            ("scan_batch (100 rows)",
             f"{thr.scan_batch_total_ms / max(1, thr.n_scan):6.1f} ms/call",
             f"({thr.scan_ops_per_sec:.0f} rows/s)",
             "        —"),
        ]
        col_w = [26, 16, 14, 13]
        for row in rows:
            print("  " + "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_w)))
        total_writes = thr.n_episodic + thr.n_semantic
        print(f"\n  Rust storage writes:  {thr.write_episodic_raw_ms:.1f} ms median "
              f"  ({1000 / max(0.1, thr.write_episodic_raw_ms):.0f} ops/s)")
        print(f"  Encoding overhead:    ~{max(0.0, enc_cost):.1f} ms per write, "
              f"~{max(0.0, enc_search_cost):.1f} ms per search")

    print(f"\n{'═' * 70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="CogDB benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=["all", "tri-memory", "token-efficiency", "consistency", "throughput"],
        default="all",
        help="Which suite to run (default: all)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip OpenAI judge, use keyword-matching fallback",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        metavar="N",
        help="Questions to use in token-efficiency suite (default: 10)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="JSON output path (default: benchmarks/results/bench_<ts>.json)",
    )
    args = parser.parse_args()
    use_llm = not args.no_llm

    run_tri = args.suite in ("all", "tri-memory")
    run_tok = args.suite in ("all", "token-efficiency")
    run_con = args.suite in ("all", "consistency")
    run_thr = args.suite in ("all", "throughput")

    tri_result: Optional[TriMemoryResult] = None
    tok_result: Optional[TokenEfficiencyResult] = None
    con_result: Optional[ConsistencyResult] = None
    thr_result: Optional[ThroughputResult] = None

    # Suites 1 and 2 share a populated database
    if run_tri or run_tok:
        tmpdir = tempfile.mkdtemp(prefix="cogdb_bench_main_")
        try:
            db = CognitiveDB(db_path=tmpdir)
            print("[Setup] Populating database with synthetic DevOps scenario…")
            t0 = time.perf_counter()
            populate_db(db)
            elapsed = round(time.perf_counter() - t0, 2)
            stats = db.stats()
            print(f"[Setup] Done in {elapsed}s — {stats}")

            if run_tri:
                tri_result = run_tri_memory_suite(db, use_llm=use_llm)
            if run_tok:
                tok_result = run_token_efficiency_suite(db, use_llm=use_llm, sample_size=args.questions)
        finally:
            try:
                db._episodic._client.reset()
            except Exception:
                pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    if run_con:
        con_result = run_consistency_suite()

    if run_thr:
        thr_result = run_throughput_suite()

    print_report(tri_result, tok_result, con_result, thr_result)

    # ── Write JSON results ─────────────────────────────────────────────────────
    out_path = args.out
    if out_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(results_dir / f"bench_{ts}.json")

    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "suites_run": args.suite,
        "llm_judge": use_llm and _OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY")),
    }
    if tri_result:
        payload["tri_memory"] = {
            "avg_score": tri_result.avg_score,
            "by_category": tri_result.avg_by_category(),
            "questions": [asdict(r) for r in tri_result.scores],
        }
    if tok_result:
        payload["token_efficiency"] = {
            "avg_tokens": tok_result.avg_tokens(),
            "avg_density": tok_result.avg_density(),
            "per_question": tok_result.per_question,
        }
    if con_result:
        payload["consistency"] = asdict(con_result)
    if thr_result:
        payload["throughput"] = asdict(thr_result)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()

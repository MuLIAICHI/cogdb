"""Importance scoring — heuristic baseline (Phase 0) + learned model (Phase 2).

Phase 2 adds ImportanceModel: Ridge regression trained on a synthetic dataset.
Features beyond the heuristic: version-pattern presence, metric-value presence,
technical entity density, numeric token specificity.

Falls back to the heuristic transparently if scikit-learn is not installed.
"""

from __future__ import annotations

import math
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from cogdb.models import MemoryType, MemoryUnit


# ── Keyword patterns (shared by heuristic and ML feature extraction) ──────────

_HIGH_IMPORTANCE_PATTERNS = [
    r"\b(error|fail|crash|critical|urgent|important|always|never)\b",
    r"\b(prefer|must|require|need|expect)\b",
    r"\b(password|secret|token|key|credential)\b",
    r"\b(deadline|due|asap|priority)\b",
]

_LOW_IMPORTANCE_PATTERNS = [
    r"\b(maybe|perhaps|might|could|possibly)\b",
    r"\b(test|example|sample|dummy|placeholder)\b",
]

_HIGH_RE = re.compile("|".join(_HIGH_IMPORTANCE_PATTERNS), re.IGNORECASE)
_LOW_RE = re.compile("|".join(_LOW_IMPORTANCE_PATTERNS), re.IGNORECASE)

# Phase 2: additional feature patterns
_VERSION_RE = re.compile(r"v\d+\.\d+")
_METRIC_RE = re.compile(r"\d+\s*(ms|%|MB|GB|KB|req|ops|min\b|s\b)", re.IGNORECASE)
_TECH_TERMS = frozenset({
    "api", "db", "sql", "jwt", "oauth", "nginx", "postgres", "postgresql",
    "redis", "docker", "kubernetes", "github", "ci", "cd", "deploy", "rollback",
    "migration", "alembic", "fastapi", "flask", "django", "react", "typescript",
    "grafana", "prometheus", "elasticsearch", "kafka", "rabbitmq", "celery",
    "lambda", "s3", "ec2", "rds", "gke", "eks", "terraform", "ansible",
    "cors", "ssl", "tls", "https", "webhook", "cron", "orm", "wal",
})


# ── ML model ──────────────────────────────────────────────────────────────────

# Synthetic training set: 60 diverse (content, memory_type, importance) examples.
# Covers the full 0.2–0.95 range with clear content-to-importance signal.
# Does NOT include benchmark fixture data (no eval leakage).
_TRAINING_SAMPLES: list[dict] = [
    # ── High importance: incidents, security, specific versions ───────────────
    {"content": "Critical production outage: database connection refused, all API calls failing with 502", "memory_type": MemoryType.EPISODIC, "importance": 0.92},
    {"content": "v2.4.1 deployed to production — includes critical security patch for authentication bypass", "memory_type": MemoryType.EPISODIC, "importance": 0.90},
    {"content": "SQL injection vulnerability found: user search endpoint concatenating raw user input into query", "memory_type": MemoryType.EPISODIC, "importance": 0.93},
    {"content": "Memory leak in background worker: RSS growing 50 MB per hour, pod OOM-killed after 6 hours", "memory_type": MemoryType.EPISODIC, "importance": 0.88},
    {"content": "API response time: p95 jumped from 120ms to 2400ms after deploy — N+1 query in order endpoint", "memory_type": MemoryType.EPISODIC, "importance": 0.89},
    {"content": "Redis connection pool exhausted after traffic spike: maxconn=50 hit at 800 req/s, 503 errors", "memory_type": MemoryType.EPISODIC, "importance": 0.87},
    {"content": "Emergency rollback of v2.4.0: authentication broken for 15% of users for 23 minutes", "memory_type": MemoryType.EPISODIC, "importance": 0.91},
    {"content": "SSL certificate expired at 03:14 UTC — site down 47 minutes until manual renewal completed", "memory_type": MemoryType.EPISODIC, "importance": 0.94},
    {"content": "Kubernetes pod OOM-killed: worker allocated 2Gi limit, peaked at 2.1Gi during batch processing", "memory_type": MemoryType.EPISODIC, "importance": 0.89},
    {"content": "Data migration completed: 14.7 million rows migrated with zero failures in 8 minutes", "memory_type": MemoryType.EPISODIC, "importance": 0.82},
    # ── Medium-high: meaningful changes, deployments, monitoring ──────────────
    {"content": "Implemented JWT refresh token rotation: 15-minute access tokens, 7-day refresh in HttpOnly cookies", "memory_type": MemoryType.EPISODIC, "importance": 0.75},
    {"content": "Added rate limiting: 100 req/min per user, 1000 req/min per IP using Redis sliding window", "memory_type": MemoryType.EPISODIC, "importance": 0.72},
    {"content": "PostgreSQL query plan changed after VACUUM ANALYZE — index scan replaced table scan, 10x speedup", "memory_type": MemoryType.EPISODIC, "importance": 0.74},
    {"content": "Docker image size reduced from 1.2 GB to 340 MB by switching to python:3.12-slim base", "memory_type": MemoryType.EPISODIC, "importance": 0.70},
    {"content": "CI pipeline time reduced from 18 min to 7 min after splitting test suites across 3 parallel runners", "memory_type": MemoryType.EPISODIC, "importance": 0.68},
    {"content": "Grafana alert configured: fires when error rate exceeds 5% or p95 latency exceeds 500ms for 2 minutes", "memory_type": MemoryType.EPISODIC, "importance": 0.73},
    {"content": "Load balancer health check endpoint updated from /health to /api/v2/health after route refactor", "memory_type": MemoryType.EPISODIC, "importance": 0.66},
    {"content": "Security scan with bandit: 3 medium findings fixed, dependency audit clean for known CVEs", "memory_type": MemoryType.EPISODIC, "importance": 0.65},
    {"content": "Deploy of v1.5.0: adds WebSocket support, Redis pub/sub backend, connection limit 5000", "memory_type": MemoryType.EPISODIC, "importance": 0.76},
    {"content": "Alembic migration added email_verified column to users table — zero-downtime, staging verified", "memory_type": MemoryType.EPISODIC, "importance": 0.71},
    # ── Medium: preferences, routine tasks, approvals ─────────────────────────
    {"content": "Team standup: Alice on dashboard filters, Bob on notifications, Carol writing documentation", "memory_type": MemoryType.EPISODIC, "importance": 0.45},
    {"content": "Code review of PR: minor style comments left, no blocking issues found, approved", "memory_type": MemoryType.EPISODIC, "importance": 0.50},
    {"content": "Updated README with new installation instructions and troubleshooting guide section", "memory_type": MemoryType.EPISODIC, "importance": 0.40},
    {"content": "Scheduled planning session for Thursday — team lead to prepare roadmap presentation slides", "memory_type": MemoryType.EPISODIC, "importance": 0.38},
    {"content": "Lead developer prefers working on backend services rather than frontend UI work", "memory_type": MemoryType.EPISODIC, "importance": 0.55},
    {"content": "Team agreed on Conventional Commits format for all commit messages going forward", "memory_type": MemoryType.EPISODIC, "importance": 0.52},
    {"content": "Environment variables for staging environment updated in configuration file", "memory_type": MemoryType.EPISODIC, "importance": 0.58},
    {"content": "Sprint demo: stakeholders happy with new analytics dashboard, requested one tweak", "memory_type": MemoryType.EPISODIC, "importance": 0.40},
    {"content": "New developer joined team today and set up local development environment successfully", "memory_type": MemoryType.EPISODIC, "importance": 0.42},
    {"content": "Installed prettier as a pre-commit hook to enforce consistent code formatting", "memory_type": MemoryType.EPISODIC, "importance": 0.48},
    # ── Low-medium: vague, speculative, test/placeholder ──────────────────────
    {"content": "Maybe we should consider moving to microservices architecture in the future", "memory_type": MemoryType.EPISODIC, "importance": 0.28},
    {"content": "The new feature might possibly improve user retention, hard to say without data yet", "memory_type": MemoryType.EPISODIC, "importance": 0.25},
    {"content": "Test: checking if WebSocket connections work correctly in the sandbox environment", "memory_type": MemoryType.EPISODIC, "importance": 0.30},
    {"content": "Example email for the notification template — just a placeholder for now", "memory_type": MemoryType.EPISODIC, "importance": 0.22},
    {"content": "Team meeting notes — general discussion, no concrete decisions or action items", "memory_type": MemoryType.EPISODIC, "importance": 0.35},
    {"content": "Looked into perhaps adopting a different ORM but could be significant refactor work", "memory_type": MemoryType.EPISODIC, "importance": 0.27},
    {"content": "Sample test data for demo purposes only — not real production records", "memory_type": MemoryType.EPISODIC, "importance": 0.20},
    {"content": "Could potentially add dark mode to the settings page at some point", "memory_type": MemoryType.EPISODIC, "importance": 0.25},
    {"content": "Dummy entry inserted to verify deployment pipeline connectivity works", "memory_type": MemoryType.EPISODIC, "importance": 0.18},
    {"content": "Brainstorming session — explored a few ideas, no firm direction decided yet", "memory_type": MemoryType.EPISODIC, "importance": 0.32},
    # ── Semantic facts ─────────────────────────────────────────────────────────
    {"content": "user_service prefers_storage PostgreSQL with connection pooling", "memory_type": MemoryType.SEMANTIC, "importance": 0.75},
    {"content": "authentication method JWT refresh tokens HttpOnly cookies", "memory_type": MemoryType.SEMANTIC, "importance": 0.72},
    {"content": "backend language Python FastAPI framework deployed Kubernetes", "memory_type": MemoryType.SEMANTIC, "importance": 0.68},
    {"content": "monitoring tool Grafana alerts configured for RSS and latency", "memory_type": MemoryType.SEMANTIC, "importance": 0.70},
    {"content": "cache backend Redis pub/sub for real-time WebSocket messages", "memory_type": MemoryType.SEMANTIC, "importance": 0.68},
    {"content": "ci_cd pipeline GitHub Actions runs on every pull request merge", "memory_type": MemoryType.SEMANTIC, "importance": 0.65},
    {"content": "lead_dev prefers dark IDE theme", "memory_type": MemoryType.SEMANTIC, "importance": 0.45},
    {"content": "team_name BlueBird Project backend PostgreSQL microservices", "memory_type": MemoryType.SEMANTIC, "importance": 0.60},
    {"content": "api_gateway deployed_version v3.1.2 stable production Kubernetes", "memory_type": MemoryType.SEMANTIC, "importance": 0.78},
    {"content": "database_migrations tool Alembic zero-downtime PostgreSQL", "memory_type": MemoryType.SEMANTIC, "importance": 0.72},
    # ── Procedural templates ───────────────────────────────────────────────────
    {"content": "Procedure database_backup: critical pg_dump procedure for PostgreSQL production. Steps: pg_dump → verify → upload S3", "memory_type": MemoryType.PROCEDURAL, "importance": 0.88},
    {"content": "Procedure incident_response: emergency production outage steps. Steps: alert → diagnose kubectl logs → rollback or fix → post-mortem", "memory_type": MemoryType.PROCEDURAL, "importance": 0.90},
    {"content": "Procedure deploy_flow: standard production deploy. Steps: CI green → merge main → docker build → kubectl rolling update → smoke test", "memory_type": MemoryType.PROCEDURAL, "importance": 0.85},
    {"content": "Procedure security_audit: pre-release bandit scan, dependency audit, SQL injection check, rate limit verification", "memory_type": MemoryType.PROCEDURAL, "importance": 0.82},
    {"content": "Procedure code_review: review diff for N+1 queries, SQL injection, missing rate limits, test coverage drop", "memory_type": MemoryType.PROCEDURAL, "importance": 0.80},
    # ── Access count variation (same content, different access_count) ──────────
    {"content": "Team uses Slack for async communication and GitHub for code reviews", "memory_type": MemoryType.EPISODIC, "importance": 0.42, "access_count": 0},
    {"content": "Team uses Slack for async communication and GitHub for code reviews", "memory_type": MemoryType.EPISODIC, "importance": 0.55, "access_count": 15},
    {"content": "Deploy runbook for production system — standard steps", "memory_type": MemoryType.EPISODIC, "importance": 0.62, "access_count": 0},
    {"content": "Deploy runbook for production system — standard steps", "memory_type": MemoryType.EPISODIC, "importance": 0.72, "access_count": 30},
    # ── Recency variation ──────────────────────────────────────────────────────
    {"content": "Fixed the login timeout bug affecting 2% of users on mobile Safari", "memory_type": MemoryType.EPISODIC, "importance": 0.70, "recency_hours": 0},
    {"content": "Fixed the login timeout bug affecting 2% of users on mobile Safari", "memory_type": MemoryType.EPISODIC, "importance": 0.62, "recency_hours": 72},
]


class ImportanceModel:
    """Lightweight Ridge regression importance predictor trained on access patterns.

    Extracts 11 features from memory content and metadata, trains a Ridge
    regression model, and predicts importance in 0–1. Falls back to the
    heuristic transparently if scikit-learn is not installed.

    Features captured beyond the heuristic:
    - Version pattern presence (v\\d+\\.\\d+) — specific release data
    - Metric value presence (\\d+ms, \\d+%) — concrete performance/incident values
    - Technical entity density — domain vocabulary depth
    - Numeric token specificity — factual detail ratio

    Args:
        None. Use get_default_model() for the pre-trained singleton.

    Example:
        >>> model = ImportanceModel()
        >>> model.train(TRAINING_SAMPLES)
        >>> score = model.predict("v1.2.3 deployed with JWT auth", MemoryType.EPISODIC)
        >>> 0.0 <= score <= 1.0
        True
    """

    FEATURE_NAMES = [
        "memory_type_norm",  # 0.0=episodic, 0.5=semantic, 1.0=procedural
        "word_count_norm",   # word_count / 100
        "has_version",       # v\d+\.\d+ present (0/1)
        "has_metric",        # \d+ms, \d+% etc. present (0/1)
        "high_kw_density",   # high-importance keywords / 5 (capped at 1.0)
        "low_kw_density",    # low-importance keywords / 5 (capped at 1.0)
        "entity_density",    # capitalized tokens / total tokens (capped at 1.0)
        "specificity",       # numeric tokens / total tokens (capped at 1.0)
        "tech_density",      # tech vocabulary hits / 5 (capped at 1.0)
        "access_log",        # log1p(access_count)
        "recency_decay",     # exp(-recency_hours / 48)
    ]

    def __init__(self) -> None:
        self._coef: Optional[list[float]] = None
        self._intercept: float = 0.0
        self._scaler_mean: Optional[list[float]] = None
        self._scaler_std: Optional[list[float]] = None

    @property
    def is_fitted(self) -> bool:
        """True if the model has been trained."""
        return self._coef is not None

    @staticmethod
    def extract_features(
        content: str,
        memory_type: MemoryType,
        access_count: int = 0,
        recency_hours: float = 0.0,
    ) -> list[float]:
        """Extract 11 numerical features from a memory.

        Args:
            content: Raw text content.
            memory_type: Episodic, semantic, or procedural.
            access_count: Times this memory has been retrieved.
            recency_hours: Hours since creation (0 = just created).

        Returns:
            List of 11 floats in the range [0, ∞) (unbounded for access_log).

        Example:
            >>> feats = ImportanceModel.extract_features("v1.2.3 deployed", MemoryType.EPISODIC)
            >>> len(feats)
            11
        """
        tokens_raw = content.split()
        tokens_lower = content.lower().split()
        n = max(1, len(tokens_lower))

        type_norm = {MemoryType.EPISODIC: 0.0, MemoryType.SEMANTIC: 0.5, MemoryType.PROCEDURAL: 1.0}[memory_type]
        has_version = 1.0 if _VERSION_RE.search(content) else 0.0
        has_metric = 1.0 if _METRIC_RE.search(content) else 0.0
        high_hits = len(_HIGH_RE.findall(content))
        low_hits = len(_LOW_RE.findall(content))
        high_kw = min(1.0, high_hits / 5.0)
        low_kw = min(1.0, low_hits / 5.0)

        # Entity density: capitalized, alpha-only tokens (proper nouns/names)
        cap_tokens = sum(1 for w in tokens_raw if w and w[0].isupper() and len(w) > 1 and w.isalpha())
        entity_density = min(1.0, cap_tokens / n)

        # Specificity: tokens containing at least one digit
        numeric_tokens = sum(1 for w in tokens_lower if any(c.isdigit() for c in w))
        specificity = min(1.0, numeric_tokens / n)

        # Technical vocabulary density
        tech_hits = sum(1 for w in tokens_lower if w.rstrip(".,;:()") in _TECH_TERMS)
        tech_density = min(1.0, tech_hits / 5.0)

        access_log = math.log1p(access_count)
        recency_decay = math.exp(-max(0.0, recency_hours) / 48.0)

        return [
            type_norm,
            n / 100.0,
            has_version,
            has_metric,
            high_kw,
            low_kw,
            entity_density,
            specificity,
            tech_density,
            access_log,
            recency_decay,
        ]

    def train(self, samples: list[dict]) -> "ImportanceModel":
        """Fit the model on labeled samples.

        Args:
            samples: List of dicts with keys: content (str), memory_type (MemoryType),
                importance (float 0-1), and optionally access_count (int),
                recency_hours (float).

        Returns:
            self (for chaining).

        Example:
            >>> model = ImportanceModel().train(_TRAINING_SAMPLES)
            >>> model.is_fitted
            True
        """
        try:
            import numpy as np
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return self

        X = np.array([
            self.extract_features(
                s["content"],
                s["memory_type"],
                s.get("access_count", 0),
                s.get("recency_hours", 1.0),
            )
            for s in samples
        ])
        y = np.array([float(s["importance"]) for s in samples])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        self._coef = model.coef_.tolist()
        self._intercept = float(model.intercept_)
        self._scaler_mean = scaler.mean_.tolist()
        self._scaler_std = scaler.scale_.tolist()
        return self

    def predict(
        self,
        content: str,
        memory_type: MemoryType,
        access_count: int = 0,
        recency_hours: float = 0.0,
    ) -> float:
        """Predict importance score for a memory.

        Args:
            content: Raw text content.
            memory_type: Episodic, semantic, or procedural.
            access_count: Times this memory has been retrieved.
            recency_hours: Hours since creation.

        Returns:
            Importance score clamped to [0.0, 1.0].

        Example:
            >>> model = ImportanceModel().train(_TRAINING_SAMPLES)
            >>> score = model.predict("CRITICAL: prod down, 502 errors", MemoryType.EPISODIC)
            >>> score > 0.7
            True
        """
        if not self.is_fitted:
            return _heuristic_score(content, memory_type, access_count, recency_hours)
        try:
            import numpy as np
        except ImportError:
            return _heuristic_score(content, memory_type, access_count, recency_hours)

        features = np.array(self.extract_features(content, memory_type, access_count, recency_hours))
        scaled = (features - np.array(self._scaler_mean)) / np.array(self._scaler_std)
        raw = float(np.dot(scaled, np.array(self._coef)) + self._intercept)
        return max(0.0, min(1.0, raw))

    def partial_fit(self, samples: list[dict]) -> "ImportanceModel":
        """Update model from new labeled samples (online learning from access patterns).

        Uses SGD warm-started from current Ridge weights. Call this as access
        data accumulates to refine the model for a specific deployment's patterns.

        Args:
            samples: New labeled samples in the same format as train().

        Returns:
            self (for chaining).

        Example:
            >>> model.partial_fit([{"content": "...", "memory_type": ..., "importance": 0.8}])
        """
        if not self.is_fitted:
            return self.train(samples)
        try:
            import numpy as np
            from sklearn.linear_model import SGDRegressor
        except ImportError:
            return self

        X = np.array([
            self.extract_features(
                s["content"],
                s["memory_type"],
                s.get("access_count", 0),
                s.get("recency_hours", 1.0),
            )
            for s in samples
        ])
        y = np.array([float(s["importance"]) for s in samples])
        X_scaled = (X - np.array(self._scaler_mean)) / np.array(self._scaler_std)

        sgd = SGDRegressor(eta0=0.001, learning_rate="constant", max_iter=1)
        sgd.coef_ = np.array(self._coef)
        sgd.intercept_ = np.array([self._intercept])
        sgd.partial_fit(X_scaled, y)

        self._coef = sgd.coef_.tolist()
        self._intercept = float(sgd.intercept_[0])
        return self

    def save(self, path: str) -> None:
        """Persist model parameters to a JSON file.

        Args:
            path: File path for JSON output.

        Example:
            >>> model.save("./cogdb_data/importance_model.json")
        """
        import json
        from pathlib import Path
        data = {
            "coef": self._coef,
            "intercept": self._intercept,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> "ImportanceModel":
        """Load model parameters from a JSON file.

        Args:
            path: File path previously written by save().

        Returns:
            self (for chaining).

        Example:
            >>> model = ImportanceModel().load("./cogdb_data/importance_model.json")
        """
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        self._coef = data["coef"]
        self._intercept = float(data["intercept"])
        self._scaler_mean = data["scaler_mean"]
        self._scaler_std = data["scaler_std"]
        return self


# ── Singleton default model ───────────────────────────────────────────────────

_model_lock = threading.Lock()
_default_model: Optional[ImportanceModel] = None


def get_default_model() -> Optional[ImportanceModel]:
    """Return the lazily-trained default ImportanceModel.

    Trains on _TRAINING_SAMPLES on first call (requires scikit-learn).
    Returns None if sklearn is not installed.

    Returns:
        Fitted ImportanceModel, or None if sklearn unavailable.

    Example:
        >>> model = get_default_model()
        >>> model is None or model.is_fitted
        True
    """
    global _default_model
    if _default_model is not None:
        return _default_model
    with _model_lock:
        if _default_model is None:
            m = ImportanceModel()
            m.train(_TRAINING_SAMPLES)
            if m.is_fitted:
                _default_model = m
    return _default_model


# ── Public scoring functions ──────────────────────────────────────────────────


def score_importance(
    content: str,
    memory_type: MemoryType,
    access_count: int = 0,
    recency_hours: float = 0.0,
    explicit_importance: float | None = None,
) -> float:
    """Compute an importance score for a memory unit.

    Uses the ML model when scikit-learn is available; falls back to the
    heuristic otherwise. If explicit_importance is provided, it is blended
    in with 60% weight regardless of which scorer is used.

    Args:
        content: Raw text content of the memory.
        memory_type: Episodic, semantic, or procedural.
        access_count: How many times this memory has been retrieved.
        recency_hours: Hours since creation (0 = just created).
        explicit_importance: If provided, blend it with the computed score.

    Returns:
        Importance score between 0.0 and 1.0.

    Example:
        >>> score_importance("User prefers dark mode", MemoryType.EPISODIC)
        0.55
        >>> score_importance("CRITICAL: API key expired", MemoryType.EPISODIC)
        0.82
    """
    model = get_default_model()
    if model is not None:
        score = model.predict(content, memory_type, access_count, recency_hours)
    else:
        score = _heuristic_score(content, memory_type, access_count, recency_hours)

    score = max(0.0, min(1.0, score))

    if explicit_importance is not None:
        score = 0.6 * explicit_importance + 0.4 * score

    return round(score, 4)


def score_memory_unit(unit: MemoryUnit) -> float:
    """Convenience wrapper to score a MemoryUnit directly.

    Args:
        unit: The memory unit to score.

    Returns:
        Importance score between 0.0 and 1.0.

    Example:
        >>> unit = MemoryUnit(content="deploy failed", memory_type=MemoryType.EPISODIC, agent_id="a1")
        >>> score_memory_unit(unit)
        0.71
    """
    now = datetime.now(timezone.utc)
    recency_hours = (now - unit.created_at).total_seconds() / 3600.0
    return score_importance(
        content=unit.content,
        memory_type=unit.memory_type,
        access_count=unit.access_count,
        recency_hours=recency_hours,
    )


# ── Heuristic implementation (kept as fallback) ───────────────────────────────


def _heuristic_score(
    content: str,
    memory_type: MemoryType,
    access_count: int = 0,
    recency_hours: float = 0.0,
) -> float:
    """Original heuristic scorer — used as ML fallback."""
    score = _base_score(memory_type)
    score += _content_signal(content)
    score += _access_boost(access_count)
    score += _recency_boost(recency_hours)
    return max(0.0, min(1.0, score))


def _base_score(memory_type: MemoryType) -> float:
    return {
        MemoryType.PROCEDURAL: 0.6,
        MemoryType.SEMANTIC: 0.55,
        MemoryType.EPISODIC: 0.4,
    }[memory_type]


def _content_signal(content: str) -> float:
    high_hits = len(_HIGH_RE.findall(content))
    low_hits = len(_LOW_RE.findall(content))
    boost = min(0.25, high_hits * 0.05)
    penalty = min(0.15, low_hits * 0.03)
    return boost - penalty


def _access_boost(access_count: int) -> float:
    if access_count <= 0:
        return 0.0
    return min(0.15, 0.05 * math.log1p(access_count))


def _recency_boost(recency_hours: float) -> float:
    if recency_hours <= 0:
        return 0.05
    if recency_hours >= 48:
        return 0.0
    return 0.05 * (1.0 - recency_hours / 48.0)

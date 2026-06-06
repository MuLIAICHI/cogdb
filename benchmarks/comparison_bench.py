"""CogDB Comparison Benchmark — CogDB vs ChromaDB, Mem0, Zep.

Usage:
    python -m benchmarks.comparison_bench
    python -m benchmarks.comparison_bench --out results/my_run.json
    python -m benchmarks.comparison_bench --no-chroma --no-mem0 --no-zep

Requires: cogdb (with compiled cogdb_engine)
Optional: mem0, zep-python, OPENAI_API_KEY (for Mem0), ZEP_API_URL (for Zep)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ── Scenario data ──────────────────────────────────────────────────────────────

MEMORIES: list[tuple[str, float]] = [
    ("Deployed v2.1.0 API to production using Kubernetes blue-green deployment. Zero downtime.", 0.9),
    ("User authentication broke after the v2.1.0 deploy — JWT secret was not rotated in prod.", 0.95),
    ("Fixed JWT issue by syncing secret from vault. Rotated all active sessions.", 0.9),
    ("Database migration ran but left orphaned rows in the sessions table.", 0.7),
    ("Cron job added to clean orphaned sessions every 6 hours.", 0.6),
    ("Redis cache hit rate dropped from 92% to 71% after deployment.", 0.75),
    ("Root cause: cache key prefix changed in the new code, invalidating all warm keys.", 0.85),
    ("Warming script deployed; cache hit rate recovered to 89% within 20 minutes.", 0.7),
    ("New feature: rate limiting on /api/v2/search endpoint, 100 req/min per user.", 0.65),
    ("Load test showed p99 latency of 180ms at 500 concurrent users — within SLA.", 0.8),
    ("On-call rotation updated: Alice covers Mon-Wed, Bob covers Thu-Sun.", 0.5),
    ("Incident #2041: API gateway returned 502 for 4 minutes due to upstream timeout.", 0.95),
    ("Post-mortem: upstream ML model server ran out of memory under load.", 0.9),
    ("Permanent fix: added circuit breaker with 30s timeout and fallback to cached predictions.", 0.85),
    ("Monitoring alert threshold for memory usage lowered from 90% to 75%.", 0.6),
    ("New microservice 'recommender' deployed to staging. Not yet in production.", 0.55),
    ("API documentation updated for v2.1 endpoints — now includes auth examples.", 0.45),
    ("Cost optimization: switched to spot instances for batch jobs, saving $800/month.", 0.5),
    ("Compliance audit passed. No PII found in logs.", 0.65),
    ("Q3 planning: next sprint will focus on search performance and recommender go-live.", 0.6),
]

QUERIES: list[tuple[str, list[str]]] = [
    ("What happened with JWT authentication after the deployment?", ["JWT", "auth", "secret", "rotated"]),
    ("How was the Redis cache issue resolved?", ["cache", "warming", "hit rate", "prefix"]),
    ("What was the root cause of Incident 2041?", ["ML model", "memory", "upstream", "502"]),
    ("What circuit breaker configuration was put in place?", ["circuit breaker", "timeout", "30s", "fallback"]),
    ("What is the current on-call rotation?", ["Alice", "Bob", "Mon", "Thu"]),
    ("What is the p99 latency under load?", ["180ms", "500", "p99", "latency"]),
    ("What rate limiting is in place for search?", ["rate limit", "search", "100 req"]),
    ("What is the status of the recommender service?", ["staging", "not production", "recommender"]),
    ("How were orphaned sessions handled?", ["orphaned", "sessions", "cron", "clean"]),
    ("What cost savings were achieved?", ["spot", "800", "saving", "cost"]),
]

AGENT_ID = "bench-agent"
TOKEN_BUDGET = 500


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SystemResult:
    """Per-system benchmark result.

    Args:
        system: Display name of the memory system.
        avg_score: Mean keyword recall score across all queries (0.0–1.0).
        avg_latency_ms: Mean query latency in milliseconds.
        scores: Per-query keyword recall scores.
        status: "ok" | "skipped" | "not_available" | "error"
        notes: Short human-readable annotation (e.g. "vector only").
        avg_token_efficiency: CogDB-only. Mean tokens_used / token_budget.
        types_retrieved: CogDB-only. Unique memory_type values seen in results.

    Example:
        result = SystemResult(
            system="CogDB",
            avg_score=0.84,
            avg_latency_ms=8.2,
            scores=[1.0, 0.75, ...],
            status="ok",
            notes="tri-memory, scoped",
            avg_token_efficiency=0.31,
            types_retrieved=["episodic"],
        )
    """

    system: str
    avg_score: float
    avg_latency_ms: float
    scores: list[float]
    status: str
    notes: str
    avg_token_efficiency: Optional[float] = None
    types_retrieved: Optional[list[str]] = None


# ── Scoring helpers ────────────────────────────────────────────────────────────

def keyword_score(results: list[str], keywords: list[str]) -> float:
    """Score 0-1: fraction of keywords found in the combined results.

    Args:
        results: Retrieved text strings from the memory system.
        keywords: Expected keywords that indicate a correct retrieval.

    Returns:
        Float in [0.0, 1.0] — fraction of keywords present.

    Example:
        score = keyword_score(["JWT secret rotated"], ["JWT", "rotated"])
        assert score == 1.0
    """
    combined = " ".join(results).lower()
    found = sum(1 for kw in keywords if kw.lower() in combined)
    return found / len(keywords)


def chroma_distance_to_score(distances: list[list[float]]) -> float:
    """Convert ChromaDB L2 distances to a 0-1 similarity score.

    Typical max L2 distance for normalized embeddings is ~2.0.

    Args:
        distances: Nested list as returned by collection.query() — distances[0]
                   is the list of distances for the first query.

    Returns:
        Float in [0.0, 1.0] where 1.0 means identical.

    Example:
        score = chroma_distance_to_score([[0.2, 0.4, 0.6]])
        assert 0.0 <= score <= 1.0
    """
    if not distances or not distances[0]:
        return 0.0
    avg_dist = sum(distances[0]) / len(distances[0])
    return max(0.0, 1.0 - avg_dist / 2.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _not_available(system: str, reason: str) -> SystemResult:
    return SystemResult(
        system=system,
        avg_score=0.0,
        avg_latency_ms=0.0,
        scores=[],
        status="not_available",
        notes=reason,
    )


def _skipped(system: str, reason: str) -> SystemResult:
    return SystemResult(
        system=system,
        avg_score=0.0,
        avg_latency_ms=0.0,
        scores=[],
        status="skipped",
        notes=reason,
    )


# ── System runners ─────────────────────────────────────────────────────────────

def run_cogdb() -> SystemResult:
    """Run the comparison scenario against CogDB.

    Returns:
        SystemResult with keyword recall scores, latency, token efficiency,
        and retrieved memory types.

    Example:
        result = run_cogdb()
        assert result.status in ("ok", "error")
    """
    try:
        from cogdb.core import CognitiveDB
        from cogdb.models import MemoryScope
        from cogdb.utils.tokenizer import count_tokens
    except ImportError as exc:
        print(f"  CogDB: cogdb_engine not built. Run: cd cogdb_engine && maturin develop")
        return SystemResult(
            system="CogDB",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="error",
            notes=f"ImportError: {exc}",
        )

    tmpdir = tempfile.mkdtemp(prefix="cogdb_comparison_")
    db: Any = None
    try:
        db = CognitiveDB(db_path=tmpdir)

        for content, importance in MEMORIES:
            db.remember(
                content=content,
                agent_id=AGENT_ID,
                importance=importance,
                scope=MemoryScope.ORGANIZATION,
            )

        scores: list[float] = []
        latencies: list[float] = []
        token_efficiencies: list[float] = []
        all_types: set[str] = set()

        for query, keywords in QUERIES:
            t0 = time.perf_counter()
            memories = db.recall(
                query=query,
                agent_id=AGENT_ID,
                token_budget=TOKEN_BUDGET,
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            texts = [m.content for m in memories]
            scores.append(keyword_score(texts, keywords))

            tokens_used = count_tokens(" ".join(texts))
            token_efficiencies.append(tokens_used / TOKEN_BUDGET)

            for m in memories:
                all_types.add(m.memory_type.value)

        return SystemResult(
            system="CogDB",
            avg_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            scores=[round(s, 4) for s in scores],
            status="ok",
            notes="tri-memory, scoped",
            avg_token_efficiency=round(
                sum(token_efficiencies) / len(token_efficiencies), 3
            ) if token_efficiencies else None,
            types_retrieved=sorted(all_types),
        )

    except Exception as exc:
        return SystemResult(
            system="CogDB",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="error",
            notes=str(exc)[:80],
        )
    finally:
        if db is not None:
            try:
                db._episodic._client.reset()
            except Exception:
                pass
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_chromadb() -> SystemResult:
    """Run the comparison scenario against ChromaDB (raw vector store, no budget).

    Returns:
        SystemResult with keyword recall scores and latency.

    Example:
        result = run_chromadb()
        assert result.status in ("ok", "not_available", "error")
    """
    try:
        import chromadb
    except ImportError:
        print("  ChromaDB: not installed. Install with: pip install chromadb")
        return _not_available("ChromaDB", "not installed")

    chroma_client: Any = None
    try:
        try:
            chroma_client = chromadb.EphemeralClient()
        except AttributeError:
            chroma_client = chromadb.Client()

        collection = chroma_client.create_collection("bench_comparison")

        collection.add(
            documents=[content for content, _ in MEMORIES],
            ids=[f"mem_{i}" for i in range(len(MEMORIES))],
            metadatas=[{"importance": importance} for _, importance in MEMORIES],
        )

        scores: list[float] = []
        latencies: list[float] = []

        for query, keywords in QUERIES:
            t0 = time.perf_counter()
            result = collection.query(query_texts=[query], n_results=5)
            latencies.append((time.perf_counter() - t0) * 1000)
            docs = result.get("documents", [[]])[0]
            scores.append(keyword_score(docs, keywords))

        return SystemResult(
            system="ChromaDB",
            avg_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            scores=[round(s, 4) for s in scores],
            status="ok",
            notes="vector only",
        )

    except Exception as exc:
        return SystemResult(
            system="ChromaDB",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="error",
            notes=str(exc)[:80],
        )
    finally:
        if chroma_client is not None:
            try:
                chroma_client.reset()
            except Exception:
                pass


def run_mem0() -> SystemResult:
    """Run the comparison scenario against Mem0 (requires OPENAI_API_KEY).

    Returns:
        SystemResult. status="not_available" if mem0 not installed or no API key.

    Example:
        result = run_mem0()
        assert result.status in ("ok", "not_available", "error")
    """
    try:
        from mem0 import Memory
    except ImportError:
        return _not_available("Mem0", "not installed")

    if not os.environ.get("OPENAI_API_KEY"):
        return SystemResult(
            system="Mem0",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="not_available",
            notes="OPENAI_API_KEY not set",
        )

    try:
        m = Memory()

        for content, _ in MEMORIES:
            m.add(content, user_id=AGENT_ID)

        scores: list[float] = []
        latencies: list[float] = []

        for query, keywords in QUERIES:
            t0 = time.perf_counter()
            raw = m.search(query, user_id=AGENT_ID)
            latencies.append((time.perf_counter() - t0) * 1000)

            # Handle both list and dict response shapes across Mem0 versions
            if isinstance(raw, dict):
                items = raw.get("results", raw.get("memories", []))
            else:
                items = raw if isinstance(raw, list) else []

            texts = [
                item.get("memory", item.get("text", str(item)))
                if isinstance(item, dict) else str(item)
                for item in items
            ]
            scores.append(keyword_score(texts, keywords))

        return SystemResult(
            system="Mem0",
            avg_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            scores=[round(s, 4) for s in scores],
            status="ok",
            notes="LLM-based extraction",
        )

    except Exception as exc:
        return SystemResult(
            system="Mem0",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="error",
            notes=str(exc)[:80],
        )


def run_zep() -> SystemResult:
    """Run the comparison scenario against Zep (zep-python or zep-cloud).

    Requires ZEP_API_URL (zep-python) or ZEP_API_KEY (zep-cloud).

    Returns:
        SystemResult. status="not_available" if client not installed or no endpoint.

    Example:
        result = run_zep()
        assert result.status in ("ok", "not_available", "error")
    """
    ZepClientClass: Any = None
    client_kwargs: dict[str, Any] = {}

    try:
        from zep_python import ZepClient as _ZepOss

        ZepClientClass = _ZepOss
        api_url = os.environ.get("ZEP_API_URL")
        if not api_url:
            return SystemResult(
                system="Zep",
                avg_score=0.0,
                avg_latency_ms=0.0,
                scores=[],
                status="not_available",
                notes="ZEP_API_URL not set",
            )
        client_kwargs = {"api_url": api_url}
    except ImportError:
        pass

    if ZepClientClass is None:
        try:
            from zep_cloud.client import Zep as _ZepCloud

            ZepClientClass = _ZepCloud
            api_key = os.environ.get("ZEP_API_KEY")
            if not api_key:
                return SystemResult(
                    system="Zep",
                    avg_score=0.0,
                    avg_latency_ms=0.0,
                    scores=[],
                    status="not_available",
                    notes="ZEP_API_KEY not set",
                )
            client_kwargs = {"api_key": api_key}
        except ImportError:
            pass

    if ZepClientClass is None:
        return _not_available("Zep", "not installed")

    import uuid

    try:
        client = ZepClientClass(**client_kwargs)
        session_id = f"bench-{uuid.uuid4().hex[:8]}"

        for content, _ in MEMORIES:
            try:
                from zep_python.memory import Memory as ZepMem
                from zep_python.memory import Message as ZepMsg

                client.memory.add(
                    session_id=session_id,
                    memory=ZepMem(messages=[ZepMsg(role="user", content=content)]),
                )
            except Exception:
                client.memory.add(
                    session_id=session_id,
                    messages=[{"role": "user", "content": content}],
                )

        scores: list[float] = []
        latencies: list[float] = []

        for query, keywords in QUERIES:
            t0 = time.perf_counter()
            try:
                result = client.memory.search(
                    session_id=session_id, text=query, limit=5
                )
                texts = [
                    r.message.content if hasattr(r, "message") else str(r)
                    for r in (result or [])
                ]
            except Exception:
                mem = client.memory.get(session_id=session_id)
                texts = [
                    m.content
                    for m in (getattr(mem, "messages", None) or [])
                ]
            latencies.append((time.perf_counter() - t0) * 1000)
            scores.append(keyword_score(texts, keywords))

        return SystemResult(
            system="Zep",
            avg_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            scores=[round(s, 4) for s in scores],
            status="ok",
            notes="session-based memory",
        )

    except Exception as exc:
        return SystemResult(
            system="Zep",
            avg_score=0.0,
            avg_latency_ms=0.0,
            scores=[],
            status="error",
            notes=str(exc)[:80],
        )


# ── Output ─────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[SystemResult]) -> None:
    """Print a side-by-side comparison table using box-drawing characters.

    Args:
        results: List of SystemResult objects, one per system.

    Example:
        print_comparison_table([cogdb_result, chroma_result])
    """
    W_SYS, W_SCR, W_LAT, W_NOT = 18, 14, 14, 20
    # Total inner width (between outer │ chars): column widths + 3 inner separators
    W_INNER = W_SYS + W_SCR + W_LAT + W_NOT + 3  # = 69

    TITLE = "CogDB Memory Benchmark — Comparison Suite"

    def hline(left: str, mid: str, right: str) -> str:
        return (
            left
            + "─" * W_SYS
            + mid
            + "─" * W_SCR
            + mid
            + "─" * W_LAT
            + mid
            + "─" * W_NOT
            + right
        )

    def row(a: str, b: str, c: str, d: str) -> str:
        return (
            f"│ {a:<{W_SYS - 2}} │ {b:^{W_SCR - 2}} │ {c:^{W_LAT - 2}} │ {d:<{W_NOT - 2}} │"
        )

    print()
    print(f"┌{'─' * W_INNER}┐")
    print(f"│ {TITLE:<{W_INNER - 1}}│")
    print(hline("├", "┬", "┤"))
    print(row("System", "Avg Score", "Latency (ms)", "Notes"))
    print(hline("├", "┼", "┤"))

    for r in results:
        if r.status == "ok":
            scr = f"{r.avg_score:.2f}"
            lat = f"{r.avg_latency_ms:.1f}"
        elif r.status in ("skipped", "not_available"):
            scr = "(skipped)"
            lat = "—"
        else:
            scr = "(error)"
            lat = "—"
        print(row(r.system, scr, lat, r.notes[: W_NOT - 2]))

    print(hline("└", "┴", "┘"))

    for r in results:
        if r.system == "CogDB" and r.status == "ok":
            print(f"\n  CogDB extras:")
            if r.avg_token_efficiency is not None:
                print(
                    f"    token_efficiency  : {r.avg_token_efficiency:.3f}"
                    f"  (avg tokens_used / budget {TOKEN_BUDGET})"
                )
            if r.types_retrieved:
                print(f"    types_retrieved   : {', '.join(r.types_retrieved)}")
    print()


def _save_results(results: list[SystemResult], out_path: str) -> None:
    """Serialize results to a JSON file.

    Args:
        results: List of SystemResult objects.
        out_path: Destination file path. Parent directories are created if needed.

    Example:
        _save_results(results, "benchmarks/results/comparison_20260606.json")
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "CogDB Comparison Suite",
        "timestamp": datetime.now().isoformat(),
        "scenario": {
            "memories": len(MEMORIES),
            "queries": len(QUERIES),
            "agent_id": AGENT_ID,
            "token_budget": TOKEN_BUDGET,
        },
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results written to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the comparison benchmark CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="CogDB comparison benchmark — CogDB vs ChromaDB, Mem0, Zep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="JSON output path (default: benchmarks/results/comparison_<timestamp>.json)",
    )
    parser.add_argument("--no-chroma", action="store_true", help="Skip ChromaDB")
    parser.add_argument("--no-mem0", action="store_true", help="Skip Mem0")
    parser.add_argument("--no-zep", action="store_true", help="Skip Zep")
    args = parser.parse_args()

    print(
        f"\n[Comparison Bench] Starting — "
        f"{len(MEMORIES)} memories, {len(QUERIES)} queries, "
        f"agent_id={AGENT_ID!r}"
    )

    results: list[SystemResult] = []

    print("\n[1/4] CogDB")
    results.append(run_cogdb())

    if not args.no_chroma:
        print("\n[2/4] ChromaDB")
        results.append(run_chromadb())
    else:
        results.append(_skipped("ChromaDB", "--no-chroma flag"))

    if not args.no_mem0:
        print("\n[3/4] Mem0")
        results.append(run_mem0())
    else:
        results.append(_skipped("Mem0", "--no-mem0 flag"))

    if not args.no_zep:
        print("\n[4/4] Zep")
        results.append(run_zep())
    else:
        results.append(_skipped("Zep", "--no-zep flag"))

    print_comparison_table(results)

    out_path = args.out
    if out_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(results_dir / f"comparison_{ts}.json")

    _save_results(results, out_path)


if __name__ == "__main__":
    main()

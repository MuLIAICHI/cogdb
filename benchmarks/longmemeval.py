"""CogDB adapter for the LongMemEval benchmark.

LongMemEval (Wu et al., 2024) tests five memory abilities:
  1. Single-session QA  — answer from a single past session
  2. Multi-session QA   — synthesize across sessions
  3. Temporal reasoning — reason about when things happened
  4. Knowledge update   — prefer newer contradicting information
  5. Absence detection  — correctly say "unknown" for missing info

Usage:
    # With synthetic built-in data:
    python -m benchmarks.longmemeval --synthetic

    # With real LongMemEval data (download separately):
    python -m benchmarks.longmemeval --data path/to/longmemeval.json

    # Compare with keyword baseline:
    python -m benchmarks.longmemeval --synthetic --compare-baseline
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class LMEMessage:
    role: str          # "user" | "assistant"
    content: str
    timestamp: str     # ISO 8601


@dataclass
class LMESession:
    session_id: str
    agent_id: str
    messages: list[LMEMessage]
    date: str          # "YYYY-MM-DD"


@dataclass
class LMEQuestion:
    question_id: str
    question: str
    expected_answer: str
    category: str      # "single_session" | "multi_session" | "temporal" | "knowledge_update" | "absence"
    agent_id: str
    related_session_ids: list[str]


@dataclass
class LMEResult:
    question_id: str
    question: str
    category: str
    cogdb_answer: str
    expected_answer: str
    keyword_score: float
    exact_match: bool


# ── Scoring functions ────────────────────────────────────────────────────────

def keyword_score(prediction: str, expected: str) -> float:
    """Token overlap score (recall-oriented).

    Args:
        prediction: Model output string.
        expected: Gold answer string.

    Returns:
        Fraction of expected tokens found in the prediction.

    Example:
        >>> keyword_score("PostgreSQL was the database choice", "PostgreSQL")
        1.0
    """
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not exp_tokens:
        return 1.0
    overlap = pred_tokens & exp_tokens
    return len(overlap) / len(exp_tokens)


def exact_match(prediction: str, expected: str) -> bool:
    """Case-insensitive exact match after stripping punctuation.

    Args:
        prediction: Model output string.
        expected: Gold answer string.

    Returns:
        True if the cleaned strings are identical.

    Example:
        >>> exact_match("PostgreSQL.", "postgresql")
        True
    """
    clean = lambda s: re.sub(r"[^\w\s]", "", s.lower()).strip()
    return clean(prediction) == clean(expected)


# ── Synthetic dataset ────────────────────────────────────────────────────────

def _make_synthetic_dataset() -> tuple[list[LMESession], list[LMEQuestion]]:
    """Build the 5-session, 10-question synthetic dataset.

    Returns:
        Tuple of (sessions, questions).
    """
    agent = "lme-agent"

    sessions = [
        LMESession(
            session_id="s1",
            agent_id=agent,
            date="2026-01-10",
            messages=[
                LMEMessage("user", "I'm starting a new backend project. What database should I pick?", "2026-01-10T09:00:00Z"),
                LMEMessage("assistant", "For a FastAPI project with relational data needs, PostgreSQL is an excellent choice. It offers strong ACID guarantees, rich JSON support, and great ecosystem tooling like SQLAlchemy.", "2026-01-10T09:01:00Z"),
                LMEMessage("user", "Let's go with PostgreSQL then. I'll set up FastAPI with SQLAlchemy ORM.", "2026-01-10T09:03:00Z"),
                LMEMessage("assistant", "Great call. I'll help you scaffold the FastAPI app with SQLAlchemy. We'll use async sessions for better performance under load.", "2026-01-10T09:04:00Z"),
                LMEMessage("user", "Sounds good. Let's also add alembic for migrations.", "2026-01-10T09:06:00Z"),
                LMEMessage("assistant", "Done. Project is set up: FastAPI + PostgreSQL + SQLAlchemy (async) + Alembic. The DB_URL defaults to postgresql+asyncpg://localhost/myapp.", "2026-01-10T09:08:00Z"),
            ],
        ),
        LMESession(
            session_id="s2",
            agent_id=agent,
            date="2026-01-12",
            messages=[
                LMEMessage("user", "We're seeing a strange issue in staging. The app slows to a crawl after a few hours and then crashes with 'QueuePool limit of size 5 overflow 10 reached'.", "2026-01-12T14:00:00Z"),
                LMEMessage("assistant", "That's a connection pool exhaustion error from SQLAlchemy. It means your sessions are not being closed after use, so connections pile up until the pool limit is hit.", "2026-01-12T14:02:00Z"),
                LMEMessage("user", "Why would sessions not close? We're using async context managers.", "2026-01-12T14:04:00Z"),
                LMEMessage("assistant", "Even with context managers, if an unhandled exception exits the block before the session commit/rollback, the connection may be returned in a broken state. The pool then holds it as 'checked out' indefinitely.", "2026-01-12T14:06:00Z"),
                LMEMessage("user", "So the bug is that SQLAlchemy sessions are not closing properly, causing connection pool exhaustion?", "2026-01-12T14:08:00Z"),
                LMEMessage("assistant", "Exactly. This is a known SQLAlchemy pitfall. We need to fix the session lifecycle to guarantee closure even on exceptions.", "2026-01-12T14:09:00Z"),
            ],
        ),
        LMESession(
            session_id="s3",
            agent_id=agent,
            date="2026-01-15",
            messages=[
                LMEMessage("user", "Following up on the pool exhaustion issue — what's the fix?", "2026-01-15T10:00:00Z"),
                LMEMessage("assistant", "Two-part fix: first, set pool_pre_ping=True in your SQLAlchemy engine config — this recycles stale connections before handing them out. Second, always call session.close() explicitly in a finally block, even inside an async context manager.", "2026-01-15T10:02:00Z"),
                LMEMessage("user", "So pool_pre_ping=True plus explicit session.close()?", "2026-01-15T10:03:00Z"),
                LMEMessage("assistant", "Yes. pool_pre_ping=True guards against dead connections at checkout; session.close() ensures the connection is released back to the pool immediately after use, not left dangling.", "2026-01-15T10:05:00Z"),
                LMEMessage("user", "Applied both fixes. Staging has been running clean for 2 days with no pool errors.", "2026-01-15T10:30:00Z"),
                LMEMessage("assistant", "Excellent. Those two together fully resolve connection pool exhaustion in this pattern.", "2026-01-15T10:31:00Z"),
            ],
        ),
        LMESession(
            session_id="s4",
            agent_id=agent,
            date="2026-01-20",
            messages=[
                LMEMessage("user", "Ready to deploy the app. What's the recommended infrastructure for a FastAPI app?", "2026-01-20T09:00:00Z"),
                LMEMessage("assistant", "For initial deployment, AWS EC2 t3.medium with Docker Compose is a solid choice. It gives you full control, predictable pricing, and easy SSH access for debugging.", "2026-01-20T09:02:00Z"),
                LMEMessage("user", "Let's go with EC2 t3.medium and Docker Compose then.", "2026-01-20T09:04:00Z"),
                LMEMessage("assistant", "I'll help you write the docker-compose.yml and set up the EC2 instance. We'll expose port 8000 via an Nginx reverse proxy and store secrets in AWS Secrets Manager.", "2026-01-20T09:06:00Z"),
                LMEMessage("user", "Deployed! The app is live on EC2 t3.medium with Docker Compose.", "2026-01-20T11:00:00Z"),
                LMEMessage("assistant", "Great. Monitor CloudWatch metrics for CPU and memory usage. If you see sustained high CPU, consider moving to a t3.large or switching architectures.", "2026-01-20T11:01:00Z"),
            ],
        ),
        LMESession(
            session_id="s5",
            agent_id=agent,
            date="2026-01-25",
            messages=[
                LMEMessage("user", "The EC2 instance is sitting mostly idle but we're still paying ~$60/month. Looking at cost savings.", "2026-01-25T09:00:00Z"),
                LMEMessage("assistant", "For a mostly-idle FastAPI service, AWS Lambda with Mangum is a great fit. You pay per request instead of per hour, so idle time costs nothing. Trade-off is cold starts.", "2026-01-25T09:02:00Z"),
                LMEMessage("user", "What's the cold start like for our app?", "2026-01-25T09:04:00Z"),
                LMEMessage("assistant", "With a slim Docker image and lazy imports, expect around 1.2 seconds for a cold start. Warm requests respond in the same sub-10ms range as EC2.", "2026-01-25T09:06:00Z"),
                LMEMessage("user", "1.2s cold start is acceptable for our use case. Let's migrate from EC2 to Lambda for cost savings.", "2026-01-25T10:00:00Z"),
                LMEMessage("assistant", "Migration complete. The app now runs on AWS Lambda with Mangum. Cold start is 1.2s; warm invocations are sub-10ms. EC2 instance has been stopped.", "2026-01-25T12:00:00Z"),
            ],
        ),
    ]

    questions = [
        LMEQuestion(
            question_id="q1",
            question="What database did we choose for the project?",
            expected_answer="PostgreSQL",
            category="single_session",
            agent_id=agent,
            related_session_ids=["s1"],
        ),
        LMEQuestion(
            question_id="q2",
            question="What was the SQLAlchemy fix for pool exhaustion?",
            expected_answer="pool_pre_ping=True and explicit session.close()",
            category="single_session",
            agent_id=agent,
            related_session_ids=["s3"],
        ),
        LMEQuestion(
            question_id="q3",
            question="What sequence of events led to the Lambda migration?",
            expected_answer="The app was deployed on EC2 t3.medium with Docker Compose, but the instance was mostly idle and cost $60/month, so we migrated to AWS Lambda for cost savings",
            category="multi_session",
            agent_id=agent,
            related_session_ids=["s4", "s5"],
        ),
        LMEQuestion(
            question_id="q4",
            question="When was the connection pool bug discovered?",
            expected_answer="2026-01-12",
            category="temporal",
            agent_id=agent,
            related_session_ids=["s2"],
        ),
        LMEQuestion(
            question_id="q5",
            question="Was the Lambda deployment before or after the pool fix?",
            expected_answer="after",
            category="temporal",
            agent_id=agent,
            related_session_ids=["s3", "s5"],
        ),
        LMEQuestion(
            question_id="q6",
            question="What is the current deployment platform?",
            expected_answer="AWS Lambda",
            category="knowledge_update",
            agent_id=agent,
            related_session_ids=["s4", "s5"],
        ),
        LMEQuestion(
            question_id="q7",
            question="What was the initial deployment, and was it changed?",
            expected_answer="EC2, later changed to Lambda",
            category="knowledge_update",
            agent_id=agent,
            related_session_ids=["s4", "s5"],
        ),
        LMEQuestion(
            question_id="q8",
            question="What monitoring tool do we use?",
            expected_answer="I don't know",
            category="absence",
            agent_id=agent,
            related_session_ids=[],
        ),
        LMEQuestion(
            question_id="q9",
            question="What is the team size?",
            expected_answer="I don't know",
            category="absence",
            agent_id=agent,
            related_session_ids=[],
        ),
        LMEQuestion(
            question_id="q10",
            question="What infrastructure cost concern drove the architecture change?",
            expected_answer="cost savings",
            category="multi_session",
            agent_id=agent,
            related_session_ids=["s5"],
        ),
    ]

    return sessions, questions


# ── Keyword baseline ─────────────────────────────────────────────────────────

class KeywordBaselineRunner:
    """Simple keyword-search baseline that doesn't use CogDB.

    Stores all messages in memory and answers by finding the message
    with the highest token overlap with the question.

    Args:
        sessions: Sessions to index.
    """

    def __init__(self) -> None:
        self._corpus: list[tuple[str, str]] = []  # (session_date, content)

    def load_sessions(self, sessions: list[LMESession]) -> None:
        """Index all messages from sessions.

        Args:
            sessions: LMESession objects to index.

        Returns:
            None

        Example:
            >>> runner = KeywordBaselineRunner()
            >>> runner.load_sessions(sessions)
        """
        for session in sessions:
            for msg in session.messages:
                self._corpus.append((session.date, f"[{session.date}] {msg.role}: {msg.content}"))

    def answer_question(self, question: LMEQuestion) -> str:
        """Answer by returning the corpus entry with highest keyword overlap.

        Args:
            question: The question to answer.

        Returns:
            The most relevant corpus snippet, or "I don't know".

        Example:
            >>> answer = runner.answer_question(q)
        """
        if not self._corpus:
            return "I don't know"

        q_tokens = set(question.question.lower().split())
        best_score = 0.0
        best_content = ""
        for _, content in self._corpus:
            c_tokens = set(content.lower().split())
            if not c_tokens:
                continue
            score = len(q_tokens & c_tokens) / len(q_tokens)
            if score > best_score:
                best_score = score
                best_content = content

        if best_score < 0.1:
            return "I don't know"
        return best_content

    def run(self, questions: list[LMEQuestion]) -> list[LMEResult]:
        """Run all questions and return scored results.

        Args:
            questions: Questions to evaluate.

        Returns:
            List of LMEResult objects.

        Example:
            >>> results = runner.run(questions)
        """
        results = []
        for q in questions:
            answer = self.answer_question(q)
            ks = keyword_score(answer, q.expected_answer)
            em = exact_match(answer, q.expected_answer)
            results.append(LMEResult(
                question_id=q.question_id,
                question=q.question,
                category=q.category,
                cogdb_answer=answer,
                expected_answer=q.expected_answer,
                keyword_score=ks,
                exact_match=em,
            ))
        return results


# ── CogDB runner ─────────────────────────────────────────────────────────────

class CogDBLongMemEvalRunner:
    """Run the LongMemEval benchmark using CogDB as the memory backend.

    Args:
        db_path: Path to the CogDB storage directory. Uses a temp dir if None.
        token_budget: Max tokens for each recall query.

    Example:
        >>> runner = CogDBLongMemEvalRunner(token_budget=1000)
        >>> runner.load_sessions(sessions)
        >>> results = runner.run(questions)
        >>> runner.print_report(results)
        >>> runner.close()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_budget: int = 1000,
    ) -> None:
        from cogdb.core import CognitiveDB

        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        if db_path is None:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="cogdb_lme_")
            db_path = self._tmp_dir.name

        self._db = CognitiveDB(db_path=db_path)
        self._token_budget = token_budget

    def load_sessions(self, sessions: list[LMESession]) -> None:
        """Store all session messages in CogDB episodic memory.

        Assistant messages are stored with importance 0.6 (answers are
        more information-dense); user messages with 0.5.

        Args:
            sessions: LMESession objects to load.

        Returns:
            None

        Example:
            >>> runner.load_sessions(sessions)
        """
        for session in sessions:
            for msg in session.messages:
                importance = 0.6 if msg.role == "assistant" else 0.5
                content = f"[{session.date}] {msg.role}: {msg.content}"
                self._db.remember(
                    content,
                    agent_id=session.agent_id,
                    importance=importance,
                    metadata={
                        "session_id": session.session_id,
                        "role": msg.role,
                        "date": session.date,
                        "timestamp": msg.timestamp,
                    },
                )

    def load_from_file(self, path: str) -> tuple[list[LMESession], list[LMEQuestion]]:
        """Load sessions and questions from a LongMemEval JSON file.

        Expected format::

            {
                "sessions": [
                    {
                        "session_id": "s1",
                        "agent_id": "agent",
                        "date": "2026-01-10",
                        "messages": [
                            {"role": "user", "content": "...", "timestamp": "..."},
                            ...
                        ]
                    }
                ],
                "questions": [
                    {
                        "question_id": "q1",
                        "question": "...",
                        "expected_answer": "...",
                        "category": "single_session",
                        "agent_id": "agent",
                        "related_session_ids": ["s1"]
                    }
                ]
            }

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            Tuple of (sessions, questions).

        Example:
            >>> sessions, questions = runner.load_from_file("lme.json")
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sessions = [
            LMESession(
                session_id=s["session_id"],
                agent_id=s["agent_id"],
                date=s["date"],
                messages=[
                    LMEMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
                    )
                    for m in s["messages"]
                ],
            )
            for s in data.get("sessions", [])
        ]

        questions = [
            LMEQuestion(
                question_id=q["question_id"],
                question=q["question"],
                expected_answer=q["expected_answer"],
                category=q["category"],
                agent_id=q["agent_id"],
                related_session_ids=q.get("related_session_ids", []),
            )
            for q in data.get("questions", [])
        ]

        return sessions, questions

    def answer_question(self, question: LMEQuestion) -> str:
        """Retrieve context from CogDB and return the best answer.

        Queries episodic memory using the question text as the query,
        then returns the content of the top result. Returns "I don't know"
        if no memories are found above the relevance threshold.

        Args:
            question: The LMEQuestion to answer.

        Returns:
            A string answer derived from retrieved memories.

        Example:
            >>> answer = runner.answer_question(q)
        """
        memories = self._db.recall(
            query=question.question,
            agent_id=question.agent_id,
            token_budget=self._token_budget,
        )
        if not memories:
            return "I don't know"

        # Return the content of the top-ranked memory (already sorted by importance)
        return memories[0].content

    def run(self, questions: list[LMEQuestion]) -> list[LMEResult]:
        """Run all questions against CogDB and return scored results.

        Args:
            questions: Questions to evaluate.

        Returns:
            List of LMEResult with keyword_score and exact_match filled in.

        Example:
            >>> results = runner.run(questions)
        """
        results = []
        for q in questions:
            answer = self.answer_question(q)
            ks = keyword_score(answer, q.expected_answer)
            em = exact_match(answer, q.expected_answer)
            results.append(LMEResult(
                question_id=q.question_id,
                question=q.question,
                category=q.category,
                cogdb_answer=answer,
                expected_answer=q.expected_answer,
                keyword_score=ks,
                exact_match=em,
            ))
        return results

    def score(self, results: list[LMEResult]) -> dict:
        """Compute per-category and overall keyword scores.

        Args:
            results: List of LMEResult objects from run().

        Returns:
            Dict with "overall" and one key per category, each 0.0–1.0.

        Example:
            >>> scores = runner.score(results)
            >>> print(scores["overall"])
        """
        categories: dict[str, list[float]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r.keyword_score)

        scores: dict[str, float] = {}
        all_scores: list[float] = []
        for cat, cat_scores in sorted(categories.items()):
            avg = sum(cat_scores) / len(cat_scores)
            scores[cat] = round(avg, 4)
            all_scores.extend(cat_scores)

        scores["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
        return scores

    def print_report(self, results: list[LMEResult], label: str = "CogDB") -> None:
        """Print a formatted benchmark report to stdout.

        Args:
            results: List of LMEResult objects from run().
            label: Runner name shown in the report header.

        Returns:
            None

        Example:
            >>> runner.print_report(results)
        """
        scores = self.score(results)
        width = 60

        print(f"\n{'=' * width}")
        print(f"  LongMemEval Results — {label}")
        print(f"{'=' * width}")

        print(f"\n{'Category':<25} {'Keyword Score':>14} {'N':>4}")
        print(f"{'-' * 45}")

        categories: dict[str, list[LMEResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        for cat in sorted(categories):
            cat_results = categories[cat]
            avg_ks = sum(r.keyword_score for r in cat_results) / len(cat_results)
            print(f"  {cat:<23} {avg_ks:>13.3f} {len(cat_results):>4}")

        print(f"{'-' * 45}")
        print(f"  {'OVERALL':<23} {scores['overall']:>13.3f} {len(results):>4}")

        print(f"\n{'Per-question detail':}")
        print(f"{'-' * width}")
        for r in results:
            em_mark = "✓" if r.exact_match else " "
            print(f"  [{em_mark}] {r.question_id:>3}  ks={r.keyword_score:.2f}  {r.category}")
            print(f"       Q: {r.question[:55]}")
            print(f"       A: {r.cogdb_answer[:55]}")
            print(f"       E: {r.expected_answer[:55]}")

        print(f"\n{'=' * width}\n")

    def close(self) -> None:
        """Close the CogDB connection and release any temp directory.

        Returns:
            None

        Example:
            >>> runner.close()
        """
        self._db.close() if hasattr(self._db, "close") else None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CogDB LongMemEval Adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--synthetic", action="store_true", help="Use built-in synthetic dataset")
    parser.add_argument("--data", type=str, help="Path to LongMemEval JSON file")
    parser.add_argument("--db-path", type=str, default=None, help="CogDB storage path (temp if not set)")
    parser.add_argument("--token-budget", type=int, default=1000, help="Token budget per recall query")
    parser.add_argument("--out", type=str, help="Save results JSON to this path")
    parser.add_argument("--compare-baseline", action="store_true", help="Also run keyword search baseline")
    args = parser.parse_args()

    if not args.synthetic and not args.data:
        parser.error("Provide --synthetic or --data <path>")

    # Load dataset
    runner = CogDBLongMemEvalRunner(db_path=args.db_path, token_budget=args.token_budget)

    if args.synthetic:
        sessions, questions = _make_synthetic_dataset()
        print(f"Loaded synthetic dataset: {len(sessions)} sessions, {len(questions)} questions")
    else:
        sessions, questions = runner.load_from_file(args.data)
        print(f"Loaded {len(sessions)} sessions, {len(questions)} questions from {args.data}")

    # Store sessions in CogDB
    print("Storing sessions in CogDB...")
    runner.load_sessions(sessions)

    # Run CogDB evaluation
    print("Evaluating questions...")
    results = runner.run(questions)
    runner.print_report(results, label="CogDB")

    # Keyword baseline comparison
    if args.compare_baseline:
        baseline = KeywordBaselineRunner()
        baseline.load_sessions(sessions)
        baseline_results = baseline.run(questions)

        # Reuse print_report via a temporary runner (no db needed)
        _print_baseline_report(baseline_results)

    # Save results
    if args.out:
        output = {
            "runner": "CogDB",
            "token_budget": args.token_budget,
            "scores": runner.score(results),
            "results": [asdict(r) for r in results],
        }
        if args.compare_baseline:
            baseline_runner_tmp = CogDBLongMemEvalRunner.__new__(CogDBLongMemEvalRunner)
            output["baseline_scores"] = _compute_scores(baseline_results)
            output["baseline_results"] = [asdict(r) for r in baseline_results]

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.out}")

    runner.close()


def _compute_scores(results: list[LMEResult]) -> dict:
    """Compute scores from a result list (standalone, no runner needed).

    Args:
        results: LMEResult objects to score.

    Returns:
        Score dict identical to CogDBLongMemEvalRunner.score().
    """
    categories: dict[str, list[float]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r.keyword_score)

    scores: dict[str, float] = {}
    all_scores: list[float] = []
    for cat, cat_scores in sorted(categories.items()):
        avg = sum(cat_scores) / len(cat_scores)
        scores[cat] = round(avg, 4)
        all_scores.extend(cat_scores)

    scores["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    return scores


def _print_baseline_report(results: list[LMEResult]) -> None:
    """Print a report for the keyword baseline without a runner instance.

    Args:
        results: Baseline LMEResult objects.
    """
    scores = _compute_scores(results)
    width = 60
    print(f"\n{'=' * width}")
    print("  LongMemEval Results — Keyword Baseline")
    print(f"{'=' * width}")

    categories: dict[str, list[LMEResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print(f"\n{'Category':<25} {'Keyword Score':>14} {'N':>4}")
    print(f"{'-' * 45}")
    for cat in sorted(categories):
        cat_results = categories[cat]
        avg_ks = sum(r.keyword_score for r in cat_results) / len(cat_results)
        print(f"  {cat:<23} {avg_ks:>13.3f} {len(cat_results):>4}")

    print(f"{'-' * 45}")
    print(f"  {'OVERALL':<23} {scores['overall']:>13.3f} {len(results):>4}")
    print(f"\n{'=' * width}\n")


if __name__ == "__main__":
    main()

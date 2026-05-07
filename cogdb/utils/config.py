"""Configuration for CogDB instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CogDBConfig:
    """Configuration for a CogDB instance.

    Args:
        db_path: Root directory for all CogDB storage files.
        embedding_model: Sentence-transformers model name for embeddings.
        embedding_dim: Dimensionality of the embedding model output.
        default_token_budget: Default max tokens for retrieval responses.
        l0_token_budget: Token budget for L0 (identity) context.
        l1_token_budget: Token budget for L1 (critical facts) context.
        l2_token_budget: Token budget for L2 (task-relevant) context.
        decay_half_life_hours: Hours until a memory's decay score halves.
        consolidation_threshold: Min episodic memories before consolidation triggers.
        contradiction_check: Whether to check for contradicting facts on write.

    Example:
        >>> config = CogDBConfig(db_path="./my_agent_memory")
        >>> db = CognitiveDB(config=config)
    """

    db_path: str = "./cogdb_data"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    default_token_budget: int = 1000
    l0_token_budget: int = 50
    l1_token_budget: int = 200
    l2_token_budget: int = 500
    decay_half_life_hours: float = 168.0  # 1 week
    consolidation_threshold: int = 10
    contradiction_check: bool = True
    max_results_per_store: int = 20

    # Retrieval blending: weight given to HNSW similarity rank in final sort.
    # 0.0 = pure importance ranking; 0.2 = 20% HNSW relevance + 80% importance.
    hnsw_blend_alpha: float = 0.2

    # Maximum number of procedural memories included per recall() call.
    # Keeps procedures from crowding out episodic memories in the token budget.
    # 1 is the right default: the most relevant procedure is selected via
    # match-count sorting; additional procedures displace episodic memories.
    max_procedures_per_query: int = 1

    # ChromaDB settings (kept for config compatibility)
    chroma_collection_name: str = "cogdb_episodic"

    # Agent identity defaults
    default_agent_id: str = "default"

    # Consolidation: enable optional LLM-powered SPO extraction (requires API key).
    use_llm_consolidation: bool = False

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        root = Path(self.db_path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "semantic").mkdir(exist_ok=True)
        (root / "procedural").mkdir(exist_ok=True)

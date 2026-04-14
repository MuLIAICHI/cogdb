"""Token counting and budget management utilities.

Uses tiktoken for accurate token counting. Falls back to
a word-based approximation if tiktoken is unavailable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_encoder() -> Optional[tiktoken.Encoding]:
    """Get a cached tiktoken encoder (cl100k_base, used by GPT-4/Claude)."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text string.

    Uses tiktoken's cl100k_base encoding for accuracy.
    Falls back to len(text) // 4 if tiktoken unavailable.

    Args:
        text: The text to count tokens for.

    Returns:
        Approximate token count.

    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    # Rough approximation: ~4 chars per token for English
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to fit within a token budget.

    Truncates at token boundaries (not character boundaries)
    and appends '...' if truncated.

    Args:
        text: The text to truncate.
        token_budget: Maximum tokens allowed.

    Returns:
        Text truncated to fit within the budget.

    Example:
        >>> truncate_to_budget("A very long text...", token_budget=5)
        'A very long...'
    """
    if token_budget <= 0:
        return ""

    current_count = count_tokens(text)
    if current_count <= token_budget:
        return text

    encoder = _get_encoder()
    if encoder is not None:
        tokens = encoder.encode(text)
        truncated_tokens = tokens[: token_budget - 1]  # Leave room for "..."
        return encoder.decode(truncated_tokens) + "..."

    # Fallback: character-based approximation
    char_budget = token_budget * 4
    return text[: char_budget - 3] + "..."


def fits_budget(text: str, token_budget: int) -> bool:
    """Check if text fits within a token budget.

    Args:
        text: The text to check.
        token_budget: Maximum tokens allowed.

    Returns:
        True if the text fits within the budget.
    """
    return count_tokens(text) <= token_budget

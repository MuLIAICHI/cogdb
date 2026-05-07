"""Singleton Engine cache — one Rust Engine per db_path.

All three Python stores (EpisodicStore, SemanticStore, ProceduralStore) share
a single PyEngine for the same db_path to avoid WAL and file-lock conflicts.
"""
from __future__ import annotations

import os
import threading
from typing import Any

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _key(db_path: str) -> str:
    """Canonical, normalized absolute path — handles Windows case-insensitivity."""
    return os.path.normcase(os.path.realpath(os.path.abspath(db_path)))


def get_engine(
    db_path: str,
    embedding_dim: int = 384,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 200,
    contradiction_check: bool = True,
) -> Any:
    """Return the existing PyEngine for db_path, or create one."""
    from cogdb_engine import PyEngine  # deferred import — extension may not be built yet

    k = _key(db_path)
    with _lock:
        if k not in _cache:
            _cache[k] = PyEngine(
                k,
                embedding_dim,
                hnsw_m,
                hnsw_ef_construction,
                contradiction_check,
            )
        return _cache[k]


def release_engine(db_path: str) -> None:
    """Close and evict the engine for db_path.

    Called by _ClientShim.reset() during test teardown.  After this returns,
    all SQLite file handles are closed so shutil.rmtree succeeds on Windows.
    """
    k = _key(db_path)
    with _lock:
        engine = _cache.pop(k, None)
    if engine is not None:
        try:
            engine.close()
        except Exception:
            pass

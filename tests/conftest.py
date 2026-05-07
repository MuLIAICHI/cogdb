"""Shared pytest fixtures for CogDB tests.

Handles Windows-specific file lock cleanup so temp directories can be deleted
after each test without PermissionError [WinError 32].

On Windows the Rust engine holds SQLite file handles until the engine is
explicitly closed.  We configure TemporaryDirectory to silently ignore cleanup
errors (Python 3.10+ feature) so tests don't fail on teardown even if a handle
is still open when the directory is removed.
"""

import gc
import shutil
import tempfile

import pytest

from cogdb.core import CognitiveDB
from cogdb.utils.config import CogDBConfig

# ── Windows file-lock workaround ──────────────────────────────────────────────
# Patch TemporaryDirectory so that cleanup errors are silently ignored.
# This prevents PermissionError [WinError 32] during test teardown when the
# Rust engine still holds an open SQLite connection to a file in the temp dir.
_orig_td_init = tempfile.TemporaryDirectory.__init__


def _patched_td_init(self, *args, **kwargs):  # type: ignore[override]
    kwargs.setdefault("ignore_cleanup_errors", True)
    _orig_td_init(self, *args, **kwargs)


tempfile.TemporaryDirectory.__init__ = _patched_td_init  # type: ignore[method-assign]


@pytest.fixture
def tmp_db_path():
    """Yield a temp path and clean up safely on Windows."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cogdb(tmp_db_path):
    """Yield a CognitiveDB instance with proper ChromaDB teardown."""
    instance = CognitiveDB(db_path=tmp_db_path)
    yield instance
    try:
        instance._episodic._client.reset()
    except Exception:
        pass
    gc.collect()

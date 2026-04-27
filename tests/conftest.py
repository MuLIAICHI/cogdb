"""Shared pytest fixtures for CogDB tests.

Handles Windows-specific ChromaDB file lock cleanup so temp directories
can be deleted after each test without PermissionError [WinError 32].
"""

import gc
import shutil
import tempfile

import pytest

from cogdb.core import CognitiveDB
from cogdb.utils.config import CogDBConfig


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

"""CogDB framework adapters — lazy imports to avoid hard dependencies."""

from __future__ import annotations

__all__: list[str] = []

try:
    from cogdb.adapters.autogen import CogDBMemory

    __all__.append("CogDBMemory")
except ImportError:
    pass

try:
    from cogdb.adapters.crewai import CogDBCrewAIStorage

    __all__.append("CogDBCrewAIStorage")
except ImportError:
    pass

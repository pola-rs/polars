"""
Resolve engine arguments and configured affinity.

Engine implementations must not import this module. `GPUEngine` is re-exported
for compatibility with its documented import path.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Final

from polars.lazyframe.engine import (
    Engine,
    GPUEngine,
    InMemoryEngine,
    StreamingEngine,
    _AutoEngine,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import get_engine_affinity

if TYPE_CHECKING:
    from polars._typing import EngineType, EngineTypeName


SUPPORTED_ENGINE_NAMES: Final[tuple[str, ...]] = (
    "auto",
    "in-memory",
    "streaming",
    "gpu",
)


_AUTO_ENGINE: Final = _AutoEngine()
_IN_MEMORY_ENGINE: Final = InMemoryEngine()
_STREAMING_ENGINE: Final = StreamingEngine()

_ENGINE_BY_NAME: Final[dict[str, Engine]] = {
    "auto": _AUTO_ENGINE,
    "in-memory": _IN_MEMORY_ENGINE,
    # Legacy alias.
    "cpu": _IN_MEMORY_ENGINE,
    "streaming": _STREAMING_ENGINE,
}

# Engine objects cannot be stored in `POLARS_ENGINE_AFFINITY`; `Config` keeps
# the object and name forms mutually exclusive.
_ENGINE_AFFINITY_OVERRIDE: Engine | None = None


def get_engine_affinity_override() -> Engine | None:
    """Return the configured engine override."""
    return _ENGINE_AFFINITY_OVERRIDE


def set_engine_affinity_override(engine: Engine | None) -> None:
    """Set the configured engine override."""
    global _ENGINE_AFFINITY_OVERRIDE
    _ENGINE_AFFINITY_OVERRIDE = engine


def _eager_engine() -> Engine:
    """Return the engine used for internal eager operations."""
    return _IN_MEMORY_ENGINE


def _engine_from_name(engine: EngineTypeName) -> Engine:
    """Resolve an explicit engine name without applying the configured affinity."""
    if engine == "gpu":
        return GPUEngine()

    if (selected := _ENGINE_BY_NAME.get(engine)) is None:
        msg = f"Invalid engine argument {engine=}"
        raise ValueError(msg)
    return selected


def _select_engine(engine: EngineType) -> Engine:
    """
    Resolve an engine argument or configured affinity to an `Engine`.

    An `"auto"` affinity remains unresolved for Rust to select at execution time.
    """
    if isinstance(engine, Engine):
        return engine

    if engine == "auto":
        if _ENGINE_AFFINITY_OVERRIDE is not None:
            return _ENGINE_AFFINITY_OVERRIDE
        engine = get_engine_affinity()

    return _engine_from_name(engine)


__all__ = [
    "GPUEngine",
    "SUPPORTED_ENGINE_NAMES",
    "get_engine_affinity_override",
    "set_engine_affinity_override",
]

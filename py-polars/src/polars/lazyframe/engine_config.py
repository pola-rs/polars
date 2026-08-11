"""
Engine selection.

Resolves an `engine=` argument, or the configured engine affinity, to one of the
engine classes in `polars.lazyframe.engine`. This is a strict consumer of that
module -- nothing there may import this one.

`GPUEngine` is re-exported here because `polars.lazyframe.engine_config` is its
documented import path.
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
    from polars._typing import EngineType


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
    # kept for backwards compatibility
    "cpu": _IN_MEMORY_ENGINE,
    "streaming": _STREAMING_ENGINE,
}

# A configured engine cannot be represented by `POLARS_ENGINE_AFFINITY`, which holds
# only a name, so an object-valued affinity is held here instead. The two are kept
# mutually exclusive by `Config.set_engine_affinity`.
_ENGINE_AFFINITY_OVERRIDE: Engine | None = None


def get_engine_affinity_override() -> Engine | None:
    """Return the object-valued default engine, if one is configured."""
    return _ENGINE_AFFINITY_OVERRIDE


def set_engine_affinity_override(engine: Engine | None) -> None:
    """Set (or clear, with `None`) the object-valued default engine."""
    global _ENGINE_AFFINITY_OVERRIDE
    _ENGINE_AFFINITY_OVERRIDE = engine


def _select_engine(engine: EngineType) -> Engine:
    """
    Resolve an `engine` argument to an `Engine` instance.

    `"auto"` is resolved against the engine affinity, which may be a configured
    engine object, or a name -- including `"auto"` itself, which Rust resolves at
    execution time.
    """
    if isinstance(engine, Engine):
        return engine

    if engine == "auto":
        if _ENGINE_AFFINITY_OVERRIDE is not None:
            return _ENGINE_AFFINITY_OVERRIDE
        engine = get_engine_affinity()

    if engine == "gpu":
        return GPUEngine()

    if (selected := _ENGINE_BY_NAME.get(engine)) is None:
        msg = f"Invalid engine argument {engine=}"
        raise ValueError(msg)
    return selected


__all__ = [
    "GPUEngine",
    "SUPPORTED_ENGINE_NAMES",
    "get_engine_affinity_override",
    "set_engine_affinity_override",
]

"""
Resolve engine arguments and configured affinity.

Engine implementations must not import this module. `GPUEngine` is re-exported
for compatibility with its documented import path.
"""

from __future__ import annotations

import contextlib
from importlib.metadata import entry_points
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

# The entry point group a package registers an `Engine` subclass under. The
# entry point's name is the engine name; its value is the class, or any callable
# that takes no arguments and returns an engine.
ENGINE_ENTRY_POINT_GROUP: Final = "polars.engines"

# One instance per registered name, created on first use.
_REGISTERED_ENGINES: dict[str, Engine] = {}

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


def registered_engine(name: str) -> Engine | None:
    """
    Return the engine an installed package registered under `name`, if any.

    A package registers an engine through an entry point in the
    `polars.engines` group. The engine is created without arguments the first
    time its name is used and the same instance is returned afterwards.
    """
    if (engine := _REGISTERED_ENGINES.get(name)) is not None:
        return engine

    registered = tuple(entry_points(group=ENGINE_ENTRY_POINT_GROUP, name=name))
    if not registered:
        return None
    if len(registered) > 1:
        sources = sorted(entry_point.value for entry_point in registered)
        msg = (
            f"more than one installed package registers the engine {name!r}: {sources}"
        )
        raise ValueError(msg)

    engine = registered[0].load()()
    if not isinstance(engine, Engine):
        msg = f"the entry point for the engine {name!r} must create a `polars.Engine`, got {type(engine).__name__!r}"
        raise TypeError(msg)
    _REGISTERED_ENGINES[name] = engine
    return engine


def _engine_from_name(engine: EngineTypeName | str) -> Engine:
    """Resolve an explicit engine name without applying the configured affinity."""
    if engine == "gpu":
        return GPUEngine()

    if (selected := _ENGINE_BY_NAME.get(engine)) is not None:
        return selected
    # Built-in names are resolved first, so a package cannot replace one.
    if (selected := registered_engine(engine)) is not None:
        return selected
    msg = f"Invalid engine argument {engine=}"
    raise ValueError(msg)


def _validate_engine(engine: EngineType) -> None:
    """Validate an engine argument without resolving the configured affinity."""
    if isinstance(engine, Engine):
        return
    _engine_from_name(engine)


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
    "ENGINE_ENTRY_POINT_GROUP",
    "GPUEngine",
    "SUPPORTED_ENGINE_NAMES",
    "get_engine_affinity_override",
    "registered_engine",
    "set_engine_affinity_override",
]

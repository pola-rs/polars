from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T")


def expired_error(
    obj: T, name: str, *, version: str = "2.0", hint: str | None = None
) -> AttributeError:
    """
    Raise an `AttributeError` for a removed attribute.

    Parameters
    ----------
    obj
        The object from which the attribute was removed.
    name
        The name of the removed attribute.
    version
        The version in which the attribute was removed.
    hint
        A hint at what to use instead of the removed attribute.
    """
    msg = f"`{name}` was removed in version {version}"
    msg = f"{msg}." if hint is None else f"{msg}; {hint}"
    return AttributeError(msg, name=name, obj=obj)


def expired_fallthrough(obj: T, name: str) -> AttributeError:
    """
    Raise an `AttributeError` for a non-existent attribute.

    Parameters
    ----------
    obj
        The object on which the attribute was accessed.
    name
        The name of the non-existent attribute.
    """
    msg = f"{type(obj).__name__!r} object has no attribute {name!r}"
    return AttributeError(msg, name=name, obj=obj)

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T")


def raise_expired_error(
    obj: T, name: str, *, version: str = "2.0", hint: str | None = None
) -> NoReturn:
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
    raise AttributeError(msg, name=name, obj=obj)


def expired_fallthrough(obj: T, name: str) -> Any:
    """
    Raise an `AttributeError` for a non-existent attribute.

    Parameters
    ----------
    obj
        The object on which the attribute was accessed.
    name
        The name of the non-existent attribute.
    """
    if fallback := getattr(super(obj.__class__, obj), "__getattr__", None) is not None:
        return fallback(name)
    else:
        msg = f"{type(obj).__name__!r} object has no attribute {name!r}"
        raise AttributeError(msg, name=name, obj=obj)

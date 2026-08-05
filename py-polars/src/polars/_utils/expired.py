from __future__ import annotations

from typing import TYPE_CHECKING

from polars.exceptions import AttributeRemovedError

if TYPE_CHECKING:
    from typing import Any, NoReturn


def raise_item_removed_error(
    obj: object, name: str, *, version: str = "2.0", hint: str | None = None
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
    raise AttributeRemovedError(msg, name=name, obj=obj)


def getattr_fallback(obj: object, superclass: object, name: str) -> Any:
    """
    Raise an `AttributeError` for a non-existent attribute.

    Parameters
    ----------
    obj
        The object on which the attribute was accessed.
    superclass
        The superclass of the object used to attempt to access the attribute.
    name
        The name of the non-existent attribute.
    """
    if (super_getattr := getattr(superclass, "__getattr__", None)) is not None:
        return super_getattr(name)
    else:
        msg = f"{type(obj).__name__!r} object has no attribute {name!r}"
        raise AttributeError(msg, name=name, obj=obj)

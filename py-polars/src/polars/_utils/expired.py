from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from polars.exceptions import AttributeRemovedError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ParamSpec

    from polars._utils.various import IdentityFunction

    P = ParamSpec("P")
    T = TypeVar("T")


def raise_for_removed_attributes(
    obj: object,
    name: str,
    attributes: dict[str, str | None],
    *,
    version: str,
) -> None:
    """
    Raise an `AttributeError` for a removed attribute.

    Parameters
    ----------
    obj
        The object from which the attribute was removed.
    name
        The name of the removed attribute.
    attributes
        A dictionary mapping removed attribute names to hints for replacement.
    version
        The version in which the attribute was removed.
    """
    if name in attributes:
        hint = attributes[name]
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


def removed_renamed_parameter(
    old_name: str,
    new_name: str,
    *,
    deprecated_in: str | None = None,
    removed_in: str,
) -> IdentityFunction:
    """
    Decorator to mark a function parameter as removed due to being renamed.

    Use as follows:

        @removed_renamed_parameter("old_name", new_name="new_name")
        def myfunc(new_name): ...
    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _removed_keyword_argument(
                old_name=old_name,
                new_name=new_name,
                kwargs=kwargs,
                func_name=function.__qualname__,
                deprecated_version=deprecated_in,
                removed_version=removed_in,
            )
            return function(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _removed_keyword_argument(
    *,
    old_name: str,
    new_name: str,
    kwargs: dict[str, object],
    func_name: str,
    deprecated_version: str | None = None,
    removed_version: str,
) -> None:
    """Rename a keyword argument of a function."""
    if old_name in kwargs:
        deprecated_and = (
            f"was deprecated in version {deprecated_version} and "
            if deprecated_version is not None
            else ""
        )
        if new_name in kwargs:
            msg = (
                f"`{func_name!r}` received both `{old_name!r}` and `{new_name!r}` as arguments;"
                f" `{old_name!r}` {deprecated_and}has been removed in version {removed_version},"
                f" use `{new_name!r}` instead"
            )
            raise TypeError(msg)

        msg = (
            f"the argument `{old_name}` for `{func_name}` {deprecated_and}has been removed in {removed_version}. "
            f"It was renamed to `{new_name}`."
        )
        raise TypeError(msg)

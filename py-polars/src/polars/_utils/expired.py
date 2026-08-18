from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, NoReturn, TypeVar

from polars.exceptions import ArgumentRemovedError, AttributeRemovedError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ParamSpec

    from polars._utils.various import IdentityFunction

    P = ParamSpec("P")
    T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class RemovedParameter:
    name: str
    deprecated_in: str | None = None
    removed_in: str
    hint: str | None = None


@dataclass(frozen=True, kw_only=True)
class RenamedParameter:
    name: str
    new_name: str
    deprecated_in: str | None = None
    removed_in: str
    hint: str | None = None


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


def removed_parameters(
    *params: RemovedParameter | RenamedParameter,
) -> IdentityFunction:
    """
    Decorator to mark function parameters.

    This decorator expects a number of `RemovedParameter` or `RenamedParameter`
    instances that describe each of the removed parameters of the method.
    """
    assert len(params) == len({p.name for p in params}), (
        "duplicate parameter in removed parameter list"
    )
    params_dict = {p.name: p for p in params}

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for name in kwargs:
                if name in params_dict:
                    _raise_removed_argument_error(
                        params_dict[name],
                        func_name=function.__name__,
                        kwargs=kwargs,
                    )
            return function(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _raise_removed_argument_error(
    param: RemovedParameter | RenamedParameter,
    *,
    func_name: str,
    kwargs: dict[str, object],
) -> NoReturn:
    was_deprecated_and = (
        f" was deprecated in version {param.deprecated_in} and"
        if param.deprecated_in is not None
        else ""
    )
    if isinstance(param, RenamedParameter):
        if param.name in kwargs and param.new_name in kwargs:
            msg = (
                f"{func_name!r} received both {param.name!r} and {param.new_name!r} as arguments;"
                f" {param.name!r}{was_deprecated_and} has been renamed to"
                f" {param.new_name!r} in version {param.removed_in}."
            )
            raise ArgumentRemovedError(msg)
        else:
            msg = (
                f"the argument {param.name!r} for {func_name!r}{was_deprecated_and}"
                f" has been removed in version {param.removed_in}."
                f" It was renamed to {param.new_name!r}."
            )
            msg = msg if param.hint is None else f"{msg} {param.hint}"
            raise ArgumentRemovedError(msg)
    elif isinstance(param, RemovedParameter):
        msg = (
            f"the argument {param.name!r} for {func_name!r}{was_deprecated_and}"
            f" has been removed in version {param.removed_in}."
        )
        msg = msg if param.hint is None else f"{msg} {param.hint}"
        raise ArgumentRemovedError(msg)
    else:
        msg = f"Unexpected parameter type: {type(param)!r}"
        raise TypeError(msg)

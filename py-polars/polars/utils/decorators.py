from __future__ import annotations

import inspect
import warnings
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")


def deprecated_alias(**aliases: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Deprecate a function or method argument.

    Decorator for deprecated function and method arguments. Use as follows:

    @deprecated_alias(old_arg='new_arg')
    def myfunc(new_arg):
        ...
    """

    def deco(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _rename_kwargs(function.__name__, kwargs, aliases)
            return function(*args, **kwargs)

        return wrapper

    return deco


def warn_closed_future_change() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Warn that user should pass in 'closed' as default value will change.

    Decorator for rolling function. Use as follows:

    @warn_closed_future_change()
    def myfunc():
        ...
    """

    def deco(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # we only warn if 'by' is passed in, otherwise 'closed' is not used
            if (kwargs.get("by") is not None) and ("closed" not in kwargs):
                warnings.warn(
                    message=(
                        "The default argument for closed, 'left', will be changed to 'right' in the future."
                        "Fix this warning by explicitly passing in a value for closed"
                    ),
                    category=FutureWarning,
                    stacklevel=find_stacklevel(),
                )

            return function(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(
    func_name: str,
    kwargs: dict[str, object],
    aliases: dict[str, str],
) -> None:
    """
    Rename the keyword arguments of a function.

    Helper function for deprecating function and method arguments.
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                    f" `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=find_stacklevel(),
            )
            kwargs[new] = kwargs.pop(alias)


def deprecate_nonkeyword_arguments(
    allowed_args: list[str] | None = None,
    message: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to deprecate the use of non-keyword arguments of a function.

    Parameters
    ----------
    allowed_args
        The names of some first arguments of the decorated function that are allowed to
        be given as positional arguments. Should include "self" when decorating class
        methods. If set to None (default), equal to all arguments that do not have a
        default value.
    message
        Optionally overwrite the default warning message.
    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        old_sig = inspect.signature(function)

        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
        new_params.sort(key=lambda p: p.kind)

        new_sig = old_sig.replace(parameters=new_params)

        num_allowed_args = len(allow_args)
        if message is None:
            msg_format = (
                f"All arguments of {function.__qualname__}{{except_args}} will be keyword-only in the next breaking release."
                " Use keyword arguments to silence this warning."
            )
            msg = msg_format.format(except_args=_format_argument_list(allow_args))
        else:
            msg = message

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if len(args) > num_allowed_args:
                warnings.warn(msg, DeprecationWarning, stacklevel=find_stacklevel())
            return function(*args, **kwargs)

        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _format_argument_list(allowed_args: list[str]) -> str:
    """
    Format allowed arguments list for use in the warning message of
    ``deprecate_nonkeyword_arguments``.
    """  # noqa: D205
    if "self" in allowed_args:
        allowed_args.remove("self")
    if not allowed_args:
        return ""
    elif len(allowed_args) == 1:
        return f" except for {allowed_args[0]!r}"
    else:
        last = allowed_args[-1]
        args = ", ".join([f"{x!r}" for x in allowed_args[:-1]])
        return f" except for {args} and {last!r}"


def redirect(
    from_to: dict[str, str | tuple[str, dict[str, Any]]]
) -> Callable[[type[T]], type[T]]:
    """
    Class decorator allowing deprecation/transition from one method name to another.

    The parameters must be the same (unless they are being renamed, in which case
    you can use this in conjunction with @deprecated_alias). If you need to redirect
    with custom kwargs, can redirect to a method name and associated kwargs dict.
    """

    def _redirecting_getattr_(obj: T, item: Any) -> Any:
        if isinstance(item, str) and item in from_to:
            new_item = from_to[item]
            new_item_name = new_item if isinstance(new_item, str) else new_item[0]
            warnings.warn(
                f"`{type(obj).__name__}.{item}` has been renamed; this"
                f" redirect is temporary, please use `.{new_item_name}` instead",
                category=DeprecationWarning,
                stacklevel=find_stacklevel(),
            )
            item = new_item_name

        attr = obj.__getattribute__(item)
        if isinstance(new_item, tuple):
            attr = partial(attr, **new_item[1])
        return attr

    def _cls_(cls: type[T]) -> type[T]:
        # note: __getattr__ is only invoked if item isn't found on the class
        cls.__getattr__ = _redirecting_getattr_  # type: ignore[attr-defined]
        return cls

    return _cls_

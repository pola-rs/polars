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


def issue_deprecation_warning(message: str, *, version: str) -> None:
    """
    Issue a deprecation warning.

    Parameters
    ----------
    message
        The message associated with the warning.
    version
        The Polars version number in which the warning is first issued.
        This argument is used to help developers determine when to remove the
        deprecated functionality.

    """
    warnings.warn(message, DeprecationWarning, stacklevel=find_stacklevel())


def deprecate_function(
    message: str, *, version: str
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark a function as deprecated."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            issue_deprecation_warning(
                f"`{function.__name__}` is deprecated. {message}",
                version=version,
            )
            return function(*args, **kwargs)

        return wrapper

    return decorate


def deprecate_renamed_function(
    new_name: str, *, version: str
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to mark a function as deprecated due to being renamed.

    Notes
    -----
    For deprecating renamed class methods, use the ``deprecate_renamed_methods``
    class decorator instead.

    """
    return deprecate_function(f"It has been renamed to `{new_name}`.", version=version)


def deprecate_renamed_parameter(
    old_name: str, new_name: str, *, version: str
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to mark a function argument as deprecated due to being renamed.

    Use as follows::

        @deprecate_renamed_parameter("old_name", "new_name", version="0.1.2")
        def myfunc(new_name):
            ...

    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _rename_keyword_argument(
                old_name, new_name, kwargs, function.__name__, version
            )
            return function(*args, **kwargs)

        return wrapper

    return decorate


def _rename_keyword_argument(
    old_name: str,
    new_name: str,
    kwargs: dict[str, object],
    func_name: str,
    version: str,
) -> None:
    """Rename a keyword argument of a function."""
    if old_name in kwargs:
        if new_name in kwargs:
            raise TypeError(
                f"`{func_name}` received both `{old_name}` and `{new_name}` as arguments."
                f" `{old_name}` is deprecated, use `{new_name}` instead."
            )
        issue_deprecation_warning(
            f"`the argument {old_name}` for `{func_name}` is deprecated."
            f" It has been renamed to `{new_name}`.",
            version=version,
        )
        kwargs[new_name] = kwargs.pop(old_name)


def deprecate_renamed_methods(
    mapping: dict[str, str | tuple[str, dict[str, Any]]], *, versions: dict[str, str]
) -> Callable[[type[T]], type[T]]:
    """
    Class decorator to mark methods as deprecated due to being renamed.

    This allows for the deprecated method to be deleted. It will remain available
    to users, but will no longer show up in auto-complete suggestions.

    If the arguments of the method are being renamed as well, use in conjunction with
    `deprecate_renamed_parameter`.

    If the new method has different default values for some keyword arguments, supply
    the old default values as a dictionary in the mapping like so::

        @deprecate_renamed_methods(
            {"old_method": ("new_method", {"flag": False})},
            versions={"old_method": "1.0.0"},
        )
        class Foo:
            def new_method(flag=True):
                ...

    Parameters
    ----------
    mapping
        Mapping of deprecated method names to new method names.
    versions
        For each deprecated method name, the Polars version number in which it was
        deprecated. This argument is used to help developers determine when to remove
        the deprecated functionality.

    """

    def _redirecting_getattr_(obj: T, item: Any) -> Any:
        if isinstance(item, str) and item in mapping:
            new_item = mapping[item]
            new_item_name = new_item if isinstance(new_item, str) else new_item[0]
            class_name = type(obj).__name__
            issue_deprecation_warning(
                f"`{class_name}.{item}` is deprecated."
                f" It has been renamed to `{class_name}.{new_item_name}`.",
                version=versions[item],
            )
            item = new_item_name

        attr = obj.__getattribute__(item)
        if isinstance(new_item, tuple):
            attr = partial(attr, **new_item[1])
        return attr

    def decorate(cls: type[T]) -> type[T]:
        # note: __getattr__ is only invoked if item isn't found on the class
        cls.__getattr__ = _redirecting_getattr_  # type: ignore[attr-defined]
        return cls

    return decorate


def deprecate_nonkeyword_arguments(
    allowed_args: list[str] | None = None, message: str | None = None, *, version: str
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
    version
        The Polars version number in which the warning is first issued.
        This argument is used to help developers determine when to remove the
        deprecated functionality.

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
                issue_deprecation_warning(msg, version=version)
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


def warn_closed_future_change() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Issue a warning to specify a value for `closed` as the default value will change.

    Decorator for rolling functions. Use as follows::

        @warn_closed_future_change()
        def rolling_min():
            ...

    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # we only warn if 'by' is passed in, otherwise 'closed' is not used
            if (kwargs.get("by") is not None) and ("closed" not in kwargs):
                issue_deprecation_warning(
                    "The default value for `closed` will change from 'left' to 'right' in a future version."
                    " Explicitly pass a value for `closed` to silence this warning.",
                    version="0.18.4",
                )
            return function(*args, **kwargs)

        return wrapper

    return decorate

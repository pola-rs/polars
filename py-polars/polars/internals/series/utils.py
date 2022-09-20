from __future__ import annotations

import inspect
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import polars.internals as pli
from polars.datatypes import PolarsDataType, dtype_to_ffiname

if TYPE_CHECKING:
    from polars.polars import PySeries

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")
    SeriesMethod = Callable[..., pli.Series]


class _EmptyBytecodeHelper:
    def __init__(self) -> None:
        # generate bytecode for empty functions with/without a docstring
        def _empty_with_docstring() -> None:
            """"""

        def _empty_without_docstring() -> None:
            pass

        self.empty_bytecode = (
            _empty_with_docstring.__code__.co_code,
            _empty_without_docstring.__code__.co_code,
        )

    def __contains__(self, item: bytes) -> bool:
        return item in self.empty_bytecode


_EMPTY_BYTECODE = _EmptyBytecodeHelper()


def _is_empty_method(func: SeriesMethod) -> bool:
    """
    Confirm that the given function has no implementation.

    Definitions of empty:

    - only has a docstring (body is empty)
    - has no docstring and just contains 'pass' (or equivalent)
    """
    fc = func.__code__
    return (fc.co_code in _EMPTY_BYTECODE) and (
        len(fc.co_consts) == 2 and fc.co_consts[1] is None
    )


def _expr_lookup(namespace: str | None) -> set[tuple[str | None, str, tuple[str, ...]]]:
    """Create lookup of potential Expr methods (in the given namespace)."""
    # dummy Expr object that we can introspect
    expr = pli.Expr()
    expr._pyexpr = None

    # optional indirection to "expr.str", "expr.dt", etc
    if namespace is not None:
        expr = getattr(expr, namespace)

    lookup = set()
    for name in dir(expr):
        if not name.startswith("_"):
            m = getattr(expr, name)
            if callable(m):
                # add function signature (argument names only) to the lookup
                # as a _possible_ candidate for expression-dispatch
                args = m.__code__.co_varnames[: m.__code__.co_argcount]
                lookup.add((namespace, name, args))
    return lookup


def call_expr(func: SeriesMethod) -> SeriesMethod:
    """Dispatch Series method to an expression implementation."""

    @wraps(func)  # type: ignore[arg-type]
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> pli.Series:
        s = pli.wrap_s(self._s)
        expr = pli.col(s.name)
        namespace = getattr(self, "_accessor", None)
        if namespace is not None:
            expr = getattr(expr, namespace)
        f = getattr(expr, func.__name__)
        return s.to_frame().select(f(*args, **kwargs)).to_series()

    # note: applying explicit '__signature__' helps IDEs (especially PyCharm)
    # with proper autocomplete, in addition to what @functools.wraps does
    setattr(wrapper, "__signature__", inspect.signature(func))  # noqa: B010
    return wrapper


def expr_dispatch(cls: type[T]) -> type[T]:
    """
    Series/NameSpace class decorator that sets up expression dispatch.

    * Applied to the Series class, and/or any Series 'NameSpace' classes.
    * Walks the class attributes, looking for methods that have empty function
      bodies, with signatures compatible with an existing Expr function.
    * IIF both conditions are met, the empty method is decorated with @call_expr.
    """
    # create lookup of expression functions in this namespace
    namespace = getattr(cls, "_accessor", None)
    expr_lookup = _expr_lookup(namespace)

    for name in dir(cls):
        if not name.startswith("_"):
            attr = getattr(cls, name)
            if callable(attr):
                # note: `co_varnames` starts with the function args, but needs to be
                # constrained by `co_argcount` as it also includes function-level consts
                args = attr.__code__.co_varnames[: attr.__code__.co_argcount]
                # if an expression method with compatible method exists, further check
                # that the series implementation has an empty function body
                if (namespace, name, args) in expr_lookup and _is_empty_method(attr):
                    setattr(cls, name, call_expr(attr))
    return cls


def get_ffi_func(
    name: str, dtype: PolarsDataType, obj: PySeries
) -> Callable[..., Any] | None:
    """
    Dynamically obtain the proper ffi function/ method.

    Parameters
    ----------
    name
        function or method name where dtype is replaced by <>
        for example
            "call_foo_<>"
    dtype
        polars dtype.
    obj
        Object to find the method for.

    Returns
    -------
    ffi function, or None if not found

    """
    ffi_name = dtype_to_ffiname(dtype)
    fname = name.replace("<>", ffi_name)
    return getattr(obj, fname, None)

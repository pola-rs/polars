from __future__ import annotations

import inspect
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import polars.internals as pli
from polars.datatypes import DataType, dtype_to_ffiname

if TYPE_CHECKING:
    from polars.polars import PySeries

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")
    SeriesMethod = Callable[..., pli.Series]
    ExprLookup = set[tuple[str | None, str, tuple[str, ...]]]


def _empty_() -> None:
    """"""


def _is_empty_method(func: SeriesMethod) -> bool:
    """Confirm function ONLY has a docstring (eg: body is empty/pass)"""
    ec, fc = _empty_.__code__, func.__code__
    return (ec.co_code == fc.co_code) and (
        len(fc.co_consts) == 2 and fc.co_consts[1] is None
    )


def _expr_lookup(namespace: str | None) -> ExprLookup:
    """Create lookup of potential Expr methods (in the same namespace)"""
    expr = pli.Expr()
    expr._pyexpr = None
    if namespace:
        expr = getattr(expr, namespace)

    lookup = set()
    for name in dir(expr):
        if not name.startswith("_"):
            m = getattr(expr, name)
            if callable(m):
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

    setattr(wrapper, "__signature__", inspect.signature(func))  # noqa: B010
    return wrapper


def expr_dispatch(cls: type[T]) -> type[T]:
    """Series/NameSpace class decorator that sets up expression dispatch."""
    namespace = getattr(cls, "_accessor", None)
    expr_lookup = _expr_lookup(namespace)

    for name in dir(cls):
        if not name.startswith("_"):
            attr = getattr(cls, name)
            if callable(attr):
                args = attr.__code__.co_varnames[: attr.__code__.co_argcount]
                if (namespace, name, args) in expr_lookup and _is_empty_method(attr):
                    setattr(cls, name, call_expr(attr))
    return cls


def get_ffi_func(
    name: str, dtype: type[DataType], obj: PySeries
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

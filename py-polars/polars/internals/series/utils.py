from __future__ import annotations

import sys
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import polars.internals as pli
from polars.datatypes import DataType, dtype_to_ffiname

if TYPE_CHECKING:
    from polars.polars import PySeries

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")


def call_expr(func: Callable[P, pli.Series]) -> Callable[P, pli.Series]:
    """Dispatch Series method to an expression implementation."""

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> pli.Series:
        s = pli.wrap_s(self._s)
        expr = pli.col(s.name)
        namespace_accessor = getattr(self, "_accessor", None)
        if namespace_accessor is not None:
            expr = getattr(expr, namespace_accessor)
        f = getattr(expr, func.__name__)
        return s.to_frame().select(f(*args, **kwargs)).to_series()

    return wrapper  # type: ignore[return-value]


def get_ffi_func(
    name: str,
    dtype: type[DataType],
    obj: PySeries,
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

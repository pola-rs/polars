from __future__ import annotations

import inspect
import sys
from functools import reduce, wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np

import polars._reexport as pl
from polars import functions as F
from polars._utils.wrap import wrap_s
from polars.datatypes import dtype_to_ffiname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import Series
    from polars._typing import PolarsDataType
    from polars.polars import PySeries

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")
    SeriesMethod = Callable[..., Series]


def expr_dispatch(cls: type[T]) -> type[T]:
    """
    Series/NameSpace class decorator that sets up expression dispatch.

    * Applied to the Series class, and/or any Series 'NameSpace' classes.
    * Walks the class attributes, looking for methods that have empty function
      bodies, with signatures compatible with an existing Expr function.
    * IFF both conditions are met, the empty method is decorated with @call_expr.
    """
    # create lookup of expression functions in this namespace
    namespace = getattr(cls, "_accessor", None)
    expr_lookup = _expr_lookup(namespace)

    for name in dir(cls):
        if (
            # private
            not name.startswith("_")
            # Avoid error when building docs
            # https://github.com/pola-rs/polars/pull/13238#discussion_r1438787093
            # TODO: is there a better way to do this?
            and name != "plot"
        ):
            attr = getattr(cls, name)
            if callable(attr):
                attr = _undecorated(attr)
                # note: `co_varnames` starts with the function args, but needs to be
                # constrained by `co_argcount` as it also includes function-level consts
                args = attr.__code__.co_varnames[: attr.__code__.co_argcount]
                # if an expression method with compatible method exists, further check
                # that the series implementation has an empty function body
                if (namespace, name, args) in expr_lookup and _is_empty_method(attr):
                    setattr(cls, name, call_expr(attr))
    return cls


def _expr_lookup(namespace: str | None) -> set[tuple[str | None, str, tuple[str, ...]]]:
    """Create lookup of potential Expr methods (in the given namespace)."""
    # dummy Expr object that we can introspect
    expr = pl.Expr()
    expr._pyexpr = None

    # optional indirection to "expr.str", "expr.dt", etc
    if namespace is not None:
        expr = getattr(expr, namespace)

    lookup = set()
    for name in dir(expr):
        if not name.startswith("_"):
            try:
                m = getattr(expr, name)
            except AttributeError:  # may raise for @property methods
                continue
            if callable(m):
                # add function signature (argument names only) to the lookup
                # as a _possible_ candidate for expression-dispatch
                m = _undecorated(m)
                args = m.__code__.co_varnames[: m.__code__.co_argcount]
                lookup.add((namespace, name, args))
    return lookup


def _undecorated(function: Callable[P, T]) -> Callable[P, T]:
    """Return the given function without any decorators."""
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def call_expr(func: SeriesMethod) -> SeriesMethod:
    """Dispatch Series method to an expression implementation."""

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Series:
        s = wrap_s(self._s)
        expr = F.col(s.name)
        if (namespace := getattr(self, "_accessor", None)) is not None:
            expr = getattr(expr, namespace)
        f = getattr(expr, func.__name__)
        return s.to_frame().select_seq(f(*args, **kwargs)).to_series()

    # note: applying explicit '__signature__' helps IDEs (especially PyCharm)
    # with proper autocomplete, in addition to what @functools.wraps does
    setattr(wrapper, "__signature__", inspect.signature(func))  # noqa: B010
    return wrapper


def _is_empty_method(func: SeriesMethod) -> bool:
    """
    Confirm that the given function has no implementation.

    Definitions of empty:

    - only has a docstring (body is empty)
    - has no docstring and just contains 'pass' (or equivalent)
    """
    fc = func.__code__
    return (fc.co_code in _EMPTY_BYTECODE) and (
        (len(fc.co_consts) == 2 and fc.co_consts[1] is None)
        # account for optimized-out docstrings (eg: running 'python -OO')
        or (sys.flags.optimize == 2 and fc.co_consts == (None,))
    )


class _EmptyBytecodeHelper:
    def __init__(self) -> None:
        # generate bytecode for empty functions with/without a docstring
        def _empty_with_docstring() -> None:
            """"""  # noqa: D419

        def _empty_without_docstring() -> None:
            pass

        self.empty_bytecode = (
            _empty_with_docstring.__code__.co_code,
            _empty_without_docstring.__code__.co_code,
        )

    def __contains__(self, item: bytes) -> bool:
        return item in self.empty_bytecode


_EMPTY_BYTECODE = _EmptyBytecodeHelper()


def np_common(*types: str) -> str:
    """
    Find supertype of numpy types.

    Parameters
    ----------
    *types
        The char version of numpy types

    Returns
    -------
    str
        The dtype in common
    """
    if hasattr(np, "promote_types"):

        def _np_super(*args: np.dtype[Any]) -> np.dtype[Any]:
            return reduce(np.promote_types, args)
    else:
        # for pre numpy 2.0
        def _np_super(*args: np.dtype[Any]) -> np.dtype[Any]:
            return np.find_common_type(args, [])  # type: ignore[attr-defined]

    dtypes = [np.dtype(ch) for ch in types]
    return _np_super(*dtypes).char


def match_in_out_types(
    ufunc_types: Sequence[str],
    *,
    args: Sequence[int | float | np.ndarray[Any, Any]] | None = None,
    args_types: Sequence[str] | None = None,
) -> str:
    """
    Obtain the proper output type based on ufunc signature and input types.

    Parameters
    ----------
    ufunc_types
        The output of ufunc.types
            for example ['e->e', 'f->f'] or ['d->L', 'f->I']
    args_types
        The types of the inputs

    Returns
    -------
    str
        The dtype to use for choosing the ffi func
    """
    NP_INTs = ["b", "h", "i", "l", "B", "H", "I", "L"]
    if args is None and args_types is None:
        msg = "must use one of args or args_types"
        raise ValueError(msg)
    if args is not None and args_types is not None:
        msg = "only specify one of args or args_types"
        raise ValueError(msg)
    if args_types is None and args is not None:
        prefer_int = None
        prefer_float = None
        args_types = []
        for arg in args:
            if isinstance(arg, float):
                if prefer_float is not None:
                    args_types.append(prefer_float)
                else:
                    args_types.append("d")
            elif isinstance(arg, int):
                if prefer_int is not None:
                    args_types.append(prefer_int)
                else:
                    args_types.append("l")
            else:
                if arg.dtype.char in NP_INTs:
                    prefer_int = arg.dtype.char
                elif arg.dtype.char in ["d", "f"]:
                    prefer_float = arg.dtype.char
                args_types.append(arg.dtype.char)
    in_out = {
        (splt := _type.split("->", maxsplit=1))[0]: splt[1] for _type in ufunc_types
    }
    assert args_types is not None
    args_str = "".join(args_types)
    if args_str in in_out:
        return in_out[args_str]
    elif len(set(in_out.values())) == 1:
        return next(iter(in_out.values()))
    elif all(v not in NP_INTs for v in in_out.values()):
        if any(x == "f" for x in args_types) and all(x != "d" for x in args_types):
            return "f"
        else:
            return "d"
    elif (
        len(args_types) == 2
        and args_types[0] in ["f", "d"]
        and args_types[1] in NP_INTs
    ):
        return args_types[0]
    elif all(
        len(set(k)) == 1 and (num_ins := len(k)) > 1 and len(args_str) > 1
        for k in in_out
    ):
        super_in = np_common(*args_types)
        super_in_repeated = [super_in for _ in range(num_ins)]
        return match_in_out_types(ufunc_types, args_types=super_in_repeated)
    else:
        msg = "no matching input to output dtype combination found. Try manually setting dtype"
        raise TypeError(msg)


def get_ffi_func(
    name: str, dtype: PolarsDataType, obj: PySeries
) -> Callable[..., Any] | None:
    """
    Dynamically obtain the proper FFI function/ method.

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
    callable or None
        FFI function, or None if not found.
    """
    ffi_name = dtype_to_ffiname(dtype)
    fname = name.replace("<>", ffi_name)
    return getattr(obj, fname, None)


def _with_no_check_length(func: Callable[..., Any]) -> Any:
    from polars.polars import check_length

    # Catch any error so that we can be sure that we always restore length checks
    try:
        check_length(False)
        result = func()
        check_length(True)
    except Exception:
        check_length(True)
        raise
    else:
        return result

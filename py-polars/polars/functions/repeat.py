from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, NoReturn, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import Float64
from polars.utils import no_default
from polars.utils._wrap import wrap_expr
from polars.utils.various import find_stacklevel

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    import sys

    from polars import Expr, Series
    from polars.type_aliases import PolarsDataType, PythonLiteral
    from polars.utils import NoDefault

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


@overload
def repeat(
    value: PythonLiteral | None,
    n: Expr | int,
    *,
    dtype: PolarsDataType | None = ...,
    name: str | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def repeat(
    value: PythonLiteral | None,
    n: int,
    *,
    dtype: PolarsDataType | None = ...,
    name: str | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def repeat(
    value: PythonLiteral | None,
    n: Expr,
    *,
    dtype: PolarsDataType | None = ...,
    name: str | None = ...,
    eager: Literal[True],
) -> NoReturn:
    ...


@overload
def repeat(
    value: PythonLiteral | None,
    n: Expr | int,
    *,
    dtype: PolarsDataType | None = ...,
    name: str | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


def repeat(
    value: PythonLiteral | None,
    n: Expr | int,
    *,
    dtype: PolarsDataType | None = None,
    name: str | None = None,
    eager: bool = False,
) -> Expr | Series:
    """
    Construct a column of length `n` filled with the given value.

    Parameters
    ----------
    value
        Value to repeat.
    n
        Length of the resulting column.
    dtype
        Data type of the resulting column. If set to ``None`` (default), data type is
        inferred from the given value. Defaults to Int32 for integer values, unless
        Int64 is required to fit the given value. Defaults to Float64 for float values.
    name
        Name of the resulting column.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    Notes
    -----
    If you want to construct a column in lazy mode and do not need a pre-determined
    length, use :func:`lit` instead.

    See Also
    --------
    lit

    Examples
    --------
    Construct a column with a repeated value in a lazy context.

    >>> pl.select(pl.repeat("z", n=3)).to_series()
    shape: (3,)
    Series: 'literal' [str]
    [
            "z"
            "z"
            "z"
    ]

    Generate a Series directly by setting ``eager=True``.

    >>> pl.repeat(3, n=3, eager=True, name="three", dtype=pl.Int8)
    shape: (3,)
    Series: 'three' [i8]
    [
            3
            3
            3
    ]

    """
    if eager:
        if not isinstance(n, int):
            raise ValueError(
                "`n` must be an integer when using `repeat` in an eager context."
            )
        return pl.Series._repeat(value, n, name=name, dtype=dtype)
    else:
        return _repeat_lazy(value, n, dtype=dtype, name=name)


def _repeat_lazy(
    value: PythonLiteral | None,
    n: Expr | int,
    *,
    dtype: PolarsDataType | None = None,
    name: str | None = None,
) -> Expr:
    if isinstance(n, int):
        n = F.lit(n)
    expr = wrap_expr(plr.repeat(value, n._pyexpr))
    if name is not None:
        expr = expr.alias(name)
    if dtype is not None:
        expr = expr.cast(dtype)
    return expr


@overload
def ones(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: Literal[False],
) -> Expr:
    ...


@overload
def ones(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: Literal[True] = ...,
) -> Series:
    ...


@overload
def ones(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


def ones(
    n: int,
    dtype: PolarsDataType = Float64,
    *,
    name: str | None = None,
    eager: bool | NoDefault = no_default,
) -> Expr | Series:
    """
    Construct a column of length `n` filled with ones.

    Syntactic sugar for ``repeat(1.0, ...)``.

    Parameters
    ----------
    n
        Length of the resulting column.
    dtype
        Data type of the resulting column. Defaults to Float64.
    name
        Name of the resulting column.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False``,
        return an expression instead.

    Notes
    -----
    If you want to construct a column in lazy mode and do not need a pre-determined
    length, use :func:`lit` instead.

    See Also
    --------
    repeat
    lit

    Examples
    --------
    >>> pl.ones(3, pl.Int8, eager=True)
    shape: (3,)
    Series: '' [i8]
    [
        1
        1
        1
    ]

    """
    if eager is no_default:
        warnings.warn(
            "In a future version, the default behaviour for `ones` will change from `eager=True` to `eager=False`. "
            "To silence this warning, please:\n"
            "- set `eager=False` to opt in to the new default behaviour, or\n"
            "- set `eager=True` to retain the old one.",
            FutureWarning,
            stacklevel=find_stacklevel(),
        )
        eager = True
    return repeat(1.0, n=n, dtype=dtype, name=name, eager=eager)


@overload
def zeros(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: Literal[False],
) -> Expr:
    ...


@overload
def zeros(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: Literal[True] = ...,
) -> Series:
    ...


@overload
def zeros(
    n: int,
    dtype: PolarsDataType = ...,
    *,
    name: str | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


def zeros(
    n: int,
    dtype: PolarsDataType = Float64,
    *,
    name: str | None = None,
    eager: bool | NoDefault = no_default,
) -> Expr | Series:
    """
    Construct a column of length `n` filled with zeros.

    Syntactic sugar for ``repeat(0.0, ...)``.

    Parameters
    ----------
    n
        Length of the resulting column.
    dtype
        Data type of the resulting column. Defaults to Float64.
    name
        Name of the resulting column.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False``,
        return an expression instead.

    Notes
    -----
    If you want to construct a column in lazy mode and do not need a pre-determined
    length, use :func:`lit` instead.

    See Also
    --------
    repeat
    lit

    Examples
    --------
    >>> pl.zeros(3, pl.Int8, eager=True)
    shape: (3,)
    Series: '' [i8]
    [
        0
        0
        0
    ]

    """
    if eager is no_default:
        warnings.warn(
            "In a future version, the default behaviour for `ones` will change from `eager=True` to `eager=False`. "
            "To silence this warning, please:\n"
            "- set `eager=False` to opt in to the new default behaviour, or\n"
            "- set `eager=True` to retain the old one.",
            FutureWarning,
            stacklevel=find_stacklevel(),
        )
        eager = True
    return repeat(0.0, n=n, dtype=dtype, name=name, eager=eager)

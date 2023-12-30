from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, overload

from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    Array,
    Boolean,
    Float64,
    List,
    Utf8,
)
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import IntoExpr, PolarsDataType


@overload
def repeat(
    value: IntoExpr | None,
    n: int | Expr,
    *,
    dtype: PolarsDataType | None = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def repeat(
    value: IntoExpr | None,
    n: int | Expr,
    *,
    dtype: PolarsDataType | None = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def repeat(
    value: IntoExpr | None,
    n: int | Expr,
    *,
    dtype: PolarsDataType | None = ...,
    eager: bool,
) -> Expr | Series:
    ...


def repeat(
    value: IntoExpr | None,
    n: int | Expr,
    *,
    dtype: PolarsDataType | None = None,
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
        Data type of the resulting column. If set to `None` (default), data type is
        inferred from the given value. Defaults to Int32 for integer values, unless
        Int64 is required to fit the given value. Defaults to Float64 for float values.
    eager
        Evaluate immediately and return a `Series`. If set to `False` (default),
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
    Series: 'repeat' [str]
    [
            "z"
            "z"
            "z"
    ]

    Generate a Series directly by setting `eager=True`.

    >>> pl.repeat(3, n=3, dtype=pl.Int8, eager=True)
    shape: (3,)
    Series: 'repeat' [i8]
    [
            3
            3
            3
    ]

    """
    if isinstance(n, int):
        n = F.lit(n)
    value = parse_as_expression(value, str_as_lit=True, list_as_lit=True, dtype=dtype)
    expr = wrap_expr(plr.repeat(value, n._pyexpr, dtype))
    if eager:
        return F.select(expr).to_series()
    return expr


# dtypes that have a reasonable one/zero mapping;
# for anything more elaborate should use `repeat`
_ones_zeros = {
    Utf8: ("0", "1"),
    Boolean: (False, True),
    List(Utf8): (["0"], ["1"]),
    List(Boolean): ([False], [True]),
}
for dtype in INTEGER_DTYPES | FLOAT_DTYPES:
    _ones_zeros[dtype] = (0, 1)
    _ones_zeros[List(dtype)] = ([0], [1])
    _ones_zeros[Array(dtype, width=1)] = ([0], [1])


@overload
def ones(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def ones(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: Literal[True],
) -> Series:
    ...


@overload
def ones(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: bool,
) -> Expr | Series:
    ...


def ones(
    n: int | Expr,
    dtype: PolarsDataType = Float64,
    *,
    eager: bool = False,
) -> Expr | Series:
    """
    Construct a column of length `n` filled with ones.

    This is syntactic sugar for the `repeat` function.

    Parameters
    ----------
    n
        Length of the resulting column.
    dtype
        Data type of the resulting column. Defaults to Float64.
    eager
        Evaluate immediately and return a `Series`. If set to `False`,
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
    Series: 'ones' [i8]
    [
        1
        1
        1
    ]

    """
    if (one_zero := _ones_zeros.get(dtype)) is None:
        raise TypeError(f"invalid dtype for `ones`; found {dtype})")

    one: Any = one_zero[1]
    return repeat(one, n=n, dtype=dtype, eager=eager).alias("ones")


@overload
def zeros(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def zeros(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: Literal[True],
) -> Series:
    ...


@overload
def zeros(
    n: int | Expr,
    dtype: PolarsDataType = ...,
    *,
    eager: bool,
) -> Expr | Series:
    ...


def zeros(
    n: int | Expr,
    dtype: PolarsDataType = Float64,
    *,
    eager: bool = False,
) -> Expr | Series:
    """
    Construct a column of length `n` filled with zeros.

    This is syntactic sugar for the `repeat` function.

    Parameters
    ----------
    n
        Length of the resulting column.
    dtype
        Data type of the resulting column. Defaults to Float64.
    eager
        Evaluate immediately and return a `Series`. If set to `False`,
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
    Series: 'zeros' [i8]
    [
        0
        0
        0
    ]

    """
    if (one_zero := _ones_zeros.get(dtype)) is None:
        raise TypeError(f"invalid dtype for `zeros`; found {dtype})")

    zero: Any = one_zero[0]
    return repeat(zero, n=n, dtype=dtype, eager=eager).alias("zeros")

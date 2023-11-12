from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

from polars import functions as F
from polars.datatypes import Int64
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from typing import Literal

    from polars import Expr, Series
    from polars.type_aliases import IntoExprColumn, PolarsIntegerType


@overload
def arange(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def arange(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def arange(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: bool,
) -> Expr | Series:
    ...


def arange(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = 1,
    *,
    dtype: PolarsIntegerType = Int64,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers.

    Alias for :func:`int_range`.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Data type of the range. Defaults to `Int64`.
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Column of data type `dtype`.

    See Also
    --------
    int_range : Generate a range of integers.
    int_ranges : Generate a range of integers for each row of the input columns.

    Examples
    --------
    >>> pl.arange(0, 3, eager=True)
    shape: (3,)
    Series: 'int' [i64]
    [
            0
            1
            2
    ]

    """
    return int_range(start, end, step, dtype=dtype, eager=eager)


@overload
def int_range(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def int_range(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def int_range(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: bool,
) -> Expr | Series:
    ...


def int_range(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = 1,
    *,
    dtype: PolarsIntegerType = Int64,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Data type of the range. Defaults to `Int64`.
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Int64`.

    See Also
    --------
    int_ranges : Generate a range of integers for each row of the input columns.

    Examples
    --------
    >>> pl.int_range(0, 3, eager=True)
    shape: (3,)
    Series: 'int' [i64]
    [
            0
            1
            2
    ]

    """
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_range(start, end, step, dtype))

    if eager:
        return F.select(result).to_series()

    return result


@overload
def int_ranges(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[False] = ...,
) -> Expr:
    ...


@overload
def int_ranges(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: Literal[True],
) -> Series:
    ...


@overload
def int_ranges(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = ...,
    *,
    dtype: PolarsIntegerType = ...,
    eager: bool,
) -> Expr | Series:
    ...


def int_ranges(
    start: int | IntoExprColumn,
    end: int | IntoExprColumn,
    step: int = 1,
    *,
    dtype: PolarsIntegerType = Int64,
    eager: bool = False,
) -> Expr | Series:
    """
    Generate a range of integers for each row of the input columns.

    Parameters
    ----------
    start
        Lower bound of the range (inclusive).
    end
        Upper bound of the range (exclusive).
    step
        Step size of the range.
    dtype
        Integer data type of the ranges. Defaults to `Int64`.
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type `List(dtype)`.

    See Also
    --------
    int_range : Generate a single range of integers.

    Examples
    --------
    >>> df = pl.DataFrame({"start": [1, -1], "end": [3, 2]})
    >>> df.with_columns(pl.int_ranges("start", "end"))
    shape: (2, 3)
    ┌───────┬─────┬────────────┐
    │ start ┆ end ┆ int_range  │
    │ ---   ┆ --- ┆ ---        │
    │ i64   ┆ i64 ┆ list[i64]  │
    ╞═══════╪═════╪════════════╡
    │ 1     ┆ 3   ┆ [1, 2]     │
    │ -1    ┆ 2   ┆ [-1, 0, 1] │
    └───────┴─────┴────────────┘

    """
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_ranges(start, end, step, dtype))

    if eager:
        return F.select(result).to_series()

    return result

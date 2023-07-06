from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterable, overload

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import (
    UInt32,
)
from polars.utils._parse_expr_input import (
    parse_as_list_of_expressions,
)
from polars.utils._wrap import wrap_expr
from polars.utils.decorators import deprecated_alias

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:

    from polars import Expr, Series
    from polars.type_aliases import (
        IntoExpr,
        PythonLiteral,
    )


@overload
def any(exprs: Series) -> bool:  # type: ignore[misc]
    ...


@overload
def any(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecated_alias(columns="exprs")
def any(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr | bool:
    """
    Evaluate a bitwise OR operation.

    If a single string is passed, this is an alias for ``pl.col(name).any()``.
    If a single Series is passed, this is an alias for ``Series.any()``.

    Otherwise, this function computes the bitwise OR horizontally across multiple
    columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [True, False, True],
    ...         "b": [False, False, False],
    ...         "c": [False, True, False],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌───────┬───────┬───────┐
    │ a     ┆ b     ┆ c     │
    │ ---   ┆ ---   ┆ ---   │
    │ bool  ┆ bool  ┆ bool  │
    ╞═══════╪═══════╪═══════╡
    │ true  ┆ false ┆ false │
    │ false ┆ false ┆ true  │
    │ true  ┆ false ┆ false │
    └───────┴───────┴───────┘

    Compares the values (in binary format) and return true if any value in the column
    is true.

    >>> df.select(pl.any("*"))
    shape: (1, 3)
    ┌──────┬───────┬──────┐
    │ a    ┆ b     ┆ c    │
    │ ---  ┆ ---   ┆ ---  │
    │ bool ┆ bool  ┆ bool │
    ╞══════╪═══════╪══════╡
    │ true ┆ false ┆ true │
    └──────┴───────┴──────┘

    Across multiple columns:

    >>> df.select(pl.any("a", "b"))
    shape: (3, 1)
    ┌───────┐
    │ any   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    │ true  │
    └───────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            return exprs.any()
        elif isinstance(exprs, str):
            return F.col(exprs).any()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return F.fold(
        F.lit(False), lambda a, b: a.cast(bool) | b.cast(bool), exprs_wrapped
    ).alias("any")


@overload
def all(exprs: Series) -> bool:  # type: ignore[misc]
    ...


@overload
def all(
    exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr
) -> Expr:
    ...


@deprecated_alias(columns="exprs")
def all(
    exprs: IntoExpr | Iterable[IntoExpr] | None = None, *more_exprs: IntoExpr
) -> Expr | bool:
    """
    Either return an expression representing all columns, or evaluate a bitwise AND operation.

    If no arguments are passed, this is an alias for ``pl.col("*")``.
    If a single string is passed, this is an alias for ``pl.col(name).any()``.
    If a single Series is passed, this is an alias for ``Series.any()``.

    Otherwise, this function computes the bitwise AND horizontally across multiple
    columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    Selecting all columns and calculating the sum:

    >>> df = pl.DataFrame(
    ...     {"a": [1, 2, 3], "b": ["hello", "foo", "bar"], "c": [1, 1, 1]}
    ... )
    >>> df.select(pl.all().sum())
    shape: (1, 3)
    ┌─────┬──────┬─────┐
    │ a   ┆ b    ┆ c   │
    │ --- ┆ ---  ┆ --- │
    │ i64 ┆ str  ┆ i64 │
    ╞═════╪══════╪═════╡
    │ 6   ┆ null ┆ 3   │
    └─────┴──────┴─────┘

    Bitwise AND across multiple columns:

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [True, False, True],
    ...         "b": [True, False, False],
    ...         "c": [False, True, False],
    ...     }
    ... )
    >>> df.select(pl.all("a", "b"))
    shape: (3, 1)
    ┌───────┐
    │ all   │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    │ false │
    └───────┘

    """  # noqa: W505
    if not more_exprs:
        if exprs is None:
            return F.col("*")
        elif isinstance(exprs, pl.Series):
            return exprs.all()
        elif isinstance(exprs, str):
            return F.col(exprs).all()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return F.fold(
        F.lit(True), lambda a, b: a.cast(bool) & b.cast(bool), exprs_wrapped
    ).alias("all")


@overload
def cumsum(exprs: Series) -> Series:  # type: ignore[misc]
    ...


@overload
def cumsum(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecated_alias(column="exprs")
def cumsum(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | Series:
    """
    Cumulatively sum all values.

    If a single string is passed, this is an alias for ``pl.col(name).cumsum()``.
    If a single Series is passed, this is an alias for ``Series.cumsum()``.

    Otherwise, this function computes the cumulative sum horizontally across multiple
    columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [3, 4],
    ...         "c": [5, 6],
    ...     }
    ... )
    >>> df
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   │
    │ 2   ┆ 4   ┆ 6   │
    └─────┴─────┴─────┘

    Cumulatively sum a column by name:

    >>> df.select(pl.cumsum("a"))
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 3   │
    └─────┘

    Cumulatively sum a list of columns/expressions horizontally:

    >>> df.with_columns(pl.cumsum("a", "c"))
    shape: (2, 4)
    ┌─────┬─────┬─────┬───────────┐
    │ a   ┆ b   ┆ c   ┆ cumsum    │
    │ --- ┆ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ struct[2] │
    ╞═════╪═════╪═════╪═══════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {1,6}     │
    │ 2   ┆ 4   ┆ 6   ┆ {2,8}     │
    └─────┴─────┴─────┴───────────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            return exprs.cumsum()
        elif isinstance(exprs, str):
            return F.col(exprs).cumsum()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    # (Expr): use u32 as that will not cast to float as eagerly
    return F.cumfold(F.lit(0).cast(UInt32), lambda a, b: a + b, exprs_wrapped).alias(
        "cumsum"
    )


@overload
def max(exprs: Series) -> PythonLiteral | None:  # type: ignore[misc]
    ...


@overload
def max(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


def max(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr | Any:
    """
    Get the maximum value.

    If a single string is passed, this is an alias for ``pl.col(name).max()``.
    If a single Series is passed, this is an alias for ``Series.max()``.

    Otherwise, this function computes the maximum value horizontally across multiple
    columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    Get the maximum value by row by passing multiple columns/expressions.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
    >>> df.select(pl.max("a", "b"))
    shape: (3, 1)
    ┌─────┐
    │ max │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 4   │
    │ 8   │
    │ 3   │
    └─────┘

    Get the maximum value of a column by passing a single column name.

    >>> df.select(pl.max("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 8   │
    └─────┘

    Get column-wise maximums for multiple columns by passing a regular expression,
    or call ``.max()`` on a multi-column expression instead.

    >>> df.select(pl.max("^a|b$"))
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 8   ┆ 5   │
    └─────┴─────┘
    >>> df.select(pl.col("a", "b").max())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 8   ┆ 5   │
    └─────┴─────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            return exprs.max()
        elif isinstance(exprs, str):
            return F.col(exprs).max()

    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.max_exprs(exprs))


@overload
def min(exprs: Series) -> PythonLiteral | None:  # type: ignore[misc]
    ...


@overload
def min(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


def min(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | PythonLiteral | None:
    """
    Get the minimum value.

    If a single string is passed, this is an alias for ``pl.col(name).min()``.
    If a single Series is passed, this is an alias for ``Series.min()``.

    Otherwise, this function computes the minimum value horizontally across multiple
    columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    Get the minimum value by row by passing multiple columns/expressions.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
    >>> df.select(pl.min("a", "b"))
    shape: (3, 1)
    ┌─────┐
    │ min │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 5   │
    │ 2   │
    └─────┘

    Get the minimum value of a column by passing a single column name.

    >>> df.select(pl.min("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    └─────┘

    Get column-wise minimums for multiple columns by passing a regular expression,
    or call ``.min()`` on a multi-column expression instead.

    >>> df.select(pl.min("^a|b$"))
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 2   │
    └─────┴─────┘
    >>> df.select(pl.col("a", "b").min())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 2   │
    └─────┴─────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            return exprs.min()
        elif isinstance(exprs, str):
            return F.col(exprs).min()

    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.min_exprs(exprs))


@overload
def sum(exprs: Series) -> int | float:  # type: ignore[misc]
    ...


@overload
def sum(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecated_alias(column="exprs")
def sum(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | int | float:
    """
    Sum all values.

    If a single string is passed, this is an alias for ``pl.col(name).sum()``.
    If a single Series is passed, this is an alias for ``Series.sum()``.

    Otherwise, this function computes the sum horizontally across multiple columns.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [3, 4],
    ...         "c": [5, 6],
    ...     }
    ... )
    >>> df
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   │
    │ 2   ┆ 4   ┆ 6   │
    └─────┴─────┴─────┘

    Sum a column by name:

    >>> df.select(pl.sum("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘

    Sum a list of columns/expressions horizontally:

    >>> df.with_columns(pl.sum("a", "c"))
    shape: (2, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   ┆ sum │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   ┆ 6   │
    │ 2   ┆ 4   ┆ 6   ┆ 8   │
    └─────┴─────┴─────┴─────┘

    Sum a series:

    >>> pl.sum(df.get_column("a"))
    3

    To aggregate the sums for more than one column/expression use ``pl.col(list).sum()``
    or a regular expression selector like ``pl.sum(regex)``:

    >>> df.select(pl.col("a", "c").sum())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ c   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 11  │
    └─────┴─────┘

    >>> df.select(pl.sum("^.*[bc]$"))
    shape: (1, 2)
    ┌─────┬─────┐
    │ b   ┆ c   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 7   ┆ 11  │
    └─────┴─────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            return exprs.sum()
        elif isinstance(exprs, str):
            return F.col(exprs).sum()

    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.sum_exprs(exprs))

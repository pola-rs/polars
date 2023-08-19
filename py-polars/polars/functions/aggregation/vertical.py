from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, overload

import polars._reexport as pl
import polars.functions as F
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.type_aliases import IntoExpr, PythonLiteral


@overload
def all(exprs: Series) -> bool:  # type: ignore[misc]
    ...


@overload
def all(
    exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr
) -> Expr:
    ...


@deprecate_renamed_parameter("columns", "exprs", version="0.18.7")
def all(
    exprs: IntoExpr | Iterable[IntoExpr] | None = None, *more_exprs: IntoExpr
) -> Expr | bool | None:
    """
    Either return an expression representing all columns, or evaluate a bitwise AND operation.

    If no arguments are passed, this is an alias for ``pl.col("*")``.
    If a single string is passed, this is an alias for ``pl.col(name).any()``.

    If a single Series is passed, this is an alias for ``Series.any()``.
    **This functionality is deprecated**.

    Otherwise, this function computes the bitwise AND horizontally across multiple
    columns.
    **This functionality is deprecated**, use ``pl.all_horizontal`` instead.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    See Also
    --------
    all_horizontal

    Examples
    --------
    Selecting all columns.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [True, False, True],
    ...         "b": [False, False, False],
    ...     }
    ... )
    >>> df.select(pl.all().sum())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ u32 ┆ u32 │
    ╞═════╪═════╡
    │ 2   ┆ 0   │
    └─────┴─────┘

    Evaluate bitwise AND for a column:

    >>> df.select(pl.all("a"))
    shape: (1, 1)
    ┌───────┐
    │ a     │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ false │
    └───────┘

    """  # noqa: W505
    if not more_exprs:
        if exprs is None:
            return F.col("*")
        elif isinstance(exprs, pl.Series):
            issue_deprecation_warning(
                "passing a Series to `all` is deprecated. Use `Series.all()` instead.",
                version="0.18.7",
            )
            return exprs.all()
        elif isinstance(exprs, str):
            return F.col(exprs).all()

    _warn_for_deprecated_horizontal_use("all")
    return F.all_horizontal(exprs, *more_exprs)


@overload
def any(exprs: Series) -> bool:  # type: ignore[misc]
    ...


@overload
def any(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecate_renamed_parameter("columns", "exprs", version="0.18.7")
def any(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | bool | None:
    """
    Evaluate a bitwise OR operation.

    If a single string is passed, this is an alias for ``pl.col(name).any()``.

    If a single Series is passed, this is an alias for ``Series.any()``.
    **This functionality is deprecated**.

    Otherwise, this function computes the bitwise OR horizontally across multiple
    columns.
    **This functionality is deprecated**, use ``pl.any_horizontal`` instead.

    See Also
    --------
    any_horizontal

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
    ...     }
    ... )
    >>> df.select(pl.any("a"))
    shape: (1, 1)
    ┌──────┐
    │ a    │
    │ ---  │
    │ bool │
    ╞══════╡
    │ true │
    └──────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            issue_deprecation_warning(
                "passing a Series to `any` is deprecated. Use `Series.any()` instead.",
                version="0.18.7",
            )
            return exprs.any()
        elif isinstance(exprs, str):
            return F.col(exprs).any()

    _warn_for_deprecated_horizontal_use("any")
    return F.any_horizontal(exprs, *more_exprs)


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
    **This functionality is deprecated**.

    Otherwise, this function computes the maximum value horizontally across multiple
    columns.
    **This functionality is deprecated**, use ``pl.max_horizontal`` instead.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    See Also
    --------
    max_horizontal

    Examples
    --------
    Get the maximum value of a column by passing a single column name.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
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
            issue_deprecation_warning(
                "passing a Series to `max` is deprecated. Use `Series.max()` instead.",
                version="0.18.7",
            )
            return exprs.max()
        elif isinstance(exprs, str):
            return F.col(exprs).max()

    _warn_for_deprecated_horizontal_use("max")
    return F.max_horizontal(exprs, *more_exprs)


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
    **This functionality is deprecated**.

    Otherwise, this function computes the minimum value horizontally across multiple
    columns.
    **This functionality is deprecated**, use ``pl.min_horizontal`` instead.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    See Also
    --------
    min_horizontal

    Examples
    --------
    Get the minimum value of a column by passing a single column name.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 8, 3],
    ...         "b": [4, 5, 2],
    ...         "c": ["foo", "bar", "foo"],
    ...     }
    ... )
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
            issue_deprecation_warning(
                "passing a Series to `min` is deprecated. Use `Series.min()` instead.",
                version="0.18.7",
            )
            return exprs.min()
        elif isinstance(exprs, str):
            return F.col(exprs).min()

    _warn_for_deprecated_horizontal_use("min")
    return F.min_horizontal(exprs, *more_exprs)


@overload
def sum(exprs: Series) -> int | float:  # type: ignore[misc]
    ...


@overload
def sum(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecate_renamed_parameter("column", "exprs", version="0.18.7")
def sum(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | int | float:
    """
    Sum all values.

    If a single string is passed, this is an alias for ``pl.col(name).sum()``.

    If a single Series is passed, this is an alias for ``Series.sum()``.
    **This functionality is deprecated**.

    Otherwise, this function computes the sum horizontally across multiple columns.
    **This functionality is deprecated**, use ``pl.sum_horizontal`` instead.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    See Also
    --------
    sum_horizontal

    Examples
    --------
    Sum a column by name:

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2],
    ...         "b": [3, 4],
    ...         "c": [5, 6],
    ...     }
    ... )
    >>> df.select(pl.sum("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘

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
            issue_deprecation_warning(
                "passing a Series to `sum` is deprecated. Use `Series.sum()` instead.",
                version="0.18.7",
            )
            return exprs.sum()
        elif isinstance(exprs, str):
            return F.col(exprs).sum()

    _warn_for_deprecated_horizontal_use("sum")
    return F.sum_horizontal(exprs, *more_exprs)


@overload
def cumsum(exprs: Series) -> Series:  # type: ignore[misc]
    ...


@overload
def cumsum(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    ...


@deprecate_renamed_parameter("column", "exprs", version="0.18.7")
def cumsum(
    exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr
) -> Expr | Series:
    """
    Cumulatively sum all values.

    If a single string is passed, this is an alias for ``pl.col(name).cumsum()``.

    If a single Series is passed, this is an alias for ``Series.cumsum()``.
    **This functionality is deprecated**.

    Otherwise, this function computes the cumulative sum horizontally across multiple
    columns.
    **This functionality is deprecated**, use ``pl.cumsum_horizontal`` instead.

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    See Also
    --------
    cumsum_horizontal

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [4, 5, 6],
    ...     }
    ... )
    >>> df.select(pl.cumsum("a"))
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 3   │
    │ 6   │
    └─────┘

    """
    if not more_exprs:
        if isinstance(exprs, pl.Series):
            issue_deprecation_warning(
                "passing a Series to `cumsum` is deprecated. Use `Series.cumsum()` instead.",
                version="0.18.7",
            )
            return exprs.cumsum()
        elif isinstance(exprs, str):
            return F.col(exprs).cumsum()

    _warn_for_deprecated_horizontal_use("cumsum")
    return F.cumsum_horizontal(exprs, *more_exprs)


def _warn_for_deprecated_horizontal_use(name: str) -> None:
    issue_deprecation_warning(
        f"using `{name}` for horizontal computation is deprecated. Use `{name}_horizontal` instead.",
        version="0.18.7",
    )

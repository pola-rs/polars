from __future__ import annotations

import contextlib
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload

import polars._reexport as pl
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    Date,
    Datetime,
    Duration,
    Int64,
    Time,
    UInt32,
    is_polars_dtype,
)
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_as_list_of_expressions,
)
from polars.utils._wrap import wrap_df, wrap_expr
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
)
from polars.utils.decorators import deprecated_alias

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    import sys

    from polars import DataFrame, Expr, LazyFrame, Series
    from polars.type_aliases import (
        CorrelationMethod,
        EpochTimeUnit,
        IntoExpr,
        PolarsDataType,
        PythonLiteral,
        RollingInterpolationMethod,
        TimeUnit,
    )

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


def col(
    name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
    *more_names: str | PolarsDataType,
) -> Expr:
    """
    Return an expression representing column(s) in a dataframe.

    Parameters
    ----------
    name
        The name or datatype of the column(s) to represent. Accepts regular expression
        input. Regular expressions should start with ``^`` and end with ``$``.
    *more_names
        Additional names or datatypes of columns to represent, specified as positional
        arguments.

    Examples
    --------
    Pass a single column name to represent that column.

    >>> df = pl.DataFrame(
    ...     {
    ...         "ham": [1, 2, 3],
    ...         "hamburger": [11, 22, 33],
    ...         "foo": [3, 2, 1],
    ...         "bar": ["a", "b", "c"],
    ...     }
    ... )
    >>> df.select(pl.col("foo"))
    shape: (3, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    │ 2   │
    │ 1   │
    └─────┘

    Use the wildcard ``*`` to represent all columns.

    >>> df.select(pl.col("*"))
    shape: (3, 4)
    ┌─────┬───────────┬─────┬─────┐
    │ ham ┆ hamburger ┆ foo ┆ bar │
    │ --- ┆ ---       ┆ --- ┆ --- │
    │ i64 ┆ i64       ┆ i64 ┆ str │
    ╞═════╪═══════════╪═════╪═════╡
    │ 1   ┆ 11        ┆ 3   ┆ a   │
    │ 2   ┆ 22        ┆ 2   ┆ b   │
    │ 3   ┆ 33        ┆ 1   ┆ c   │
    └─────┴───────────┴─────┴─────┘
    >>> df.select(pl.col("*").exclude("ham"))
    shape: (3, 3)
    ┌───────────┬─────┬─────┐
    │ hamburger ┆ foo ┆ bar │
    │ ---       ┆ --- ┆ --- │
    │ i64       ┆ i64 ┆ str │
    ╞═══════════╪═════╪═════╡
    │ 11        ┆ 3   ┆ a   │
    │ 22        ┆ 2   ┆ b   │
    │ 33        ┆ 1   ┆ c   │
    └───────────┴─────┴─────┘

    Regular expression input is supported.

    >>> df.select(pl.col("^ham.*$"))
    shape: (3, 2)
    ┌─────┬───────────┐
    │ ham ┆ hamburger │
    │ --- ┆ ---       │
    │ i64 ┆ i64       │
    ╞═════╪═══════════╡
    │ 1   ┆ 11        │
    │ 2   ┆ 22        │
    │ 3   ┆ 33        │
    └─────┴───────────┘

    Multiple columns can be represented by passing a list of names.

    >>> df.select(pl.col(["hamburger", "foo"]))
    shape: (3, 2)
    ┌───────────┬─────┐
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    │ 22        ┆ 2   │
    │ 33        ┆ 1   │
    └───────────┴─────┘

    Or use positional arguments to represent multiple columns in the same way.

    >>> df.select(pl.col("hamburger", "foo"))
    shape: (3, 2)
    ┌───────────┬─────┐
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    │ 22        ┆ 2   │
    │ 33        ┆ 1   │
    └───────────┴─────┘

    Easily select all columns that match a certain data type by passing that datatype.

    >>> df.select(pl.col(pl.Utf8))
    shape: (3, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    │ b   │
    │ c   │
    └─────┘
    >>> df.select(pl.col(pl.Int64, pl.Float64))
    shape: (3, 3)
    ┌─────┬───────────┬─────┐
    │ ham ┆ hamburger ┆ foo │
    │ --- ┆ ---       ┆ --- │
    │ i64 ┆ i64       ┆ i64 │
    ╞═════╪═══════════╪═════╡
    │ 1   ┆ 11        ┆ 3   │
    │ 2   ┆ 22        ┆ 2   │
    │ 3   ┆ 33        ┆ 1   │
    └─────┴───────────┴─────┘

    """
    if more_names:
        if isinstance(name, str):
            names_str = [name]
            names_str.extend(more_names)  # type: ignore[arg-type]
            return wrap_expr(plr.cols(names_str))
        elif is_polars_dtype(name):
            dtypes = [name]
            dtypes.extend(more_names)
            return wrap_expr(plr.dtype_cols(dtypes))
        else:
            raise TypeError(
                f"Invalid input for `col`. Expected `str` or `DataType`, got {type(name)!r}"
            )

    if isinstance(name, str):
        return wrap_expr(plr.col(name))
    elif is_polars_dtype(name):
        return wrap_expr(plr.dtype_cols([name]))
    elif isinstance(name, Iterable):
        names = list(name)
        if not names:
            return wrap_expr(plr.cols(names))

        item = names[0]
        if isinstance(item, str):
            return wrap_expr(plr.cols(names))
        elif is_polars_dtype(item):
            return wrap_expr(plr.dtype_cols(names))
        else:
            raise TypeError(
                "Invalid input for `col`. Expected iterable of type `str` or `DataType`,"
                f" got iterable of type {type(item)!r}"
            )
    else:
        raise TypeError(
            f"Invalid input for `col`. Expected `str` or `DataType`, got {type(name)!r}"
        )


def element() -> Expr:
    """
    Alias for an element being evaluated in an `eval` expression.

    Examples
    --------
    A horizontal rank computation by taking the elements of a list

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
    >>> df.with_columns(
    ...     pl.concat_list(["a", "b"]).list.eval(pl.element().rank()).alias("rank")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬────────────┐
    │ a   ┆ b   ┆ rank       │
    │ --- ┆ --- ┆ ---        │
    │ i64 ┆ i64 ┆ list[f32]  │
    ╞═════╪═════╪════════════╡
    │ 1   ┆ 4   ┆ [1.0, 2.0] │
    │ 8   ┆ 5   ┆ [2.0, 1.0] │
    │ 3   ┆ 2   ┆ [2.0, 1.0] │
    └─────┴─────┴────────────┘

    A mathematical operation on array elements

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
    >>> df.with_columns(
    ...     pl.concat_list(["a", "b"]).list.eval(pl.element() * 2).alias("a_b_doubled")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────────────┐
    │ a   ┆ b   ┆ a_b_doubled │
    │ --- ┆ --- ┆ ---         │
    │ i64 ┆ i64 ┆ list[i64]   │
    ╞═════╪═════╪═════════════╡
    │ 1   ┆ 4   ┆ [2, 8]      │
    │ 8   ┆ 5   ┆ [16, 10]    │
    │ 3   ┆ 2   ┆ [6, 4]      │
    └─────┴─────┴─────────────┘

    """
    return col("")


@overload
def count(column: str) -> Expr:
    ...


@overload
def count(column: Series) -> int:
    ...


@overload
def count(column: None = None) -> Expr:
    ...


def count(column: str | Series | None = None) -> Expr | int:
    """
    Count the number of values in this column/context.

    .. warning::
        `null` is deemed a value in this context.

    Parameters
    ----------
    column
        If dtype is:

        * ``pl.Series`` : count the values in the series.
        * ``str`` : count the values in this column.
        * ``None`` : count the number of values in this context.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.count())
    shape: (1, 1)
    ┌───────┐
    │ count │
    │ ---   │
    │ u32   │
    ╞═══════╡
    │ 3     │
    └───────┘
    >>> df.groupby("c", maintain_order=True).agg(pl.count())
    shape: (2, 2)
    ┌─────┬───────┐
    │ c   ┆ count │
    │ --- ┆ ---   │
    │ str ┆ u32   │
    ╞═════╪═══════╡
    │ foo ┆ 2     │
    │ bar ┆ 1     │
    └─────┴───────┘

    """
    if column is None:
        return wrap_expr(plr.count())

    if isinstance(column, pl.Series):
        return column.len()
    return col(column).count()


def implode(name: str) -> Expr:
    """
    Aggregate all column values into a list.

    Parameters
    ----------
    name
        Name of the column that should be imploded.

    """
    return col(name).implode()


@overload
def std(column: str, ddof: int = 1) -> Expr:
    ...


@overload
def std(column: Series, ddof: int = 1) -> float | None:
    ...


def std(column: str | Series, ddof: int = 1) -> Expr | float | None:
    """
    Get the standard deviation.

    Parameters
    ----------
    column
        Column to get the standard deviation from.
    ddof
        “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
        where N represents the number of elements.
        By default ddof is 1.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.std("a"))
    shape: (1, 1)
    ┌──────────┐
    │ a        │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 3.605551 │
    └──────────┘
    >>> df["a"].std()
    3.605551275463989

    """
    if isinstance(column, pl.Series):
        return column.std(ddof)
    return col(column).std(ddof)


@overload
def var(column: str, ddof: int = 1) -> Expr:
    ...


@overload
def var(column: Series, ddof: int = 1) -> float | None:
    ...


def var(column: str | Series, ddof: int = 1) -> Expr | float | None:
    """
    Get the variance.

    Parameters
    ----------
    column
        Column to get the variance of.
    ddof
        “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
        where N represents the number of elements.
        By default ddof is 1.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.var("a"))
    shape: (1, 1)
    ┌──────┐
    │ a    │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 13.0 │
    └──────┘
    >>> df["a"].var()
    13.0

    """
    if isinstance(column, pl.Series):
        return column.var(ddof)
    return col(column).var(ddof)


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
            return col(exprs).max()

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
            return col(exprs).min()

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
            return col(exprs).sum()

    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.sum_exprs(exprs))


@overload
def mean(column: str) -> Expr:
    ...


@overload
def mean(column: Series) -> float:
    ...


def mean(column: str | Series) -> Expr | float | None:
    """
    Get the mean value.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.mean("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 4.0 │
    └─────┘
    >>> pl.mean(df["a"])
    4.0

    """
    if isinstance(column, pl.Series):
        return column.mean()
    return col(column).mean()


@overload
def avg(column: str) -> Expr:
    ...


@overload
def avg(column: Series) -> float:
    ...


def avg(column: str | Series) -> Expr | float:
    """
    Alias for mean.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.avg("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 4.0 │
    └─────┘
    >>> pl.avg(df["a"])
    4.0

    """
    return mean(column)


@overload
def median(column: str) -> Expr:
    ...


@overload
def median(column: Series) -> float | int:
    ...


def median(column: str | Series) -> Expr | float | int | None:
    """
    Get the median value.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.median("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 3.0 │
    └─────┘
    >>> pl.median(df["a"])
    3.0

    """
    if isinstance(column, pl.Series):
        return column.median()
    return col(column).median()


@overload
def n_unique(column: str) -> Expr:
    ...


@overload
def n_unique(column: Series) -> int:
    ...


def n_unique(column: str | Series) -> Expr | int:
    """
    Count unique values.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 1], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.n_unique("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    └─────┘
    >>> pl.n_unique(df["a"])
    2

    """
    if isinstance(column, pl.Series):
        return column.n_unique()
    return col(column).n_unique()


def approx_unique(column: str | Expr) -> Expr:
    """
    Approx count unique values.

    This is done using the HyperLogLog++ algorithm for cardinality estimation.

    Parameters
    ----------
    column
        Column name or Series.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 1], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.approx_unique("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    └─────┘

    """
    if isinstance(column, pl.Expr):
        return column.approx_unique()
    return col(column).approx_unique()


@overload
def first(column: str) -> Expr:
    ...


@overload
def first(column: Series) -> Any:
    ...


@overload
def first(column: None = None) -> Expr:
    ...


def first(column: str | Series | None = None) -> Expr | Any:
    """
    Get the first value.

    Depending on the input type this function does different things:

    input:

    - None -> expression to take first column of a context.
    - str -> syntactic sugar for `pl.col(..).first()`
    - Series -> Take first value in `Series`

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.first())
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 8   │
    │ 3   │
    └─────┘
    >>> df.select(pl.first("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    └─────┘
    >>> pl.first(df["a"])
    1

    """
    if column is None:
        return wrap_expr(plr.first())

    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


@overload
def last(column: str) -> Expr:
    ...


@overload
def last(column: Series) -> Any:
    ...


@overload
def last(column: None = None) -> Expr:
    ...


def last(column: str | Series | None = None) -> Expr:
    """
    Get the last value.

    Depending on the input type this function does different things:

    - None -> expression to take last column of a context.
    - str -> syntactic sugar for `pl.col(..).last()`
    - Series -> Take last value in `Series`

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.last())
    shape: (3, 1)
    ┌─────┐
    │ c   │
    │ --- │
    │ str │
    ╞═════╡
    │ foo │
    │ bar │
    │ foo │
    └─────┘
    >>> df.select(pl.last("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    └─────┘
    >>> pl.last(df["a"])
    3

    """
    if column is None:
        return wrap_expr(plr.last())

    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


@overload
def head(column: str, n: int = ...) -> Expr:
    ...


@overload
def head(column: Series, n: int = ...) -> Series:
    ...


def head(column: str | Series, n: int = 10) -> Expr | Series:
    """
    Get the first `n` rows.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to return.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.head("a"))
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 8   │
    │ 3   │
    └─────┘
    >>> df.select(pl.head("a", 2))
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 8   │
    └─────┘
    >>> pl.head(df["a"], 2)
    shape: (2,)
    Series: 'a' [i64]
    [
        1
        8
    ]

    """
    if isinstance(column, pl.Series):
        return column.head(n)
    return col(column).head(n)


@overload
def tail(column: str, n: int = ...) -> Expr:
    ...


@overload
def tail(column: Series, n: int = ...) -> Series:
    ...


def tail(column: str | Series, n: int = 10) -> Expr | Series:
    """
    Get the last `n` rows.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to return.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.tail("a"))
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 8   │
    │ 3   │
    └─────┘
    >>> df.select(pl.tail("a", 2))
    shape: (2, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 8   │
    │ 3   │
    └─────┘
    >>> pl.tail(df["a"], 2)
    shape: (2,)
    Series: 'a' [i64]
    [
        8
        3
    ]

    """
    if isinstance(column, pl.Series):
        return column.tail(n)
    return col(column).tail(n)


def lit(
    value: Any, dtype: PolarsDataType | None = None, *, allow_object: bool = False
) -> Expr:
    """
    Return an expression representing a literal value.

    Parameters
    ----------
    value
        Value that should be used as a `literal`.
    dtype
        Optionally define a dtype.
    allow_object
        If type is unknown use an 'object' type.
        By default, we will raise a `ValueException`
        if the type is unknown.

    Examples
    --------
    Literal scalar values:

    >>> pl.lit(1)  # doctest: +IGNORE_RESULT
    >>> pl.lit(5.5)  # doctest: +IGNORE_RESULT
    >>> pl.lit(None)  # doctest: +IGNORE_RESULT
    >>> pl.lit("foo_bar")  # doctest: +IGNORE_RESULT
    >>> pl.lit(date(2021, 1, 20))  # doctest: +IGNORE_RESULT
    >>> pl.lit(datetime(2023, 3, 31, 10, 30, 45))  # doctest: +IGNORE_RESULT

    Literal list/Series data (1D):

    >>> pl.lit([1, 2, 3])  # doctest: +IGNORE_RESULT
    >>> pl.lit(pl.Series("x", [1, 2, 3]))  # doctest: +IGNORE_RESULT

    Literal list/Series data (2D):

    >>> pl.lit([[1, 2], [3, 4]])  # doctest: +IGNORE_RESULT
    >>> pl.lit(pl.Series("y", [[1, 2], [3, 4]]))  # doctest: +IGNORE_RESULT

    Expected datatypes

    - ''pl.lit([])'' -> empty  Series Float32
    - ''pl.lit([1, 2, 3])'' -> Series Int64
    - ''pl.lit([[]])''-> empty  Series List<Null>
    - ''pl.lit([[1, 2, 3]])'' -> Series List<i64>
    - ''pl.lit(None)'' -> Series Null

    """
    time_unit: TimeUnit

    if isinstance(value, datetime):
        time_unit = "us" if dtype is None else getattr(dtype, "time_unit", "us")
        time_zone = (
            value.tzinfo
            if getattr(dtype, "time_zone", None) is None
            else getattr(dtype, "time_zone", None)
        )
        if (
            value.tzinfo is not None
            and getattr(dtype, "time_zone", None) is not None
            and dtype.time_zone != str(value.tzinfo)  # type: ignore[union-attr]
        ):
            raise TypeError(
                f"Time zone of dtype ({dtype.time_zone}) differs from time zone of value ({value.tzinfo})."  # type: ignore[union-attr]
            )
        e = lit(_datetime_to_pl_timestamp(value, time_unit)).cast(Datetime(time_unit))
        if time_zone is not None:
            return e.dt.replace_time_zone(str(time_zone))
        else:
            return e

    elif isinstance(value, timedelta):
        time_unit = "us" if dtype is None else getattr(dtype, "time_unit", "us")
        return lit(_timedelta_to_pl_timedelta(value, time_unit)).cast(
            Duration(time_unit)
        )

    elif isinstance(value, time):
        return lit(_time_to_pl_time(value)).cast(Time)

    elif isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)

    elif isinstance(value, pl.Series):
        name = value.name
        value = value._s
        e = wrap_expr(plr.lit(value, allow_object))
        if name == "":
            return e
        return e.alias(name)

    elif (_check_for_numpy(value) and isinstance(value, np.ndarray)) or isinstance(
        value, (list, tuple)
    ):
        return lit(pl.Series("", value))

    elif dtype:
        return wrap_expr(plr.lit(value, allow_object)).cast(dtype)

    try:
        # numpy literals like np.float32(0) have item/dtype
        item = value.item()

        # numpy item() is py-native datetime/timedelta when units < 'ns'
        if isinstance(item, (datetime, timedelta)):
            return lit(item)

        # handle 'ns' units
        if isinstance(item, int) and hasattr(value, "dtype"):
            dtype_name = value.dtype.name
            if dtype_name.startswith(("datetime64[", "timedelta64[")):
                time_unit = dtype_name[11:-1]
                return lit(item).cast(
                    Datetime(time_unit)
                    if dtype_name.startswith("date")
                    else Duration(time_unit)
                )

    except AttributeError:
        item = value
    return wrap_expr(plr.lit(item, allow_object))


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
            return col(exprs).cumsum()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    # (Expr): use u32 as that will not cast to float as eagerly
    return cumfold(lit(0).cast(UInt32), lambda a, b: a + b, exprs_wrapped).alias(
        "cumsum"
    )


def corr(
    a: str | Expr,
    b: str | Expr,
    *,
    method: CorrelationMethod = "pearson",
    ddof: int = 1,
    propagate_nans: bool = False,
) -> Expr:
    """
    Compute the Pearson's or Spearman rank correlation correlation between two columns.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
        where N represents the number of elements.
        By default ddof is 1.
    method : {'pearson', 'spearman'}
        Correlation method.
    propagate_nans
        If `True` any `NaN` encountered will lead to `NaN` in the output.
        Defaults to `False` where `NaN` are regarded as larger than any finite number
        and thus lead to the highest rank.

    Examples
    --------
    Pearson's correlation:

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.corr("a", "b"))
    shape: (1, 1)
    ┌──────────┐
    │ a        │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 0.544705 │
    └──────────┘

    Spearman rank correlation:

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.corr("a", "b", method="spearman"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f32 │
    ╞═════╡
    │ 0.5 │
    └─────┘
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)

    if method == "pearson":
        return wrap_expr(plr.pearson_corr(a._pyexpr, b._pyexpr, ddof))
    elif method == "spearman":
        return wrap_expr(
            plr.spearman_rank_corr(a._pyexpr, b._pyexpr, ddof, propagate_nans)
        )
    else:
        raise ValueError(
            f"method must be one of {{'pearson', 'spearman'}}, got {method!r}"
        )


def cov(a: str | Expr, b: str | Expr) -> Expr:
    """
    Compute the covariance between two columns/ expressions.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.cov("a", "b"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 3.0 │
    └─────┘

    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(plr.cov(a._pyexpr, b._pyexpr))


def map(
    exprs: Sequence[str] | Sequence[Expr],
    function: Callable[[Sequence[Series]], Series],
    return_dtype: PolarsDataType | None = None,
) -> Expr:
    """
    Map a custom function over multiple columns/expressions.

    Produces a single Series result.

    Parameters
    ----------
    exprs
        Input Series to f
    function
        Function to apply over the input
    return_dtype
        dtype of the output Series

    Returns
    -------
    Expr

    Examples
    --------
    >>> def test_func(a, b, c):
    ...     return a + b + c
    ...
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3, 4],
    ...         "b": [4, 5, 6, 7],
    ...     }
    ... )
    >>>
    >>> df.with_columns(
    ...     (
    ...         pl.struct(["a", "b"]).map(
    ...             lambda x: test_func(x.struct.field("a"), x.struct.field("b"), 1)
    ...         )
    ...     ).alias("a+b+c")
    ... )
    shape: (4, 3)
    ┌─────┬─────┬───────┐
    │ a   ┆ b   ┆ a+b+c │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ i64 ┆ i64   │
    ╞═════╪═════╪═══════╡
    │ 1   ┆ 4   ┆ 6     │
    │ 2   ┆ 5   ┆ 8     │
    │ 3   ┆ 6   ┆ 10    │
    │ 4   ┆ 7   ┆ 12    │
    └─────┴─────┴───────┘
    """
    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(
        plr.map_mul(
            exprs, function, return_dtype, apply_groups=False, returns_scalar=False
        )
    )


def apply(
    exprs: Sequence[str | Expr],
    function: Callable[[Sequence[Series]], Series | Any],
    return_dtype: PolarsDataType | None = None,
    *,
    returns_scalar: bool = True,
) -> Expr:
    """
    Apply a custom/user-defined function (UDF) in a GroupBy context.

    Depending on the context it has the following behavior:

    * Select
        Don't use apply, use `map`
    * GroupBy
        expected type `f`: Callable[[Series], Series]
        Applies a python function over each group.

    Parameters
    ----------
    exprs
        Input Series to f
    function
        Function to apply over the input
    return_dtype
        dtype of the output Series
    returns_scalar
        If the function returns a single scalar as output.

    Returns
    -------
    Expr

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [7, 2, 3, 4],
    ...         "b": [2, 5, 6, 7],
    ...     }
    ... )
    >>> df
    shape: (4, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 7   ┆ 2   │
    │ 2   ┆ 5   │
    │ 3   ┆ 6   │
    │ 4   ┆ 7   │
    └─────┴─────┘

    Calculate product of ``a``.

    >>> df.with_columns(pl.col("a").apply(lambda x: x * x).alias("product_a"))
    shape: (4, 3)
    ┌─────┬─────┬───────────┐
    │ a   ┆ b   ┆ product_a │
    │ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64       │
    ╞═════╪═════╪═══════════╡
    │ 7   ┆ 2   ┆ 49        │
    │ 2   ┆ 5   ┆ 4         │
    │ 3   ┆ 6   ┆ 9         │
    │ 4   ┆ 7   ┆ 16        │
    └─────┴─────┴───────────┘
    """
    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(
        plr.map_mul(
            exprs,
            function,
            return_dtype,
            apply_groups=True,
            returns_scalar=returns_scalar,
        )
    )


def fold(
    acc: IntoExpr,
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
) -> Expr:
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    acc
        Accumulator Expression. This is the value that will be initialized when the fold
        starts. For a sum this could for instance be lit(0).
    function
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    Notes
    -----
    If you simply want the first encountered expression as accumulator,
    consider using ``reduce``.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [3, 4, 5],
    ...         "c": [5, 6, 7],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   │
    │ 2   ┆ 4   ┆ 6   │
    │ 3   ┆ 5   ┆ 7   │
    └─────┴─────┴─────┘

    Horizontally sum over all columns and add 1.

    >>> df.select(
    ...     pl.fold(
    ...         acc=pl.lit(1), function=lambda acc, x: acc + x, exprs=pl.col("*")
    ...     ).alias("sum"),
    ... )
    shape: (3, 1)
    ┌─────┐
    │ sum │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 10  │
    │ 13  │
    │ 16  │
    └─────┘

    You can also apply a condition/predicate on all columns:

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [0, 1, 2],
    ...     }
    ... )
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 0   │
    │ 2   ┆ 1   │
    │ 3   ┆ 2   │
    └─────┴─────┘

    >>> df.filter(
    ...     pl.fold(
    ...         acc=pl.lit(True),
    ...         function=lambda acc, x: acc & x,
    ...         exprs=pl.col("*") > 1,
    ...     )
    ... )
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 2   │
    └─────┴─────┘
    """
    # in case of pl.col("*")
    acc = parse_as_expression(acc, str_as_lit=True)
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.fold(acc, function, exprs))


def reduce(
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
) -> Expr:
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    function
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    Notes
    -----
    See ``fold`` for the version with an explicit accumulator.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [0, 1, 2],
    ...     }
    ... )
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 0   │
    │ 2   ┆ 1   │
    │ 3   ┆ 2   │
    └─────┴─────┘

    Horizontally sum over all columns.

    >>> df.select(
    ...     pl.reduce(function=lambda acc, x: acc + x, exprs=pl.col("*")).alias("sum"),
    ... )
    shape: (3, 1)
    ┌─────┐
    │ sum │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 3   │
    │ 5   │
    └─────┘

    """
    # in case of pl.col("*")
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.reduce(function, exprs))


def cumfold(
    acc: IntoExpr,
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
    *,
    include_init: bool = False,
) -> Expr:
    """
    Cumulatively accumulate over multiple columns horizontally/ row wise with a left fold.

    Every cumulative result is added as a separate field in a Struct column.

    Parameters
    ----------
    acc
        Accumulator Expression. This is the value that will be initialized when the fold
        starts. For a sum this could for instance be lit(0).
    function
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.
    include_init
        Include the initial accumulator state as struct field.

    Notes
    -----
    If you simply want the first encountered expression as accumulator,
    consider using ``cumreduce``.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [3, 4, 5],
    ...         "c": [5, 6, 7],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   │
    │ 2   ┆ 4   ┆ 6   │
    │ 3   ┆ 5   ┆ 7   │
    └─────┴─────┴─────┘

    >>> df.select(
    ...     pl.cumfold(
    ...         acc=pl.lit(1), function=lambda acc, x: acc + x, exprs=pl.col("*")
    ...     ).alias("cumfold"),
    ... )
    shape: (3, 1)
    ┌───────────┐
    │ cumfold   │
    │ ---       │
    │ struct[3] │
    ╞═══════════╡
    │ {2,5,10}  │
    │ {3,7,13}  │
    │ {4,9,16}  │
    └───────────┘

    """  # noqa: W505
    # in case of pl.col("*")
    acc = parse_as_expression(acc, str_as_lit=True)
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cumfold(acc, function, exprs, include_init))


def cumreduce(
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
) -> Expr:
    """
    Cumulatively accumulate over multiple columns horizontally/ row wise with a left fold.

    Every cumulative result is added as a separate field in a Struct column.

    Parameters
    ----------
    function
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [3, 4, 5],
    ...         "c": [5, 6, 7],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   │
    │ 2   ┆ 4   ┆ 6   │
    │ 3   ┆ 5   ┆ 7   │
    └─────┴─────┴─────┘

    >>> df.select(
    ...     pl.cumreduce(function=lambda acc, x: acc + x, exprs=pl.col("*")).alias(
    ...         "cumreduce"
    ...     ),
    ... )
    shape: (3, 1)
    ┌───────────┐
    │ cumreduce │
    │ ---       │
    │ struct[3] │
    ╞═══════════╡
    │ {1,4,9}   │
    │ {2,6,12}  │
    │ {3,8,15}  │
    └───────────┘
    """  # noqa: W505
    # in case of pl.col("*")
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cumreduce(function, exprs))


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
            return col(exprs).any()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return fold(
        lit(False), lambda a, b: a.cast(bool) | b.cast(bool), exprs_wrapped
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
            return col("*")
        elif isinstance(exprs, pl.Series):
            return exprs.all()
        elif isinstance(exprs, str):
            return col(exprs).all()

    pyexprs = parse_as_list_of_expressions(exprs, *more_exprs)
    exprs_wrapped = [wrap_expr(e) for e in pyexprs]

    return fold(
        lit(True), lambda a, b: a.cast(bool) & b.cast(bool), exprs_wrapped
    ).alias("all")


def exclude(
    columns: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
    *more_columns: str | PolarsDataType,
) -> Expr:
    """
    Represent all columns except for the given columns.

    Syntactic sugar for ``pl.all().exclude(columns)``.

    Parameters
    ----------
    columns
        The name or datatype of the column(s) to exclude. Accepts regular expression
        input. Regular expressions should start with ``^`` and end with ``$``.
    *more_columns
        Additional names or datatypes of columns to exclude, specified as positional
        arguments.

    Examples
    --------
    Exclude by column name(s):

    >>> df = pl.DataFrame(
    ...     {
    ...         "aa": [1, 2, 3],
    ...         "ba": ["a", "b", None],
    ...         "cc": [None, 2.5, 1.5],
    ...     }
    ... )
    >>> df.select(pl.exclude("ba"))
    shape: (3, 2)
    ┌─────┬──────┐
    │ aa  ┆ cc   │
    │ --- ┆ ---  │
    │ i64 ┆ f64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    │ 2   ┆ 2.5  │
    │ 3   ┆ 1.5  │
    └─────┴──────┘

    Exclude by regex, e.g. removing all columns whose names end with the letter "a":

    >>> df.select(pl.exclude("^.*a$"))
    shape: (3, 1)
    ┌──────┐
    │ cc   │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ null │
    │ 2.5  │
    │ 1.5  │
    └──────┘

    Exclude by dtype(s), e.g. removing all columns of type Int64 or Float64:

    >>> df.select(pl.exclude([pl.Int64, pl.Float64]))
    shape: (3, 1)
    ┌──────┐
    │ ba   │
    │ ---  │
    │ str  │
    ╞══════╡
    │ a    │
    │ b    │
    │ null │
    └──────┘

    """
    return col("*").exclude(columns, *more_columns)


def groups(column: str) -> Expr:
    """Syntactic sugar for `pl.col("foo").agg_groups()`."""
    return col(column).agg_groups()


def quantile(
    column: str,
    quantile: float | Expr,
    interpolation: RollingInterpolationMethod = "nearest",
) -> Expr:
    """
    Syntactic sugar for `pl.col("foo").quantile(..)`.

    Parameters
    ----------
    column
        Column name.
    quantile
        Quantile between 0.0 and 1.0.
    interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
        Interpolation method.

    """
    return col(column).quantile(quantile, interpolation)


def arg_sort_by(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    descending: bool | Sequence[bool] = False,
) -> Expr:
    """
    Return the row indices that would sort the columns.

    Parameters
    ----------
    exprs
        Column(s) to arg sort by. Accepts expression input. Strings are parsed as column
        names.
    *more_exprs
        Additional columns to arg sort by, specified as positional arguments.
    descending
        Sort in descending order. When sorting by multiple columns, can be specified
        per column by passing a sequence of booleans.

    Examples
    --------
    Pass a single column name to compute the arg sort by that column.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [0, 1, 1, 0],
    ...         "b": [3, 2, 3, 2],
    ...     }
    ... )
    >>> df.select(pl.arg_sort_by("a"))
    shape: (4, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 0   │
    │ 3   │
    │ 1   │
    │ 2   │
    └─────┘

    Compute the arg sort by multiple columns by either passing a list of columns, or by
    specifying each column as a positional argument.

    >>> df.select(pl.arg_sort_by(["a", "b"], descending=True))
    shape: (4, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    │ 1   │
    │ 0   │
    │ 3   │
    └─────┘

    """
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)

    if isinstance(descending, bool):
        descending = [descending] * len(exprs)
    elif len(exprs) != len(descending):
        raise ValueError(
            f"the length of `descending` ({len(descending)}) does not match the length of `exprs` ({len(exprs)})"
        )
    return wrap_expr(plr.arg_sort_by(exprs, descending))


def collect_all(
    lazy_frames: Sequence[LazyFrame],
    *,
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    no_optimization: bool = False,
    slice_pushdown: bool = True,
    common_subplan_elimination: bool = True,
    streaming: bool = False,
) -> list[DataFrame]:
    """
    Collect multiple LazyFrames at the same time.

    This runs all the computation graphs in parallel on Polars threadpool.

    Parameters
    ----------
    lazy_frames
        A list of LazyFrames to collect.
    type_coercion
        Do type coercion optimization.
    predicate_pushdown
        Do predicate pushdown optimization.
    projection_pushdown
        Do projection pushdown optimization.
    simplify_expression
        Run simplify expressions optimization.
    no_optimization
        Turn off optimizations.
    slice_pushdown
        Slice pushdown optimization.
    common_subplan_elimination
        Will try to cache branching subplans that occur on self-joins or unions.
    streaming
        Run parts of the query in a streaming fashion (this is in an alpha state)

    Returns
    -------
    List[DataFrame]

    """
    if no_optimization:
        predicate_pushdown = False
        projection_pushdown = False
        slice_pushdown = False
        common_subplan_elimination = False

    prepared = []

    for lf in lazy_frames:
        ldf = lf._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            common_subplan_elimination,
            streaming,
        )
        prepared.append(ldf)

    out = plr.collect_all(prepared)

    # wrap the pydataframes into dataframe
    result = [wrap_df(pydf) for pydf in out]

    return result


def select(*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> DataFrame:
    """
    Run polars expressions without a context.

    This is syntactic sugar for running ``df.select`` on an empty DataFrame.

    Parameters
    ----------
    *exprs
        Column(s) to select, specified as positional arguments.
        Accepts expression input. Strings are parsed as column names,
        other non-expression inputs are parsed as literals.
    **named_exprs
        Additional columns to select, specified as keyword arguments.
        The columns will be renamed to the keyword used.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> foo = pl.Series("foo", [1, 2, 3])
    >>> bar = pl.Series("bar", [3, 2, 1])
    >>> pl.select(pl.min([foo, bar]))
    shape: (3, 1)
    ┌─────┐
    │ min │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 1   │
    └─────┘

    """
    return pl.DataFrame().select(*exprs, **named_exprs)


@overload
def arg_where(condition: Expr | Series, *, eager: Literal[False] = ...) -> Expr:
    ...


@overload
def arg_where(condition: Expr | Series, *, eager: Literal[True]) -> Series:
    ...


@overload
def arg_where(condition: Expr | Series, *, eager: bool) -> Expr | Series:
    ...


def arg_where(condition: Expr | Series, *, eager: bool = False) -> Expr | Series:
    """
    Return indices where `condition` evaluates `True`.

    Parameters
    ----------
    condition
        Boolean expression to evaluate
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    >>> df.select(
    ...     [
    ...         pl.arg_where(pl.col("a") % 2 == 0),
    ...     ]
    ... ).to_series()
    shape: (2,)
    Series: 'a' [u32]
    [
        1
        3
    ]

    See Also
    --------
    Series.arg_true : Return indices where Series is True

    """
    if eager:
        if not isinstance(condition, pl.Series):
            raise ValueError(
                "expected 'Series' in 'arg_where' if 'eager=True', got"
                f" {type(condition)}"
            )
        return condition.to_frame().select(arg_where(col(condition.name))).to_series()
    else:
        condition = parse_as_expression(condition)
        return wrap_expr(plr.arg_where(condition))


def coalesce(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """
    Folds the columns from left to right, keeping the first non-null value.

    Parameters
    ----------
    exprs
        Columns to coalesce. Accepts expression input. Strings are parsed as column
        names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to coalesce, specified as positional arguments.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, None, None, None],
    ...         "b": [1, 2, None, None],
    ...         "c": [5, None, 3, None],
    ...     }
    ... )
    >>> df.with_columns(pl.coalesce(["a", "b", "c", 10]).alias("d"))
    shape: (4, 4)
    ┌──────┬──────┬──────┬─────┐
    │ a    ┆ b    ┆ c    ┆ d   │
    │ ---  ┆ ---  ┆ ---  ┆ --- │
    │ i64  ┆ i64  ┆ i64  ┆ i64 │
    ╞══════╪══════╪══════╪═════╡
    │ 1    ┆ 1    ┆ 5    ┆ 1   │
    │ null ┆ 2    ┆ null ┆ 2   │
    │ null ┆ null ┆ 3    ┆ 3   │
    │ null ┆ null ┆ null ┆ 10  │
    └──────┴──────┴──────┴─────┘
    >>> df.with_columns(pl.coalesce(pl.col(["a", "b", "c"]), 10.0).alias("d"))
    shape: (4, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ d    │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ 5    ┆ 1.0  │
    │ null ┆ 2    ┆ null ┆ 2.0  │
    │ null ┆ null ┆ 3    ┆ 3.0  │
    │ null ┆ null ┆ null ┆ 10.0 │
    └──────┴──────┴──────┴──────┘

    """
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.coalesce(exprs))


@overload
def from_epoch(column: str | Expr, time_unit: EpochTimeUnit = ...) -> Expr:
    ...


@overload
def from_epoch(
    column: Series | Sequence[int], time_unit: EpochTimeUnit = ...
) -> Series:
    ...


def from_epoch(
    column: str | Expr | Series | Sequence[int], time_unit: EpochTimeUnit = "s"
) -> Expr | Series:
    """
    Utility function that parses an epoch timestamp (or Unix time) to Polars Date(time).

    Depending on the `time_unit` provided, this function will return a different dtype:
    - time_unit="d" returns pl.Date
    - time_unit="s" returns pl.Datetime["us"] (pl.Datetime's default)
    - time_unit="ms" returns pl.Datetime["ms"]
    - time_unit="us" returns pl.Datetime["us"]
    - time_unit="ns" returns pl.Datetime["ns"]

    Parameters
    ----------
    column
        Series or expression to parse integers to pl.Datetime.
    time_unit
        The unit of time of the timesteps since epoch time.

    Examples
    --------
    >>> df = pl.DataFrame({"timestamp": [1666683077, 1666683099]}).lazy()
    >>> df.select(pl.from_epoch(pl.col("timestamp"), time_unit="s")).collect()
    shape: (2, 1)
    ┌─────────────────────┐
    │ timestamp           │
    │ ---                 │
    │ datetime[μs]        │
    ╞═════════════════════╡
    │ 2022-10-25 07:31:17 │
    │ 2022-10-25 07:31:39 │
    └─────────────────────┘

    The function can also be used in an eager context by passing a Series.

    >>> s = pl.Series([12345, 12346])
    >>> pl.from_epoch(s, time_unit="d")
    shape: (2,)
    Series: '' [date]
    [
            2003-10-20
            2003-10-21
    ]

    """
    if isinstance(column, str):
        column = col(column)
    elif not isinstance(column, (pl.Series, pl.Expr)):
        column = pl.Series(column)  # Sequence input handled by Series constructor

    if time_unit == "d":
        return column.cast(Date)
    elif time_unit == "s":
        return (column.cast(Int64) * 1_000_000).cast(Datetime("us"))
    elif time_unit in DTYPE_TEMPORAL_UNITS:
        return column.cast(Datetime(time_unit))
    else:
        raise ValueError(
            f"'time_unit' must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got {time_unit!r}."
        )


def rolling_cov(
    a: str | Expr,
    b: str | Expr,
    *,
    window_size: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> Expr:
    """
    Compute the rolling covariance between two columns/ expressions.

    The window at a given row includes the row itself and the
    `window_size - 1` elements before it.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    window_size
        The length of the window.
    min_periods
        The number of values in the window that should be non-null before computing
        a result. If None, it will be set equal to window size.
    ddof
        Delta degrees of freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.

    """
    if min_periods is None:
        min_periods = window_size
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(
        plr.rolling_cov(a._pyexpr, b._pyexpr, window_size, min_periods, ddof)
    )


def rolling_corr(
    a: str | Expr,
    b: str | Expr,
    *,
    window_size: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> Expr:
    """
    Compute the rolling correlation between two columns/ expressions.

    The window at a given row includes the row itself and the
    `window_size - 1` elements before it.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    window_size
        The length of the window.
    min_periods
        The number of values in the window that should be non-null before computing
        a result. If None, it will be set equal to window size.
    ddof
        Delta degrees of freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.

    """
    if min_periods is None:
        min_periods = window_size
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(
        plr.rolling_corr(a._pyexpr, b._pyexpr, window_size, min_periods, ddof)
    )


def sql_expr(sql: str) -> Expr:
    """
    Parse a SQL expression to a polars expression.

    Parameters
    ----------
    sql
        SQL expression

    Examples
    --------
    >>> df = pl.DataFrame({"a": [2, 1]})
    >>> expr = pl.sql_expr("MAX(a)")
    >>> df.select(expr)
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 2   │
    └─────┘
    """
    return wrap_expr(plr.sql_expr(sql))

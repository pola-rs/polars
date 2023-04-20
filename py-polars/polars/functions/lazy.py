from __future__ import annotations

import contextlib
import warnings
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload

from polars import internals as pli
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    Date,
    Datetime,
    Duration,
    Int32,
    Int64,
    Struct,
    Time,
    UInt32,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.utils._parse_expr_input import expr_to_lit_or_expr, selection_to_pyexpr_list
from polars.utils._wrap import wrap_df, wrap_expr
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
)
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import arange as pyarange
    from polars.polars import arg_sort_by as py_arg_sort_by
    from polars.polars import arg_where as py_arg_where
    from polars.polars import as_struct as _as_struct
    from polars.polars import coalesce_exprs as _coalesce_exprs
    from polars.polars import col as pycol
    from polars.polars import collect_all as _collect_all
    from polars.polars import cols as pycols
    from polars.polars import concat_lst as _concat_lst
    from polars.polars import concat_str as _concat_str
    from polars.polars import count as _count
    from polars.polars import cov as pycov
    from polars.polars import cumfold as pycumfold
    from polars.polars import cumreduce as pycumreduce
    from polars.polars import dtype_cols as _dtype_cols
    from polars.polars import first as _first
    from polars.polars import fold as pyfold
    from polars.polars import last as _last
    from polars.polars import lit as pylit
    from polars.polars import map_mul as _map_mul
    from polars.polars import max_exprs as _max_exprs
    from polars.polars import min_exprs as _min_exprs
    from polars.polars import pearson_corr as pypearson_corr
    from polars.polars import py_datetime, py_duration
    from polars.polars import reduce as pyreduce
    from polars.polars import repeat as _repeat
    from polars.polars import spearman_rank_corr as pyspearman_rank_corr
    from polars.polars import sum_exprs as _sum_exprs


if TYPE_CHECKING:
    import sys

    from polars.dataframe import DataFrame
    from polars.expr.expr import Expr
    from polars.lazyframe import LazyFrame
    from polars.series import Series
    from polars.type_aliases import (
        CorrelationMethod,
        EpochTimeUnit,
        IntoExpr,
        PolarsDataType,
        PythonLiteral,
        RollingInterpolationMethod,
        SchemaDict,
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
            return wrap_expr(pycols(names_str))
        elif is_polars_dtype(name):
            dtypes = [name]
            dtypes.extend(more_names)
            return wrap_expr(_dtype_cols(dtypes))
        else:
            raise TypeError(
                f"Invalid input for `col`. Expected `str` or `DataType`, got {type(name)!r}"
            )

    if isinstance(name, str):
        return wrap_expr(pycol(name))
    elif is_polars_dtype(name):
        return wrap_expr(_dtype_cols([name]))
    elif isinstance(name, Iterable):
        names = list(name)
        if not names:
            return wrap_expr(pycols(names))

        item = names[0]
        if isinstance(item, str):
            return wrap_expr(pycols(names))
        elif is_polars_dtype(item):
            return wrap_expr(_dtype_cols(names))
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
    ...     pl.concat_list(["a", "b"]).arr.eval(pl.element().rank()).alias("rank")
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
    ...     pl.concat_list(["a", "b"]).arr.eval(pl.element() * 2).alias("a_b_doubled")
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
        return wrap_expr(_count())

    if isinstance(column, pli.Series):
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


def list_(name: str) -> Expr:
    """
    Aggregate to list.

    .. deprecated:: 0.17.3
        ``list`` will be removed in favor of ``implode``.

    Parameters
    ----------
    name
        Name of the column that should be aggregated into a list.

    """
    warnings.warn(
        "`pl.list` is deprecated, please use `pl.implode` instead.",
        DeprecationWarning,
        stacklevel=find_stacklevel(),
    )
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
    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Series):
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

    If a single column is passed, get the maximum value of that column (vertical).
    If multiple columns are passed, get the maximum value of each row (horizontal).

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    Get the maximum value by columns with a string column name.

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.max("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 8   │
    └─────┘

    Get the maximum value by row with a list of columns/expressions.

    >>> df.select(pl.max(["a", "b"]))
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

    To aggregate maximums for more than one column/expression use ``pl.col(list).max()``
    or a regular expression selector like ``pl.sum(regex)``:

    >>> df.select(pl.col(["a", "b"]).max())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 8   ┆ 5   │
    └─────┴─────┘

    >>> df.select(pl.max("^.*[ab]$"))
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
        if isinstance(exprs, pli.Series):
            return exprs.max()
        elif isinstance(exprs, str):
            return col(exprs).max()
        elif not isinstance(exprs, Iterable):
            return lit(exprs).max()

    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    return wrap_expr(_max_exprs(exprs))


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

    If a single column is passed, get the minimum value of that column (vertical).
    If multiple columns are passed, get the minimum value of each row (horizontal).

    Parameters
    ----------
    exprs
        Column(s) to use in the aggregation. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to use in the aggregation, specified as positional arguments.

    Examples
    --------
    Get the minimum value by columns with a string column name.

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

    Get the minimum value by row with a list of columns/expressions.

    >>> df.select(pl.min(["a", "b"]))
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

    To aggregate minimums for more than one column/expression use ``pl.col(list).min()``
    or a regular expression selector like ``pl.sum(regex)``:

    >>> df.select(pl.col(["a", "b"]).min())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 2   │
    └─────┴─────┘

    >>> df.select(pl.min("^.*[ab]$"))
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
        if isinstance(exprs, pli.Series):
            return exprs.min()
        elif isinstance(exprs, str):
            return col(exprs).min()
        elif not isinstance(exprs, Iterable):
            return lit(exprs).min()

    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    return wrap_expr(_min_exprs(exprs))


@overload
def sum(column: str | Sequence[Expr | str] | Expr) -> Expr:
    ...


@overload
def sum(column: Series) -> int | float:
    ...


def sum(
    column: str | Sequence[Expr | str] | Series | Expr,
) -> Expr | Any:
    """
    Sum values in a column/Series, or horizontally across list of columns/expressions.

    ``pl.sum(str)`` is syntactic sugar for:

    >>> pl.col(str).sum()  # doctest: +SKIP

    ``pl.sum(list)`` is syntactic sugar for:

    >>> pl.fold(pl.lit(0), lambda x, y: x + y, list).alias("sum")  # doctest: +SKIP

    Parameters
    ----------
    column
        Column(s) to be used in aggregation.
        This can be:

        - a column name, or Series -> aggregate the sum value of that column/Series.
        - a List[Expr] -> aggregate the sum value horizontally across the Expr result.

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

    >>> df.with_columns(pl.sum(["a", "c"]))
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

    >>> df.select(pl.col(["a", "c"]).sum())
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
    if isinstance(column, pli.Series):
        return column.sum()
    elif isinstance(column, str):
        return col(column).sum()
    elif isinstance(column, Sequence):
        exprs = selection_to_pyexpr_list(column)
        return wrap_expr(_sum_exprs(exprs))
    else:
        # (Expr): use u32 as that will not cast to float as eagerly
        return fold(lit(0).cast(UInt32), lambda a, b: a + b, column).alias("sum")


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
    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Expr):
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
        return wrap_expr(_first())

    if isinstance(column, pli.Series):
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
        return wrap_expr(_last())

    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Series):
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
    if isinstance(column, pli.Series):
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
        if value.tzinfo is not None and getattr(dtype, "time_zone", None) is not None:
            raise TypeError(
                "Cannot cast tz-aware value to tz-aware dtype. "
                "Please drop the time zone from the dtype."
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

    elif isinstance(value, pli.Series):
        name = value.name
        value = value._s
        e = wrap_expr(pylit(value, allow_object))
        if name == "":
            return e
        return e.alias(name)

    elif (_check_for_numpy(value) and isinstance(value, np.ndarray)) or isinstance(
        value, (list, tuple)
    ):
        return lit(pli.Series("", value))

    elif dtype:
        return wrap_expr(pylit(value, allow_object)).cast(dtype)

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
    return wrap_expr(pylit(item, allow_object))


@overload
def cumsum(column: str | Sequence[Expr | str] | Expr) -> Expr:
    ...


@overload
def cumsum(column: Series) -> int | float:
    ...


def cumsum(
    column: str | Sequence[Expr | str] | Series | Expr,
) -> Expr | Any:
    """
    Cumulatively sum values in a column/Series, or horizontally across list of columns/expressions.

    ``pl.cumsum(str)`` is syntactic sugar for:

    >>> pl.col(str).cumsum()  # doctest: +SKIP

    ``pl.cumsum(list)`` is syntactic sugar for:

    >>> pl.cumfold(pl.lit(0), lambda x, y: x + y, list).alias(
    ...     "cumsum"
    ... )  # doctest: +SKIP

    Parameters
    ----------
    column
        Column(s) to be used in aggregation.
        This can be:

        - a column name, or Series -> aggregate the sum value of that column/Series.
        - a List[Expr] -> aggregate the sum value horizontally across the Expr result.

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

    >>> df.with_columns(pl.cumsum(["a", "c"]))
    shape: (2, 4)
    ┌─────┬─────┬─────┬───────────┐
    │ a   ┆ b   ┆ c   ┆ cumsum    │
    │ --- ┆ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ struct[2] │
    ╞═════╪═════╪═════╪═══════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {1,6}     │
    │ 2   ┆ 4   ┆ 6   ┆ {2,8}     │
    └─────┴─────┴─────┴───────────┘

    """  # noqa: W505
    if isinstance(column, pli.Series):
        return column.cumsum()
    elif isinstance(column, str):
        return col(column).cumsum()
    # (Expr): use u32 as that will not cast to float as eagerly
    return cumfold(lit(0).cast(UInt32), lambda a, b: a + b, column).alias("cumsum")


def spearman_rank_corr(
    a: str | Expr, b: str | Expr, ddof: int = 1, *, propagate_nans: bool = False
) -> Expr:
    """
    Compute the spearman rank correlation between two columns.

    Missing data will be excluded from the computation.

    .. deprecated:: 0.16.10
        ``spearman_rank_corr`` will be removed in favor of
        ``corr(..., method="spearman")``.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
        where N represents the number of elements.
        By default ddof is 1.
    propagate_nans
        If `True` any `NaN` encountered will lead to `NaN` in the output.
        Defaults to `False` where `NaN` are regarded as larger than any finite number
        and thus lead to the highest rank.

    See Also
    --------
    corr

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.spearman_rank_corr("a", "b"))  # doctest: +SKIP
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 0.5 │
    └─────┘
    """
    warnings.warn(
        "`spearman_rank_corr()` is deprecated in favor of `corr()`",
        DeprecationWarning,
        stacklevel=find_stacklevel(),
    )
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(pyspearman_rank_corr(a._pyexpr, b._pyexpr, ddof, propagate_nans))


def pearson_corr(a: str | Expr, b: str | Expr, ddof: int = 1) -> Expr:
    """
    Compute the pearson's correlation between two columns.

    .. deprecated:: 0.16.10
        ``pearson_corr`` will be removed in favor of ``corr(..., method="pearson")``.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
        where N represents the number of elements.
        By default ddof is 1.

    See Also
    --------
    corr

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.pearson_corr("a", "b"))  # doctest: +SKIP
    shape: (1, 1)
    ┌──────────┐
    │ a        │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 0.544705 │
    └──────────┘
    """
    warnings.warn(
        "`pearson_corr()` is deprecated in favor of `corr()`",
        DeprecationWarning,
        stacklevel=find_stacklevel(),
    )
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr, ddof))


def corr(
    a: str | Expr,
    b: str | Expr,
    *,
    method: CorrelationMethod = "pearson",
    ddof: int = 1,
    propagate_nans: bool = False,
) -> Expr:
    """
    Compute the pearson's or spearman rank correlation correlation between two columns.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
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
    │ f64 │
    ╞═════╡
    │ 0.5 │
    └─────┘
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)

    if method == "pearson":
        return wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr, ddof))
    elif method == "spearman":
        return wrap_expr(
            pyspearman_rank_corr(a._pyexpr, b._pyexpr, ddof, propagate_nans)
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
    return wrap_expr(pycov(a._pyexpr, b._pyexpr))


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
    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(
        _map_mul(
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
    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(
        _map_mul(
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
    acc = expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(pyfold(acc._pyexpr, function, exprs))


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
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(pyreduce(function, exprs))


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
    acc = expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(pycumfold(acc._pyexpr, function, exprs, include_init))


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
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = selection_to_pyexpr_list(exprs)
    return wrap_expr(pycumreduce(function, exprs))


def any(columns: str | Sequence[str] | Sequence[Expr] | Expr) -> Expr:
    """
    Evaluate columnwise or elementwise with a bitwise OR operation.

    Parameters
    ----------
    columns
        If given this function will apply a bitwise or on the columns.

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

    """
    if isinstance(columns, str):
        return col(columns).any()
    else:
        return fold(
            lit(False), lambda a, b: a.cast(bool) | b.cast(bool), columns
        ).alias("any")


def all(columns: str | Sequence[Expr] | Expr | None = None) -> Expr:
    """
    Do one of two things.

    * function can do a columnwise or elementwise AND operation
    * a wildcard column selection

    Parameters
    ----------
    columns
        If given this function will apply a bitwise & on the columns.

    Examples
    --------
    Sum all columns

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

    """
    if columns is None:
        return col("*")
    elif isinstance(columns, str):
        return col(columns).all()
    else:
        return fold(lit(True), lambda a, b: a.cast(bool) & b.cast(bool), columns).alias(
            "all"
        )


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


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    eager: Literal[False],
) -> Expr:
    ...


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    eager: Literal[True],
    dtype: PolarsDataType | None = ...,
) -> Series:
    ...


@overload
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = ...,
    *,
    eager: bool = ...,
    dtype: PolarsDataType | None = ...,
) -> Expr | Series:
    ...


@deprecated_alias(low="start", high="end")
def arange(
    start: int | Expr | Series,
    end: int | Expr | Series,
    step: int = 1,
    *,
    eager: bool = False,
    dtype: PolarsDataType | None = None,
) -> Expr | Series:
    """
    Create a range expression (or Series).

    This can be used in a `select`, `with_column` etc. Be sure that the resulting
    range size is equal to the length of the DataFrame you are collecting.

    Examples
    --------
    >>> df.lazy().filter(pl.col("foo") < pl.arange(0, 100)).collect()  # doctest: +SKIP

    Parameters
    ----------
    start
        Lower bound of range.
    end
        Upper bound of range.
    step
        Step size of the range.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    dtype
        Apply an explicit integer dtype to the resulting expression (default is Int64).

    """
    start = expr_to_lit_or_expr(start, str_to_lit=False)
    end = expr_to_lit_or_expr(end, str_to_lit=False)
    range_expr = wrap_expr(pyarange(start._pyexpr, end._pyexpr, step))

    if dtype is not None and dtype != Int64:
        range_expr = range_expr.cast(dtype)
    if not eager:
        return range_expr
    else:
        return (
            pli.DataFrame()
            .select(range_expr)
            .to_series()
            .rename("arange", in_place=True)
        )


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
    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))

    if isinstance(descending, bool):
        descending = [descending] * len(exprs)
    elif len(exprs) != len(descending):
        raise ValueError(
            f"the length of `descending` ({len(descending)}) does not match the length of `exprs` ({len(exprs)})"
        )
    return wrap_expr(py_arg_sort_by(exprs, descending))


def duration(
    *,
    days: Expr | str | int | None = None,
    seconds: Expr | str | int | None = None,
    nanoseconds: Expr | str | int | None = None,
    microseconds: Expr | str | int | None = None,
    milliseconds: Expr | str | int | None = None,
    minutes: Expr | str | int | None = None,
    hours: Expr | str | int | None = None,
    weeks: Expr | str | int | None = None,
) -> Expr:
    """
    Create polars `Duration` from distinct time components.

    Returns
    -------
    Expr of type `pl.Duration`

    Examples
    --------
    >>> from datetime import datetime
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
    ...         "add": [1, 2],
    ...     }
    ... )
    >>> print(df)
    shape: (2, 2)
    ┌─────────────────────┬─────┐
    │ dt                  ┆ add │
    │ ---                 ┆ --- │
    │ datetime[μs]        ┆ i64 │
    ╞═════════════════════╪═════╡
    │ 2022-01-01 00:00:00 ┆ 1   │
    │ 2022-01-02 00:00:00 ┆ 2   │
    └─────────────────────┴─────┘
    >>> with pl.Config(tbl_width_chars=120):
    ...     df.select(
    ...         (pl.col("dt") + pl.duration(weeks="add")).alias("add_weeks"),
    ...         (pl.col("dt") + pl.duration(days="add")).alias("add_days"),
    ...         (pl.col("dt") + pl.duration(seconds="add")).alias("add_seconds"),
    ...         (pl.col("dt") + pl.duration(milliseconds="add")).alias("add_millis"),
    ...         (pl.col("dt") + pl.duration(hours="add")).alias("add_hours"),
    ...     )
    ...
    shape: (2, 5)
    ┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────────┬─────────────────────┐
    │ add_weeks           ┆ add_days            ┆ add_seconds         ┆ add_millis              ┆ add_hours           │
    │ ---                 ┆ ---                 ┆ ---                 ┆ ---                     ┆ ---                 │
    │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]            ┆ datetime[μs]        │
    ╞═════════════════════╪═════════════════════╪═════════════════════╪═════════════════════════╪═════════════════════╡
    │ 2022-01-08 00:00:00 ┆ 2022-01-02 00:00:00 ┆ 2022-01-01 00:00:01 ┆ 2022-01-01 00:00:00.001 ┆ 2022-01-01 01:00:00 │
    │ 2022-01-16 00:00:00 ┆ 2022-01-04 00:00:00 ┆ 2022-01-02 00:00:02 ┆ 2022-01-02 00:00:00.002 ┆ 2022-01-02 02:00:00 │
    └─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┴─────────────────────┘

    """  # noqa: W505
    if hours is not None:
        hours = expr_to_lit_or_expr(hours, str_to_lit=False)._pyexpr
    if minutes is not None:
        minutes = expr_to_lit_or_expr(minutes, str_to_lit=False)._pyexpr
    if seconds is not None:
        seconds = expr_to_lit_or_expr(seconds, str_to_lit=False)._pyexpr
    if milliseconds is not None:
        milliseconds = expr_to_lit_or_expr(milliseconds, str_to_lit=False)._pyexpr
    if microseconds is not None:
        microseconds = expr_to_lit_or_expr(microseconds, str_to_lit=False)._pyexpr
    if nanoseconds is not None:
        nanoseconds = expr_to_lit_or_expr(nanoseconds, str_to_lit=False)._pyexpr
    if days is not None:
        days = expr_to_lit_or_expr(days, str_to_lit=False)._pyexpr
    if weeks is not None:
        weeks = expr_to_lit_or_expr(weeks, str_to_lit=False)._pyexpr

    return wrap_expr(
        py_duration(
            days,
            seconds,
            nanoseconds,
            microseconds,
            milliseconds,
            minutes,
            hours,
            weeks,
        )
    )


def datetime_(
    year: Expr | str | int,
    month: Expr | str | int,
    day: Expr | str | int,
    hour: Expr | str | int | None = None,
    minute: Expr | str | int | None = None,
    second: Expr | str | int | None = None,
    microsecond: Expr | str | int | None = None,
) -> Expr:
    """
    Create a Polars literal expression of type Datetime.

    Parameters
    ----------
    year
        column or literal.
    month
        column or literal, ranging from 1-12.
    day
        column or literal, ranging from 1-31.
    hour
        column or literal, ranging from 1-23.
    minute
        column or literal, ranging from 1-59.
    second
        column or literal, ranging from 1-59.
    microsecond
        column or literal, ranging from 1-999999.

    Returns
    -------
    Expr of type `pl.Datetime`

    """
    year_expr = expr_to_lit_or_expr(year, str_to_lit=False)
    month_expr = expr_to_lit_or_expr(month, str_to_lit=False)
    day_expr = expr_to_lit_or_expr(day, str_to_lit=False)

    if hour is not None:
        hour = expr_to_lit_or_expr(hour, str_to_lit=False)._pyexpr
    if minute is not None:
        minute = expr_to_lit_or_expr(minute, str_to_lit=False)._pyexpr
    if second is not None:
        second = expr_to_lit_or_expr(second, str_to_lit=False)._pyexpr
    if microsecond is not None:
        microsecond = expr_to_lit_or_expr(microsecond, str_to_lit=False)._pyexpr

    return wrap_expr(
        py_datetime(
            year_expr._pyexpr,
            month_expr._pyexpr,
            day_expr._pyexpr,
            hour,
            minute,
            second,
            microsecond,
        )
    )


def date_(
    year: Expr | str | int,
    month: Expr | str | int,
    day: Expr | str | int,
) -> Expr:
    """
    Create a Polars literal expression of type Date.

    Parameters
    ----------
    year
        column or literal.
    month
        column or literal, ranging from 1-12.
    day
        column or literal, ranging from 1-31.

    Returns
    -------
    Expr of type pl.Date

    """
    return datetime_(year, month, day).cast(Date).alias("date")


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
) -> Expr:
    """
    Horizontally concatenate columns into a single string column.

    Operates in linear time.

    Parameters
    ----------
    exprs
        Columns to concatenate into a single string column. Accepts expression input.
        Strings are parsed as column names, other non-expression inputs are parsed as
        literals. Non-``Utf8`` columns are cast to ``Utf8``.
    *more_exprs
        Additional columns to concatenate into a single string column, specified as
        positional arguments.
    separator
        String that will be used to separate the values of each column.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": ["dogs", "cats", None],
    ...         "c": ["play", "swim", "walk"],
    ...     }
    ... )
    >>> df.with_columns(
    ...     pl.concat_str(
    ...         [
    ...             pl.col("a") * 2,
    ...             pl.col("b"),
    ...             pl.col("c"),
    ...         ],
    ...         separator=" ",
    ...     ).alias("full_sentence"),
    ... )
    shape: (3, 4)
    ┌─────┬──────┬──────┬───────────────┐
    │ a   ┆ b    ┆ c    ┆ full_sentence │
    │ --- ┆ ---  ┆ ---  ┆ ---           │
    │ i64 ┆ str  ┆ str  ┆ str           │
    ╞═════╪══════╪══════╪═══════════════╡
    │ 1   ┆ dogs ┆ play ┆ 2 dogs play   │
    │ 2   ┆ cats ┆ swim ┆ 4 cats swim   │
    │ 3   ┆ null ┆ walk ┆ null          │
    └─────┴──────┴──────┴───────────────┘

    """
    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    return wrap_expr(_concat_str(exprs, separator))


def format(f_string: str, *args: Expr | str) -> Expr:
    """
    Format expressions as a string.

    Parameters
    ----------
    f_string
        A string that with placeholders.
        For example: "hello_{}" or "{}_world
    args
        Expression(s) that fill the placeholders

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": ["a", "b", "c"],
    ...         "b": [1, 2, 3],
    ...     }
    ... )
    >>> df.select(
    ...     [
    ...         pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt"),
    ...     ]
    ... )
    shape: (3, 1)
    ┌─────────────┐
    │ fmt         │
    │ ---         │
    │ str         │
    ╞═════════════╡
    │ foo_a_bar_1 │
    │ foo_b_bar_2 │
    │ foo_c_bar_3 │
    └─────────────┘

    """
    if f_string.count("{}") != len(args):
        raise ValueError("number of placeholders should equal the number of arguments")

    exprs = []

    arguments = iter(args)
    for i, s in enumerate(f_string.split("{}")):
        if i > 0:
            e = expr_to_lit_or_expr(next(arguments), str_to_lit=False)
            exprs.append(e)

        if len(s) > 0:
            exprs.append(lit(s))

    return concat_str(exprs, separator="")


def concat_list(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """
    Horizontally concatenate columns into a single list column.

    Operates in linear time.

    Parameters
    ----------
    exprs
        Columns to concatenate into a single list column. Accepts expression input.
        Strings are parsed as column names, other non-expression inputs are parsed as
        literals.
    *more_exprs
        Additional columns to concatenate into a single list column, specified as
        positional arguments.

    Examples
    --------
    Create lagged columns and collect them into a list. This mimics a rolling window.

    >>> df = pl.DataFrame({"A": [1.0, 2.0, 9.0, 2.0, 13.0]})
    >>> df = df.select([pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)])
    >>> df.select(
    ...     pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")
    ... )
    shape: (5, 1)
    ┌───────────────────┐
    │ A_rolling         │
    │ ---               │
    │ list[f64]         │
    ╞═══════════════════╡
    │ [null, null, 1.0] │
    │ [null, 1.0, 2.0]  │
    │ [1.0, 2.0, 9.0]   │
    │ [2.0, 9.0, 2.0]   │
    │ [9.0, 2.0, 13.0]  │
    └───────────────────┘

    """
    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    return wrap_expr(_concat_lst(exprs))


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

    out = _collect_all(prepared)

    # wrap the pydataframes into dataframe
    result = [wrap_df(pydf) for pydf in out]

    return result


def select(
    exprs: IntoExpr | Iterable[IntoExpr] | None = None,
    *more_exprs: IntoExpr,
    **named_exprs: IntoExpr,
) -> DataFrame:
    """
    Run polars expressions without a context.

    This is syntactic sugar for running ``df.select`` on an empty DataFrame.

    Parameters
    ----------
    exprs
        Expression or expressions to run.
    *more_exprs
        Additional expressions to run, specified as positional arguments.
    **named_exprs
        Additional expressions to run, specified as keyword arguments. The expressions
        will be renamed to the keyword used.

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
    return pli.DataFrame().select(exprs, *more_exprs, **named_exprs)


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: Literal[False] = ...,
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Expr:
    ...


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: Literal[True],
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Series:
    ...


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: bool,
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Expr | Series:
    ...


def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = None,
    *more_exprs: IntoExpr,
    eager: bool = False,
    schema: SchemaDict | None = None,
    **named_exprs: IntoExpr,
) -> Expr | Series:
    """
    Collect columns into a struct column.

    Parameters
    ----------
    exprs
        Column(s) to collect into a struct column. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to collect into the struct column, specified as positional
        arguments.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    schema
        Optional schema that explicitly defines the struct field dtypes.
    **named_exprs
        Additional columns to collect into the struct column, specified as keyword
        arguments. The columns will be renamed to the keyword used.

    Examples
    --------
    Collect all columns of a dataframe into a struct by passing ``pl.all()``.

    >>> df = pl.DataFrame(
    ...     {
    ...         "int": [1, 2],
    ...         "str": ["a", "b"],
    ...         "bool": [True, None],
    ...         "list": [[1, 2], [3]],
    ...     }
    ... )
    >>> df.select(pl.struct(pl.all()).alias("my_struct"))
    shape: (2, 1)
    ┌─────────────────────┐
    │ my_struct           │
    │ ---                 │
    │ struct[4]           │
    ╞═════════════════════╡
    │ {1,"a",true,[1, 2]} │
    │ {2,"b",null,[3]}    │
    └─────────────────────┘

    Collect selected columns into a struct by either passing a list of columns, or by
    specifying each column as a positional argument.

    >>> df.select(pl.struct("int", False).alias("my_struct"))
    shape: (2, 1)
    ┌───────────┐
    │ my_struct │
    │ ---       │
    │ struct[2] │
    ╞═══════════╡
    │ {1,false} │
    │ {2,false} │
    └───────────┘

    Use keyword arguments to easily name each struct field.

    >>> df.select(pl.struct(p="int", q="bool").alias("my_struct")).schema
    {'my_struct': Struct([Field('p', Int64), Field('q', Boolean)])}

    """
    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    if named_exprs:
        exprs.extend(
            expr_to_lit_or_expr(expr, name=name, str_to_lit=False)._pyexpr
            for name, expr in named_exprs.items()
        )

    expr = wrap_expr(_as_struct(exprs))
    if schema:
        expr = expr.cast(Struct(schema), strict=False)

    if eager:
        return select(expr).to_series()
    else:
        return expr


@overload
def repeat(
    value: float | int | str | bool | None,
    n: Expr | int,
    *,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> Expr:
    ...


@overload
def repeat(
    value: float | int | str | bool | None,
    n: Expr | int,
    *,
    eager: Literal[True],
    name: str | None = ...,
) -> Series:
    ...


@overload
def repeat(
    value: float | int | str | bool | None,
    n: Expr | int,
    *,
    eager: bool,
    name: str | None,
) -> Expr | Series:
    ...


def repeat(
    value: float | int | str | bool | None,
    n: Expr | int,
    *,
    eager: bool = False,
    name: str | None = None,
) -> Expr | Series:
    """
    Repeat a single value n times.

    Parameters
    ----------
    value
        Value to repeat.
    n
        repeat `n` times
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    name
        Only used in `eager` mode. As expression, use `alias`

    """
    if eager:
        if name is None:
            name = ""
        dtype = py_type_to_dtype(type(value))
        if (
            dtype == Int64
            and isinstance(value, int)
            and -(2**31) <= value <= 2**31 - 1
        ):
            dtype = Int32
        s = pli.Series._repeat(name, value, n, dtype)  # type: ignore[arg-type]
        return s
    else:
        if isinstance(n, int):
            n = lit(n)
        return wrap_expr(_repeat(value, n._pyexpr))


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
        if not isinstance(condition, pli.Series):
            raise ValueError(
                "expected 'Series' in 'arg_where' if 'eager=True', got"
                f" {type(condition)}"
            )
        return condition.to_frame().select(arg_where(col(condition.name))).to_series()
    else:
        condition = expr_to_lit_or_expr(condition, str_to_lit=True)
        return wrap_expr(py_arg_where(condition._pyexpr))


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
    exprs = selection_to_pyexpr_list(exprs)
    if more_exprs:
        exprs.extend(selection_to_pyexpr_list(more_exprs))
    return wrap_expr(_coalesce_exprs(exprs))


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
    elif not isinstance(column, (pli.Series, pli.Expr)):
        column = pli.Series(column)  # Sequence input handled by Series constructor

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

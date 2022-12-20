from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable, Sequence, overload

from polars import internals as pli
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    DataType,
    DataTypeClass,
    Date,
    Datetime,
    Duration,
    Int64,
    PolarsDataType,
    Time,
    UInt32,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.dependencies import _NUMPY_TYPE
from polars.dependencies import numpy as np
from polars.internals.type_aliases import EpochTimeUnit
from polars.utils import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
    deprecated_alias,
)

try:
    from polars.polars import arange as pyarange
    from polars.polars import arg_where as py_arg_where
    from polars.polars import argsort_by as pyargsort_by
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

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from polars.internals.type_aliases import (
        IntoExpr,
        RollingInterpolationMethod,
        TimeUnit,
    )


def col(
    name: str | Sequence[str] | Sequence[PolarsDataType] | pli.Series | PolarsDataType,
) -> pli.Expr:
    """
    Return an expression representing a column in a DataFrame.

    Can be used to select:

    - a single column by name
    - all columns by using a wildcard `"*"`
    - column by regular expression if the regex starts with `^` and ends with `$`
    - all columns with the same dtype by using a Polars type

    Parameters
    ----------
    name
        A string that holds the name of the column

    Examples
    --------
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
    ├╌╌╌╌╌┤
    │ 2   │
    ├╌╌╌╌╌┤
    │ 1   │
    └─────┘
    >>> df.select(pl.col("*"))
    shape: (3, 4)
    ┌─────┬───────────┬─────┬─────┐
    │ ham ┆ hamburger ┆ foo ┆ bar │
    │ --- ┆ ---       ┆ --- ┆ --- │
    │ i64 ┆ i64       ┆ i64 ┆ str │
    ╞═════╪═══════════╪═════╪═════╡
    │ 1   ┆ 11        ┆ 3   ┆ a   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 22        ┆ 2   ┆ b   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 33        ┆ 1   ┆ c   │
    └─────┴───────────┴─────┴─────┘
    >>> df.select(pl.col("^ham.*$"))
    shape: (3, 2)
    ┌─────┬───────────┐
    │ ham ┆ hamburger │
    │ --- ┆ ---       │
    │ i64 ┆ i64       │
    ╞═════╪═══════════╡
    │ 1   ┆ 11        │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2   ┆ 22        │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 33        │
    └─────┴───────────┘
    >>> df.select(pl.col("*").exclude("ham"))
    shape: (3, 3)
    ┌───────────┬─────┬─────┐
    │ hamburger ┆ foo ┆ bar │
    │ ---       ┆ --- ┆ --- │
    │ i64       ┆ i64 ┆ str │
    ╞═══════════╪═════╪═════╡
    │ 11        ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 22        ┆ 2   ┆ b   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 33        ┆ 1   ┆ c   │
    └───────────┴─────┴─────┘
    >>> df.select(pl.col(["hamburger", "foo"]))
    shape: (3, 2)
    ┌───────────┬─────┐
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 22        ┆ 2   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 33        ┆ 1   │
    └───────────┴─────┘
    >>> # Select columns with a dtype
    >>> df.select(pl.col(pl.Utf8))
    shape: (3, 1)
    ┌─────┐
    │ bar │
    │ --- │
    │ str │
    ╞═════╡
    │ a   │
    ├╌╌╌╌╌┤
    │ b   │
    ├╌╌╌╌╌┤
    │ c   │
    └─────┘
    >>> # Select columns from a list of dtypes
    >>> df.select(pl.col([pl.Int64, pl.Float64]))
    shape: (3, 3)
    ┌─────┬───────────┬─────┐
    │ ham ┆ hamburger ┆ foo │
    │ --- ┆ ---       ┆ --- │
    │ i64 ┆ i64       ┆ i64 │
    ╞═════╪═══════════╪═════╡
    │ 1   ┆ 11        ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 22        ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 33        ┆ 1   │
    └─────┴───────────┴─────┘

    """
    if isinstance(name, pli.Series):
        name = name.to_list()  # type: ignore[assignment]

    if isinstance(name, DataTypeClass):
        name = [name]

    if isinstance(name, DataType):
        return pli.wrap_expr(_dtype_cols([name]))

    elif not isinstance(name, str) and isinstance(name, Sequence):
        if len(name) == 0 or isinstance(name[0], str):
            return pli.wrap_expr(pycols(name))
        elif is_polars_dtype(name[0]):
            return pli.wrap_expr(_dtype_cols(name))
        else:
            raise ValueError("Expected list values to be all `str` or all `DataType`")
    return pli.wrap_expr(pycol(name))


def element() -> pli.Expr:
    """
    Alias for an element being evaluated in an `eval` expression.

    Examples
    --------
    A horizontal rank computation by taking the elements of a list

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
    >>> df.with_column(
    ...     pl.concat_list(["a", "b"]).arr.eval(pl.element().rank()).alias("rank")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬────────────┐
    │ a   ┆ b   ┆ rank       │
    │ --- ┆ --- ┆ ---        │
    │ i64 ┆ i64 ┆ list[f32]  │
    ╞═════╪═════╪════════════╡
    │ 1   ┆ 4   ┆ [1.0, 2.0] │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 8   ┆ 5   ┆ [2.0, 1.0] │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 2   ┆ [2.0, 1.0] │
    └─────┴─────┴────────────┘

    A mathematical operation on array elements

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
    >>> df.with_column(
    ...     pl.concat_list(["a", "b"]).arr.eval(pl.element() * 2).alias("a_b_doubled")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────────────┐
    │ a   ┆ b   ┆ a_b_doubled │
    │ --- ┆ --- ┆ ---         │
    │ i64 ┆ i64 ┆ list[i64]   │
    ╞═════╪═════╪═════════════╡
    │ 1   ┆ 4   ┆ [2, 8]      │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 8   ┆ 5   ┆ [16, 10]    │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 2   ┆ [6, 4]      │
    └─────┴─────┴─────────────┘

    """
    return col("")


@overload
def count(column: str) -> pli.Expr:
    ...


@overload
def count(column: pli.Series) -> int:
    ...


@overload
def count(column: None = None) -> pli.Expr:
    ...


def count(column: str | pli.Series | None = None) -> pli.Expr | int:
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
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
    │ bar ┆ 1     │
    └─────┴───────┘

    """
    if column is None:
        return pli.wrap_expr(_count())

    if isinstance(column, pli.Series):
        return column.len()
    return col(column).count()


def to_list(name: str) -> pli.Expr:
    """
    Aggregate to list.

    Re-exported as `pl.list()`

    """
    return col(name).list()


@overload
def std(column: str, ddof: int = 1) -> pli.Expr:
    ...


@overload
def std(column: pli.Series, ddof: int = 1) -> float | None:
    ...


def std(column: str | pli.Series, ddof: int = 1) -> pli.Expr | float | None:
    """
    Get the standard deviation.

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
def var(column: str, ddof: int = 1) -> pli.Expr:
    ...


@overload
def var(column: pli.Series, ddof: int = 1) -> float | None:
    ...


def var(column: str | pli.Series, ddof: int = 1) -> pli.Expr | float | None:
    """
    Get the variance.

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
def max(column: str | Sequence[pli.Expr | str]) -> pli.Expr:
    ...


@overload
def max(column: pli.Series) -> int | float:
    ...


def max(column: str | Sequence[pli.Expr | str] | pli.Series) -> pli.Expr | Any:
    """
    Get the maximum value. Can be used horizontally or vertically.

    Parameters
    ----------
    column
        Column(s) to be used in aggregation. Will lead to different behavior based on
        the input:
        - Union[str, Series] -> aggregate the maximum value of that column.
        - List[Expr] -> aggregate the maximum value horizontally.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})

    Get the maximum value by columns with a string column name

    >>> df.select(pl.max("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 8   │
    └─────┘

    Get the maximum value by row with a list of columns/expressions

    >>> df.select(pl.max(["a", "b"]))
    shape: (3, 1)
    ┌─────┐
    │ max │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 4   │
    ├╌╌╌╌╌┤
    │ 8   │
    ├╌╌╌╌╌┤
    │ 3   │
    └─────┘

    To aggregate the maximums for more than one column/expression
    use ``pl.col(list).max()`` instead:

    >>> df.select(pl.col(["a", "b"]).max())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 8   ┆ 5   │
    └─────┴─────┘

    """
    if isinstance(column, pli.Series):
        return column.max()
    elif isinstance(column, str):
        return col(column).max()
    else:
        exprs = pli.selection_to_pyexpr_list(column)
        return pli.wrap_expr(_max_exprs(exprs))


@overload
def min(
    column: str | Sequence[pli.Expr | str | date | datetime | int | float],
) -> pli.Expr:
    ...


@overload
def min(column: pli.Series) -> int | float:
    ...


def min(
    column: str | Sequence[pli.Expr | str | date | datetime | int | float] | pli.Series,
) -> pli.Expr | Any:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on
        the input:
        - Union[str, Series] -> aggregate the sum value of that column.
        - List[Expr] -> aggregate the min value horizontally.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})

    Get the minimum value by columns with a string column name

    >>> df.select(pl.min("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    └─────┘

    Get the minimum value by row with a list of columns/expressions

    >>> df.select(pl.min(["a", "b"]))
    shape: (3, 1)
    ┌─────┐
    │ min │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    ├╌╌╌╌╌┤
    │ 5   │
    ├╌╌╌╌╌┤
    │ 2   │
    └─────┘

    To aggregate the minimums for more than one column/expression
    use ``pl.col(list).min()`` instead:

    >>> df.select(pl.col(["a", "b"]).min())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 2   │
    └─────┴─────┘

    """
    if isinstance(column, pli.Series):
        return column.min()
    elif isinstance(column, str):
        return col(column).min()
    else:
        exprs = pli.selection_to_pyexpr_list(column)
        return pli.wrap_expr(_min_exprs(exprs))


@overload
def sum(column: str | Sequence[pli.Expr | str] | pli.Expr) -> pli.Expr:
    ...


@overload
def sum(column: pli.Series) -> int | float:
    ...


def sum(
    column: str | Sequence[pli.Expr | str] | pli.Series | pli.Expr,
) -> pli.Expr | Any:
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
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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

    >>> df.with_column(pl.sum(["a", "c"]))
    shape: (2, 4)
    ┌─────┬─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   ┆ sum │
    │ --- ┆ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   ┆ 6   │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   ┆ 6   ┆ 8   │
    └─────┴─────┴─────┴─────┘

    Sum a series:

    >>> pl.sum(df.get_column("a"))
    3

    To aggregate the sums for more than one column/expression use ``pl.col(list).sum()``
    instead:

    >>> df.select(pl.col(["a", "c"]).sum())
    shape: (1, 2)
    ┌─────┬─────┐
    │ a   ┆ c   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 11  │
    └─────┴─────┘

    """
    if isinstance(column, pli.Series):
        return column.sum()
    elif isinstance(column, str):
        return col(column).sum()
    elif isinstance(column, Sequence):
        exprs = pli.selection_to_pyexpr_list(column)
        return pli.wrap_expr(_sum_exprs(exprs))
    else:
        # (Expr): use u32 as that will not cast to float as eagerly
        return fold(lit(0).cast(UInt32), lambda a, b: a + b, column).alias("sum")


@overload
def mean(column: str) -> pli.Expr:
    ...


@overload
def mean(column: pli.Series) -> float:
    ...


def mean(column: str | pli.Series) -> pli.Expr | float:
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
def avg(column: str) -> pli.Expr:
    ...


@overload
def avg(column: pli.Series) -> float:
    ...


def avg(column: str | pli.Series) -> pli.Expr | float:
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
def median(column: str) -> pli.Expr:
    ...


@overload
def median(column: pli.Series) -> float | int:
    ...


def median(column: str | pli.Series) -> pli.Expr | float | int:
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
def n_unique(column: str) -> pli.Expr:
    ...


@overload
def n_unique(column: pli.Series) -> int:
    ...


def n_unique(column: str | pli.Series) -> pli.Expr | int:
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


@overload
def first(column: str) -> pli.Expr:
    ...


@overload
def first(column: pli.Series) -> Any:
    ...


@overload
def first(column: None = None) -> pli.Expr:
    ...


def first(column: str | pli.Series | None = None) -> pli.Expr | Any:
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
    ├╌╌╌╌╌┤
    │ 8   │
    ├╌╌╌╌╌┤
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
        return pli.wrap_expr(_first())

    if isinstance(column, pli.Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


@overload
def last(column: str) -> pli.Expr:
    ...


@overload
def last(column: pli.Series) -> Any:
    ...


@overload
def last(column: None = None) -> pli.Expr:
    ...


def last(column: str | pli.Series | None = None) -> pli.Expr:
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
    ├╌╌╌╌╌┤
    │ bar │
    ├╌╌╌╌╌┤
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
        return pli.wrap_expr(_last())

    if isinstance(column, pli.Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


@overload
def head(column: str, n: int = 10) -> pli.Expr:
    ...


@overload
def head(column: pli.Series, n: int = 10) -> pli.Series:
    ...


def head(column: str | pli.Series, n: int = 10) -> pli.Expr | pli.Series:
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
    ├╌╌╌╌╌┤
    │ 8   │
    ├╌╌╌╌╌┤
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
    ├╌╌╌╌╌┤
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
def tail(column: str, n: int = 10) -> pli.Expr:
    ...


@overload
def tail(column: pli.Series, n: int = 10) -> pli.Series:
    ...


def tail(column: str | pli.Series, n: int = 10) -> pli.Expr | pli.Series:
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
    ├╌╌╌╌╌┤
    │ 8   │
    ├╌╌╌╌╌┤
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
    ├╌╌╌╌╌┤
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
    value: Any, dtype: type[DataType] | None = None, allow_object: bool = False
) -> pli.Expr:
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
    Literal integer:

    >>> pl.lit(1)  # doctest: +IGNORE_RESULT

    Literal str:

    >>> pl.lit("foo")  # doctest: +IGNORE_RESULT

    Literal datetime:

    >>> from datetime import datetime
    >>> pl.lit(datetime(2021, 1, 20))  # doctest: +IGNORE_RESULT

    Literal Null:

    >>> pl.lit(None)  # doctest: +IGNORE_RESULT

    Literal eager Series:

    >>> pl.lit(pl.Series("a", [1, 2, 3]))  # doctest: +IGNORE_RESULT

    """
    tu: TimeUnit
    if isinstance(value, datetime):
        tu = "us"
        e = lit(_datetime_to_pl_timestamp(value, tu)).cast(Datetime(tu))
        if value.tzinfo is not None:
            return e.dt.tz_localize(str(value.tzinfo))
        else:
            return e

    elif isinstance(value, timedelta):
        tu = "us"
        return lit(_timedelta_to_pl_timedelta(value, tu)).cast(Duration(tu))

    elif isinstance(value, time):
        return lit(_time_to_pl_time(value)).cast(Time)

    elif isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)

    elif isinstance(value, pli.Series):
        name = value.name
        value = value._s
        e = pli.wrap_expr(pylit(value, allow_object))
        if name == "":
            return e
        return e.alias(name)

    if _NUMPY_TYPE(value) and isinstance(value, np.ndarray):
        return lit(pli.Series("", value))

    if dtype:
        return pli.wrap_expr(pylit(value, allow_object)).cast(dtype)

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
                tu = dtype_name[11:-1]
                return lit(item).cast(
                    Datetime(tu) if dtype_name.startswith("date") else Duration(tu)
                )

    except AttributeError:
        item = value
    return pli.wrap_expr(pylit(item, allow_object))


@overload
def cumsum(column: str | Sequence[pli.Expr | str] | pli.Expr) -> pli.Expr:
    ...


@overload
def cumsum(column: pli.Series) -> int | float:
    ...


def cumsum(
    column: str | Sequence[pli.Expr | str] | pli.Series | pli.Expr,
) -> pli.Expr | Any:
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
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
    ├╌╌╌╌╌┤
    │ 3   │
    └─────┘

    Cumulatively sum a list of columns/expressions horizontally:

    >>> df.with_column(pl.cumsum(["a", "c"]))
    shape: (2, 4)
    ┌─────┬─────┬─────┬───────────┐
    │ a   ┆ b   ┆ c   ┆ cumsum    │
    │ --- ┆ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ struct[2] │
    ╞═════╪═════╪═════╪═══════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {1,6}     │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2   ┆ 4   ┆ 6   ┆ {2,8}     │
    └─────┴─────┴─────┴───────────┘

    """  # noqa E501
    if isinstance(column, pli.Series):
        return column.cumsum()
    elif isinstance(column, str):
        return col(column).cumsum()
    # (Expr): use u32 as that will not cast to float as eagerly
    return cumfold(lit(0).cast(UInt32), lambda a, b: a + b, column).alias("cumsum")


def spearman_rank_corr(
    a: str | pli.Expr, b: str | pli.Expr, ddof: int = 1, propagate_nans: bool = False
) -> pli.Expr:
    """
    Compute the spearman rank correlation between two columns.

    Missing data will be excluded from the computation.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        Delta degrees of freedom
    propagate_nans
        If `True` any `NaN` encountered will lead to `NaN` in the output.
        Defaults to `False` where `NaN` are regarded as larger than any finite number
        and thus lead to the highest rank.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.spearman_rank_corr("a", "b"))
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
    return pli.wrap_expr(
        pyspearman_rank_corr(a._pyexpr, b._pyexpr, ddof, propagate_nans)
    )


def pearson_corr(a: str | pli.Expr, b: str | pli.Expr, ddof: int = 1) -> pli.Expr:
    """
    Compute the pearson's correlation between two columns.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        Delta degrees of freedom

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.pearson_corr("a", "b"))
    shape: (1, 1)
    ┌──────────┐
    │ a        │
    │ ---      │
    │ f64      │
    ╞══════════╡
    │ 0.544705 │
    └──────────┘

    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pli.wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr, ddof))


def cov(
    a: str | pli.Expr,
    b: str | pli.Expr,
) -> pli.Expr:
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
    return pli.wrap_expr(pycov(a._pyexpr, b._pyexpr))


def map(
    exprs: Sequence[str] | Sequence[pli.Expr],
    f: Callable[[Sequence[pli.Series]], pli.Series],
    return_dtype: type[DataType] | None = None,
) -> pli.Expr:
    """
    Map a custom function over multiple columns/expressions.

    Produces a single Series result.

    Parameters
    ----------
    exprs
        Input Series to f
    f
        Function to apply over the input
    return_dtype
        dtype of the output Series

    Returns
    -------
    Expr

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(
        _map_mul(exprs, f, return_dtype, apply_groups=False, returns_scalar=False)
    )


def apply(
    exprs: Sequence[str | pli.Expr],
    f: Callable[[Sequence[pli.Series]], pli.Series | Any],
    return_dtype: type[DataType] | None = None,
    returns_scalar: bool = True,
) -> pli.Expr:
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
    f
        Function to apply over the input
    return_dtype
        dtype of the output Series
    returns_scalar
        If the function returns a single scalar as output.

    Returns
    -------
    Expr

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(
        _map_mul(
            exprs, f, return_dtype, apply_groups=True, returns_scalar=returns_scalar
        )
    )


def fold(
    acc: IntoExpr,
    f: Callable[[pli.Series, pli.Series], pli.Series],
    exprs: Sequence[pli.Expr | str] | pli.Expr,
) -> pli.Expr:
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    acc
        Accumulator Expression. This is the value that will be initialized when the fold
        starts. For a sum this could for instance be lit(0).
    f
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    Notes
    -----
    If you simply want the first encountered expression as accumulator,
    consider using ``reduce``.

    """
    # in case of pl.col("*")
    acc = pli.expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pyfold(acc._pyexpr, f, exprs))


def reduce(
    f: Callable[[pli.Series, pli.Series], pli.Series],
    exprs: Sequence[pli.Expr | str] | pli.Expr,
) -> pli.Expr:
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    f
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    Notes
    -----
    See ``fold`` for the version with an explicit accumulator.

    """
    # in case of pl.col("*")
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pyreduce(f, exprs))


def cumfold(
    acc: IntoExpr,
    f: Callable[[pli.Series, pli.Series], pli.Series],
    exprs: Sequence[pli.Expr | str] | pli.Expr,
    include_init: bool = False,
) -> pli.Expr:
    """
    Cumulatively accumulate over multiple columns horizontally/ row wise with a left fold.

    Every cumulative result is added as a separate field in a Struct column.

    Parameters
    ----------
    acc
        Accumulator Expression. This is the value that will be initialized when the fold
        starts. For a sum this could for instance be lit(0).
    f
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

    """  # noqa E501
    # in case of pl.col("*")
    acc = pli.expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pycumfold(acc._pyexpr, f, exprs, include_init))


def cumreduce(
    f: Callable[[pli.Series, pli.Series], pli.Series],
    exprs: Sequence[pli.Expr | str] | pli.Expr,
) -> pli.Expr:
    """
    Cumulatively accumulate over multiple columns horizontally/ row wise with a left fold.

    Every cumulative result is added as a separate field in a Struct column.

    Parameters
    ----------
    f
        Function to apply over the accumulator and the value.
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over. May also be a wildcard expression.

    """  # noqa E501
    # in case of pl.col("*")
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pycumreduce(f, exprs))


def any(name: str | Sequence[str] | Sequence[pli.Expr] | pli.Expr) -> pli.Expr:
    """Evaluate columnwise or elementwise with a bitwise OR operation."""
    if isinstance(name, str):
        return col(name).any()
    else:
        return fold(lit(False), lambda a, b: a.cast(bool) | b.cast(bool), name).alias(
            "any"
        )


def exclude(
    columns: (
        str
        | Sequence[str]
        | DataType
        | type[DataType]
        | DataType
        | Sequence[DataType | type[DataType]]
    ),
) -> pli.Expr:
    """
    Exclude certain columns from a wildcard/regex selection.

    Syntactic sugar for:

    >>> pl.all().exclude(columns)  # doctest: +SKIP

    Parameters
    ----------
    columns
        Column(s) to exclude from selection
        This can be:

        - a column name, or multiple column names
        - a regular expression starting with `^` and ending with `$`
        - a dtype or multiple dtypes

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "aa": [1, 2, 3],
    ...         "ba": ["a", "b", None],
    ...         "cc": [None, 2.5, 1.5],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌─────┬──────┬──────┐
    │ aa  ┆ ba   ┆ cc   │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ str  ┆ f64  │
    ╞═════╪══════╪══════╡
    │ 1   ┆ a    ┆ null │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2   ┆ b    ┆ 2.5  │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 3   ┆ null ┆ 1.5  │
    └─────┴──────┴──────┘

    Exclude by column name(s):

    >>> df.select(pl.exclude("ba"))
    shape: (3, 2)
    ┌─────┬──────┐
    │ aa  ┆ cc   │
    │ --- ┆ ---  │
    │ i64 ┆ f64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2   ┆ 2.5  │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┤
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
    ├╌╌╌╌╌╌┤
    │ 2.5  │
    ├╌╌╌╌╌╌┤
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
    ├╌╌╌╌╌╌┤
    │ b    │
    ├╌╌╌╌╌╌┤
    │ null │
    └──────┘

    """
    return col("*").exclude(columns)


def all(name: str | Sequence[pli.Expr] | pli.Expr | None = None) -> pli.Expr:
    """
    Do one of two things.

    * function can do a columnwise or elementwise AND operation
    * a wildcard column selection

    Parameters
    ----------
    name
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
    if name is None:
        return col("*")
    elif isinstance(name, str):
        return col(name).all()
    else:
        return fold(lit(True), lambda a, b: a.cast(bool) & b.cast(bool), name).alias(
            "all"
        )


def groups(column: str) -> pli.Expr:
    """Syntactic sugar for `pl.col("foo").agg_groups()`."""
    return col(column).agg_groups()


def quantile(
    column: str,
    quantile: float | pli.Expr,
    interpolation: RollingInterpolationMethod = "nearest",
) -> pli.Expr:
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
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = ...,
    *,
    eager: Literal[False],
) -> pli.Expr:
    ...


@overload
def arange(
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = ...,
    *,
    eager: Literal[True],
    dtype: PolarsDataType | None = ...,
) -> pli.Series:
    ...


@overload
def arange(
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = ...,
    *,
    eager: bool = False,
    dtype: PolarsDataType | None = ...,
) -> pli.Expr | pli.Series:
    ...


def arange(
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = 1,
    *,
    eager: bool = False,
    dtype: PolarsDataType | None = None,
) -> pli.Expr | pli.Series:
    """
    Create a range expression (or Series).

    This can be used in a `select`, `with_column` etc. Be sure that the resulting
    range size is equal to the length of the DataFrame you are collecting.

    Examples
    --------
    >>> df.lazy().filter(pl.col("foo") < pl.arange(0, 100)).collect()  # doctest: +SKIP

    Parameters
    ----------
    low
        Lower bound of range.
    high
        Upper bound of range.
    step
        Step size of the range.
    eager
        If eager evaluation is `True`, a Series is returned instead of an Expr.
    dtype
        Apply an explicit integer dtype to the resulting expression (default is Int64).

    """
    low = pli.expr_to_lit_or_expr(low, str_to_lit=False)
    high = pli.expr_to_lit_or_expr(high, str_to_lit=False)
    range_expr = pli.wrap_expr(pyarange(low._pyexpr, high._pyexpr, step))

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


def argsort_by(
    exprs: pli.Expr | str | Sequence[pli.Expr | str],
    reverse: Sequence[bool] | bool = False,
) -> pli.Expr:
    """
    Find the indexes that would sort the columns.

    Argsort by multiple columns. The first column will be used for the ordering.
    If there are duplicates in the first column, the second column will be used to
    determine the ordering and so on.

    Parameters
    ----------
    exprs
        Columns use to determine the ordering.
    reverse
        Default is ascending.

    """
    if isinstance(exprs, str) or not isinstance(exprs, Sequence):
        exprs = [exprs]
    if isinstance(reverse, bool):
        reverse = [reverse] * len(exprs)
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pyargsort_by(exprs, reverse))


def duration(
    *,
    days: pli.Expr | str | int | None = None,
    seconds: pli.Expr | str | int | None = None,
    nanoseconds: pli.Expr | str | int | None = None,
    microseconds: pli.Expr | str | int | None = None,
    milliseconds: pli.Expr | str | int | None = None,
    minutes: pli.Expr | str | int | None = None,
    hours: pli.Expr | str | int | None = None,
    weeks: pli.Expr | str | int | None = None,
) -> pli.Expr:
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
    ...         "datetime": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
    ...         "add": [1, 2],
    ...     }
    ... )
    >>> df.select(
    ...     [
    ...         (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
    ...         (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
    ...         (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
    ...         (pl.col("datetime") + pl.duration(milliseconds="add")).alias(
    ...             "add_milliseconds"
    ...         ),
    ...         (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
    ...     ]
    ... )  # doctest: +IGNORE_RESULT
    shape: (2, 5)
    ┌────────────┬────────────┬─────────────────────┬──────────────┬─────────────────────┐
    │ add_weeks  ┆ add_days   ┆ add_seconds         ┆ add_millisec ┆ add_hours           │
    │ ---        ┆ ---        ┆ ---                 ┆ onds         ┆ ---                 │
    │ datetime[m ┆ datetime[m ┆ datetime[ms]        ┆ ---          ┆ datetime[ms]        │
    │ s]         ┆ s]         ┆                     ┆ datetime[ms] ┆                     │
    ╞════════════╪════════════╪═════════════════════╪══════════════╪═════════════════════╡
    │ 2022-01-08 ┆ 2022-01-02 ┆ 2022-01-01 00:00:01 ┆ 2022-01-01   ┆ 2022-01-01 01:00:00 │
    │ 00:00:00   ┆ 00:00:00   ┆                     ┆ 00:00:00.001 ┆                     │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2022-01-16 ┆ 2022-01-04 ┆ 2022-01-02 00:00:02 ┆ 2022-01-02   ┆ 2022-01-02 02:00:00 │
    │ 00:00:00   ┆ 00:00:00   ┆                     ┆ 00:00:00.002 ┆                     │
    └────────────┴────────────┴─────────────────────┴──────────────┴─────────────────────┘

    """  # noqa: E501
    if hours is not None:
        hours = pli.expr_to_lit_or_expr(hours, str_to_lit=False)._pyexpr
    if minutes is not None:
        minutes = pli.expr_to_lit_or_expr(minutes, str_to_lit=False)._pyexpr
    if seconds is not None:
        seconds = pli.expr_to_lit_or_expr(seconds, str_to_lit=False)._pyexpr
    if milliseconds is not None:
        milliseconds = pli.expr_to_lit_or_expr(milliseconds, str_to_lit=False)._pyexpr
    if microseconds is not None:
        microseconds = pli.expr_to_lit_or_expr(microseconds, str_to_lit=False)._pyexpr
    if nanoseconds is not None:
        nanoseconds = pli.expr_to_lit_or_expr(nanoseconds, str_to_lit=False)._pyexpr
    if days is not None:
        days = pli.expr_to_lit_or_expr(days, str_to_lit=False)._pyexpr
    if weeks is not None:
        weeks = pli.expr_to_lit_or_expr(weeks, str_to_lit=False)._pyexpr

    return pli.wrap_expr(
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


def _datetime(
    year: pli.Expr | str | int,
    month: pli.Expr | str | int,
    day: pli.Expr | str | int,
    hour: pli.Expr | str | int | None = None,
    minute: pli.Expr | str | int | None = None,
    second: pli.Expr | str | int | None = None,
    microsecond: pli.Expr | str | int | None = None,
) -> pli.Expr:
    """
    Create polars `Datetime` from distinct time components.

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
    year_expr = pli.expr_to_lit_or_expr(year, str_to_lit=False)
    month_expr = pli.expr_to_lit_or_expr(month, str_to_lit=False)
    day_expr = pli.expr_to_lit_or_expr(day, str_to_lit=False)

    if hour is not None:
        hour = pli.expr_to_lit_or_expr(hour, str_to_lit=False)._pyexpr
    if minute is not None:
        minute = pli.expr_to_lit_or_expr(minute, str_to_lit=False)._pyexpr
    if second is not None:
        second = pli.expr_to_lit_or_expr(second, str_to_lit=False)._pyexpr
    if microsecond is not None:
        microsecond = pli.expr_to_lit_or_expr(microsecond, str_to_lit=False)._pyexpr

    return pli.wrap_expr(
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


def _date(
    year: pli.Expr | str | int,
    month: pli.Expr | str | int,
    day: pli.Expr | str | int,
) -> pli.Expr:
    """
    Create polars Date from distinct time components.

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
    return _datetime(year, month, day).cast(Date).alias("date")


def concat_str(exprs: Sequence[pli.Expr | str] | pli.Expr, sep: str = "") -> pli.Expr:
    """
    Horizontally concat Utf8 Series in linear time. Non-Utf8 columns are cast to Utf8.

    Parameters
    ----------
    exprs
        Columns to concat into a Utf8 Series.
    sep
        String value that will be used to separate the values.

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
    ...     [
    ...         pl.concat_str(
    ...             [
    ...                 pl.col("a") * 2,
    ...                 pl.col("b"),
    ...                 pl.col("c"),
    ...             ],
    ...             sep=" ",
    ...         ).alias("full_sentence"),
    ...     ]
    ... )
    shape: (3, 4)
    ┌─────┬──────┬──────┬───────────────┐
    │ a   ┆ b    ┆ c    ┆ full_sentence │
    │ --- ┆ ---  ┆ ---  ┆ ---           │
    │ i64 ┆ str  ┆ str  ┆ str           │
    ╞═════╪══════╪══════╪═══════════════╡
    │ 1   ┆ dogs ┆ play ┆ 2 dogs play   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2   ┆ cats ┆ swim ┆ 4 cats swim   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ null ┆ walk ┆ null          │
    └─────┴──────┴──────┴───────────────┘

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_concat_str(exprs, sep))


def format(fstring: str, *args: pli.Expr | str) -> pli.Expr:
    """
    Format expressions as a string.

    Parameters
    ----------
    fstring
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
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ foo_b_bar_2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ foo_c_bar_3 │
    └─────────────┘

    """
    if fstring.count("{}") != len(args):
        raise ValueError("number of placeholders should equal the number of arguments")

    exprs = []

    arguments = iter(args)
    for i, s in enumerate(fstring.split("{}")):
        if i > 0:
            e = pli.expr_to_lit_or_expr(next(arguments), str_to_lit=False)
            exprs.append(e)

        if len(s) > 0:
            exprs.append(lit(s))

    return concat_str(exprs, sep="")


def concat_list(exprs: Sequence[str | pli.Expr | pli.Series] | pli.Expr) -> pli.Expr:
    """
    Concat the arrays in a Series dtype List in linear time.

    Parameters
    ----------
    exprs
        Columns to concat into a List Series

    Examples
    --------
    Create lagged columns and collect them into a list. This mimics a rolling window.

    >>> df = pl.DataFrame(
    ...     {
    ...         "A": [1.0, 2.0, 9.0, 2.0, 13.0],
    ...     }
    ... )
    >>> (
    ...     df.with_columns(
    ...         [pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)]
    ...     ).select(
    ...         [
    ...             pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias(
    ...                 "A_rolling"
    ...             )
    ...         ]
    ...     )
    ... )
    shape: (5, 1)
    ┌───────────────────┐
    │ A_rolling         │
    │ ---               │
    │ list[f64]         │
    ╞═══════════════════╡
    │ [null, null, 1.0] │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [null, 1.0, 2.0]  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [1.0, 2.0, 9.0]   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [2.0, 9.0, 2.0]   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [9.0, 2.0, 13.0]  │
    └───────────────────┘

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_concat_lst(exprs))


@deprecated_alias(allow_streaming="streaming")
def collect_all(
    lazy_frames: Sequence[pli.LazyFrame],
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    string_cache: bool = False,
    no_optimization: bool = False,
    slice_pushdown: bool = True,
    common_subplan_elimination: bool = True,
    streaming: bool = False,
) -> list[pli.DataFrame]:
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
    string_cache
        This argument is deprecated and will be ignored
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
    result = [pli.wrap_df(pydf) for pydf in out]

    return result


def select(
    exprs: str | pli.Expr | Sequence[str | pli.Expr] | pli.Series,
) -> pli.DataFrame:
    """
    Run polars expressions without a context.

    This is syntactic sugar for running `df.select` on an empty DataFrame.

    Parameters
    ----------
    exprs
        Expressions to run

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> foo = pl.Series("foo", [1, 2, 3])
    >>> bar = pl.Series("bar", [3, 2, 1])
    >>> pl.select(
    ...     [
    ...         pl.min([foo, bar]),
    ...     ]
    ... )
    shape: (3, 1)
    ┌─────┐
    │ min │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    ├╌╌╌╌╌┤
    │ 2   │
    ├╌╌╌╌╌┤
    │ 1   │
    └─────┘

    """
    return pli.DataFrame([]).select(exprs)


@overload
def struct(
    exprs: Sequence[pli.Expr | str | pli.Series] | pli.Expr | pli.Series,
    eager: Literal[True],
) -> pli.Series:
    ...


@overload
def struct(
    exprs: Sequence[pli.Expr | str | pli.Series] | pli.Expr | pli.Series,
    eager: Literal[False],
) -> pli.Expr:
    ...


@overload
def struct(
    exprs: Sequence[pli.Expr | str | pli.Series] | pli.Expr | pli.Series,
    eager: bool = False,
) -> pli.Expr | pli.Series:
    ...


def struct(
    exprs: Sequence[pli.Expr | str | pli.Series] | pli.Expr | pli.Series,
    eager: bool = False,
) -> pli.Expr | pli.Series:
    """
    Collect several columns into a Series of dtype Struct.

    Parameters
    ----------
    exprs
        Columns/Expressions to collect into a Struct
    eager
        Evaluate immediately

    Examples
    --------
    >>> pl.DataFrame(
    ...     {
    ...         "int": [1, 2],
    ...         "str": ["a", "b"],
    ...         "bool": [True, None],
    ...         "list": [[1, 2], [3]],
    ...     }
    ... ).select([pl.struct(pl.all()).alias("my_struct")])
    shape: (2, 1)
    ┌─────────────────────┐
    │ my_struct           │
    │ ---                 │
    │ struct[4]           │
    ╞═════════════════════╡
    │ {1,"a",true,[1, 2]} │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ {2,"b",null,[3]}    │
    └─────────────────────┘

    Only collect specific columns as a struct:

    >>> df = pl.DataFrame(
    ...     {"a": [1, 2, 3, 4], "b": ["one", "two", "three", "four"], "c": [9, 8, 7, 6]}
    ... )
    >>> df.with_column(pl.struct(pl.col(["a", "b"])).alias("a_and_b"))
    shape: (4, 4)
    ┌─────┬───────┬─────┬─────────────┐
    │ a   ┆ b     ┆ c   ┆ a_and_b     │
    │ --- ┆ ---   ┆ --- ┆ ---         │
    │ i64 ┆ str   ┆ i64 ┆ struct[2]   │
    ╞═════╪═══════╪═════╪═════════════╡
    │ 1   ┆ one   ┆ 9   ┆ {1,"one"}   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2   ┆ two   ┆ 8   ┆ {2,"two"}   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ three ┆ 7   ┆ {3,"three"} │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 4   ┆ four  ┆ 6   ┆ {4,"four"}  │
    └─────┴───────┴─────┴─────────────┘

    """
    if eager:
        return pli.select(struct(exprs, eager=False)).to_series()
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_as_struct(exprs))


@overload
def repeat(
    value: float | int | str | bool | None,
    n: pli.Expr | int,
    *,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> pli.Expr:
    ...


@overload
def repeat(
    value: float | int | str | bool | None,
    n: pli.Expr | int,
    *,
    eager: Literal[True],
    name: str | None = ...,
) -> pli.Series:
    ...


@overload
def repeat(
    value: float | int | str | bool | None,
    n: pli.Expr | int,
    *,
    eager: bool,
    name: str | None,
) -> pli.Expr | pli.Series:
    ...


def repeat(
    value: float | int | str | bool | None,
    n: pli.Expr | int,
    *,
    eager: bool = False,
    name: str | None = None,
) -> pli.Expr | pli.Series:
    """
    Repeat a single value n times.

    Parameters
    ----------
    value
        Value to repeat.
    n
        repeat `n` times
    eager
        Run eagerly and collect into a `Series`
    name
        Only used in `eager` mode. As expression, us `alias`

    """
    if eager:
        if name is None:
            name = ""
        dtype = py_type_to_dtype(type(value))
        s = pli.Series._repeat(name, value, n, dtype)  # type: ignore[arg-type]
        return s
    else:
        if isinstance(n, int):
            n = lit(n)
        return pli.wrap_expr(_repeat(value, n._pyexpr))


@overload
def arg_where(
    condition: pli.Expr | pli.Series,
    eager: Literal[False] = ...,
) -> pli.Expr:
    ...


@overload
def arg_where(condition: pli.Expr | pli.Series, eager: Literal[True]) -> pli.Series:
    ...


@overload
def arg_where(condition: pli.Expr | pli.Series, eager: bool) -> pli.Expr | pli.Series:
    ...


def arg_where(
    condition: pli.Expr | pli.Series, eager: bool = False
) -> pli.Expr | pli.Series:
    """
    Return indices where `condition` evaluates `True`.

    Parameters
    ----------
    condition
        Boolean expression to evaluate
    eager
        Whether to apply this function eagerly (as opposed to lazily).

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

    """
    if eager:
        if not isinstance(condition, pli.Series):
            raise ValueError(
                "expected 'Series' in 'arg_where' if 'eager=True', got"
                f" {type(condition)}"
            )
        return (
            condition.to_frame().select(arg_where(pli.col(condition.name))).to_series()
        )
    else:
        condition = pli.expr_to_lit_or_expr(condition, str_to_lit=True)
        return pli.wrap_expr(py_arg_where(condition._pyexpr))


def coalesce(
    exprs: Sequence[
        pli.Expr | str | date | datetime | timedelta | int | float | bool | pli.Series
    ]
    | pli.Expr,
) -> pli.Expr:
    """
    Folds the expressions from left to right, keeping the first non-null value.

    Parameters
    ----------
    exprs
        Expressions to coalesce.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     data=[
    ...         (None, 1.0, 1.0),
    ...         (None, 2.0, 2.0),
    ...         (None, None, 3.0),
    ...         (None, None, None),
    ...     ],
    ...     columns=[("a", pl.Float64), ("b", pl.Float64), ("c", pl.Float64)],
    ... )
    >>> df.with_column(pl.coalesce(["a", "b", "c", 99.9]).alias("d"))
    shape: (4, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ d    │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ 1.0  ┆ 1.0  ┆ 1.0  │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ null ┆ 2.0  ┆ 2.0  ┆ 2.0  │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ null ┆ null ┆ 3.0  ┆ 3.0  │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ null ┆ null ┆ null ┆ 99.9 │
    └──────┴──────┴──────┴──────┘

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_coalesce_exprs(exprs))


@overload
def from_epoch(
    column: str | pli.Expr | pli.Series,
    unit: EpochTimeUnit = ...,
    *,
    eager: Literal[False],
) -> pli.Expr:
    ...


@overload
def from_epoch(
    column: str | pli.Expr | pli.Series | Sequence[int],
    unit: EpochTimeUnit = ...,
    *,
    eager: Literal[True],
) -> pli.Series:
    ...


@overload
def from_epoch(
    column: pli.Series | Sequence[int],
    unit: EpochTimeUnit = ...,
    *,
    eager: Literal[True] = ...,
) -> pli.Series:
    ...


@overload
def from_epoch(
    column: str | pli.Expr,
    unit: EpochTimeUnit = ...,
    *,
    eager: Literal[False] = ...,
) -> pli.Expr:
    ...


@overload
def from_epoch(
    column: str | pli.Expr | pli.Series | Sequence[int],
    unit: EpochTimeUnit = ...,
    *,
    eager: bool = ...,
) -> pli.Expr | pli.Series:
    ...


def from_epoch(
    column: str | pli.Expr | pli.Series | Sequence[int],
    unit: EpochTimeUnit = "s",
    *,
    eager: bool = False,
) -> pli.Expr | pli.Series:
    """
    Utility function that parses an epoch timestamp (or Unix time) to Polars Date(time).

    Depending on the `unit` provided, this function will return a different dtype:
    - unit="d" returns pl.Date
    - unit="s" returns pl.Datetime["us"] (pl.Datetime's default)
    - unit="ms" returns pl.Datetime["ms"]
    - unit="us" returns pl.Datetime["us"]
    - unit="ns" returns pl.Datetime["ns"]

    Parameters
    ----------
    column
        Series or expression to parse integers to pl.Datetime.
    unit
        The unit of the timesteps since epoch time.
    eager
        If eager evaluation is `True`, a Series is returned instead of an Expr.

    Examples
    --------
    >>> df = pl.DataFrame({"timestamp": [1666683077, 1666683099]}).lazy()
    >>> df.select(pl.from_epoch(pl.col("timestamp"), unit="s")).collect()
    shape: (2, 1)
    ┌─────────────────────┐
    │ timestamp           │
    │ ---                 │
    │ datetime[μs]        │
    ╞═════════════════════╡
    │ 2022-10-25 07:31:17 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2022-10-25 07:31:39 │
    └─────────────────────┘

    """
    if isinstance(column, str):
        column = col(column)
    elif not isinstance(column, (pli.Series, pli.Expr)):
        column = pli.Series(column)  # Sequence input handled by Series constructor

    if unit == "d":
        expr = column.cast(Date)
    elif unit == "s":
        expr = (column.cast(Int64) * 1_000_000).cast(Datetime("us"))
    elif unit in DTYPE_TEMPORAL_UNITS:
        expr = column.cast(Datetime(unit))
    else:
        raise ValueError(
            f"'unit' must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got '{unit}'."
        )

    if eager:
        if not isinstance(column, pli.Series):
            raise ValueError(
                "expected 'Series or Sequence' in 'from_epoch' if 'eager=True', got"
                f" {type(column)}"
            )
        else:
            return column.to_frame().select(expr).to_series()
    else:
        return expr

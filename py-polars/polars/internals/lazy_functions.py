from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast, overload

from polars import internals as pli
from polars.datatypes import (
    DataType,
    Date,
    Datetime,
    Duration,
    PolarsDataType,
    UInt32,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.utils import _datetime_to_pl_timestamp, _timedelta_to_pl_timedelta

try:
    from polars.polars import arange as pyarange
    from polars.polars import arg_where as py_arg_where
    from polars.polars import argsort_by as pyargsort_by
    from polars.polars import as_struct as _as_struct
    from polars.polars import col as pycol
    from polars.polars import collect_all as _collect_all
    from polars.polars import cols as pycols
    from polars.polars import concat_lst as _concat_lst
    from polars.polars import concat_str as _concat_str
    from polars.polars import count as _count
    from polars.polars import cov as pycov
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
    from polars.polars import repeat as _repeat
    from polars.polars import spearman_rank_corr as pyspearman_rank_corr
    from polars.polars import sum_exprs as _sum_exprs

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from polars.internals.type_aliases import InterpolationMethod, IntoExpr, TimeUnit


def col(
    name: str | Sequence[str] | Sequence[PolarsDataType] | pli.Series | PolarsDataType,
) -> pli.Expr:
    """
    Return an expression representing a column in a DataFrame.

    Can be used to select:

    - a single column by name
    - all columns by using a wildcard `"*"`
    - column by regular expression if the regex starts with `^` and ends with `$`

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

    """
    if isinstance(name, pli.Series):
        name = name.to_list()  # type: ignore[assignment]

    # note: we need the typing.cast call here twice to make mypy happy under Python 3.7
    # On Python 3.10, it is not needed. We use cast as it works across versions,
    # ignoring the typing error would lead to unneeded ignores under Python 3.10.
    if isclass(name) and issubclass(cast(type, name), DataType):
        name = [cast(type, name)]

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
    Alias for an element in evaluated in an `eval` expression.

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
    """Get the standard deviation."""
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
    """Get the variance."""
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

    """
    if isinstance(column, pli.Series):
        return column.max()
    elif isinstance(column, str):
        return col(column).max()
    else:
        exprs = pli.selection_to_pyexpr_list(column)
        return pli.wrap_expr(_max_exprs(exprs))


@overload
def min(column: str | Sequence[pli.Expr | str]) -> pli.Expr:
    ...


@overload
def min(column: pli.Series) -> int | float:
    ...


def min(column: str | Sequence[pli.Expr | str] | pli.Series) -> pli.Expr | Any:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on
        the input:
        - Union[str, Series] -> aggregate the sum value of that column.
        - List[Expr] -> aggregate the sum value horizontally.

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
    """Get the mean value."""
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
    """Alias for mean."""
    return mean(column)


@overload
def median(column: str) -> pli.Expr:
    ...


@overload
def median(column: pli.Series) -> float | int:
    ...


def median(column: str | pli.Series) -> pli.Expr | float | int:
    """Get the median value."""
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
    """Count unique values."""
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
        return lit(_datetime_to_pl_timestamp(value, tu)).cast(Datetime(tu))

    elif isinstance(value, timedelta):
        tu = "us"
        return lit(_timedelta_to_pl_timedelta(value, tu)).cast(Duration(tu))

    elif isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)

    elif isinstance(value, pli.Series):
        name = value.name
        value = value._s
        e = pli.wrap_expr(pylit(value, allow_object))
        if name == "":
            return e
        return e.alias(name)

    if _NUMPY_AVAILABLE and isinstance(value, np.ndarray):
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


def spearman_rank_corr(a: str | pli.Expr, b: str | pli.Expr, ddof: int = 1) -> pli.Expr:
    """
    Compute the spearman rank correlation between two columns.

    Parameters
    ----------
    a
        Column name or Expression.
    b
        Column name or Expression.
    ddof
        Delta degrees of freedom

    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pli.wrap_expr(pyspearman_rank_corr(a._pyexpr, b._pyexpr, ddof))


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
    return pli.wrap_expr(_map_mul(exprs, f, return_dtype, apply_groups=False))


def apply(
    exprs: Sequence[str | pli.Expr],
    f: Callable[[Sequence[pli.Series]], pli.Series | Any],
    return_dtype: type[DataType] | None = None,
) -> pli.Expr:
    """
    Apply a custom function in a GroupBy context.

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

    Returns
    -------
    Expr

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_map_mul(exprs, f, return_dtype, apply_groups=True))


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

    """
    # in case of pl.col("*")
    acc = pli.expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pli.Expr):
        exprs = [exprs]

    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pyfold(acc._pyexpr, f, exprs))


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
    column: str, quantile: float, interpolation: InterpolationMethod = "nearest"
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
) -> pli.Series:
    ...


@overload
def arange(
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = ...,
    *,
    eager: bool = False,
) -> pli.Expr | pli.Series:
    ...


def arange(
    low: int | pli.Expr | pli.Series,
    high: int | pli.Expr | pli.Series,
    step: int = 1,
    *,
    eager: bool = False,
) -> pli.Expr | pli.Series:
    """
    Create a range expression.

    This can be used in a `select`, `with_column` etc.

    Be sure that the range size is equal to the DataFrame you are collecting.

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

    """
    low = pli.expr_to_lit_or_expr(low, str_to_lit=False)
    high = pli.expr_to_lit_or_expr(high, str_to_lit=False)

    if eager:
        df = pli.DataFrame({"a": [1]})
        return df.select(arange(low, high, step).alias("arange"))["arange"]

    return pli.wrap_expr(pyarange(low._pyexpr, high._pyexpr, step))


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
    days: pli.Expr | str | None = None,
    seconds: pli.Expr | str | None = None,
    nanoseconds: pli.Expr | str | None = None,
    milliseconds: pli.Expr | str | None = None,
    minutes: pli.Expr | str | None = None,
    hours: pli.Expr | str | None = None,
    weeks: pli.Expr | str | None = None,
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
    if nanoseconds is not None:
        nanoseconds = pli.expr_to_lit_or_expr(nanoseconds, str_to_lit=False)._pyexpr
    if days is not None:
        days = pli.expr_to_lit_or_expr(days, str_to_lit=False)._pyexpr
    if weeks is not None:
        weeks = pli.expr_to_lit_or_expr(weeks, str_to_lit=False)._pyexpr
    return pli.wrap_expr(
        py_duration(days, seconds, nanoseconds, milliseconds, minutes, hours, weeks)
    )


def _datetime(
    year: pli.Expr | str,
    month: pli.Expr | str,
    day: pli.Expr | str,
    hour: pli.Expr | str | None = None,
    minute: pli.Expr | str | None = None,
    second: pli.Expr | str | None = None,
    millisecond: pli.Expr | str | None = None,
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
        column or literal, ranging from 1-24.
    minute
        column or literal, ranging from 1-60.
    second
        column or literal, ranging from 1-60.
    millisecond
        column or literal, ranging from 1-1000.

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
    if millisecond is not None:
        millisecond = pli.expr_to_lit_or_expr(millisecond, str_to_lit=False)._pyexpr
    return pli.wrap_expr(
        py_datetime(
            year_expr._pyexpr,
            month_expr._pyexpr,
            day_expr._pyexpr,
            hour,
            minute,
            second,
            millisecond,
        )
    )


def _date(
    year: pli.Expr | str,
    month: pli.Expr | str,
    day: pli.Expr | str,
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


def collect_all(
    lazy_frames: Sequence[pli.LazyFrame],
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    string_cache: bool = False,
    no_optimization: bool = False,
    slice_pushdown: bool = False,
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

    Returns
    -------
    List[DataFrame]

    """
    if no_optimization:
        predicate_pushdown = False
        projection_pushdown = False
        slice_pushdown = False

    prepared = []

    for lf in lazy_frames:
        ldf = lf._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
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

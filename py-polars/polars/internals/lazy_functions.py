from datetime import date, datetime, timedelta
from inspect import isclass
from typing import Any, Callable, List, Optional, Sequence, Type, Union, cast, overload

import numpy as np

from polars import internals as pli
from polars.datatypes import DataType, Date, Datetime, Duration
from polars.utils import (
    _datetime_to_pl_timestamp,
    _timedelta_to_pl_timedelta,
    in_nanoseconds_window,
    timedelta_in_nanoseconds_window,
)

try:
    from polars.polars import arange as pyarange
    from polars.polars import argsort_by as pyargsort_by
    from polars.polars import binary_function as pybinary_function
    from polars.polars import col as pycol
    from polars.polars import collect_all as _collect_all
    from polars.polars import cols as pycols
    from polars.polars import concat_lst as _concat_lst
    from polars.polars import concat_str as _concat_str
    from polars.polars import cov as pycov
    from polars.polars import dtype_cols as _dtype_cols
    from polars.polars import fold as pyfold
    from polars.polars import lit as pylit
    from polars.polars import map_mul as _map_mul
    from polars.polars import pearson_corr as pypearson_corr
    from polars.polars import py_datetime
    from polars.polars import spearman_rank_corr as pyspearman_rank_corr

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


def col(
    name: Union[str, List[str], List[Type[DataType]], "pli.Series", Type[DataType]]
) -> "pli.Expr":
    """
    A column in a DataFrame.
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
        name = name.to_list()  # type: ignore

    # note: we need the typing.cast call here twice to make mypy happy under Python 3.7
    # On Python 3.10, it is not needed. We use cast as it works across versions, ignoring
    # the typing error would lead to unneeded ignores under Python 3.10.
    if isclass(name) and issubclass(cast(type, name), DataType):
        name = [cast(type, name)]

    if isinstance(name, list):
        if len(name) == 0 or isinstance(name[0], str):
            return pli.wrap_expr(pycols(name))
        elif isclass(name[0]) and issubclass(name[0], DataType):
            return pli.wrap_expr(_dtype_cols(name))
        else:
            raise ValueError("did expect argument of List[str] or List[DataType]")
    return pli.wrap_expr(pycol(name))


@overload
def count(column: str) -> "pli.Expr":
    ...


@overload
def count(column: "pli.Series") -> int:
    ...


def count(column: Union[str, "pli.Series"] = "") -> Union["pli.Expr", int]:
    """
    Count the number of values in this column.
    """
    if isinstance(column, pli.Series):
        return column.len()
    return col(column).count()


def to_list(name: str) -> "pli.Expr":
    """
    Aggregate to list.

    Re-exported as `pl.list()`
    """
    return col(name).list()


@overload
def std(column: str) -> "pli.Expr":
    ...


@overload
def std(column: "pli.Series") -> Optional[float]:
    ...


def std(column: Union[str, "pli.Series"]) -> Union["pli.Expr", Optional[float]]:
    """
    Get the standard deviation.
    """
    if isinstance(column, pli.Series):
        return column.std()
    return col(column).std()


@overload
def var(column: str) -> "pli.Expr":
    ...


@overload
def var(column: "pli.Series") -> Optional[float]:
    ...


def var(column: Union[str, "pli.Series"]) -> Union["pli.Expr", Optional[float]]:
    """
    Get the variance.
    """
    if isinstance(column, pli.Series):
        return column.var()
    return col(column).var()


@overload
def max(column: Union[str, List[Union["pli.Expr", str]]]) -> "pli.Expr":
    ...


@overload
def max(column: "pli.Series") -> Union[int, float]:
    ...


def max(
    column: Union[str, List[Union["pli.Expr", str]], "pli.Series"]
) -> Union["pli.Expr", Any]:
    """
    Get the maximum value. Can be used horizontally or vertically.

    Parameters
    ----------
    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
        - Union[str, Series] -> aggregate the maximum value of that column.
        - List[Expr] -> aggregate the maximum value horizontally.
    """
    if isinstance(column, pli.Series):
        return column.max()
    elif isinstance(column, list):

        def max_(
            acc: "pli.Series", val: "pli.Series"
        ) -> "pli.Series":  # pragma: no cover
            mask = acc > val
            return acc.zip_with(mask, val)

        first = column[0]
        if isinstance(first, str):
            first = col(first)
        return fold(first, max_, column[1:]).alias("max")
    else:
        return col(column).max()


@overload
def min(column: Union[str, List[Union["pli.Expr", str]]]) -> "pli.Expr":
    ...


@overload
def min(column: "pli.Series") -> Union[int, float]:
    ...


def min(
    column: Union[str, List[Union["pli.Expr", str]], "pli.Series"]
) -> Union["pli.Expr", Any]:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
        - Union[str, Series] -> aggregate the sum value of that column.
        - List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pli.Series):
        return column.min()
    elif isinstance(column, list):

        def min_(
            acc: "pli.Series", val: "pli.Series"
        ) -> "pli.Series":  # pragma: no cover
            mask = acc < val
            return acc.zip_with(mask, val)

        first = column[0]
        if isinstance(first, str):
            first = col(first)
        return fold(first, min_, column[1:]).alias("min")
    else:
        return col(column).min()


@overload
def sum(column: Union[str, List[Union["pli.Expr", str]]]) -> "pli.Expr":
    ...


@overload
def sum(column: "pli.Series") -> Union[int, float]:
    ...


def sum(
    column: Union[str, List[Union["pli.Expr", str]], "pli.Series"]
) -> Union["pli.Expr", Any]:
    """
    Get the sum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
        - Union[str, Series] -> aggregate the sum value of that column.
        - List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pli.Series):
        return column.sum()
    elif isinstance(column, list):
        first = column[0]
        if isinstance(first, str):
            first = col(first)
        return fold(first, lambda a, b: a + b, column[1:]).alias("sum")
    else:
        return col(column).sum()


@overload
def mean(column: str) -> "pli.Expr":
    ...


@overload
def mean(column: "pli.Series") -> float:
    ...


def mean(column: Union[str, "pli.Series"]) -> Union["pli.Expr", float]:
    """
    Get the mean value.
    """
    if isinstance(column, pli.Series):
        return column.mean()
    return col(column).mean()


@overload
def avg(column: str) -> "pli.Expr":
    ...


@overload
def avg(column: "pli.Series") -> float:
    ...


def avg(column: Union[str, "pli.Series"]) -> Union["pli.Expr", float]:
    """
    Alias for mean.
    """
    return mean(column)


@overload
def median(column: str) -> "pli.Expr":
    ...


@overload
def median(column: "pli.Series") -> Union[float, int]:
    ...


def median(column: Union[str, "pli.Series"]) -> Union["pli.Expr", float, int]:
    """
    Get the median value.
    """
    if isinstance(column, pli.Series):
        return column.median()
    return col(column).median()


@overload
def n_unique(column: str) -> "pli.Expr":
    ...


@overload
def n_unique(column: "pli.Series") -> int:
    ...


def n_unique(column: Union[str, "pli.Series"]) -> Union["pli.Expr", int]:
    """Count unique values."""
    if isinstance(column, pli.Series):
        return column.n_unique()
    return col(column).n_unique()


@overload
def first(column: str) -> "pli.Expr":
    ...


@overload
def first(column: "pli.Series") -> Any:
    ...


def first(column: Union[str, "pli.Series"]) -> Union["pli.Expr", Any]:
    """
    Get the first value.
    """
    if isinstance(column, pli.Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


@overload
def last(column: str) -> "pli.Expr":
    ...


@overload
def last(column: "pli.Series") -> Any:
    ...


def last(column: Union[str, "pli.Series"]) -> "pli.Expr":
    """
    Get the last value.
    """
    if isinstance(column, pli.Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


@overload
def head(column: str, n: Optional[int]) -> "pli.Expr":
    ...


@overload
def head(column: "pli.Series", n: Optional[int]) -> "pli.Series":
    ...


def head(
    column: Union[str, "pli.Series"], n: Optional[int] = None
) -> Union["pli.Expr", "pli.Series"]:
    """
    Get the first n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pli.Series):
        return column.head(n)
    return col(column).head(n)


@overload
def tail(column: str, n: Optional[int]) -> "pli.Expr":
    ...


@overload
def tail(column: "pli.Series", n: Optional[int]) -> "pli.Series":
    ...


def tail(
    column: Union[str, "pli.Series"], n: Optional[int] = None
) -> Union["pli.Expr", "pli.Series"]:
    """
    Get the last n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pli.Series):
        return column.tail(n)
    return col(column).tail(n)


def lit(
    value: Optional[Union[float, int, str, date, datetime, "pli.Series"]],
    dtype: Optional[Type[DataType]] = None,
) -> "pli.Expr":
    """
    A literal value.

    Parameters
    ----------
    value
        Value that should be used as a `literal`.
    dtype
        Optionally define a dtype.

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
    if isinstance(value, datetime):
        if in_nanoseconds_window(value):
            tu = "ns"
        else:
            tu = "ms"
        return (
            lit(_datetime_to_pl_timestamp(value, tu))
            .cast(Datetime)
            .dt.and_time_unit(tu)
        )
    if isinstance(value, timedelta):
        if timedelta_in_nanoseconds_window(value):
            tu = "ns"
        else:
            tu = "ms"
        return (
            lit(_timedelta_to_pl_timedelta(value, tu))
            .cast(Duration)
            .dt.and_time_unit(tu, dtype=Duration)
        )

    if isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)

    if isinstance(value, pli.Series):
        name = value.name
        value = value._s
        e = pli.wrap_expr(pylit(value))
        if name == "":
            return e
        return e.alias(name)

    if isinstance(value, np.ndarray):
        return lit(pli.Series("", value))

    if dtype:
        return pli.wrap_expr(pylit(value)).cast(dtype)
    return pli.wrap_expr(pylit(value))


def spearman_rank_corr(
    a: Union[str, "pli.Expr"],
    b: Union[str, "pli.Expr"],
) -> "pli.Expr":
    """
    Compute the spearman rank correlation between two columns.

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
    return pli.wrap_expr(pyspearman_rank_corr(a._pyexpr, b._pyexpr))


def pearson_corr(
    a: Union[str, "pli.Expr"],
    b: Union[str, "pli.Expr"],
) -> "pli.Expr":
    """
    Compute the pearson's correlation between two columns.

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
    return pli.wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr))


def cov(
    a: Union[str, "pli.Expr"],
    b: Union[str, "pli.Expr"],
) -> "pli.Expr":
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
    exprs: Union[List[str], List["pli.Expr"]],
    f: Callable[[List["pli.Series"]], "pli.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pli.Expr":
    """
    Map a custom function over multiple columns/expressions and produce a single Series result.

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
    exprs: List[Union[str, "pli.Expr"]],
    f: Callable[[List["pli.Series"]], Union["pli.Series", Any]],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pli.Expr":
    """
    Apply a custom function in a GroupBy context.

    Depending on the context it has the following behavior:

    ## Context

    * Select/Project
        Don't do this, use `map`
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


def map_binary(
    a: Union[str, "pli.Expr"],
    b: Union[str, "pli.Expr"],
    f: Callable[["pli.Series", "pli.Series"], "pli.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pli.Expr":
    """
     .. deprecated:: 0.10.4
       use `map` or `apply`
    Map a custom function over two columns and produce a single Series result.

    Parameters
    ----------
    a
        Input Series a.
    b
        Input Series b.
    f
        Function to apply.
    return_dtype
        Output type of the udf.
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return pli.wrap_expr(pybinary_function(a._pyexpr, b._pyexpr, f, return_dtype))


def fold(
    acc: "pli.Expr",
    f: Callable[["pli.Series", "pli.Series"], "pli.Series"],
    exprs: Union[Sequence[Union["pli.Expr", str]], "pli.Expr"],
) -> "pli.Expr":
    """
    Accumulate over multiple columns horizontally/ row wise with a left fold.

    Parameters
    ----------
    acc
     Accumulator Expression. This is the value that will be initialized when the fold starts.
     For a sum this could for instance be lit(0).

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


def any(name: Union[str, List["pli.Expr"]]) -> "pli.Expr":
    """
    Evaluate columnwise or elementwise with a bitwise OR operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a | b, name).alias("any")
    return col(name).sum() > 0


def exclude(columns: Union[str, List[str]]) -> "pli.Expr":
    """
    Exclude certain columns from a wildcard expression.

    Syntactic sugar for:

    >>> pl.col("*").exclude(columns)  # doctest: +SKIP

    Parameters
    ----------
    columns
        Column(s) to exclude from selection


    Examples
    --------

    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": ["a", "b", None],
    ...         "c": [None, 2, 1],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌─────┬──────┬──────┐
    │ a   ┆ b    ┆ c    │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ str  ┆ i64  │
    ╞═════╪══════╪══════╡
    │ 1   ┆ a    ┆ null │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2   ┆ b    ┆ 2    │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 3   ┆ null ┆ 1    │
    └─────┴──────┴──────┘

    >>> df.select(pl.exclude("b"))
    shape: (3, 2)
    ┌─────┬──────┐
    │ a   ┆ c    │
    │ --- ┆ ---  │
    │ i64 ┆ i64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2   ┆ 2    │
    ├╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 3   ┆ 1    │
    └─────┴──────┘

    """
    return col("*").exclude(columns)


def all(name: Optional[Union[str, List["pli.Expr"]]] = None) -> "pli.Expr":
    """
    This function is two things

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
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a & b, name).alias("all")
    return col(name).cast(bool).sum() == col(name).count()


def groups(column: str) -> "pli.Expr":
    """
    Syntactic sugar for `pl.col("foo").agg_groups()`.
    """
    return col(column).agg_groups()


def quantile(
    column: str, quantile: float, interpolation: str = "nearest"
) -> "pli.Expr":
    """
    Syntactic sugar for `pl.col("foo").quantile(..)`.
    """
    return col(column).quantile(quantile, interpolation)


def arange(
    low: Union[int, "pli.Expr", "pli.Series"],
    high: Union[int, "pli.Expr", "pli.Series"],
    step: int = 1,
    eager: bool = False,
) -> Union["pli.Expr", "pli.Series"]:
    """
    Create a range expression. This can be used in a `select`, `with_column` etc.
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
        Step size of the range
    eager
        If eager evaluation is `True`, a Series is returned instead of an Expr
    """
    low = pli.expr_to_lit_or_expr(low, str_to_lit=False)
    high = pli.expr_to_lit_or_expr(high, str_to_lit=False)

    if eager:
        df = pli.DataFrame({"a": [1]})
        return df.select(arange(low, high, step).alias("arange"))["arange"]

    return pli.wrap_expr(pyarange(low._pyexpr, high._pyexpr, step))


def argsort_by(
    exprs: List[Union["pli.Expr", str]], reverse: Union[List[bool], bool] = False
) -> "pli.Expr":
    """
    Find the indexes that would sort the columns.

    Argsort by multiple columns. The first column will be used for the ordering.
    If there are duplicates in the first column, the second column will be used to determine the ordering
    and so on.

    Parameters
    ----------
    exprs
        Columns use to determine the ordering.
    reverse
        Default is ascending.
    """
    if not isinstance(reverse, list):
        reverse = [reverse] * len(exprs)
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(pyargsort_by(exprs, reverse))


def _datetime(
    year: Union["pli.Expr", str],
    month: Union["pli.Expr", str],
    day: Union["pli.Expr", str],
    hour: Optional[Union["pli.Expr", str]] = None,
    minute: Optional[Union["pli.Expr", str]] = None,
    second: Optional[Union["pli.Expr", str]] = None,
    millisecond: Optional[Union["pli.Expr", str]] = None,
) -> "pli.Expr":
    """
    Create polars Datetime from distinct time components.

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
    Expr of type pl.Datetime
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
    year: Union["pli.Expr", str],
    month: Union["pli.Expr", str],
    day: Union["pli.Expr", str],
) -> "pli.Expr":
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


def concat_str(exprs: Sequence[Union["pli.Expr", str]], sep: str = "") -> "pli.Expr":
    """
    Concat Utf8 Series in linear time. Non utf8 columns are cast to utf8.

    Parameters
    ----------
    exprs
        Columns to concat into a Utf8 Series
    sep
        String value that will be used to separate the values.
    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_concat_str(exprs, sep))


def format(fstring: str, *args: Union["pli.Expr", str]) -> "pli.Expr":
    """
    String format utility for expressions

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


def concat_list(exprs: Sequence[Union[str, "pli.Expr", "pli.Series"]]) -> "pli.Expr":
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
    ┌─────────────────┐
    │ A_rolling       │
    │ ---             │
    │ list [f64]      │
    ╞═════════════════╡
    │ [null, null, 1] │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [null, 1, 2]    │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [1, 2, 9]       │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [2, 9, 2]       │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ [9, 2, 13]      │
    └─────────────────┘

    """
    exprs = pli.selection_to_pyexpr_list(exprs)
    return pli.wrap_expr(_concat_lst(exprs))


def collect_all(
    lazy_frames: "List[pli.LazyFrame]",
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    string_cache: bool = False,
    no_optimization: bool = False,
    slice_pushdown: bool = False,
) -> "List[pli.DataFrame]":
    """
    Collect multiple LazyFrames at the same time. This runs all the computation graphs in parallel on
    Polars threadpool.

    Parameters
    ----------
    type_coercion
        Do type coercion optimization.
    predicate_pushdown
        Do predicate pushdown optimization.
    projection_pushdown
        Do projection pushdown optimization.
    simplify_expression
        Run simplify expressions optimization.
    string_cache
        Use a global string cache in this query.
        This is needed if you want to join on categorical columns.

        Caution!
            If you already have set a global string cache, set this to `False` as this will reset the
            global cache when the query is finished.
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
            string_cache,
            slice_pushdown,
        )
        prepared.append(ldf)

    out = _collect_all(prepared)

    # wrap the pydataframes into dataframe
    result = [pli.wrap_df(pydf) for pydf in out]

    return result


def select(
    exprs: Union[str, "pli.Expr", Sequence[str], Sequence["pli.Expr"], "pli.Series"]
) -> "pli.DataFrame":
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

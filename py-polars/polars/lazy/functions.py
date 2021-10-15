import typing as tp
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Type, Union

import numpy as np

import polars as pl

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
    from polars.polars import fold as pyfold
    from polars.polars import lit as pylit
    from polars.polars import map_mul as _map_mul
    from polars.polars import pearson_corr as pypearson_corr
    from polars.polars import spearman_rank_corr as pyspearman_rank_corr

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

from ..datatypes import DataType

__all__ = [
    "col",
    "count",
    "to_list",
    "std",
    "var",
    "max",
    "min",
    "sum",
    "mean",
    "avg",
    "median",
    "n_unique",
    "first",
    "last",
    "head",
    "tail",
    "lit",
    "pearson_corr",
    "spearman_rank_corr",
    "cov",
    "map",
    "apply",
    "map_binary",
    "fold",
    "any",
    "all",
    "groups",
    "quantile",
    "arange",
    "argsort_by",
    "concat_str",
    "concat_list",
    "collect_all",
    "exclude",
]


def col(name: Union[str, tp.List[str]]) -> "pl.Expr":
    """
    A column in a DataFrame.
    Can be used to select:

     * a single column by name
     * all columns by using a wildcard `"*"`
     * column by regular expression if the regex starts with `^` and ends with `$`

    Parameters
    col
        A string that holds the name of the column

    Examples
    -------

    >>> df = pl.DataFrame({
    >>> "ham": [1, 2, 3],
    >>> "hamburger": [11, 22, 33],
    >>> "foo": [3, 2, 1]})
    >>> df.select(col("foo"))
    shape: (3, 1)
    ╭─────╮
    │ foo │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    ├╌╌╌╌╌┤
    │ 2   │
    ├╌╌╌╌╌┤
    │ 1   │
    ╰─────╯
    >>> df.select(col("*"))
    shape: (3, 3)
    ╭─────┬───────────┬─────╮
    │ ham ┆ hamburger ┆ foo │
    │ --- ┆ ---       ┆ --- │
    │ i64 ┆ i64       ┆ i64 │
    ╞═════╪═══════════╪═════╡
    │ 1   ┆ 11        ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 22        ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 33        ┆ 1   │
    ╰─────┴───────────┴─────╯
    >>> df.select(col("^ham.*$"))
    shape: (3, 2)
    ╭─────┬───────────╮
    │ ham ┆ hamburger │
    │ --- ┆ ---       │
    │ i64 ┆ i64       │
    ╞═════╪═══════════╡
    │ 1   ┆ 11        │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2   ┆ 22        │
    ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
    │ 3   ┆ 33        │
    ╰─────┴───────────╯
    >>> df.select(col("*").exclude("ham"))
    shape: (3, 2)
    ╭───────────┬─────╮
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 22        ┆ 2   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 33        ┆ 1   │
    ╰───────────┴─────╯
    >>> df.select(col(["hamburger", "foo"])
    shape: (3, 2)
    ╭───────────┬─────╮
    │ hamburger ┆ foo │
    │ ---       ┆ --- │
    │ i64       ┆ i64 │
    ╞═══════════╪═════╡
    │ 11        ┆ 3   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 22        ┆ 2   │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ 33        ┆ 1   │
    ╰───────────┴─────╯

    """
    if isinstance(name, list):
        return pl.lazy.expr.wrap_expr(pycols(name))
    return pl.lazy.expr.wrap_expr(pycol(name))


def count(column: Union[str, "pl.Series"] = "") -> Union["pl.Expr", int]:
    """
    Count the number of values in this column.
    """
    if isinstance(column, pl.Series):
        return column.len()
    return col(column).count()


def to_list(name: str) -> "pl.Expr":
    """
    Aggregate to list.

    Re-exported as `pl.list()`
    """
    return col(name).list()


def std(column: Union[str, "pl.Series"]) -> Union["pl.Expr", Optional[float]]:
    """
    Get the standard deviation.
    """
    if isinstance(column, pl.Series):
        return column.std()
    return col(column).std()


def var(column: Union[str, "pl.Series"]) -> Union["pl.Expr", Optional[float]]:
    """
    Get the variance.
    """
    if isinstance(column, pl.Series):
        return column.var()
    return col(column).var()


def max(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the maximum value. Can be used horizontally or vertically.

    Parameters
    ----------
    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the maximum value of that column.
            - tp.List[Expr] -> aggregate the maximum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.max()
    elif isinstance(column, list):

        def max_(acc: "pl.Series", val: "pl.Series") -> "pl.Series":
            mask = acc < val
            return acc.zip_with(mask, val)

        return fold(lit(0), max_, column).alias("max")
    else:
        return col(column).max()


def min(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.min()
    elif isinstance(column, list):

        def min_(acc: "pl.Series", val: "pl.Series") -> "pl.Series":
            mask = acc > val
            return acc.zip_with(mask, val)

        return fold(lit(0), min_, column).alias("min")
    else:
        return col(column).min()


def sum(column: Union[str, tp.List["pl.Expr"], "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the sum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, pl.Series):
        return column.sum()
    elif isinstance(column, list):
        return fold(lit(0), lambda a, b: a + b, column).alias("sum")
    else:
        return col(column).sum()


def mean(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Get the mean value.
    """
    if isinstance(column, pl.Series):
        return column.mean()
    return col(column).mean()


def avg(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float]:
    """
    Alias for mean.
    """
    return mean(column)


def median(column: Union[str, "pl.Series"]) -> Union["pl.Expr", float, int]:
    """
    Get the median value.
    """
    if isinstance(column, pl.Series):
        return column.median()
    return col(column).median()


def n_unique(column: Union[str, "pl.Series"]) -> Union["pl.Expr", int]:
    """Count unique values."""
    if isinstance(column, pl.Series):
        return column.n_unique()
    return col(column).n_unique()


def first(column: Union[str, "pl.Series"]) -> Union["pl.Expr", Any]:
    """
    Get the first value.
    """
    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


def last(column: Union[str, "pl.Series"]) -> "pl.Expr":
    """
    Get the last value.
    """
    if isinstance(column, pl.Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


def head(
    column: Union[str, "pl.Series"], n: Optional[int] = None
) -> Union["pl.Expr", "pl.Series"]:
    """
    Get the first n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pl.Series):
        return column.head(n)
    return col(column).head(n)


def tail(
    column: Union[str, "pl.Series"], n: Optional[int] = None
) -> Union["pl.Expr", "pl.Series"]:
    """
    Get the last n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, pl.Series):
        return column.tail(n)
    return col(column).tail(n)


def lit(
    value: Optional[Union[float, int, str, datetime, "pl.Series"]],
    dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
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

    >>> # literal integer
    >>> lit(1)

    >>> # literal str.
    >>> lit("foo")

    >>> # literal datetime
    >>> lit(datetime(2021, 1, 20))

    >>> # literal Null
    >>> lit(None)

    >>> # literal eager Series
    >>> lit(Series("a", [1, 2, 3])
    """
    if isinstance(value, datetime):
        return lit(int((value.replace(tzinfo=timezone.utc)).timestamp() * 1e3)).cast(
            pl.Datetime
        )

    if isinstance(value, pl.Series):
        name = value.name
        value = value._s
        return pl.lazy.expr.wrap_expr(pylit(value)).alias(name)

    if isinstance(value, np.ndarray):
        return lit(pl.Series("", value))

    if dtype:
        return pl.lazy.expr.wrap_expr(pylit(value)).cast(dtype)
    return pl.lazy.expr.wrap_expr(pylit(value))


def spearman_rank_corr(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
) -> "pl.Expr":
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
    return pl.lazy.expr.wrap_expr(pyspearman_rank_corr(a._pyexpr, b._pyexpr))


def pearson_corr(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
) -> "pl.Expr":
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
    return pl.lazy.expr.wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr))


def cov(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
) -> "pl.Expr":
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
    return pl.lazy.expr.wrap_expr(pycov(a._pyexpr, b._pyexpr))


def map(
    exprs: Union[tp.List[str], tp.List["pl.Expr"]],
    f: Callable[[tp.List["pl.Series"]], "pl.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
    """
    Map a custom function over multiple columns/expressions and produce a single Series result.

    Parameters
    ----------
    columns
        Input Series to f
    f
        Function to apply over the input
    return_dtype
        dtype of the output Series

    Returns
    -------
    Expr
    """
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(_map_mul(exprs, f, return_dtype, apply_groups=False))


def apply(
    exprs: Union[tp.List[str], tp.List["pl.Expr"]],
    f: Callable[[tp.List["pl.Series"]], "pl.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
    """
    Apply a custom function in a GroupBy context.

    Depending on the context it has the following behavior:

    ## Context

    * Select/Project
        Don't do this, use `map_mul`
    * GroupBy
        expected type `f`: Callable[[Series], Series]
        Applies a python function over each group.

    Parameters
    ----------
    columns
        Input Series to f
    f
        Function to apply over the input
    return_dtype
        dtype of the output Series

    Returns
    -------
    Expr
    """
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(_map_mul(exprs, f, return_dtype, apply_groups=True))


def map_binary(
    a: Union[str, "pl.Expr"],
    b: Union[str, "pl.Expr"],
    f: Callable[["pl.Series", "pl.Series"], "pl.Series"],
    return_dtype: Optional[Type[DataType]] = None,
) -> "pl.Expr":
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
    return pl.lazy.expr.wrap_expr(
        pybinary_function(a._pyexpr, b._pyexpr, f, return_dtype)
    )


def fold(
    acc: "pl.Expr",
    f: Callable[["pl.Series", "pl.Series"], "pl.Series"],
    exprs: Union[tp.List["pl.Expr"], "pl.Expr"],
) -> "pl.Expr":
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
    acc = pl.lazy.expr.expr_to_lit_or_expr(acc, str_to_lit=True)
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.wrap_expr(pyfold(acc._pyexpr, f, exprs))


def any(name: Union[str, tp.List["pl.Expr"]]) -> "pl.Expr":
    """
    Evaluate columnwise or elementwise with a bitwise OR operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a | b, name).alias("any")
    return col(name).sum() > 0


def exclude(columns: Union[str, tp.List[str]]) -> "pl.Expr":
    """
     Exclude certain columns from a wildcard expression.

     Syntactic sugar for:
     >>> col("*").exclude()

     Parameters
     ----------
     columns
         Column(s) to exclude from selection

     Examples
     --------

     >>> df = pl.DataFrame({
     >>>     "a": [1, 2, 3],
     >>>     "b": ["a", "b", None],
     >>>     "c": [None, 2, 1]
     >>> })
     >>> df
     shape: (3, 3)
     ╭─────┬──────┬──────╮
     │ a   ┆ b    ┆ c    │
     │ --- ┆ ---  ┆ ---  │
     │ i64 ┆ str  ┆ i64  │
     ╞═════╪══════╪══════╡
     │ 1   ┆ "a"  ┆ null │
     ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
     │ 2   ┆ "b"  ┆ 2    │
     ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
     │ 3   ┆ null ┆ 1    │
     ╰─────┴──────┴──────╯
     >>> df.select(pl.exclude("b"))
    shape: (3, 2)
     ╭─────┬──────╮
     │ a   ┆ c    │
     │ --- ┆ ---  │
     │ i64 ┆ i64  │
     ╞═════╪══════╡
     │ 1   ┆ null │
     ├╌╌╌╌╌┼╌╌╌╌╌╌┤
     │ 2   ┆ 2    │
     ├╌╌╌╌╌┼╌╌╌╌╌╌┤
     │ 3   ┆ 1    │
     ╰─────┴──────╯
    """
    return col("*").exclude(columns)


def all(name: Optional[Union[str, tp.List["pl.Expr"]]] = None) -> "pl.Expr":
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

    >>> # sum all columns
    >>> df.select(pl.all().sum())


    >>> df.select(pl.all([col(name).is_not_null() for name in df.columns]))
    """
    if name is None:
        return col("*")
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a & b, name).alias("all")
    return col(name).cast(bool).sum() == col(name).count()


def groups(column: str) -> "pl.Expr":
    """
    Syntactic sugar for `column("foo").agg_groups()`.
    """
    return col(column).agg_groups()


def quantile(column: str, quantile: float) -> "pl.Expr":
    """
    Syntactic sugar for `column("foo").quantile(..)`.
    """
    return col(column).quantile(quantile)


def arange(
    low: Union[int, "pl.Expr", "pl.Series"],
    high: Union[int, "pl.Expr", "pl.Series"],
    step: int = 1,
    eager: bool = False,
) -> Union["pl.Expr", "pl.Series"]:
    """
    Create a range expression. This can be used in a `select`, `with_column` etc.
    Be sure that the range size is equal to the DataFrame you are collecting.

     Examples
     --------

    >>> (df.lazy()
        .filter(pl.col("foo") < pl.arange(0, 100))
        .collect())

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
    low = pl.lazy.expr_to_lit_or_expr(low, str_to_lit=False)
    high = pl.lazy.expr_to_lit_or_expr(high, str_to_lit=False)

    if eager:
        df = pl.DataFrame({"a": [1]})
        return df.select(pl.arange(low, high, step).alias("arange"))["arange"]  # type: ignore

    return pl.wrap_expr(pyarange(low._pyexpr, high._pyexpr, step))


def argsort_by(
    exprs: tp.List["pl.Expr"], reverse: Union[tp.List[bool], bool] = False
) -> "pl.Expr":
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
        reverse = [reverse]
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(pyargsort_by(exprs, reverse))


def concat_str(exprs: tp.List["pl.Expr"], sep: str = "") -> "pl.Expr":
    """
    Concat Utf8 Series in linear time. Non utf8 columns are cast to utf8.

    Parameters
    ----------
    exprs
        Columns to concat into a Utf8 Series
    sep
        String value that will be used to separate the values.
    """
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(_concat_str(exprs, sep))


def concat_list(exprs: tp.List["pl.Expr"]) -> "pl.Expr":
    """
    Concat the arrays in a Series dtype List in linear time.

    Parameters
    ----------
    exprs
        Columns to concat into a List Series
    """
    exprs = pl.lazy.expr._selection_to_pyexpr_list(exprs)
    return pl.lazy.expr.wrap_expr(_concat_lst(exprs))


def collect_all(
    lazy_frames: "tp.List[pl.LazyFrame]",
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    string_cache: bool = False,
    no_optimization: bool = False,
) -> "tp.List[pl.DataFrame]":
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

    Returns
    -------
    List[DataFrame]
    """
    if no_optimization:
        predicate_pushdown = False
        projection_pushdown = False

    prepared = []

    for lf in lazy_frames:
        ldf = lf._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
        )
        prepared.append(ldf)

    out = _collect_all(prepared)

    # wrap the pydataframes into dataframe

    result = []
    for pydf in out:
        result.append(pl.eager.frame.wrap_df(pydf))

    return result

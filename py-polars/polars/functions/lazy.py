from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import DTYPE_TEMPORAL_UNITS, Date, Datetime, Int64
from polars.utils._async import _AioDataFrameResult, _GeventDataFrameResult
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_as_list_of_expressions,
)
from polars.utils._wrap import wrap_df, wrap_expr
from polars.utils.deprecation import deprecate_renamed_function

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    from typing import Awaitable, Collection, Literal

    from polars import DataFrame, Expr, LazyFrame, Series
    from polars.type_aliases import (
        CorrelationMethod,
        EpochTimeUnit,
        IntoExpr,
        PolarsDataType,
        RollingInterpolationMethod,
        SelectorType,
    )


def element() -> Expr:
    """
    An alias/placeholder for an element being evaluated in an `eval` expression.

    Also used to represent group elements when passing a custom `aggregate_function`
    to :func:`DataFrame.pivot`.

    Examples
    --------
    A horizontal rank computation by taking the elements of a list:

    >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
    >>> df.with_columns(
    ...     pl.concat_list(["a", "b"]).list.eval(pl.element().rank()).alias("rank")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬────────────┐
    │ a   ┆ b   ┆ rank       │
    │ --- ┆ --- ┆ ---        │
    │ i64 ┆ i64 ┆ list[f64]  │
    ╞═════╪═════╪════════════╡
    │ 1   ┆ 4   ┆ [1.0, 2.0] │
    │ 8   ┆ 5   ┆ [2.0, 1.0] │
    │ 3   ┆ 2   ┆ [2.0, 1.0] │
    └─────┴─────┴────────────┘

    A mathematical operation on list elements:

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

    See Also
    --------
    polars.Expr.list.eval
    polars.DataFrame.pivot

    """
    return F.col("")


def count(column: str | None = None) -> Expr:
    """
    Either return the number of rows in the context, or return the number of non-`null` values in the column.

    If no arguments are passed, returns the number of rows in the context.
    Rows containing `null` values count towards the total.
    This is similar to `COUNT(*)` in SQL.

    Otherwise, this function is syntactic sugar for `col(column).count()`.

    Parameters
    ----------
    column
        An optional column name.

    Returns
    -------
    Expr
        A :class:`UInt32` expression.

    See Also
    --------
    Expr.count

    Examples
    --------
    Return the number of rows in a context. Note that rows containing `null` values are
    counted towards the total:

    >>> df = pl.DataFrame({"a": [1, 2, None], "b": [3, None, None]})
    >>> df.select(pl.count())
    shape: (1, 1)
    ┌───────┐
    │ count │
    │ ---   │
    │ u32   │
    ╞═══════╡
    │ 3     │
    └───────┘

    Return the number of non-`null` values in a column:

    >>> df.select(pl.count("a"))
    shape: (1, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ u32 │
    ╞═════╡
    │ 2   │
    └─────┘
    """  # noqa: W505
    if column is None:
        return wrap_expr(plr.count())

    return F.col(column).count()


def implode(name: str) -> Expr:
    """
    Aggregate all of a column's values into a length-1 :class:`List` column.

    This function is syntactic sugar for `pl.col(name).implode()`.

    Parameters
    ----------
    name
        The column name.

    """
    return F.col(name).implode()


def std(column: str, ddof: int = 1) -> Expr:
    """
    Get the standard deviation of the elements in a column.

    This function is syntactic sugar for `pl.col(column).std(ddof)`.

    Parameters
    ----------
    column
        The column name.
    ddof
        "Delta Degrees of Freedom": the divisor used in the calculation is
        `N - ddof`, where `N` represents the number of elements.
        By default, `ddof` is 1.

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
    return F.col(column).std(ddof)


def var(column: str, ddof: int = 1) -> Expr:
    """
    Get the variance of the elements in a column.

    This function is syntactic sugar for `pl.col(column).var(ddof)`.

    Parameters
    ----------
    column
        The column name.
    ddof
        "Delta Degrees of Freedom": the divisor used in the calculation is
        `N - ddof`, where `N` represents the number of elements.
        By default, `ddof` is 1.

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
    return F.col(column).var(ddof)


def mean(column: str) -> Expr:
    """
    Get the mean of the elements in a column.

    This function is syntactic sugar for `pl.col(column).mean()`.

    Parameters
    ----------
    column
        The column name.

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

    """
    return F.col(column).mean()


def median(column: str) -> Expr:
    """
    Get the median of the elements in a column.

    This function is syntactic sugar for `pl.col(column).median()`.

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

    """
    return F.col(column).median()


def n_unique(column: str) -> Expr:
    """
    Get the number of unique values in a column.

    This function is syntactic sugar for `pl.col(column).n_unique()`.

    Parameters
    ----------
    column
        The column name.

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

    """
    return F.col(column).n_unique()


def approx_n_unique(column: str | Expr) -> Expr:
    """
    Get a fast approximation of the number of unique values in a column.

    This is done using the HyperLogLog++ algorithm for cardinality estimation.

    Parameters
    ----------
    column
        The column name.

    Examples
    --------
    >>> df = pl.DataFrame({"a": [1, 8, 1], "b": [4, 5, 2], "c": ["foo", "bar", "foo"]})
    >>> df.select(pl.approx_n_unique("a"))
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
        return column.approx_n_unique()
    return F.col(column).approx_n_unique()


def first(column: str | None = None) -> Expr:
    """
    Get the first value.

    This function has different behavior depending on the input type:

    - `None` -> Expression to take first column of a context.
    - `str` -> Syntactic sugar for `pl.col(column).first()`.

    Parameters
    ----------
    column
        An optional column name.

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

    """
    if column is None:
        return wrap_expr(plr.first())

    return F.col(column).first()


def last(column: str | None = None) -> Expr:
    """
    Get the last value.

    This function has different behavior depending on the input type:

    - `None` -> Expression to take last column of a context.
    - `str` -> Syntactic sugar for `pl.col(column).last()`.

    Parameters
    ----------
    column
        An optional column name.

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

    """
    if column is None:
        return wrap_expr(plr.last())

    return F.col(column).last()


def head(column: str, n: int = 10) -> Expr:
    """
    Get the first `n` elements of a column.

    This function is syntactic sugar for `pl.col(column).head(n)`.

    Parameters
    ----------
    column
        The column name.
    n
        The number of elements to return.

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

    """
    return F.col(column).head(n)


def tail(column: str, n: int = 10) -> Expr:
    """
    Get the last `n` elements of a column.

    This function is syntactic sugar for `pl.col(column).tail(n)`.

    Parameters
    ----------
    column
        The column name.
    n
        The number of elements to return.

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

    """
    return F.col(column).tail(n)


def corr(
    a: IntoExpr,
    b: IntoExpr,
    *,
    method: CorrelationMethod = "pearson",
    ddof: int = 1,
    propagate_nans: bool = False,
) -> Expr:
    """
    Compute the Pearson or Spearman rank correlation correlation between two columns.

    Parameters
    ----------
    a
        A column name or expression.
    b
        Another column name or expression.
    ddof
        "Delta Degrees of Freedom": the divisor used in the calculation is
        `N - ddof`, where `N` represents the number of elements.
        By default, `ddof` is 1.
    method : {'pearson', 'spearman'}
        The correlation method.
    propagate_nans
        If `propagate_nans=True`, any `NaN` encountered will lead to `NaN` in the
        output. If `propagate_nans=False` (the default), `NaN` is considered larger than
        any finite number and will thus lead to the highest rank.

    Examples
    --------
    Pearson correlation:

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
    a = parse_as_expression(a)
    b = parse_as_expression(b)

    if method == "pearson":
        return wrap_expr(plr.pearson_corr(a, b, ddof))
    elif method == "spearman":
        return wrap_expr(plr.spearman_rank_corr(a, b, ddof, propagate_nans))
    else:
        raise ValueError(
            f"method must be one of {{'pearson', 'spearman'}}, got {method!r}"
        )


def cov(a: IntoExpr, b: IntoExpr, ddof: int = 1) -> Expr:
    """
    Compute the covariance between two columns.

    Parameters
    ----------
    a
        A column name or expression.
    b
        Another column name or expression.
    ddof
        "Delta Degrees of Freedom": the divisor used in the calculation is
        `N - ddof`, where `N` represents the number of elements.
        By default, `ddof` is 1.

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
    a = parse_as_expression(a)
    b = parse_as_expression(b)
    return wrap_expr(plr.cov(a, b, ddof))


def map_batches(
    exprs: Sequence[str] | Sequence[Expr],
    function: Callable[[Sequence[Series]], Series],
    return_dtype: PolarsDataType | None = None,
) -> Expr:
    """
    Apply a custom Python function to a sequence of `Series` to form a single `Series`.

    Unlike :func:`DataFrame.map_batches`, which maps a single `Series` to a
    single `Series`, here the custom function must map multiple `Series` to a single
    `Series`.

    If you want to apply a custom function elementwise over single values, see
    :func:`DataFrame.map_elements`. A reasonable use case for `map_batches` is
    transforming the values represented by an expression using a third-party library
    like :mod:`numpy`.

    Parameters
    ----------
    exprs
        A sequence of column names or expressions to input to the custom function.
    function
        The function or `Callable` to apply; must take multiple `Series` (as a Python
        `list`) and return a single `Series`.
    return_dtype
        The data type of the output `Series`. If not set, will be auto-inferred.

    Returns
    -------
    Expr
        An expression of the data type given by `return_dtype`.

    Examples
    --------
    >>> def test_func(a, b, c):
    ...     return a + b + c
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3, 4],
    ...         "b": [4, 5, 6, 7],
    ...     }
    ... )
    >>>
    >>> df.with_columns(
    ...     (
    ...         pl.struct(["a", "b"]).map_batches(
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
            exprs, function, return_dtype, map_groups=False, returns_scalar=False
        )
    )


@deprecate_renamed_function("map_batches", version="0.19.0")
def map(
    exprs: Sequence[str] | Sequence[Expr],
    function: Callable[[Sequence[Series]], Series],
    return_dtype: PolarsDataType | None = None,
) -> Expr:
    """
    Map a custom function over multiple columns.

    .. deprecated:: 0.19.0
        This function has been renamed to :func:`map_batches`.

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
        Expression with the data type given by `return_dtype`.

    """
    return map_batches(exprs, function, return_dtype)


def map_groups(
    exprs: Sequence[str | Expr],
    function: Callable[[Sequence[Series]], Series | Any],
    return_dtype: PolarsDataType | None = None,
    *,
    returns_scalar: bool = True,
) -> Expr:
    """
    Apply a custom/user-defined function (UDF) in a GroupBy context.

    Unlike :func:`DataFrame.map_groups`, which maps a single `Series` to a
    single `Series`, here the custom function must map multiple `Series` to a single
    `Series`.

    .. warning::
        This method is much slower than the native expressions API.
        Only use it if you cannot implement your logic otherwise.

    Implementing logic using a Python function is almost always *significantly*
    slower and more memory intensive than implementing the same logic using
    the native expression API because:

    - The native expression engine runs in Rust; UDFs run in Python.
    - Use of Python UDFs forces the DataFrame to be materialized in memory.
    - Polars-native expressions can be parallelised (UDFs cannot).
    - Polars-native expressions can be logically optimised (UDFs cannot).

    Wherever possible you should strongly prefer the native expression API
    to achieve the best performance.

    The idiomatic way to apply custom functions over multiple columns is via:

    `pl.struct([my_columns]).map_elements(lambda struct_series: ...)`

    Parameters
    ----------
    exprs
        A sequence of column names or expressions to input to the custom function.
    function
        The function or `Callable` to apply; must take multiple `Series` (as a Python
        `list`) and return a single `Series`.
    return_dtype
        The data type of the output `Series`. If not set, will be auto-inferred.
    returns_scalar
        Whether to force the output to be a :class:`List` column (of length-1 lists)
        even when `function` returns a non-:class:`List` `Series`.

    Returns
    -------
    Expr
        An expression with the data type given by `return_dtype`.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "group": [1, 1, 2],
    ...         "a": [1, 3, 3],
    ...         "b": [5, 6, 7],
    ...     }
    ... )
    >>> df
    shape: (3, 3)
    ┌───────┬─────┬─────┐
    │ group ┆ a   ┆ b   │
    │ ---   ┆ --- ┆ --- │
    │ i64   ┆ i64 ┆ i64 │
    ╞═══════╪═════╪═════╡
    │ 1     ┆ 1   ┆ 5   │
    │ 1     ┆ 3   ┆ 6   │
    │ 2     ┆ 3   ┆ 7   │
    └───────┴─────┴─────┘
    >>> (
    ...     df.group_by("group").agg(
    ...         pl.map_groups(
    ...             exprs=["a", "b"],
    ...             function=lambda list_of_series: list_of_series[0]
    ...             / list_of_series[0].sum()
    ...             + list_of_series[1],
    ...         ).alias("my_custom_aggregation")
    ...     )
    ... ).sort("group")
    shape: (2, 2)
    ┌───────┬───────────────────────┐
    │ group ┆ my_custom_aggregation │
    │ ---   ┆ ---                   │
    │ i64   ┆ list[f64]             │
    ╞═══════╪═══════════════════════╡
    │ 1     ┆ [5.25, 6.75]          │
    │ 2     ┆ [8.0]                 │
    └───────┴───────────────────────┘

    The output for group `1` can be understood as follows:

    - group `1` contains two `Series`: `'a': [1, 3]` and `'b': [4, 5]`
    - applying the function to those lists of `Series`, one gets the output
      `[1 / 4 + 5, 3 / 4 + 6]`, i.e. `[5.25, 6.75]`
    """
    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(
        plr.map_mul(
            exprs,
            function,
            return_dtype,
            map_groups=True,
            returns_scalar=returns_scalar,
        )
    )


@deprecate_renamed_function("map_groups", version="0.19.0")
def apply(
    exprs: Sequence[str | Expr],
    function: Callable[[Sequence[Series]], Series | Any],
    return_dtype: PolarsDataType | None = None,
    *,
    returns_scalar: bool = True,
) -> Expr:
    """
    Apply a custom/user-defined function (UDF) in a GroupBy context.

    .. deprecated:: 0.19.0
        This function has been renamed to :func:`map_groups`.

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
        Expression with the data type given by `return_dtype`.

    """
    return map_groups(exprs, function, return_dtype, returns_scalar=returns_scalar)


def fold(
    acc: IntoExpr,
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
) -> Expr:
    """
    Accumulate over multiple columns horizontally/row-wise with a left fold.

    Parameters
    ----------
    acc
        An accumulator expression. This is the value that will be initialized when the
        `fold` starts. For a sum, this could for instance be `lit(0)`.
    function
        A function that takes two `Series` as arguments - the accumulated value so far,
        and the new value to accumulate with it - and returns a single `Series` with the
        new accumulated value: `function(acc, value) -> new_value`.
    exprs
        The expressions to aggregate over.

    Notes
    -----
    If you simply want the first encountered expression as the initial value, use
    :func:`reduce`.

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

    Horizontally sum over all columns and add 1:

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
    # in case of col("*")
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
    Accumulate over multiple columns horizontally/row-wise with a left fold.

    Parameters
    ----------
    function
        A function that takes two `Series` as arguments - the accumulated value so far,
        and the new value to accumulate with it - and returns a single `Series` with the
        new accumulated value: `function(acc, value) -> new_value`.
    exprs
        The expressions to aggregate over.

    Notes
    -----
    If you want to explicitly specify an initial value for the accumulation, rather than
    using the first encountered expression as the initial value, use :func:`reduce`.

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

    Horizontally sum over all columns:

    >>> df.select(
    ...     pl.reduce(function=lambda acc, x: acc + x, exprs=pl.col("*")).alias("sum")
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
    # in case of col("*")
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.reduce(function, exprs))


def cum_fold(
    acc: IntoExpr,
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
    *,
    include_init: bool = False,
) -> Expr:
    """
    Accumulate over multiple columns horizontally/row-wise with a left fold.

    Every cumulative result is added as a separate field in a :class:`Struct` column.

    Parameters
    ----------
    acc
        An accumulator expression. This is the value that will be initialized when the
        `fold` starts. For a sum, this could for instance be `lit(0)`.
    function
        A function that takes two `Series` as arguments - the accumulated value so far,
        and the new value to accumulate with it - and returns a single `Series` with the
        new accumulated value: `function(acc, value) -> new_value`.
    exprs
        The expressions to aggregate over.
    include_init
        Whether to also include the initial accumulator state as a :class:`Struct`
        field.

    Notes
    -----
    If you simply want the first encountered expression as the initial value, use
    :func:`cum_reduce`.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [3, 4, 5],
    ...         "c": [5, 6, 7],
    ...     }
    ... )
    >>> df.with_columns(
    ...     pl.cum_fold(acc=pl.lit(1), function=lambda acc, x: acc + x, exprs=pl.all())
    ... )
    shape: (3, 4)
    ┌─────┬─────┬─────┬───────────┐
    │ a   ┆ b   ┆ c   ┆ cum_fold  │
    │ --- ┆ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ struct[3] │
    ╞═════╪═════╪═════╪═══════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {2,5,10}  │
    │ 2   ┆ 4   ┆ 6   ┆ {3,7,13}  │
    │ 3   ┆ 5   ┆ 7   ┆ {4,9,16}  │
    └─────┴─────┴─────┴───────────┘

    """
    # in case of col("*")
    acc = parse_as_expression(acc, str_as_lit=True)
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cum_fold(acc, function, exprs, include_init).alias("cum_fold"))


def cum_reduce(
    function: Callable[[Series, Series], Series],
    exprs: Sequence[Expr | str] | Expr,
) -> Expr:
    """
    Accumulate over multiple columns horizontally/row-wise with a left fold.

    Every cumulative result is added as a separate field in a :class:`Struct` column.

    Parameters
    ----------
    function
        A function that takes two `Series` as arguments - the accumulated value so far,
        and the new value to accumulate with it - and returns a single `Series` with the
        new accumulated value: `function(acc, value) -> new_value`.
    exprs
        The expressions to aggregate over.

    Notes
    -----
    If you want to explicitly specify an initial value for the accumulation, rather than
    using the first encountered expression as the initial value, use :func:`cum_fold`.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": [3, 4, 5],
    ...         "c": [5, 6, 7],
    ...     }
    ... )
    >>> df.with_columns(pl.cum_reduce(function=lambda acc, x: acc + x, exprs=pl.all()))
    shape: (3, 4)
    ┌─────┬─────┬─────┬────────────┐
    │ a   ┆ b   ┆ c   ┆ cum_reduce │
    │ --- ┆ --- ┆ --- ┆ ---        │
    │ i64 ┆ i64 ┆ i64 ┆ struct[3]  │
    ╞═════╪═════╪═════╪════════════╡
    │ 1   ┆ 3   ┆ 5   ┆ {1,4,9}    │
    │ 2   ┆ 4   ┆ 6   ┆ {2,6,12}   │
    │ 3   ┆ 5   ┆ 7   ┆ {3,8,15}   │
    └─────┴─────┴─────┴────────────┘
    """
    # in case of col("*")
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cum_reduce(function, exprs).alias("cum_reduce"))


def arctan2(y: str | Expr, x: str | Expr) -> Expr:
    """
    Compute the two-argument inverse tangent in radians.

    Returns the angle (in radians) in the plane between the
    positive x-axis and the ray from the origin to `(x, y)`.

    Parameters
    ----------
    y
        A column name or expression.
    x
        Another column name or expression.

    Examples
    --------
    >>> import math
    >>> twoRootTwo = math.sqrt(2) / 2
    >>> df = pl.DataFrame(
    ...     {
    ...         "y": [twoRootTwo, -twoRootTwo, twoRootTwo, -twoRootTwo],
    ...         "x": [twoRootTwo, twoRootTwo, -twoRootTwo, -twoRootTwo],
    ...     }
    ... )
    >>> df.select(
    ...     pl.arctan2d("y", "x").alias("atan2d"), pl.arctan2("y", "x").alias("atan2")
    ... )
    shape: (4, 2)
    ┌────────┬───────────┐
    │ atan2d ┆ atan2     │
    │ ---    ┆ ---       │
    │ f64    ┆ f64       │
    ╞════════╪═══════════╡
    │ 45.0   ┆ 0.785398  │
    │ -45.0  ┆ -0.785398 │
    │ 135.0  ┆ 2.356194  │
    │ -135.0 ┆ -2.356194 │
    └────────┴───────────┘

    """
    if isinstance(y, str):
        y = F.col(y)
    if isinstance(x, str):
        x = F.col(x)
    return wrap_expr(plr.arctan2(y._pyexpr, x._pyexpr))


def arctan2d(y: str | Expr, x: str | Expr) -> Expr:
    """
    Compute the two-argument inverse tangent in degrees.

    Returns the angle (in degrees) in the plane between the positive x-axis
    and the ray from the origin to `(x, y)`.

    Parameters
    ----------
    y
        A column name or expression.
    x
        Another column name or expression.

    Examples
    --------
    >>> import math
    >>> twoRootTwo = math.sqrt(2) / 2
    >>> df = pl.DataFrame(
    ...     {
    ...         "y": [twoRootTwo, -twoRootTwo, twoRootTwo, -twoRootTwo],
    ...         "x": [twoRootTwo, twoRootTwo, -twoRootTwo, -twoRootTwo],
    ...     }
    ... )
    >>> df.select(
    ...     pl.arctan2d("y", "x").alias("atan2d"), pl.arctan2("y", "x").alias("atan2")
    ... )
    shape: (4, 2)
    ┌────────┬───────────┐
    │ atan2d ┆ atan2     │
    │ ---    ┆ ---       │
    │ f64    ┆ f64       │
    ╞════════╪═══════════╡
    │ 45.0   ┆ 0.785398  │
    │ -45.0  ┆ -0.785398 │
    │ 135.0  ┆ 2.356194  │
    │ -135.0 ┆ -2.356194 │
    └────────┴───────────┘

    """
    if isinstance(y, str):
        y = F.col(y)
    if isinstance(x, str):
        x = F.col(x)
    return wrap_expr(plr.arctan2d(y._pyexpr, x._pyexpr))


def exclude(
    columns: (
        str
        | PolarsDataType
        | SelectorType
        | Expr
        | Collection[str | PolarsDataType | SelectorType | Expr]
    ),
    *more_columns: str | PolarsDataType | SelectorType | Expr,
) -> Expr:
    """
    Select all columns except those matching the given columns, datatypes, or selectors.

    .. versionchanged:: 0.20.3
        This function is now a simple redirect to the `cs.exclude()` selector.

    Parameters
    ----------
    columns
        One or more columns, datatypes, or selectors representing the columns to
        exclude.
    *more_columns
        Additional columns, datatypes, or selectors to exclude, specified as positional
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
    >>> df.select(pl.exclude("ba", "xx"))
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

    Exclude by regex, e.g. removing all columns whose names end with the letter
    `"a"`:

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

    Exclude by dtype(s), e.g. removing all columns of type :class:`Int64` or
    :class:`Float64`:

    >>> df.select(pl.exclude(pl.Int64, pl.Float64))
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

    Exclude column using a compound selector:

    >>> import polars.selectors as cs
    >>> df.select(pl.exclude(cs.first() | cs.last()))
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
    from polars.selectors import exclude

    return exclude(columns, *more_columns)


def groups(column: str) -> Expr:
    """
    Syntactic sugar for `pl.col(column).agg_groups()`.

    Parameters
    ----------
    column
        A column name.
    """
    return F.col(column).agg_groups()


def quantile(
    column: str,
    quantile: float | Expr,
    interpolation: RollingInterpolationMethod = "nearest",
) -> Expr:
    """
    Syntactic sugar for `pl.col(column).quantile(..)`.

    Parameters
    ----------
    column
        A column name.
    quantile
        A quantile between 0.0 and 1.0.
    interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
        The interpolation method to use when the specified quantile falls between two
        values.

    """
    return F.col(column).quantile(quantile, interpolation)


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
        Whether to arg sort in descending instead of ascending order. When sorting by
        multiple columns, can be specified per column by passing a sequence of booleans.

    Examples
    --------
    Pass a single column name to arg sort by that column:

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

    Arg sort by multiple columns by either passing a list of columns, or by specifying
    each column as a positional argument:

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
    lazy_frames: Iterable[LazyFrame],
    *,
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    no_optimization: bool = False,
    slice_pushdown: bool = True,
    comm_subplan_elim: bool = True,
    comm_subexpr_elim: bool = True,
    streaming: bool = False,
) -> list[DataFrame]:
    """
    Collect multiple lazyframes at the same time.

    This runs all the computation graphs in parallel on the Polars threadpool.

    Parameters
    ----------
    lazy_frames
        A list of lazyframes to collect.
    type_coercion
        Whether to perform type coercion optimization.
    predicate_pushdown
        Whether to perform predicate pushdown optimization.
    projection_pushdown
        Whether to perform projection pushdown optimization.
    simplify_expression
        Whether to perform expression simplification optimization.
    no_optimization
        Whether to turn off (certain) optimizations.
    slice_pushdown
        Whether to perform slice pushdown optimization.
    comm_subplan_elim
        Whether to try to cache branching subplans that occur on self-joins or
        unions.
    comm_subexpr_elim
        Whether to cache and reuse common subexpressions.
    streaming
        Whether to run parts of the query in a streaming fashion (this is in an
        alpha state).

    Returns
    -------
    list of DataFrames
        The collected dataframes, returned in the same order as the input
        lazyframes.

    """
    if no_optimization:
        predicate_pushdown = False
        projection_pushdown = False
        slice_pushdown = False
        comm_subplan_elim = False
        comm_subexpr_elim = False

    prepared = []

    for lf in lazy_frames:
        ldf = lf._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim,
            comm_subexpr_elim,
            streaming,
            _eager=False,
        )
        prepared.append(ldf)

    out = plr.collect_all(prepared)

    # wrap the pydataframes into dataframe
    result = [wrap_df(pydf) for pydf in out]

    return result


@overload
def collect_all_async(
    lazy_frames: Iterable[LazyFrame],
    *,
    gevent: Literal[True],
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    no_optimization: bool = True,
    slice_pushdown: bool = True,
    comm_subplan_elim: bool = True,
    comm_subexpr_elim: bool = True,
    streaming: bool = True,
) -> _GeventDataFrameResult[list[DataFrame]]:
    ...


@overload
def collect_all_async(
    lazy_frames: Iterable[LazyFrame],
    *,
    gevent: Literal[False] = False,
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    no_optimization: bool = False,
    slice_pushdown: bool = True,
    comm_subplan_elim: bool = True,
    comm_subexpr_elim: bool = True,
    streaming: bool = False,
) -> Awaitable[list[DataFrame]]:
    ...


def collect_all_async(
    lazy_frames: Iterable[LazyFrame],
    *,
    gevent: bool = False,
    type_coercion: bool = True,
    predicate_pushdown: bool = True,
    projection_pushdown: bool = True,
    simplify_expression: bool = True,
    no_optimization: bool = False,
    slice_pushdown: bool = True,
    comm_subplan_elim: bool = True,
    comm_subexpr_elim: bool = True,
    streaming: bool = False,
) -> Awaitable[list[DataFrame]] | _GeventDataFrameResult[list[DataFrame]]:
    """
    Collect multiple lazyframes at the same time, asynchronously.

    Collects into a list of dataframes (like :func:`polars.collect_all`),
    but instead of returning them directly, they are scheduled to be collected
    inside the thread pool, while this method returns almost instantly.

    May be useful if you use `gevent <https://www.gevent.org>`_ or `asyncio
    <https://docs.python.org/3/library/asyncio.html>`_ and want to release control
    to other greenlets/tasks while lazyframes are being collected.

    Parameters
    ----------
    lazy_frames
        A list of lazyframes to collect.
    gevent
        Whether to return a wrapper to `gevent.event.AsyncResult` instead of an
        `Awaitable`.
    type_coercion
        Whether to perform type coercion optimization.
    predicate_pushdown
        Whether to perform predicate pushdown optimization.
    projection_pushdown
        Whether to perform projection pushdown optimization.
    simplify_expression
        Whether to perform expression simplification optimization.
    no_optimization
        Whether to turn off (certain) optimizations.
    slice_pushdown
        Whether to perform slice pushdown optimization.
    comm_subplan_elim
        Whether to try to cache branching subplans that occur on self-joins or unions.
    comm_subexpr_elim
        Whether to cache and reuse common subexpressions.
    streaming
        Whether to run parts of the query in a streaming fashion (this is in an alpha
        state).

    Notes
    -----
    In case of error `set_exception` is used on
    `asyncio.Future`/`gevent.event.AsyncResult` and will be reraised by them.

    Warnings
    --------
    This functionality is experimental and may change without it being considered a
    breaking change.

    See Also
    --------
    polars.collect_all : Collect multiple lazyframes at the same time.
    LazyFrame.collect_async: To collect single frame.

    Returns
    -------
    If `gevent=False` (the default), returns `Awaitable`.

    If `gevent=True`, returns a wrapper that has a
    `.get(block=True, timeout=None)` method.
    """
    if no_optimization:
        predicate_pushdown = False
        projection_pushdown = False
        slice_pushdown = False
        comm_subplan_elim = False
        comm_subexpr_elim = False

    prepared = []

    for lf in lazy_frames:
        ldf = lf._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim,
            comm_subexpr_elim,
            streaming,
            _eager=False,
        )
        prepared.append(ldf)

    result = _GeventDataFrameResult() if gevent else _AioDataFrameResult()
    plr.collect_all_with_callback(prepared, result._callback_all)  # type: ignore[attr-defined]
    return result  # type: ignore[return-value]


def select(*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> DataFrame:
    """
    Run polars expressions without a context.

    This is syntactic sugar for running `df.select` on an empty `DataFrame`.

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
    >>> pl.select(min=pl.min_horizontal(foo, bar))
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
    Return indices where `condition` evaluates to `True`.

    Parameters
    ----------
    condition
        A :class:`Boolean` expression to evaluate.
    eager
        Whether to evaluate immediately and return a `Series`, rather than returning an
        expression.

    See Also
    --------
    Series.arg_true : Return indices where Series is True

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
        if not isinstance(condition, pl.Series):
            raise ValueError(
                "expected 'Series' in 'arg_where' if 'eager=True', got"
                f" {type(condition).__name__!r}"
            )
        return condition.to_frame().select(arg_where(F.col(condition.name))).to_series()
    else:
        condition = parse_as_expression(condition)
        return wrap_expr(plr.arg_where(condition))


def coalesce(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """
    Fold columns from left to right, keeping the first non-`null` value.

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
    Parse an epoch timestamp (or Unix time) to a Polars :class:`Date`/:class`Datetime`.

    Depending on the `time_unit` provided, this function will return a different dtype:

    - `time_unit="d"` returns :class:`pl.Date`
    - `time_unit="s"` returns `pl.Datetime["us"]` (the default for :class:`pl.Datetime`)
    - `time_unit="ms"` returns `pl.Datetime["ms"]`
    - `time_unit="us"` returns `pl.Datetime["us"]`
    - `time_unit="ns"` returns `pl.Datetime["ns"]`

    Parameters
    ----------
    column
        A `Series`, expression, or sequence of integers to interpret as timestamps.
    time_unit
        The unit of time of the timesteps since epoch time (00:00:00 UTC on 1 January
        1970).

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

    The function can also be used in an eager context by passing a `Series`:

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
        column = F.col(column)
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
            f"`time_unit` must be one of {{'ns', 'us', 'ms', 's', 'd'}}, got {time_unit!r}"
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
    Get the rolling (moving) covariance between two columns.

    A window of length `window_size` will traverse the pair of columns. The covariance
    of the values that fill this window will become one element of the output.

    The window corresponding to a given row of the output will include the
    corresponding row of the input and the `window_size - 1` rows before it. This means
    that the first `window_size - 1` rows of the output will be `null`.

    Parameters
    ----------
    a
        A column name or expression.
    b
        Another column name or expression.
    window_size
        The length of the window.
    min_periods
        The number of values in the window that should be non-`null` before computing a
        result. If `None`, it will be set equal to `window_size`.
    ddof
        "Delta Degrees of Freedom": the divisor for a length-`N` window is `N - ddof`.

    """
    if min_periods is None:
        min_periods = window_size
    if isinstance(a, str):
        a = F.col(a)
    if isinstance(b, str):
        b = F.col(b)
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
    Get the rolling (moving) correlation between two columns.

    A window of length `window_size` will traverse the pair of columns. The correlation
    of the values that fill this window will become one element of the output.

    The window corresponding to a given row of the output will include the
    corresponding row of the input and the `window_size - 1` rows before it. This means
    that the first `window_size - 1` rows of the output will be `null`.

    Parameters
    ----------
    a
        A column name or expression.
    b
        Another column name or expression.
    window_size
        The length of the window.
    min_periods
        The number of values in the window that should be non-`null` before computing a
        result. If `None`, it will be set equal to `window_size`.
    ddof
        "Delta Degrees of Freedom": the divisor for a length-`N` window is `N - ddof`.

    """
    if min_periods is None:
        min_periods = window_size
    if isinstance(a, str):
        a = F.col(a)
    if isinstance(b, str):
        b = F.col(b)
    return wrap_expr(
        plr.rolling_corr(a._pyexpr, b._pyexpr, window_size, min_periods, ddof)
    )


@overload
def sql_expr(sql: str) -> Expr:  # type: ignore[overload-overlap]
    ...


@overload
def sql_expr(sql: Sequence[str]) -> list[Expr]:
    ...


def sql_expr(sql: str | Sequence[str]) -> Expr | list[Expr]:
    """
    Parse one or more SQL expressions to polars expression(s).

    Parameters
    ----------
    sql
        One or more SQL expressions.

    Examples
    --------
    Parse a single SQL expression:

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

    Parse multiple SQL expressions:

    >>> df.with_columns(
    ...     *pl.sql_expr(["POWER(a,a) AS a_a", "CAST(a AS TEXT) AS a_txt"]),
    ... )
    shape: (2, 3)
    ┌─────┬─────┬───────┐
    │ a   ┆ a_a ┆ a_txt │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ f64 ┆ str   │
    ╞═════╪═════╪═══════╡
    │ 2   ┆ 4.0 ┆ 2     │
    │ 1   ┆ 1.0 ┆ 1     │
    └─────┴─────┴───────┘
    """
    if isinstance(sql, str):
        return wrap_expr(plr.sql_expr(sql))
    else:
        return [wrap_expr(plr.sql_expr(q)) for q in sql]


@deprecate_renamed_function("cum_fold", version="0.19.14")
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
    """  # noqa: W505
    # in case of col("*")
    acc = parse_as_expression(acc, str_as_lit=True)
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cum_fold(acc, function, exprs, include_init))


@deprecate_renamed_function("cum_reduce", version="0.19.14")
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
    """  # noqa: W505
    # in case of col("*")
    if isinstance(exprs, pl.Expr):
        exprs = [exprs]

    exprs = parse_as_list_of_expressions(exprs)
    return wrap_expr(plr.cum_reduce(function, exprs))

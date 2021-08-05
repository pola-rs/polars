import typing as tp
from datetime import datetime
from typing import Any, Callable, Optional, Sequence, Type, Union

import polars as pl

try:
    from polars.polars import PyExpr

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

from ..datatypes import Boolean, DataType, Date32, Date64, Float64, Int64, Utf8
from .functions import UDF, col, lit

__all__ = [
    "Expr",
    "ExprStringNameSpace",
    "ExprDateTimeNameSpace",
    "expr_to_lit_or_expr",
]


def _selection_to_pyexpr_list(
    exprs: Union[str, "Expr", Sequence[str], Sequence["Expr"]]
) -> tp.List["PyExpr"]:
    pyexpr_list: tp.List[PyExpr]
    if isinstance(exprs, str):
        pyexpr_list = [col(exprs)._pyexpr]
    elif isinstance(exprs, Expr):
        pyexpr_list = [exprs._pyexpr]
    else:
        pyexpr_list = []
        for expr in exprs:
            if isinstance(expr, str):
                expr = col(expr)
            pyexpr_list.append(expr._pyexpr)
    return pyexpr_list


def wrap_expr(pyexpr: "PyExpr") -> "Expr":
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """
    Expressions that can be used in various contexts.
    """

    def __init__(self) -> None:
        self._pyexpr: PyExpr

    @staticmethod
    def _from_pyexpr(pyexpr: "PyExpr") -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other: Any) -> "PyExpr":
        if isinstance(other, PyExpr):
            return other
        elif isinstance(other, Expr):
            return other._pyexpr
        else:
            return lit(other)._pyexpr

    def __to_expr(self, other: Any) -> "Expr":
        if isinstance(other, Expr):
            return other
        return lit(other)

    def __invert__(self) -> "Expr":
        return self.is_not()

    def __and__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._and(other._pyexpr))

    def __or__(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr._or(other._pyexpr))

    def __add__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr + self.__to_pyexpr(other))

    def __sub__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr - self.__to_pyexpr(other))

    def __mul__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr * self.__to_pyexpr(other))

    def __truediv__(self, other: Any) -> "Expr":
        return wrap_expr(self._pyexpr / self.__to_pyexpr(other))

    def __pow__(self, power: float, modulo: None = None) -> "Expr":
        return self.pow(power)

    def __ge__(self, other: Any) -> "Expr":
        return self.gt_eq(self.__to_expr(other))

    def __le__(self, other: Any) -> "Expr":
        return self.lt_eq(self.__to_expr(other))

    def __eq__(self, other: Any) -> "Expr":  # type: ignore[override]
        return self.eq(self.__to_expr(other))

    def __ne__(self, other: Any) -> "Expr":  # type: ignore[override]
        return self.neq(self.__to_expr(other))

    def __lt__(self, other: Any) -> "Expr":
        return self.lt(self.__to_expr(other))

    def __gt__(self, other: Any) -> "Expr":
        return self.gt(self.__to_expr(other))

    def eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.eq(other._pyexpr))

    def neq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.neq(other._pyexpr))

    def gt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt(other._pyexpr))

    def gt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.gt_eq(other._pyexpr))

    def lt_eq(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt_eq(other._pyexpr))

    def lt(self, other: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.lt(other._pyexpr))

    def alias(self, name: str) -> "Expr":
        """
        Rename the output of an expression.

        Parameters
        ----------
        name
            New name.

        Examples
        --------

        >>> df = pl.DataFrame({
        >>>     "a": [1, 2, 3],
        >>>     "b": ["a", "b", None]
        >>> })
        >>> df
        shape: (3, 2)
        ╭─────┬──────╮
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ "a"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ "b"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        ╰─────┴──────╯
        >>> df.select([
        >>>     col("a").alias("bar"),
        >>>     col("b").alias("foo")
        >>> ])
        shape: (3, 2)
        ╭─────┬──────╮
        │ bar ┆ foo  │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ "a"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ "b"  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ null │
        ╰─────┴──────╯

        """
        return wrap_expr(self._pyexpr.alias(name))

    def exclude(self, columns: Union[str, tp.List[str]]) -> "Expr":
        """
         Exclude certain columns from a wildcard expression.

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
         >>> df.select(col("*").exclude("b"))
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
        if isinstance(columns, str):
            columns = [columns]
        return wrap_expr(self._pyexpr.exclude(columns))

    def keep_name(self) -> "Expr":
        """
        Keep the original root name of the expression.

        Examples
        --------

        A groupby aggregation often changes the name of a column.
        With `keep_name` we can keep the original name of the column

        >>> df = pl.DataFrame({
        >>> "a": [1, 2, 3],
        >>> "b": ["a", "b", None]
        >>> })
        >>> (df.groupby("a")
        >>> .agg(col("b").list())
        >>> .sort(by="a")
        >>> )
        shape: (3, 2)
        ╭─────┬────────────╮
        │ a   ┆ b_agg_list │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ [a]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ [b]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        ╰─────┴────────────╯
        >>> # keep the original column name
        >>> (df.groupby("a")
        >>> .agg(col("b").list().keep_name())
        >>> .sort(by="a")
        >>> )
        shape: (3, 2)
        ╭─────┬────────────╮
        │ a   ┆ b          │
        │ --- ┆ ---        │
        │ i64 ┆ list [str] │
        ╞═════╪════════════╡
        │ 1   ┆ [a]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ [b]        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ [null]     │
        ╰─────┴────────────╯

        """

        return wrap_expr(self._pyexpr.keep_name())

    def is_not(self) -> "Expr":
        """
        Negate a boolean expression.

        Examples
        --------

        >>> df = pl.DataFrame({
        >>>     "a": [True, False, False],
        >>>     "b": ["a", "b", None],
        >>> })
        shape: (3, 2)
        ╭───────┬──────╮
        │ a     ┆ b    │
        │ ---   ┆ ---  │
        │ bool  ┆ str  │
        ╞═══════╪══════╡
        │ true  ┆ "a"  │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ "b"  │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ false ┆ null │
        ╰───────┴──────╯
        >>> df.select(col("a").is_not())
        shape: (3, 1)
        ╭───────╮
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ├╌╌╌╌╌╌╌┤
        │ true  │
        ╰───────╯

        """
        return wrap_expr(self._pyexpr.is_not())

    def is_null(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression contains null values.
        """
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression does not contain null values.
        """
        return wrap_expr(self._pyexpr.is_not_null())

    def is_finite(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are finite.
        """
        return wrap_expr(self._pyexpr.is_finite())

    def is_infinite(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are infinite.
        """
        return wrap_expr(self._pyexpr.is_infinite())

    def is_nan(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are NaN (Not A Number).
        """
        return wrap_expr(self._pyexpr.is_nan())

    def is_not_nan(self) -> "Expr":
        """
        Create a boolean expression returning `True` where the expression values are not NaN (Not A Number).
        """
        return wrap_expr(self._pyexpr.is_not_nan())

    def agg_groups(self) -> "Expr":
        """
        Get the group indexes of the group by operation.
        Should be used in aggregation context only.
        """
        return wrap_expr(self._pyexpr.agg_groups())

    def count(self) -> "Expr":
        """Count the number of values in this expression"""
        return wrap_expr(self._pyexpr.count())

    def slice(self, offset: int, length: int) -> "Expr":
        """
        Slice the Series.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.
        """
        return wrap_expr(self._pyexpr.slice(offset, length))

    def drop_nulls(self) -> "Expr":
        """
        Syntactic sugar for:

        >>> col("foo").filter(col("foo").is_not_null())
        """
        return self.filter(self.is_not_null())

    def cum_sum(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cum_sum(reverse))

    def cum_min(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cum_min(reverse))

    def cum_max(self, reverse: bool = False) -> "Expr":
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.
        """
        return wrap_expr(self._pyexpr.cum_max(reverse))

    def round(self, decimals: int) -> "Expr":
        """
        Round underlying floating point data by `decimals` digits.

        Parameters
        ----------
        decimals
            Number of decimals to round by.
        """
        return wrap_expr(self._pyexpr.round(decimals))

    def dot(self, other: "Expr") -> "Expr":
        """
        Compute the dot/inner product between two Expressions

        Parameters
        ----------
        other
            Expression to compute dot product with
        """
        other = expr_to_lit_or_expr(other, str_to_lit=False)
        return wrap_expr(self._pyexpr.dot(other._pyexpr))

    def mode(self) -> "Expr":
        """
        Compute the most occurring value(s). Can return multiple Values
        """
        return wrap_expr(self._pyexpr.mode())

    def cast(self, dtype: Type[Any]) -> "Expr":
        """
        Cast an expression to a different data types.

        Parameters
        ----------
        dtype
            Output data type.
        """
        if dtype == str:
            dtype = Utf8
        elif dtype == bool:
            dtype = Boolean
        elif dtype == float:
            dtype = Float64
        elif dtype == int:
            dtype = Int64
        return wrap_expr(self._pyexpr.cast(dtype))

    def sort(self, reverse: bool = False) -> "Expr":
        """
        Sort this column. In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        reverse
            False -> order from small to large.
            True -> order from large to small.
        """
        return wrap_expr(self._pyexpr.sort(reverse))

    def arg_sort(self, reverse: bool = False) -> "Expr":
        """
        Get the index values that would sort this column.

        Parameters
        ----------
        reverse
            False -> order from small to large.
            True -> order from large to small.

        Returns
        -------
        out
            Series of type UInt32
        """
        return wrap_expr(self._pyexpr.arg_sort(reverse))

    def sort_by(self, by: Union["Expr", str], reverse: bool = False) -> "Expr":
        """
        Sort this column by the ordering of another column.
        In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        by
            The column used for sorting.
        reverse
            False -> order from small to large.
            True -> order from large to small.
        """
        if isinstance(by, str):
            by = col(by)

        return wrap_expr(self._pyexpr.sort_by(by._pyexpr, reverse))

    def take(self, index: "Expr") -> "Expr":
        """
        Take values by index.

        Parameters
        ----------
        index
            An expression that leads to a UInt32 dtyped Series.

        Returns
        -------
        Values taken by index
        """
        index = expr_to_lit_or_expr(index, str_to_lit=False)
        return wrap_expr(self._pyexpr.take(index._pyexpr))

    def shift(self, periods: int) -> "Expr":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_expr(self._pyexpr.shift(periods))

    def shift_and_fill(self, periods: int, fill_value: "Expr") -> "Expr":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.
        """
        return wrap_expr(self._pyexpr.shift_and_fill(periods, fill_value._pyexpr))

    def fill_none(self, fill_value: Union[str, int, float, "Expr"]) -> "Expr":
        """
        Fill none value with a fill value
        """
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_expr(self._pyexpr.fill_none(fill_value._pyexpr))

    def forward_fill(self) -> "Expr":
        """
        Fill missing values with the latest seen values
        """
        return wrap_expr(self._pyexpr.forward_fill())

    def backward_fill(self) -> "Expr":
        """
        Fill missing values with the next to be seen values
        """
        return wrap_expr(self._pyexpr.backward_fill())

    def reverse(self) -> "Expr":
        """
        Reverse the selection.
        """
        return wrap_expr(self._pyexpr.reverse())

    def std(self) -> "Expr":
        """
        Get standard deviation.
        """
        return wrap_expr(self._pyexpr.std())

    def var(self) -> "Expr":
        """
        Get variance.
        """
        return wrap_expr(self._pyexpr.var())

    def max(self) -> "Expr":
        """
        Get maximum value.
        """
        return wrap_expr(self._pyexpr.max())

    def min(self) -> "Expr":
        """
        Get minimum value.
        """
        return wrap_expr(self._pyexpr.min())

    def sum(self) -> "Expr":
        """
        Get sum value.
        """
        return wrap_expr(self._pyexpr.sum())

    def mean(self) -> "Expr":
        """
        Get mean value.
        """
        return wrap_expr(self._pyexpr.mean())

    def median(self) -> "Expr":
        """
        Get median value.
        """
        return wrap_expr(self._pyexpr.median())

    def n_unique(self) -> "Expr":
        """Count unique values."""
        return wrap_expr(self._pyexpr.n_unique())

    def arg_unique(self) -> "Expr":
        """Get index of first unique value."""
        return wrap_expr(self._pyexpr.arg_unique())

    def unique(self) -> "Expr":
        """Get unique values."""
        return wrap_expr(self._pyexpr.unique())

    def first(self) -> "Expr":
        """
        Get the first value.
        """
        return wrap_expr(self._pyexpr.first())

    def last(self) -> "Expr":
        """
        Get the last value.
        """
        return wrap_expr(self._pyexpr.last())

    def list(self) -> "Expr":
        """
        Aggregate to list.
        """
        return wrap_expr(self._pyexpr.list())

    def over(self, expr: Union[str, "Expr", tp.List["Expr"]]) -> "Expr":
        """
        Apply window function over a subgroup.
        This is similar to a groupby + aggregation + self join.
        Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html)

        Parameters
        ----------
        expr
            Column(s) to group by.

        Examples
        --------

        >>> df = DataFrame({
        >>>    "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
        >>>    "values": [1, 2, 3, 4, 5, 6, 7, 8, 8]
        >>>})
        >>> (df.lazy()
        >>>    .select([
        >>>       col("groups")
        >>>       sum("values").over("groups"))
        >>>   ]).collect())
            ╭────────┬────────╮
            │ groups ┆ values │
            │ ---    ┆ ---    │
            │ i32    ┆ i32    │
            ╞════════╪════════╡
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ ...    ┆ ...    │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 2      ┆ 13     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 3      ┆ 15     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 3      ┆ 15     │
            ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
            │ 1      ┆ 16     │
            ╰────────┴────────╯

        """

        pyexprs = _selection_to_pyexpr_list(expr)

        return wrap_expr(self._pyexpr.over(pyexprs))

    def is_unique(self) -> "Expr":
        """
        Get mask of unique values.
        """
        return wrap_expr(self._pyexpr.is_unique())

    def is_first(self) -> "Expr":
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series
        """
        return wrap_expr(self._pyexpr.is_first())

    def is_duplicated(self) -> "Expr":
        """
        Get mask of duplicated values.
        """
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(self, quantile: float) -> "Expr":
        """
        Get quantile value.
        """
        return wrap_expr(self._pyexpr.quantile(quantile))

    def filter(self, predicate: "Expr") -> "Expr":
        """
        Filter a single column.
        Should be used in aggregation context. If you want to filter on a DataFrame level, use `LazyFrame.filter`.

        Parameters
        ----------
        predicate
            Boolean expression.
        """
        return wrap_expr(self._pyexpr.filter(predicate._pyexpr))

    def map(
        self,
        f: Union["UDF", Callable[["pl.Series"], "pl.Series"]],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Expr":
        """
        Apply a custom python function. This function must produce a `Series`. Any other value will be stored as
        null/missing. If you want to apply a function over single values, consider using `apply`.

        [read more in the book](https://ritchie46.github.io/polars-book/how_can_i/use_custom_functions.html#lazy)

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        """
        if isinstance(f, UDF):
            return_dtype = f.return_dtype
            f = f.f
        if return_dtype == str:
            return_dtype = Utf8
        elif return_dtype == int:
            return_dtype = Int64
        elif return_dtype == float:
            return_dtype = Float64
        elif return_dtype == bool:
            return_dtype = Boolean
        return wrap_expr(self._pyexpr.map(f, return_dtype))

    def apply(
        self,
        f: Union[Callable[["pl.Series"], "pl.Series"], Callable[[Any], Any]],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Expr":
        """
        Apply a custom function in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        ## Context

        * Select/Project
            expected type `f`: Callable[[Any], Any]
            Applies a python function over each individual value in the column.
        * GroupBy
            expected type `f`: Callable[[Series], Series]
            Applies a python function over each group.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.

        Examples
        --------

        >>> df = pl.DataFrame({"a": [1,  2,  1,  1],
                   "b": ["a", "b", "c", "c"]})
        >>> (df
         .lazy()
         .groupby("b")
         .agg([col("a").apply(lambda x: x.sum())])
         .collect()
        )
        shape: (3, 2)
        ╭─────┬─────╮
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ c   ┆ 2   │
        ╰─────┴─────╯

        """

        # input x: Series of type list containing the group values
        def wrap_f(x: "pl.Series") -> "pl.Series":
            return x.apply(f, return_dtype=return_dtype)

        return self.map(wrap_f)

    def explode(self) -> "Expr":
        """
        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype
        """
        return wrap_expr(self._pyexpr.explode())

    def take_every(self, n: int) -> "Expr":
        """
        Take every nth value in the Series and return as a new Series.
        """
        return wrap_expr(self._pyexpr.take_every(n))

    def head(self, n: Optional[int] = None) -> "Expr":
        """
        Take the first n values.
        """
        return wrap_expr(self._pyexpr.head(n))

    def tail(self, n: Optional[int] = None) -> "Expr":
        """
        Take the last n values.
        """
        return wrap_expr(self._pyexpr.tail(n))

    def pow(self, exponent: float) -> "Expr":
        """
        Raise expression to the power of exponent.
        """
        return wrap_expr(self._pyexpr.pow(exponent))

    def is_in(self, other: Union["Expr", tp.List[Any]]) -> "Expr":
        """
        Check if elements of this Series are in the right Series, or List values of the right Series.

        Parameters
        ----------
        other
            Series of primitive type or List type.

        Returns
        -------
        Expr that evaluates to a Boolean Series.
        """
        if isinstance(other, list):
            other = lit(pl.Series("", other))
        return wrap_expr(self._pyexpr.is_in(other._pyexpr))

    def repeat_by(self, by: "Expr") -> "Expr":
        """
        Repeat the elements in this Series `n` times by dictated by the number given by `by`.
        The elements are expanded into a `List`

        Parameters
        ----------
        by
            Numeric column that determines how often the values will be repeated.
            The column will be coerced to UInt32. Give this dtype to make the coercion a no-op.

        Returns
        -------
        Series of type List
        """
        by = expr_to_lit_or_expr(by, False)
        return wrap_expr(self._pyexpr.repeat_by(by._pyexpr))

    def is_between(
        self, start: Union["Expr", datetime], end: Union["Expr", datetime]
    ) -> "Expr":
        """
        Check if this expression is between start and end.
        """
        cast_to_date64 = False
        if isinstance(start, datetime):
            start = lit(start)
            cast_to_date64 = True
        if isinstance(end, datetime):
            end = lit(end)
            cast_to_date64 = True
        if cast_to_date64:
            expr = self.cast(Date64)
        else:
            expr = self
        return ((expr > start) & (expr < end)).alias("is_between")

    @property
    def dt(self) -> "ExprDateTimeNameSpace":
        """
        Create an object namespace of all datetime related methods.
        """
        return ExprDateTimeNameSpace(self)

    @property
    def str(self) -> "ExprStringNameSpace":
        """
        Create an object namespace of all string related methods.
        """
        return ExprStringNameSpace(self)

    def hash(self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3) -> "pl.Expr":
        """
        Hash the Series.

        The hash value is of type `Date64`

        Parameters
        ----------
        k0
            seed parameter
        k1
            seed parameter
        k2
            seed parameter
        k3
            seed parameter
        """
        return wrap_expr(self._pyexpr.hash(k0, k1, k2, k3))

    def reinterpret(self, signed: bool) -> "pl.Expr":
        """
        Reinterpret the underlying bits as a signed/unsigned integer.
        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            True -> pl.Int64
            False -> pl.UInt64
        """
        return wrap_expr(self._pyexpr.reinterpret(signed))


class ExprStringNameSpace:
    """
    Namespace for string related expressions
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strptime(
        self,
        datatype: Union[Date32, Date64],
        fmt: Optional[str] = None,
    ) -> Expr:
        """
        Parse utf8 expression as a Date32/Date64 type.

        Parameters
        ----------
        datatype
            Date32 | Date64.
        fmt
            "yyyy-mm-dd".
        """
        if datatype == Date32:
            return wrap_expr(self._pyexpr.str_parse_date32(fmt))
        elif datatype == Date64:
            return wrap_expr(self._pyexpr.str_parse_date64(fmt))
        else:
            raise NotImplementedError

    def parse_date(
        self,
        datatype: Union[Date32, Date64],
        fmt: Optional[str] = None,
    ) -> Expr:
        """
        .. deprecated:: 0.8.7
        use `strptime`
        """
        return self.strptime(datatype, fmt)

    def lengths(self) -> Expr:
        """
        Get the length of the Strings as UInt32.
        """
        return wrap_expr(self._pyexpr.str_lengths())

    def to_uppercase(self) -> Expr:
        """
        Transform to uppercase variant.
        """
        return wrap_expr(self._pyexpr.str_to_uppercase())

    def to_lowercase(self) -> Expr:
        """
        Transform to lowercase variant.
        """
        return wrap_expr(self._pyexpr.str_to_lowercase())

    def contains(self, pattern: str) -> Expr:
        """
        Check if string contains regex.

        Parameters
        ----------
        pattern
            Regex pattern.
        """
        return wrap_expr(self._pyexpr.str_contains(pattern))

    def json_path_match(self, json_path: str) -> Expr:
        """
        Extract the first match of json string with provided JSONPath expression.
        Throw errors if encounter invalid json strings.
        All return value will be casted to Utf8 regardless of the original value.
        Documentation on JSONPath standard: https://goessner.net/articles/JsonPath/

        Parameters
        ----------
        json_path
            A valid JSON path query string

        Returns
        -------
        Utf8 array. Contain null if original value is null or the json_path return nothing.

        Examples
        --------

        >>> df = pl.DataFrame({
        'json_val' = ['{"a":"1"}',None,'{"a":2}', '{"a":2.1}', '{"a":true}']
        })
        >>> df.select(pl.col('json_val').str.json_path_match('$.a')
        shape: (5,)
        Series: 'json_val' [str]
        [
            "1"
            null
            "2"
            "2.1"
            "true"
        ]
        """
        return wrap_expr(self._pyexpr.str_json_path_match(json_path))

    def replace(self, pattern: str, value: str) -> Expr:
        """
        Replace first regex match with a string value.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        """
        return wrap_expr(self._pyexpr.str_replace(pattern, value))

    def replace_all(self, pattern: str, value: str) -> Expr:
        """
        Replace substring on all regex pattern matches.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        """
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value))

    def slice(self, start: int, length: Optional[int] = None) -> Expr:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        start
            Start of the slice (negative indexing may be used).
        length
            Optional length of the slice.

        Returns
        -------
        Series of Utf8 type
        """
        return wrap_expr(self._pyexpr.str_slice(start, length))


class ExprDateTimeNameSpace:
    """
    Namespace for datetime related expressions.
    """

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def strftime(self, fmt: str) -> Expr:
        """
        Format date32/date64 with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
        """
        return wrap_expr(self._pyexpr.strftime(fmt))

    def year(self) -> Expr:
        """
        Extract year from underlying Date representation.
        Can be performed on Date32 and Date64.

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32
        """
        return wrap_expr(self._pyexpr.year())

    def month(self) -> Expr:
        """
        Extract month from underlying Date representation.
        Can be performed on Date32 and Date64.

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32
        """
        return wrap_expr(self._pyexpr.month())

    def day(self) -> Expr:
        """
        Extract day from underlying Date representation.
        Can be performed on Date32 and Date64.

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.day())

    def ordinal_day(self) -> Expr:
        """
        Extract ordinal day from underlying Date representation.
        Can be performed on Date32 and Date64.

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.ordinal_day())

    def hour(self) -> Expr:
        """
        Extract hour from underlying DateTime representation.
        Can be performed on Date64.

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32
        """
        return wrap_expr(self._pyexpr.hour())

    def minute(self) -> Expr:
        """
        Extract minutes from underlying DateTime representation.
        Can be performed on Date64.

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32
        """
        return wrap_expr(self._pyexpr.minute())

    def second(self) -> Expr:
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64.

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32
        """
        return wrap_expr(self._pyexpr.second())

    def nanosecond(self) -> Expr:
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64.

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32
        """
        return wrap_expr(self._pyexpr.nanosecond())

    def round(self, rule: str, n: int) -> Expr:
        """
        Round the datetime.

        Parameters
        ----------
        rule
            Units of the downscaling operation.

            Any of:
                - "month"
                - "week"
                - "day"
                - "hour"
                - "minute"
                - "second"

        n
            Number of units (e.g. 5 "day", 15 "minute".
        """
        return wrap_expr(self._pyexpr).map(lambda s: s.dt.round(rule, n), None)


def expr_to_lit_or_expr(
    expr: Union[Expr, int, float, str, tp.List[Expr]], str_to_lit: bool = True
) -> Expr:
    """
    Helper function that converts args to expressions.

    Parameters
    ----------
    expr
        Any argument.
    str_to_lit
        If True string argument `"foo"` will be converted to `lit("foo")`,
        If False it will be converted to `col("foo")`

    Returns
    -------

    """
    if isinstance(expr, str) and not str_to_lit:
        return col(expr)
    elif isinstance(expr, (int, float, str)):
        return lit(expr)
    elif isinstance(expr, list):
        return [expr_to_lit_or_expr(e, str_to_lit=str_to_lit) for e in expr]  # type: ignore[return-value]
    else:
        return expr

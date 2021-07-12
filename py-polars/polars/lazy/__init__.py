"""
This module contains all expressions and classes needed for lazy computation/ query execution.
"""
import os
import shutil
import subprocess
import tempfile
import typing as tp
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ..datatypes import (
    Boolean,
    DataType,
    Date32,
    Date64,
    Float64,
    Int64,
    Utf8,
    pytype_to_polars_type,
)
from ..frame import DataFrame, wrap_df
from ..series import Series
from ..utils import _process_null_values

try:
    from ..polars import PyExpr, PyLazyFrame, PyLazyGroupBy
    from ..polars import argsort_by as pyargsort_by
    from ..polars import binary_function as pybinary_function
    from ..polars import col as pycol
    from ..polars import cov as pycov
    from ..polars import except_ as pyexcept
    from ..polars import lit as pylit
    from ..polars import pearson_corr as pypearson_corr
    from ..polars import series_from_range as _series_from_range
    from ..polars import when as pywhen
except ImportError:
    import warnings

    warnings.warn("Binary files missing.")
    __pdoc__ = {"wrap_ldf": False, "wrap_expr": False}


def _selection_to_pyexpr_list(
    exprs: Union[str, "Expr", Sequence[str], Sequence["Expr"]]
) -> tp.List[PyExpr]:
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


def wrap_ldf(ldf: PyLazyFrame) -> "LazyFrame":
    return LazyFrame._from_pyldf(ldf)


class LazyGroupBy:
    """
    Created by `df.lazy().groupby("foo)"`
    """

    def __init__(self, lgb: PyLazyGroupBy):
        self.lgb = lgb

    def agg(self, aggs: Union[tp.List["Expr"], "Expr"]) -> "LazyFrame":
        """
        Describe the aggregation that need to be done on a group.

        Parameters
        ----------
        aggs
            Single/ Multiple aggregation expression(s).

        # Example

        ```python
        (pl.scan_csv("data.csv")
            .groupby("groups")
            .agg([
                    pl.col("name").n_unique().alias("unique_names"),
                    pl.max("values")
                ])
        )
        ```
        """
        aggs = _selection_to_pyexpr_list(aggs)
        return wrap_ldf(self.lgb.agg(aggs))

    def apply(self, f: Callable[[DataFrame], DataFrame]) -> "LazyFrame":
        """
        Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
        this as materializing the `DataFrame` is quite expensive.

        Parameters
        ----------
        f
            Function to apply over the `DataFrame`.
        """
        return wrap_ldf(self.lgb.apply(f))


class LazyFrame:
    """
    Representation of a Lazy computation graph/ query.
    """

    def __init__(self) -> None:
        self._ldf: PyLazyFrame

    @staticmethod
    def _from_pyldf(ldf: PyLazyFrame) -> "LazyFrame":
        self = LazyFrame.__new__(LazyFrame)
        self._ldf = ldf
        return self

    @staticmethod
    def scan_csv(
        file: str,
        has_headers: bool = True,
        ignore_errors: bool = False,
        sep: str = ",",
        skip_rows: int = 0,
        stop_after_n_rows: Optional[int] = None,
        cache: bool = True,
        dtype: Optional[Dict[str, Type[DataType]]] = None,
        low_memory: bool = False,
        comment_char: Optional[str] = None,
        null_values: Optional[Union[str, tp.List[str], Dict[str, str]]] = None,
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_csv`
        """
        dtype_list: Optional[tp.List[Tuple[str, Type[DataType]]]] = None
        if dtype is not None:
            dtype_list = []
            for k, v in dtype.items():
                dtype_list.append((k, pytype_to_polars_type(v)))
        processed_null_values = _process_null_values(null_values)

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_csv(
            file,
            sep,
            has_headers,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            cache,
            dtype_list,
            low_memory,
            comment_char,
            processed_null_values,
        )
        return self

    @staticmethod
    def scan_parquet(
        file: str, stop_after_n_rows: Optional[int] = None, cache: bool = True
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_parquet`
        """

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_parquet(file, stop_after_n_rows, cache)
        return self

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Apply a function on Self.

        Parameters
        ----------
        func
            Callable.
        args
            Arguments.
        kwargs
            Keyword arguments.
        """
        return func(self, *args, **kwargs)

    def describe_plan(self) -> str:
        """
        A string representation of the unoptimized query plan.
        """
        return self._ldf.describe_plan()

    def describe_optimized_plan(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
    ) -> str:
        """
        A string representation of the optimized query plan.
        """

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache=False,
        )

        return ldf.describe_optimized_plan()

    def show_graph(
        self,
        optimized: bool = True,
        show: bool = True,
        output_path: Optional[str] = None,
        raw_output: bool = False,
        figsize: Tuple[float, float] = (16.0, 12.0),
    ) -> Optional[str]:
        """
        Show a plot of the query plan. Note that you should have graphviz installed.

        Parameters
        ----------
        optimized
            Optimize the query plan.
        show
            Show the figure.
        output_path
            Write the figure to disk.
        raw_output
            Return dot syntax.
        figsize
            Passed to matlotlib if `show` == True.
        """
        try:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Graphviz dot binary should be on your PATH and matplotlib should be installed to show graph."
            )
        dot = self._ldf.to_dot(optimized)
        if raw_output:
            return dot
        with tempfile.TemporaryDirectory() as tmpdir_name:
            dot_path = os.path.join(tmpdir_name, "dot")
            with open(dot_path, "w") as f:
                f.write(dot)

            subprocess.run(["dot", "-Nshape=box", "-Tpng", "-O", dot_path])
            out_path = os.path.join(tmpdir_name, "dot.png")

            if output_path is not None:
                shutil.copy(out_path, output_path)

            if show:
                plt.figure(figsize=figsize)
                img = mpimg.imread(out_path)
                plt.imshow(img)
                plt.show()
        return None

    def sort(
        self,
        by_columns: Union[str, "Expr", tp.List["Expr"]],
        reverse: Union[bool, tp.List[bool]] = False,
    ) -> "LazyFrame":
        """
        Sort the DataFrame by:

            - A single column name
            - An expression
            - Multiple expressions

        Parameters
        ----------
        by_columns
            Column (expressions) to sort by.
        reverse
            Whether or not to sort in reverse order.
        """
        if type(by_columns) is str:
            return wrap_ldf(self._ldf.sort(by_columns, reverse))
        if type(reverse) is bool:
            reverse = [reverse]

        by_columns = expr_to_lit_or_expr(by_columns, str_to_lit=False)
        by_columns = _selection_to_pyexpr_list(by_columns)
        return wrap_ldf(self._ldf.sort_by_exprs(by_columns, reverse))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Collect into a DataFrame.

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
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
        )
        return wrap_df(ldf.collect())

    def fetch(
        self,
        n_rows: int = 500,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = True,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Fetch is like a collect operation, but it overwrites the number of rows read by every scan
        operation. This is a utility that helps debug a query on a smaller number of rows.

        Note that the fetch does not guarantee the final number of rows in the DataFrame.
        Filter, join operations and a lower number of rows available in the scanned file influence
        the final number of rows.

        Parameters
        ----------
        n_rows
            Collect n_rows from the data sources.
        type_coercion
            Run type coercion optimization.
        predicate_pushdown
            Run predicate pushdown optimization.
        projection_pushdown
            Run projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        string_cache
            Use a global string cache in this query.
            This is needed if you want to join on categorical columns.
        no_optimization
            Turn off optimizations.

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
        )
        return wrap_df(ldf.fetch(n_rows))

    def cache(
        self,
    ) -> "LazyFrame":
        """
        Cache the result once the execution of the physical plan hits this node.
        """
        return wrap_ldf(self._ldf.cache())

    def filter(self, predicate: "Expr") -> "LazyFrame":
        """
        Filter the rows in the DataFrame based on a predicate expression.

        Parameters
        ----------
        predicate
            Expression that evaluates to a boolean Series.
        """
        if isinstance(predicate, str):
            predicate = col(predicate)
        return wrap_ldf(self._ldf.filter(predicate._pyexpr))

    def select(
        self, exprs: Union[str, "Expr", Sequence[str], Sequence["Expr"]]
    ) -> "LazyFrame":
        """
        Select columns from this DataFrame.

        Parameters
        ----------
        exprs
            Column or columns to select.
        """
        exprs = _selection_to_pyexpr_list(exprs)
        return wrap_ldf(self._ldf.select(exprs))

    def groupby(
        self, by: Union[str, tp.List[str], "Expr", tp.List["Expr"]]
    ) -> LazyGroupBy:
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        """
        new_by: tp.List[PyExpr]
        if isinstance(by, list):
            new_by = []
            for e in by:
                if isinstance(e, str):
                    e = col(e)
                new_by.append(e._pyexpr)
        elif isinstance(by, str):
            new_by = [col(by)._pyexpr]
        elif isinstance(by, Expr):
            new_by = [by._pyexpr]
        lgb = self._ldf.groupby(new_by)
        return LazyGroupBy(lgb)

    def join(
        self,
        ldf: "LazyFrame",
        left_on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        right_on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        on: Optional[Union[str, "Expr", tp.List[str], tp.List["Expr"]]] = None,
        how: str = "inner",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> "LazyFrame":
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        ldf
            Lazy DataFrame to join with.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be None.
        how
            one of:
                "inner"
                "left"
                "outer"
                "asof",
                "cross"

        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to the join in parallel.

        # Asof joins
        This is similar to a left-join except that we match on nearest key rather than equal keys.
        The keys must be sorted to perform an asof join

        """
        if how == "cross":
            return wrap_ldf(
                self._ldf.join(ldf._ldf, [], [], allow_parallel, force_parallel, how)
            )

        left_on_: Union[tp.List[str], tp.List[Expr], None]
        if isinstance(left_on, (str, Expr)):
            left_on_ = [left_on]  # type: ignore[assignment]
        else:
            left_on_ = left_on

        right_on_: Union[tp.List[str], tp.List[Expr], None]
        if isinstance(right_on, (str, Expr)):
            right_on_ = [right_on]  # type: ignore[assignment]
        else:
            right_on_ = right_on

        if isinstance(on, str):
            left_on_ = [on]
            right_on_ = [on]
        elif isinstance(on, list):
            left_on_ = on
            right_on_ = on

        if left_on_ is None or right_on_ is None:
            raise ValueError("You should pass the column to join on as an argument.")

        new_left_on = []
        for column in left_on_:
            if isinstance(column, str):
                column = col(column)
            new_left_on.append(column._pyexpr)
        new_right_on = []
        for column in right_on_:
            if isinstance(column, str):
                column = col(column)
            new_right_on.append(column._pyexpr)

        return wrap_ldf(
            self._ldf.join(
                ldf._ldf, new_left_on, new_right_on, allow_parallel, force_parallel, how
            )
        )

    def with_columns(self, exprs: Union[tp.List["Expr"], "Expr"]) -> "LazyFrame":
        """
        Add or overwrite multiple columns in a DataFrame.

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns.
        """
        if isinstance(exprs, Expr):
            return self.with_column(exprs)

        pyexprs = []

        for e in exprs:
            if isinstance(e, Expr):
                pyexprs.append(e._pyexpr)
            elif isinstance(e, Series):
                pyexprs.append(lit(e)._pyexpr)

        return wrap_ldf(self._ldf.with_columns(pyexprs))

    def with_column(self, expr: "Expr") -> "LazyFrame":
        """
        Add or overwrite column in a DataFrame.

        Parameters
        ----------
        expr
            Expression that evaluates to column.
        """
        return self.with_columns([expr])

    def drop_columns(self, columns: tp.List[str]) -> "LazyFrame":
        """
        Remove multiple columns from a DataFrame.

        Parameters
        ----------
        columns
            List of column names.
        """
        return wrap_ldf(self._ldf.drop_columns(columns))

    def drop_column(self, column: str) -> "LazyFrame":
        """
        Remove a column from the DataFrame.

        Parameters
        ----------
        column
            Name of the column that should be removed.
        """
        return self.drop_columns([column])

    def with_column_renamed(self, existing_name: str, new_name: str) -> "LazyFrame":
        """
        Rename a column in the DataFrame
        """
        return wrap_ldf(self._ldf.with_column_renamed(existing_name, new_name))

    def reverse(self) -> "LazyFrame":
        """
        Reverse the DataFrame.
        """
        return wrap_ldf(self._ldf.reverse())

    def shift(self, periods: int) -> "LazyFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_ldf(self._ldf.shift(periods))

    def shift_and_fill(
        self, periods: int, fill_value: Union["Expr", int, str, float]
    ) -> "LazyFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with the result of this expression.
        """
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.shift_and_fill(periods, fill_value._pyexpr))

    def slice(self, offset: int, length: int) -> "LazyFrame":
        """
        Slice the DataFrame.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.
        """
        return wrap_ldf(self._ldf.slice(offset, length))

    def limit(self, n: int) -> "LazyFrame":
        """
        Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.slice(0, n)

    def head(self, n: int) -> "LazyFrame":
        """
        Get the first `n` rows of the DataFrame
        Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.limit(n)

    def tail(self, n: int) -> "LazyFrame":
        """
        Get the last `n` rows of the DataFrame.

        Parameters
        ----------
        n
            Number of rows.
        """
        return wrap_ldf(self._ldf.tail(n))

    def last(self) -> "LazyFrame":
        """
        Get the last row of the DataFrame.
        """
        return self.tail(1)

    def first(self) -> "LazyFrame":
        """
        Get the first row of the DataFrame.
        """
        return self.slice(0, 1)

    def fill_none(self, fill_value: Union[int, str, "Expr"]) -> "LazyFrame":
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.fill_none(fill_value._pyexpr))

    def std(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their standard deviation value.
        """
        return wrap_ldf(self._ldf.std())

    def var(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their variance value.
        """
        return wrap_ldf(self._ldf.var())

    def max(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their maximum value.
        """
        return wrap_ldf(self._ldf.max())

    def min(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their minimum value.
        """
        return wrap_ldf(self._ldf.min())

    def sum(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their sum value.
        """
        return wrap_ldf(self._ldf.sum())

    def mean(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their mean value.
        """
        return wrap_ldf(self._ldf.mean())

    def median(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their median value.
        """
        return wrap_ldf(self._ldf.median())

    def quantile(self, quantile: float) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their quantile value.
        """
        return wrap_ldf(self._ldf.quantile(quantile))

    def explode(self, columns: Union[str, tp.List[str]]) -> "LazyFrame":
        """
        Explode lists to long format.
        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_ldf(self._ldf.explode(columns))

    def drop_duplicates(
        self,
        maintain_order: bool = False,
        subset: Optional[Union[tp.List[str], str]] = None,
    ) -> "LazyFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_duplicates(maintain_order, subset))

    def drop_nulls(
        self, subset: Optional[Union[tp.List[str], str]] = None
    ) -> "LazyFrame":
        """
        Drop rows with null values from this DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_nulls(subset))

    def melt(
        self, id_vars: Union[str, tp.List[str]], value_vars: Union[str, tp.List[str]]
    ) -> "LazyFrame":
        """
        Unpivot DataFrame to long format.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables.
        value_vars
            Values to use as identifier variables.
        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        return wrap_ldf(self._ldf.melt(id_vars, value_vars))

    def map(
        self,
        f: Union["UDF", Callable[[DataFrame], DataFrame]],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        no_optimizations: bool = False,
    ) -> "LazyFrame":
        """
        Apply a custom UDF. It is important that the UDF returns a Polars DataFrame.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        predicate_pushdown
            Allow predicate pushdown optimization to pass this node.
        projection_pushdown
            Allow projection pushdown optimization to pass this node.
        no_optimizations
            Turn off all optimizations past this point.
        """
        if not no_optimizations:
            predicate_pushdown = False
            projection_pushdown = False
        return wrap_ldf(self._ldf.map(f, predicate_pushdown, projection_pushdown))


def wrap_expr(pyexpr: PyExpr) -> "Expr":
    return Expr._from_pyexpr(pyexpr)


class Expr:
    """
    Expressions that can be used in various contexts.
    """

    def __init__(self) -> None:
        self._pyexpr: PyExpr

    @staticmethod
    def _from_pyexpr(pyexpr: PyExpr) -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other: Any) -> PyExpr:
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
        """
        return wrap_expr(self._pyexpr.alias(name))

    def is_not(self) -> "Expr":
        """
        Negate a boolean expression.
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

        ``` python
        df = DataFrame({
            "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 8]
        })
        print(df.lazy()
            .select([
                col("groups")
                sum("values").over("groups"))
            ]).collect())

        ```

        outputs:

        ``` text
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

        ```
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
        f: Union["UDF", Callable[[Series], Series]],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Expr":
        """
        Apply a custom UDF. It is important that the UDF returns a Polars Series.

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
        f: Callable[[Series], Series],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "Expr":
        """
        Apply a custom UDF in a GroupBy context. This is syntactic sugar for the `apply` method which operates on all
        groups at once. The UDF passed to this expression will operate on a single group.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.

        # Example

        ```python
        df = pl.DataFrame({"a": [1,  2,  1,  1],
                   "b": ["a", "b", "c", "c"]})

        (df
         .lazy()
         .groupby("b")
         .agg([col("a").apply(lambda x: x.sum())])
         .collect()
        )
        ```

        > returns

        ```text
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
        ```
        """

        # input x: Series of type list containing the group values
        def wrap_f(x: "Series") -> "Series":
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
            other = lit(Series("", other))
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


class WhenThenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthenthen: Any):
        self.pywenthenthen = pywhenthenthen

    def when(self, predicate: Expr) -> "WhenThenThen":
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self.pywenthenthen.when(predicate._pyexpr))

    def then(self, expr: Union[Expr, int, float, str]) -> "WhenThenThen":
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr_ = expr_to_lit_or_expr(expr)
        return WhenThenThen(self.pywenthenthen.then(expr_._pyexpr))

    def otherwise(self, expr: Union[Expr, int, float, str]) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self.pywenthenthen.otherwise(expr._pyexpr))


class WhenThen:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhenthen: Any):
        self._pywhenthen = pywhenthen

    def when(self, predicate: Expr) -> WhenThenThen:
        """
        Start another when, then, otherwise layer.
        """
        return WhenThenThen(self._pywhenthen.when(predicate._pyexpr))

    def otherwise(self, expr: Union[Expr, int, float, str]) -> Expr:
        """
        Values to return in case of the predicate being `False`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))


class When:
    """
    Utility class. See the `when` function.
    """

    def __init__(self, pywhen: pywhen):
        self._pywhen = pywhen

    def then(self, expr: Union[Expr, int, float, str]) -> WhenThen:
        """
        Values to return in case of the predicate being `True`.

        See Also: the `when` function.
        """
        expr = expr_to_lit_or_expr(expr)
        pywhenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(pywhenthen)


def when(expr: Expr) -> When:
    """
    Start a when, then, otherwise expression.

    # Example

    Below we add a column with the value 1, where column "foo" > 2 and the value -1 where it isn't.

    ```python
    lf.with_column(
        when(col("foo") > 2)
        .then(lit(1))
        .otherwise(lit(-1))
    )
    ```

    Or with multiple `when, thens` chained:


    ```python
    lf.with_column(
        when(col("foo") > 2).then(1)
        when(col("bar") > 2).then(4)
        .otherwise(-1)
    )
    ```
    """
    expr = expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)


def col(name: str) -> Expr:
    """
    A column in a DataFrame.
    """
    return wrap_expr(pycol(name))


def except_(name: str) -> Expr:
    """
    Exclude a column from a selection.

    # Example
    ```python
    df = pl.DataFrame({
        "ham": [1, 1, 2, 2, 3],
        "foo": [1, 1, 2, 2, 3],
        "bar": [1, 1, 2, 2, 3],
    })

    df.lazy()
        .select(["*", except_("foo")])
        .collect()
    ```
    Outputs:

    ```text
    ╭─────┬─────╮
    │ ham ┆ bar │
    │ --- ┆ --- │
    │ f64 ┆ f64 │
    ╞═════╪═════╡
    │ 1   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 1   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 2   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 3   │
    ╰─────┴─────╯
    ```
    """
    return wrap_expr(pyexcept(name))


def count(column: Union[str, Series] = "") -> Union[Expr, int]:
    """
    Count the number of values in this column.
    """
    if isinstance(column, Series):
        return column.len()
    return col(column).count()


def to_list(name: str) -> Expr:
    """
    Aggregate to list.
    """
    return col(name).list()


def std(column: Union[str, Series]) -> Union[Expr, float]:
    """
    Get the standard deviation.
    """
    if isinstance(column, Series):
        return column.std()
    return col(column).std()


def var(column: Union[str, Series]) -> Union[Expr, float]:
    """
    Get the variance.
    """
    if isinstance(column, Series):
        return column.var()
    return col(column).var()


def max(column: Union[str, tp.List[Expr], Series]) -> Union[Expr, Any]:
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
    if isinstance(column, Series):
        return column.max()
    elif isinstance(column, list):

        def max_(acc: Series, val: Series) -> Series:
            mask = acc < val
            return acc.zip_with(mask, val)

        return fold(lit(0), max_, column).alias("max")
    else:
        return col(column).max()


def min(column: Union[str, tp.List[Expr], Series]) -> Union[Expr, Any]:
    """
    Get the minimum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, Series):
        return column.min()
    elif isinstance(column, list):

        def min_(acc: Series, val: Series) -> Series:
            mask = acc > val
            return acc.zip_with(mask, val)

        return fold(lit(0), min_, column).alias("min")
    else:
        return col(column).min()


def sum(column: Union[str, tp.List[Expr], Series]) -> Union[Expr, Any]:
    """
    Get the sum value.

    column
        Column(s) to be used in aggregation. Will lead to different behavior based on the input.
        input:
            - Union[str, Series] -> aggregate the sum value of that column.
            - tp.List[Expr] -> aggregate the sum value horizontally.
    """
    if isinstance(column, Series):
        return column.sum()
    elif isinstance(column, list):
        return fold(lit(0), lambda a, b: a + b, column).alias("sum")
    else:
        return col(column).sum()


def mean(column: Union[str, Series]) -> Union[Expr, float]:
    """
    Get the mean value.
    """
    if isinstance(column, Series):
        return column.mean()
    return col(column).mean()


def avg(column: Union[str, Series]) -> Union[Expr, float]:
    """
    Alias for mean.
    """
    return mean(column)


def median(column: Union[str, Series]) -> Union[Expr, float, int]:
    """
    Get the median value.
    """
    if isinstance(column, Series):
        return column.median()
    return col(column).median()


def n_unique(column: Union[str, Series]) -> Union[Expr, int]:
    """Count unique values."""
    if isinstance(column, Series):
        return column.n_unique()
    return col(column).n_unique()


def first(column: Union[str, Series]) -> Union[Expr, Any]:
    """
    Get the first value.
    """
    if isinstance(column, Series):
        if column.len() > 0:
            return column[0]
        else:
            raise IndexError("The series is empty, so no first value can be returned.")
    return col(column).first()


def last(column: Union[str, Series]) -> Expr:
    """
    Get the last value.
    """
    if isinstance(column, Series):
        if column.len() > 0:
            return column[-1]
        else:
            raise IndexError("The series is empty, so no last value can be returned,")
    return col(column).last()


def head(column: Union[str, Series], n: Optional[int] = None) -> Union[Expr, Series]:
    """
    Get the first n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, Series):
        return column.head(n)
    return col(column).head(n)


def tail(column: Union[str, Series], n: Optional[int] = None) -> Union[Expr, Series]:
    """
    Get the last n rows of an Expression.

    Parameters
    ----------
    column
        Column name or Series.
    n
        Number of rows to take.
    """
    if isinstance(column, Series):
        return column.tail(n)
    return col(column).tail(n)


def lit_date(dt: datetime) -> Expr:
    """
    Converts a Python DateTime to a literal Expression.

    Parameters
    ----------
    dt
        datetime.datetime
    """
    return lit(int(dt.timestamp() * 1e3))


def lit(
    value: Optional[Union[float, int, str, datetime, Series]],
    dtype: Optional[Type[DataType]] = None,
) -> Expr:
    """
    A literal value.

    Parameters
    ----------
    value
        Value that should be used as a `literal`.
    dtype
        Optionally define a dtype.

    # Example

    ```python
    # literal integer
    lit(1)

    # literal str.
    lit("foo")

    # literal date64
    lit(datetime(2021, 1, 20))

    # literal Null
    lit(None)

    # literal eager Series
    lit(Series("a", [1, 2, 3])
    ```
    """
    if isinstance(value, datetime):
        return lit(int(value.timestamp() * 1e3)).cast(Date64)

    if isinstance(value, Series):
        name = value.name
        value = value._s
        return wrap_expr(pylit(value)).alias(name)

    if isinstance(value, np.ndarray):
        return lit(Series("", value))

    if dtype:
        return wrap_expr(pylit(value)).cast(dtype)
    return wrap_expr(pylit(value))


def pearson_corr(
    a: Union[str, Expr],
    b: Union[str, Expr],
) -> Expr:
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
    return wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr))


def cov(
    a: Union[str, Expr],
    b: Union[str, Expr],
) -> Expr:
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
    return wrap_expr(pycov(a._pyexpr, b._pyexpr))


def map_binary(
    a: Union[str, Expr],
    b: Union[str, Expr],
    f: Callable[[Series, Series], Series],
    return_dtype: Optional[Type[DataType]] = None,
) -> Expr:
    """
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
    return wrap_expr(pybinary_function(a._pyexpr, b._pyexpr, f, return_dtype))


def fold(
    acc: Expr, f: Callable[[Series, Series], Series], exprs: tp.List[Expr]
) -> Expr:
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
        Expressions to aggregate over.
    """
    for e in exprs:
        acc = map_binary(acc, e, f, None)
    return acc


def any(name: Union[str, tp.List[Expr]]) -> Expr:
    """
    Evaluate columnwise or elementwise with a bitwise OR operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a | b, name).alias("any")
    return col(name).sum() > 0


def all(name: Union[str, tp.List[Expr]]) -> Expr:
    """
    Evaluate columnwise or elementwise with a bitwise AND operation.
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a & b, name).alias("all")
    return col(name).cast(bool).sum() == col(name).count()


def groups(column: str) -> Expr:
    """
    Syntactic sugar for `column("foo").agg_groups()`.
    """
    return col(column).agg_groups()


def quantile(column: str, quantile: float) -> Expr:
    """
    Syntactic sugar for `column("foo").quantile(..)`.
    """
    return col(column).quantile(quantile)


class UDF:
    """
    Deprecated: don't use me
    """

    def __init__(self, f: Callable[[Series], Series], return_dtype: Type[DataType]):
        self.f = f
        self.return_dtype = return_dtype


def udf(f: Callable[[Series], Series], return_dtype: Type[DataType]) -> UDF:
    """
    Deprecated: don't use me
    """
    return UDF(f, return_dtype)


def arange(
    low: Union[int, Expr],
    high: Union[int, Expr],
    step: int = 1,
    dtype: Optional[Type[DataType]] = None,
    eager: bool = False,
) -> Union[Expr, Series]:
    """
    Create a range expression. This can be used in a `select`, `with_column` etc.
    Be sure that the range size is equal to the DataFrame you are collecting.

    # Example

    ```python
    (df.lazy()
        .filter(pl.col("foo") < pl.arange(0, 100))
        .collect())
    ```

    Parameters
    ----------
    low
        Lower bound of range.
    high
        Upper bound of range.
    step
        Step size of the range
    dtype
        DataType of the range. Valid dtypes:
            * Int32
            * Int64
            * UInt32
    eager
        If eager evaluation is `True`, a Series is returned instead of an Expr
    """
    if dtype is None:
        dtype = Int64

    def create_range(s1: Series, s2: Series) -> Series:
        assert s1.len() == 1
        assert s2.len() == 1
        return Series._from_pyseries(_series_from_range(s1[0], s2[0], step, dtype))

    # eager execution can only work if low and high are literals
    if eager:
        if not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError(
                "arguments low and high must be integers in eager execution"
            )
        return Series._from_pyseries(_series_from_range(low, high, step, dtype))

    if isinstance(low, int):
        low = lit(low)
    if isinstance(high, int):
        high = lit(high)

    return map_binary(low, high, create_range, return_dtype=dtype)


def argsort_by(
    exprs: tp.List[Expr], reverse: Union[tp.List[bool], bool] = False
) -> Expr:
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
    exprs = _selection_to_pyexpr_list(exprs)
    return wrap_expr(pyargsort_by(exprs, reverse))

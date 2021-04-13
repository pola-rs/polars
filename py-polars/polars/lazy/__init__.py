from typing import Union, List, Callable, Optional, Dict

from polars import Series
from polars.frame import DataFrame, wrap_df
from polars import datatypes
from polars.datatypes import DataType
import os
import tempfile
import subprocess
import shutil
from datetime import datetime

try:
    from ..polars import (
        PyLazyFrame,
        col as pycol,
        lit as pylit,
        # binary_expr,
        binary_function as pybinary_function,
        pearson_corr as pypearson_corr,
        cov as pycov,
        PyExpr,
        PyLazyGroupBy,
        when as pywhen,
        except_ as pyexcept,
        range as pyrange,
    )
except ImportError:
    import warnings

    warnings.warn("binary files missing")

    __pdoc__ = {"wrap_ldf": False, "wrap_expr": False}


def _selection_to_pyexpr_list(exprs) -> "List[PyExpr]":
    if not isinstance(exprs, List):
        if isinstance(exprs, str):
            exprs = col(exprs)
        exprs = [exprs._pyexpr]
    else:
        new = []
        for expr in exprs:
            if isinstance(expr, str):
                expr = col(expr)
            new.append(expr._pyexpr)
        exprs = new
    return exprs


def wrap_ldf(ldf: "PyLazyFrame") -> "LazyFrame":
    return LazyFrame._from_pyldf(ldf)


class LazyGroupBy:
    def __init__(self, lgb: "PyLazyGroupBy"):
        self.lgb = lgb

    def agg(self, aggs: "Union[List[Expr], Expr]") -> "LazyFrame":
        aggs = _selection_to_pyexpr_list(aggs)
        return wrap_ldf(self.lgb.agg(aggs))

    def apply(self, f: "Callable[[DataFrame], DataFrame]") -> "LazyFrame":
        return wrap_ldf(self.lgb.apply(f))


class LazyFrame:
    def __init__(self):
        self._ldf = None

    @staticmethod
    def _from_pyldf(ldf: "PyLazyFrame") -> "LazyFrame":
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
        stop_after_n_rows: "Optional[int]" = None,
        cache: bool = True,
        dtype: "Optional[Dict[str, DataType]]" = None,
    ):
        if dtype is not None:
            new_dtype = []
            for k, v in dtype.items():
                new_dtype.append((k, datatypes.pytype_to_polars_type(v)))
            dtype = new_dtype

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_csv(
            file,
            sep,
            has_headers,
            ignore_errors,
            skip_rows,
            stop_after_n_rows,
            cache,
            dtype,
        )
        return self

    @staticmethod
    def scan_parquet(
        file: str, stop_after_n_rows: "Optional[int]" = None, cache: bool = True
    ):

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_parquet(file, stop_after_n_rows, cache)
        return self

    def pipe(self, func: Callable, *args, **kwargs):
        """
        Apply a function on Self

        Parameters
        ----------
        func
            Callable
        args
            Arguments
        kwargs
            Keyword arguments
        """
        return func(self, *args, **kwargs)

    def describe_plan(self) -> str:
        return self._ldf.describe_plan()

    def show_graph(
        self,
        optimized: bool = True,
        show: bool = True,
        output_path: "Optional[str]" = None,
        raw_output: bool = False,
    ) -> "Optional[str]":
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
        except ImportError:
            raise ImportError(
                "graphviz dot binary should be on your PATH and matplotlib should be installed to show graph"
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
                img = mpimg.imread(out_path)
                plt.imshow(img)
                plt.show()

    def describe_optimized_plan(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
    ) -> str:

        ldf = self._ldf.optimization_toggle(
            type_coercion, predicate_pushdown, projection_pushdown, simplify_expression
        )

        return ldf.describe_optimized_plan()

    def sort(self, by_column: str, reverse: bool = False) -> "LazyFrame":
        return wrap_ldf(self._ldf.sort(by_column, reverse))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = True,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Collect into a DataFrame

        Parameters
        ----------
        type_coercion
            do type coercion optimization
        predicate_pushdown
            do predicate pushdown optimization
        projection_pushdown
            do projection pushdown optimization
        simplify_expression
            run simplify expressions optimization
        no_optimization
            Turn off optimizations

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion, predicate_pushdown, projection_pushdown, simplify_expression
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
            run type coercion optimization
        predicate_pushdown
            run predicate pushdown optimization
        projection_pushdown
            run projection pushdown optimization
        simplify_expression
            run simplify expressions optimization

        Returns
        -------
        DataFrame
        """
        ldf = self._ldf.optimization_toggle(
            type_coercion, predicate_pushdown, projection_pushdown, simplify_expression
        )
        return wrap_df(ldf.fetch(n_rows))

    def cache(
        self,
    ) -> "LazyFrame":
        """
        Cache the result once Physical plan hits this node.
        """
        return wrap_ldf(self._ldf.cache())

    def filter(self, predicate: "Expr") -> "LazyFrame":
        if isinstance(predicate, str):
            predicate = col(predicate)
        return wrap_ldf(self._ldf.filter(predicate._pyexpr))

    def select(self, exprs: "Union[str, Expr, List[str], List[Expr]]") -> "LazyFrame":
        exprs = _selection_to_pyexpr_list(exprs)
        return wrap_ldf(self._ldf.select(exprs))

    def groupby(self, by: "Union[str, List[str], Expr, List[Expr]]") -> LazyGroupBy:
        """
        Start a groupby operation

        Parameters
        ----------
        by
            Column(s) to group by.
        """
        if isinstance(by, list):
            new_by = []
            for e in by:
                if isinstance(e, str):
                    e = col(e)
                new_by.append(e._pyexpr)
            by = new_by
        elif isinstance(by, str):
            by = [col(by)._pyexpr]
        elif isinstance(by, Expr):
            by = [by._pyexpr]
        lgb = self._ldf.groupby(by)
        return LazyGroupBy(lgb)

    def join(
        self,
        ldf: "LazyFrame",
        left_on: "Union[Optional[Expr], str]" = None,
        right_on: "Union[Optional[Expr], str]" = None,
        on: "Union[Optional[Expr], str]" = None,
        how="inner",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> "LazyFrame":
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        ldf
            Lazy DataFrame to join with
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
        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan evaluate the computation of both DataFrames up to the join in parallel.
        """
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        if isinstance(on, str):
            left_on = [on]
            right_on = [on]
        elif isinstance(on, List):
            left_on = on
            right_on = on
        if left_on is None or right_on is None:
            raise ValueError("you should pass the column to join on as an argument")

        new_left_on = []
        for column in left_on:
            if isinstance(column, str):
                column = col(column)
            new_left_on.append(column._pyexpr)
        new_right_on = []
        for column in right_on:
            if isinstance(column, str):
                column = col(column)
            new_right_on.append(column._pyexpr)

        out = self._ldf.join(
            ldf._ldf, new_left_on, new_right_on, allow_parallel, force_parallel, how
        )

        return wrap_ldf(out)

    def with_columns(self, exprs: "List[Expr]") -> "LazyFrame":
        """
        Add or overwrite multiple columns in a DataFrame

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns
        """
        pyexprs = []

        for e in exprs:
            if isinstance(e, Expr):
                pyexprs.append(e._pyexpr)
            elif isinstance(e, Series):
                pyexprs.append(lit(e)._pyexpr)

        return wrap_ldf(self._ldf.with_columns(pyexprs))

    def with_column(self, expr: "Expr") -> "LazyFrame":
        """
        Add or overwrite column in a DataFrame

        Parameters
        ----------
        expr
            Expression that evaluates to column
        """
        return self.with_columns([expr])

    def drop_columns(self, columns: "List[str]") -> "LazyFrame":
        """
        Remove multiple columns from a DataFrame

        Parameters
        ----------
        columns
            List of column names
        """
        return wrap_ldf(self._ldf.drop_columns(columns))

    def drop_column(self, column: "str") -> "LazyFrame":
        """
        Remove a column from the DataFrame

        Parameters
        ----------
        column
            Name of the column that should be removed
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
        self, periods: int, fill_value: "Union[Expr, int, str, float]"
    ) -> "LazyFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with the result of this expression
        """
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.shift_and_fill(periods, fill_value._pyexpr))

    def slice(self, offset: int, length: int):
        """
        Slice the DataFrame

        Parameters
        ----------
        offset
            Start index
        length
            Length of the slice
        """
        return wrap_ldf(self._ldf.slice(offset, length))

    def limit(self, n: int):
        """
        Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.slice(0, n)

    def tail(self, n: int):
        """
        Get the last `n` rows of the DataFrame

        Parameters
        ----------
        n
            Number of rows.
        """
        return wrap_ldf(self._ldf.tail(n))

    def last(self):
        """
        Get the last row of the DataFrame
        """
        return self.tail(1)

    def first(self):
        """
        Get the first row of the DataFrame
        """
        return self.slice(0, 1)

    def fill_none(self, fill_value: "Union[int, str, Expr]"):
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_ldf(self._ldf.fill_none(fill_value._pyexpr))

    def std(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their standard deviation value
        """
        return wrap_ldf(self._ldf.std())

    def var(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their variance value
        """
        return wrap_ldf(self._ldf.var())

    def max(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their maximum value
        """
        return wrap_ldf(self._ldf.max())

    def min(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their minimum value
        """
        return wrap_ldf(self._ldf.min())

    def sum(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their sum value
        """
        return wrap_ldf(self._ldf.sum())

    def mean(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their mean value
        """
        return wrap_ldf(self._ldf.mean())

    def median(self) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their median value
        """
        return wrap_ldf(self._ldf.median())

    def quantile(self, quantile: float) -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their quantile value
        """
        return wrap_ldf(self._ldf.quantile(quantile))

    def explode(self, columns: "Union[str, List[str]]") -> "LazyFrame":
        """
        Explode lists to long format
        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_ldf(self._ldf.explode(columns))

    def drop_duplicates(
        self,
        maintain_order: bool = False,
        subset: "Optional[Union[List[str], str]]" = None,
    ) -> "LazyFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, List):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_duplicates(maintain_order, subset))

    def drop_nulls(
        self, subset: "Optional[Union[List[str], str]]" = None
    ) -> "LazyFrame":
        """
        Drop rows with null values from this DataFrame.
        """
        if subset is not None and not isinstance(subset, List):
            subset = [subset]
        return wrap_ldf(self._ldf.drop_nulls(subset))

    def melt(
        self, id_vars: "Union[List[str], str]", value_vars: "Union[List[str], str]"
    ) -> "DataFrame":
        """
        Unpivot DataFrame to long format.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables

        value_vars
            Values to use as identifier variables
        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        return wrap_ldf(self._ldf.melt(id_vars, value_vars))

    def map(
        self,
        f: "Union[UDF, Callable[[DataFrame], DataFrame]]",
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
    ) -> "LazyFrame":
        """
        Apply a custom UDF. It is important that the UDF returns a Polars DataFrame.

        Parameters
        ----------
        f
            lambda/ function to apply
        predicate_pushdown
            Allow predicate pushdown optimization to pass this node
        projection_pushdown
            Allow projection pushdown optimization to pass this node
        """
        return wrap_ldf(self._ldf.map(f, predicate_pushdown, projection_pushdown))


def wrap_expr(pyexpr: "PyExpr") -> "Expr":
    return Expr._from_pyexpr(pyexpr)


class Expr:
    def __init__(self):
        self._pyexpr = None

    @staticmethod
    def _from_pyexpr(pyexpr: "PyExpr") -> "Expr":
        self = Expr.__new__(Expr)
        self._pyexpr = pyexpr
        return self

    def __to_pyexpr(self, other):
        if isinstance(other, PyExpr):
            return other
        if isinstance(other, Expr):
            return other._pyexpr
        return lit(other)._pyexpr

    def __to_expr(self, other):
        if isinstance(other, Expr):
            return other
        return lit(other)

    def __invert__(self) -> "Expr":
        return self.is_not()

    def __and__(self, other):
        return wrap_expr(self._pyexpr._and(other._pyexpr))

    def __or__(self, other):
        return wrap_expr(self._pyexpr._or(other._pyexpr))

    def __add__(self, other):
        return wrap_expr(self._pyexpr + self.__to_pyexpr(other))

    def __sub__(self, other):
        return wrap_expr(self._pyexpr - self.__to_pyexpr(other))

    def __mul__(self, other):
        return wrap_expr(self._pyexpr * self.__to_pyexpr(other))

    def __truediv__(self, other):
        return wrap_expr(self._pyexpr / self.__to_pyexpr(other))

    def __pow__(self, power, modulo=None):
        return self.pow(power)

    def __ge__(self, other):
        return self.gt_eq(self.__to_expr(other))

    def __le__(self, other):
        return self.lt_eq(self.__to_expr(other))

    def __eq__(self, other):
        return self.eq(self.__to_expr(other))

    def __ne__(self, other):
        return self.neq(self.__to_expr(other))

    def __lt__(self, other):
        return self.lt(self.__to_expr(other))

    def __gt__(self, other):
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
        return wrap_expr(self._pyexpr.alias(name))

    def is_not(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_not())

    def is_null(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_null())

    def is_not_null(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_not_null())

    def is_finite(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_finite())

    def is_infinite(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_infinite())

    def is_nan(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_nan())

    def is_not_nan(self) -> "Expr":
        return wrap_expr(self._pyexpr.is_not_nan())

    def agg_groups(self) -> "Expr":
        return wrap_expr(self._pyexpr.agg_groups())

    def count(self) -> "Expr":
        return wrap_expr(self._pyexpr.count())

    def slice(self, offset: int, length: int):
        """
        Slice the Series

        Parameters
        ----------
        offset
            Start index
        length
            Length of the slice
        """
        return wrap_expr(self._pyexpr.slice(offset, length))

    def cum_sum(self, reverse: bool):
        """
        Get an array with the cumulative sum computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return wrap_expr(self._pyexpr.cum_sum(reverse))

    def cum_min(self, reverse: bool):
        """
        Get an array with the cumulative min computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return wrap_expr(self._pyexpr.cum_min(reverse))

    def cum_max(self, reverse: bool):
        """
        Get an array with the cumulative max computed at every element

        Parameters
        ----------
        reverse
            reverse the operation
        """
        return wrap_expr(self._pyexpr.cum_max(reverse))

    def cast(self, dtype: "DataType") -> "Expr":
        if dtype == str:
            dtype = datatypes.Utf8
        elif dtype == bool:
            dtype = datatypes.Boolean
        elif dtype == float:
            dtype = datatypes.Float64
        elif dtype == int:
            dtype = datatypes.Int64
        return wrap_expr(self._pyexpr.cast(dtype))

    def sort(self, reverse: bool = False) -> "Expr":
        """
        Sort this column. In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        reverse
            False -> order from small to large
            True -> order from large to small
        """
        return wrap_expr(self._pyexpr.sort(reverse))

    def sort_by(self, by: "Union[Expr, str]", reverse: bool = False) -> "Expr":
        """
        Sort this column by the ordering of another column.
        In projection/ selection context the whole column is sorted.
        If used in a groupby context, the groups are sorted.

        Parameters
        ----------
        by
            The column used for sorting
        reverse
            False -> order from small to large
            True -> order from large to small
        """
        if isinstance(by, str):
            by = col(by)

        return wrap_expr(self._pyexpr.sort_by(by._pyexpr, reverse))

    def shift(self, periods: int) -> "Expr":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`

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
            fill None values with the result of this expression
        """
        return wrap_expr(self._pyexpr.shift_and_fill(periods, fill_value._pyexpr))

    def fill_none(self, fill_value: "Union[str, int, float, Expr]") -> "Expr":
        if not isinstance(fill_value, Expr):
            fill_value = lit(fill_value)
        return wrap_expr(self._pyexpr.fill_none(fill_value))

    def reverse(self) -> "Expr":
        """
        Reverse the selection
        """
        return wrap_expr(self._pyexpr.reverse())

    def std(self) -> "Expr":
        """
        Get standard deviation
        """
        return wrap_expr(self._pyexpr.std())

    def var(self) -> "Expr":
        """
        Get variance
        """
        return wrap_expr(self._pyexpr.var())

    def max(self) -> "Expr":
        """
        Get maximum value
        """
        return wrap_expr(self._pyexpr.max())

    def min(self) -> "Expr":
        """
        Get minimum value
        """
        return wrap_expr(self._pyexpr.min())

    def sum(self) -> "Expr":
        """
        Get sum value
        """
        return wrap_expr(self._pyexpr.sum())

    def mean(self) -> "Expr":
        """
        Get mean value
        """
        return wrap_expr(self._pyexpr.mean())

    def median(self) -> "Expr":
        """
        Get median value
        """
        return wrap_expr(self._pyexpr.median())

    def n_unique(self) -> "Expr":
        """Count unique values"""
        return wrap_expr(self._pyexpr.n_unique())

    def first(self) -> "Expr":
        """
        Get first value
        """
        return wrap_expr(self._pyexpr.first())

    def last(self) -> "Expr":
        """
        Get last value
        """
        return wrap_expr(self._pyexpr.last())

    def list(self) -> "Expr":
        """
        Aggregate to list
        """
        return wrap_expr(self._pyexpr.list())

    def over(self, expr: "Union[str, Expr]") -> "Expr":
        """
        Apply window function over a subgroup.
        This is similar to a groupby + aggregation + self join.
        Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html).Do

        Parameters
        ----------
        expr
            Expression that evaluates to a column of groups

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
        if isinstance(expr, str):
            expr = col(expr)

        return wrap_expr(self._pyexpr.over(expr._pyexpr))

    def is_unique(self) -> "Expr":
        """
        Get mask of unique values
        """
        return wrap_expr(self._pyexpr.is_unique())

    def is_duplicated(self) -> "Expr":
        """
        Get mask of duplicated values
        """
        return wrap_expr(self._pyexpr.is_duplicated())

    def quantile(self, quantile: float) -> "Expr":
        """
        Get quantile value
        """
        return wrap_expr(self._pyexpr.quantile(quantile))

    def str_parse_date(self, datatype: "DataType", fmt: Optional[str] = None) -> "Expr":
        """
        Parse utf8 expression as a Date32/Date64 type.

        Parameters
        ----------
        datatype
            Date32 | Date64
        fmt
            "yyyy-mm-dd"
        """
        if datatype == datatypes.Date32:
            return wrap_expr(self._pyexpr.str_parse_date32(fmt))
        if datatype == datatypes.Date64:
            return wrap_expr(self._pyexpr.str_parse_date64(fmt))
        raise NotImplementedError

    def str_lengths(self) -> "Expr":
        """
        Get the length of the Strings as UInt32
        """
        return wrap_expr(self._pyexpr.str_lengths())

    def str_to_uppercase(self) -> "Expr":
        """
        Transform to uppercase variant
        """
        return wrap_expr(self._pyexpr.str_to_uppercase())

    def str_to_lowercase(self) -> "Expr":
        """
        Transform to lowercase variant
        """
        return wrap_expr(self._pyexpr.str_to_lowercase())

    def str_contains(self, pattern: str) -> "Expr":
        """
        Check if string contains regex.

        Parameters
        ----------
        pattern
            regex pattern
        """
        return wrap_expr(self._pyexpr.str_contains(pattern))

    def str_replace(self, pattern: str, value: str) -> "Expr":
        """
        Replace substring where regex pattern first matches.

        Parameters
        ----------
        pattern
            regex pattern
        value
            replacement string
        """
        return wrap_expr(self._pyexpr.str_replace(pattern, value))

    def str_replace_all(self, pattern: str, value: str) -> "Expr":
        """
        Replace substring on all regex pattern matches.

        Parameters
        ----------
        pattern
            regex pattern
        value
            replacement string
        """
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value))

    def datetime_str_fmt(self, fmt: str) -> "Expr":
        """
        Format date32/date64 with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
        """
        return wrap_expr(self._pyexpr.datetime_fmt_str(fmt))

    def year(self):
        """
        Extract year from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the year number in the calendar date.

        Returns
        -------
        Year as Int32
        """
        return wrap_expr(self._pyexpr.year())

    def month(self):
        """
        Extract month from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the month number starting from 1.
        The return value ranges from 1 to 12.

        Returns
        -------
        Month as UInt32
        """
        return wrap_expr(self._pyexpr.month())

    def day(self):
        """
        Extract day from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the day of month starting from 1.
        The return value ranges from 1 to 31. (The last day of month differs by months.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.day())

    def ordinal_day(self):
        """
        Extract ordinal day from underlying Date representation.
        Can be performed on Date32 and Date64

        Returns the day of year starting from 1.
        The return value ranges from 1 to 366. (The last day of year differs by years.)

        Returns
        -------
        Day as UInt32
        """
        return wrap_expr(self._pyexpr.ordinal_day())

    def hour(self):
        """
        Extract day from underlying DateTime representation.
        Can be performed on Date64

        Returns the hour number from 0 to 23.

        Returns
        -------
        Hour as UInt32
        """
        return wrap_expr(self._pyexpr.hour())

    def minute(self):
        """
        Extract minutes from underlying DateTime representation.
        Can be performed on Date64

        Returns the minute number from 0 to 59.

        Returns
        -------
        Minute as UInt32
        """
        return wrap_expr(self._pyexpr.minute())

    def second(self):
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64

        Returns the second number from 0 to 59.

        Returns
        -------
        Second as UInt32
        """
        return wrap_expr(self._pyexpr.second())

    def nanosecond(self):
        """
        Extract seconds from underlying DateTime representation.
        Can be performed on Date64

        Returns the number of nanoseconds since the whole non-leap second.
        The range from 1,000,000,000 to 1,999,999,999 represents the leap second.

        Returns
        -------
        Nanosecond as UInt32
        """
        return wrap_expr(self._pyexpr.nanosecond())

    def filter(self, predicate: "Expr") -> "Expr":
        """
        Filter a single column
        Should be used in aggregation context. If you want to filter on a DataFrame level, use `LazyFrame.filter`

        Parameters
        ----------
        predicate
            Boolean expression
        """
        return wrap_expr(self._pyexpr.filter(predicate._pyexpr))

    def map(
        self,
        f: "Union[UDF, Callable[[Series], Series]]",
        dtype_out: Optional["DataType"] = None,
    ) -> "Expr":
        """
        Apply a custom UDF. It is important that the UDF returns a Polars Series.

        [read more in the book](https://ritchie46.github.io/polars-book/how_can_i/use_custom_functions.html#lazy)

        Parameters
        ----------
        f
            lambda/ function to apply
        dtype_out
            dtype of the output Series
        """
        if isinstance(f, UDF):
            dtype_out = f.output_type
            f = f.f
        if dtype_out == str:
            dtype_out = datatypes.Utf8
        elif dtype_out == int:
            dtype_out = datatypes.Int64
        elif dtype_out == float:
            dtype_out = datatypes.Float64
        elif dtype_out == bool:
            dtype_out = datatypes.Boolean
        return wrap_expr(self._pyexpr.map(f, dtype_out))

    def apply(
        self,
        f: "Union[UDF, Callable[[Series], Series]]",
        dtype_out: Optional["DataType"] = None,
    ) -> "Expr":
        """
        Apply a custom UDF in a GroupBy context. This is syntactic sugar for the `apply` method which operates on all
        groups at once. The UDF passed to this expression will operate on a single group.

        Parameters
        ----------
        f
            lambda/ function to apply
        dtype_out
            dtype of the output Series

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
            return x.apply(f, dtype_out=dtype_out)

        return self.map(wrap_f)

    def explode(self):
        """
        Explode a list or utf8 Series. This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype
        """
        return wrap_expr(self._pyexpr.explode())

    def take_every(self, n: int) -> "Expr":
        """
        Take every nth value in the Series and return as a new Series
        """
        return wrap_expr(self._pyexpr.take_every(n))

    def head(self, n: "Optional[int]" = None):
        """
        Take the first n values
        """
        return wrap_expr(self._pyexpr.head(n))

    def tail(self, n: "Optional[int]" = None):
        """
        Take the last n values
        """
        return wrap_expr(self._pyexpr.tail(n))

    def pow(self, exponent: float) -> "Expr":
        """
        Raise expression to the power of exponent
        """
        return wrap_expr(self._pyexpr.pow(exponent))

    def is_in(self, list_expr: "Expr") -> "Expr":
        return wrap_expr(self._pyexpr.is_in(list_expr))

    def is_between(
        self, start: "Union[Expr, datetime]", end: "Union[Expr, datetime]"
    ) -> "Expr":
        """
        Check if this expression is between start and end
        """
        cast_to_date64 = False
        if isinstance(start, datetime):
            start = lit(start)
            cast_to_date64 = True
        if isinstance(end, datetime):
            end = lit(end)
            cast_to_date64 = True
        if cast_to_date64:
            expr = self.cast(datatypes.Date64)
        else:
            expr = self
        return ((expr > start) & (expr < end)).alias("is_between")


def expr_to_lit_or_expr(expr: Union["Expr", int, float, str]) -> "Expr":
    if isinstance(expr, (int, float, str)):
        return lit(expr)
    return expr


class WhenThen:
    def __init__(self, pywhenthen: "PyWhenThen"):  # noqa F821
        self._pywhenthen = pywhenthen

    def otherwise(self, expr: "Union[Expr, int, float, str]") -> "Expr":
        expr = expr_to_lit_or_expr(expr)
        return wrap_expr(self._pywhenthen.otherwise(expr._pyexpr))


class When:
    def __init__(self, pywhen: "pywhen"):  # noqa F821
        self._pywhen = pywhen

    def then(self, expr: "Union[Expr, int, float, str]") -> WhenThen:
        expr = expr_to_lit_or_expr(expr)
        whenthen = self._pywhen.then(expr._pyexpr)
        return WhenThen(whenthen)


def when(expr: "Expr") -> When:
    """
    Start a when, then, otherwise expression

    # Example

    Below we add a column with the value 1, where column "foo" > 2 and the value -1 where it isn't.

    ```python
    lf.with_column(
        when(col("foo") > 2)
        .then(lit(1))
        .otherwise(lit(-1))
    )
    ```
    """
    expr = expr_to_lit_or_expr(expr)
    pw = pywhen(expr._pyexpr)
    return When(pw)


def col(name: str) -> "Expr":
    """
    A column in a DataFrame
    """
    return wrap_expr(pycol(name))


def except_(name: str) -> "Expr":
    """
    Exclude a column from a selection

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


def count(name: str = "") -> "Expr":
    """
    Count the number of values in this column
    """
    return col(name).count()


def to_list(name: str) -> "Expr":
    """
    Aggregate to list
    """
    return col(name).list()


def std(name: str) -> "Expr":
    """
    Get standard deviation
    """
    return col(name).std()


def var(name: str) -> "Expr":
    """
    Get variance
    """
    return col(name).var()


def max(name: "Union[str, List[Expr]]") -> "Expr":
    """
    Get maximum value
    """
    if isinstance(name, list):

        def max_(acc: Series, val: Series) -> Series:
            mask = acc < val
            return acc.zip_with(mask, val)

        return fold(lit(0), max_, name).alias("max")
    return col(name).max()


def min(name: "Union[str, List[Expr]]") -> "Expr":
    """
    Get minimum value
    """
    if isinstance(name, list):

        def min_(acc: Series, val: Series) -> Series:
            mask = acc > val
            return acc.zip_with(mask, val)

        return fold(lit(0), min_, name).alias("min")
    return col(name).min()


def sum(name: "Union[str, List[Expr]]") -> "Expr":
    """
    Get sum value
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a + b, name).alias("sum")
    return col(name).sum()


def mean(name: str) -> "Expr":
    """
    Get mean value
    """
    return col(name).mean()


def avg(name: str) -> "Expr":
    """
    Alias for mean
    """
    return col(name).mean()


def median(name: str) -> "Expr":
    """
    Get median value
    """
    return col(name).median()


def n_unique(name: str) -> "Expr":
    """Count unique values"""
    return col(name).n_unique()


def first(name: str) -> "Expr":
    """
    Get first value
    """
    return col(name).first()


def last(name: str) -> "Expr":
    """
    Get last value
    """
    return col(name).last()


def head(name: str, n: "Optional[int]" = None) -> "Expr":
    """
    Get the first n rows of an Expression

    Parameters
    ----------
    name
        column name
    n
        number of rows to take
    """
    return col(name).head(n)


def tail(name: str, n: "Optional[int]" = None) -> "Expr":
    """
    Get the last n rows of an Expression

    Parameters
    ----------
    name
        column name
    n
        number of rows to take
    """
    return col(name).tail(n)


def lit_date(dt: "datetime") -> Expr:
    """
    Converts a Python DateTime to a literal Expression.

    Parameters
    ----------
    dt
        datetime.datetime
    """
    return lit(int(dt.timestamp() * 1e3))


def lit(value: "Optional[Union[float, int, str, datetime, Series]]") -> "Expr":
    """
    A literal value

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
        return lit(int(value.timestamp() * 1e3)).cast(datatypes.Date64)

    if isinstance(value, Series):
        value = value._s

    return wrap_expr(pylit(value))


def pearson_corr(
    a: "Union[str, Expr]",
    b: "Union[str, Expr]",
) -> "Expr":
    """
    Compute the pearson's correlation between two columns

    Parameters
    ----------
    a
        Column name or Expression
    b
        Column name or Expression
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(pypearson_corr(a._pyexpr, b._pyexpr))


def cov(
    a: "Union[str, Expr]",
    b: "Union[str, Expr]",
) -> "Expr":
    """
    Compute the covariance between two columns/ expressions.

    Parameters
    ----------
    a
        Column name or Expression
    b
        Column name or Expression
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(pycov(a._pyexpr, b._pyexpr))


def map_binary(
    a: "Union[str, Expr]",
    b: "Union[str, Expr]",
    f: Callable[[Series, Series], Series],
    output_type: "Optional[DataType]" = None,
) -> "Expr":
    """
    Map a custom function over two columns and produce a single Series result.

    Parameters
    ----------
    a
        Input Series a
    b
        Input Series b
    f
        Function to apply
    output_type
        Output type of the udf
    """
    if isinstance(a, str):
        a = col(a)
    if isinstance(b, str):
        b = col(b)
    return wrap_expr(pybinary_function(a._pyexpr, b._pyexpr, f, output_type))


def fold(acc: Expr, f: Callable[[Series, Series], Series], exprs: List[Expr]) -> Expr:
    """
    Accumulate over multiple columns horizontally / row wise with a left fold.

    Parameters
    ----------
    acc
     Accumulator Expression. This is the value that will be initialized when the fold starts.
     For a sum this could for instance be lit(0)

    f
        Function to apply over the accumulator and the value
        Fn(acc, value) -> new_value
    exprs
        Expressions to aggregate over
    """
    for e in exprs:
        acc = map_binary(acc, e, f, None)
    return acc


def any(name: "Union[str, List[Expr]]") -> "Expr":
    """
    Evaluate columnwise or elementwise with a bitwise OR operation
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a | b, name).alias("any")
    return col(name).sum() > 0


def all(name: "Union[str, List[Expr]]") -> "Expr":
    """
    Evaluate columnwise or elementwise with a bitwise OR operation
    """
    if isinstance(name, list):
        return fold(lit(0), lambda a, b: a & b, name).alias("all")
    return col(name).cast(bool).sum() == col(name).count()


class UDF:
    def __init__(self, f: Callable[[Series], Series], output_type: "DataType"):
        self.f = f
        self.output_type = output_type


def udf(f: Callable[[Series], Series], output_type: "DataType"):
    return UDF(f, output_type)


def arange(low: int, high: int, dtype: "Optional[DataType]" = None) -> "Expr":
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
        lower bound of range.
    high
        upper bound of range.
    dtype
        DataType of the range. Valid dtypes:
            * Int32
            * Int64
            * UInt32
    """
    if dtype is None:
        dtype = datatypes.Int64
    return wrap_expr(pyrange(low, high, dtype))

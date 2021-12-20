"""
This module contains all expressions and classes needed for lazy computation/ query execution.
"""
import os
import shutil
import subprocess
import tempfile
import typing as tp
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

try:
    from polars.polars import PyExpr, PyLazyFrame, PyLazyGroupBy

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars import internals as pli
from polars.datatypes import DataType, py_type_to_dtype
from polars.utils import _process_null_values


def wrap_ldf(ldf: "PyLazyFrame") -> "LazyFrame":
    return LazyFrame._from_pyldf(ldf)


def _prepare_groupby_inputs(
    by: Optional[Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]]],
) -> tp.List["PyExpr"]:
    if isinstance(by, list):
        new_by = []
        for e in by:
            if isinstance(e, str):
                e = pli.col(e)
            new_by.append(e._pyexpr)
    elif isinstance(by, str):
        new_by = [pli.col(by)._pyexpr]
    elif isinstance(by, pli.Expr):
        new_by = [by._pyexpr]
    elif by is None:
        return []
    return new_by


class LazyFrame:
    """
    Representation of a Lazy computation graph/ query.
    """

    def __init__(self) -> None:
        self._ldf: PyLazyFrame

    @staticmethod
    def _from_pyldf(ldf: "PyLazyFrame") -> "LazyFrame":
        self = LazyFrame.__new__(LazyFrame)
        self._ldf = ldf
        return self

    @staticmethod
    def scan_csv(
        file: str,
        has_header: bool = True,
        sep: str = ",",
        comment_char: Optional[str] = None,
        quote_char: Optional[str] = r'"',
        skip_rows: int = 0,
        dtypes: Optional[Dict[str, Type[DataType]]] = None,
        null_values: Optional[Union[str, tp.List[str], Dict[str, str]]] = None,
        ignore_errors: bool = False,
        cache: bool = True,
        with_column_names: Optional[Callable[[tp.List[str]], tp.List[str]]] = None,
        infer_schema_length: Optional[int] = 100,
        n_rows: Optional[int] = None,
        low_memory: bool = False,
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_csv`
        """
        dtype_list: Optional[tp.List[Tuple[str, Type[DataType]]]] = None
        if dtypes is not None:
            dtype_list = []
            for k, v in dtypes.items():
                dtype_list.append((k, py_type_to_dtype(v)))
        processed_null_values = _process_null_values(null_values)

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_csv(
            file,
            sep,
            has_header,
            ignore_errors,
            skip_rows,
            n_rows,
            cache,
            dtype_list,
            low_memory,
            comment_char,
            quote_char,
            processed_null_values,
            infer_schema_length,
            with_column_names,
        )
        return self

    @staticmethod
    def scan_parquet(
        file: str, n_rows: Optional[int] = None, cache: bool = True
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_parquet`
        """

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_parquet(file, n_rows, cache)
        return self

    @staticmethod
    def scan_ipc(
        file: str, n_rows: Optional[int] = None, cache: bool = True
    ) -> "LazyFrame":
        """
        See Also: `pl.scan_ipc`
        """

        self = LazyFrame.__new__(LazyFrame)
        self._ldf = PyLazyFrame.new_from_ipc(file, n_rows, cache)
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

    def inspect(self, fmt: str = "{}") -> "LazyFrame":
        """
        Prints the value that this node in the computation graph evaluates to and passes on the value.

        >>> df = pl.DataFrame({"foo": [1, 1, -2, 3]}).lazy()
        >>> (
        ...     df.select(
        ...         [
        ...             pl.col("foo").cumsum().alias("bar"),
        ...         ]
        ...     )
        ...     .inspect()  # print the node before the filter
        ...     .filter(pl.col("bar") == pl.col("foo"))
        ... )  # doctest: +ELLIPSIS
        <polars.internals.lazy_frame.LazyFrame object at ...>

        """

        def inspect(s: pli.DataFrame) -> pli.DataFrame:
            print(fmt.format(s))
            return s

        return self.map(inspect, predicate_pushdown=True, projection_pushdown=True)

    def sort(
        self,
        by: Union[str, "pli.Expr", tp.List[str], tp.List["pli.Expr"]],
        reverse: Union[bool, tp.List[bool]] = False,
    ) -> "LazyFrame":
        """
        Sort the DataFrame by:

            - A single column name
            - An expression
            - Multiple expressions

        Parameters
        ----------
        by
            Column (expressions) to sort by.
        reverse
            Whether or not to sort in reverse order.
        """
        if type(by) is str:
            return wrap_ldf(self._ldf.sort(by, reverse))
        if type(reverse) is bool:
            reverse = [reverse]

        by = pli.expr_to_lit_or_expr(by, str_to_lit=False)
        by = pli._selection_to_pyexpr_list(by)
        return wrap_ldf(self._ldf.sort_by_exprs(by, reverse))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
    ) -> pli.DataFrame:
        """
        Collect into a DataFrame.

        Note: use `fetch` if you want to run this query on the first `n` rows only.
        This can be a huge time saver in debugging queries.

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
        return pli.wrap_df(ldf.collect())

    def fetch(
        self,
        n_rows: int = 500,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = True,
        no_optimization: bool = False,
    ) -> pli.DataFrame:
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
        return pli.wrap_df(ldf.fetch(n_rows))

    @property
    def columns(self) -> tp.List[str]:
        """
        Get or set column names.

        Examples
        --------

        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "foo": [1, 2, 3],
        ...             "bar": [6, 7, 8],
        ...             "ham": ["a", "b", "c"],
        ...         }
        ...     )
        ...     .lazy()
        ...     .select(["foo", "bar"])
        ... )

        >>> df.columns
        ['foo', 'bar']

        """
        return self._ldf.columns()

    def cache(
        self,
    ) -> "LazyFrame":
        """
        Cache the result once the execution of the physical plan hits this node.
        """
        return wrap_ldf(self._ldf.cache())

    def filter(self, predicate: Union["pli.Expr", str]) -> "LazyFrame":
        """
        Filter the rows in the DataFrame based on a predicate expression.

        Parameters
        ----------
        predicate
            Expression that evaluates to a boolean Series.

        Examples
        --------

        >>> lf = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()

        Filter on one condition:

        >>> lf.filter(pl.col("foo") < 3).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        Filter on multiple conditions:

        >>> lf.filter((pl.col("foo") < 3) & (pl.col("ham") == "a")).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        """
        if isinstance(predicate, str):
            predicate = pli.col(predicate)
        return wrap_ldf(self._ldf.filter(predicate._pyexpr))

    def select(
        self,
        exprs: Union[
            str, "pli.Expr", Sequence[str], Sequence["pli.Expr"], "pli.Series"
        ],
    ) -> "LazyFrame":
        """
        Select columns from this DataFrame.

        Parameters
        ----------
        exprs
            Column or columns to select.
        """
        exprs = pli._selection_to_pyexpr_list(exprs)
        return wrap_ldf(self._ldf.select(exprs))

    def groupby(
        self,
        by: Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]],
        maintain_order: bool = False,
    ) -> "LazyGroupBy":
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more expensive than a default groupby.
        """
        new_by = _prepare_groupby_inputs(by)
        lgb = self._ldf.groupby(new_by, maintain_order)
        return LazyGroupBy(lgb)

    def groupby_dynamic(
        self,
        time_column: str,
        every: str,
        period: Optional[str] = None,
        offset: Optional[str] = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: str = "right",
        by: Optional[Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]]] = None,
    ) -> "LazyGroupBy":
        """
        Groups based on a time value. Time windows are calculated and rows are assigned to windows.
        Different from a normal groupby is that a row can be member of multiple groups. The time window could
        be seen as a rolling window, with a window size determined by dates/times instead of slots in the DataFrame.

        A window is defined by:

        - every: interval of the window
        - period: length of the window
        - offset: offset of the window

        The `every`, `period` and `offset` arguments are created with
        the following string language:

        - 1ns   (1 nanosecond)
        - 1us   (1 microsecond)
        - 1ms   (1 millisecond)
        - 1s    (1 second)
        - 1m    (1 minute)
        - 1h    (1 hour)
        - 1d    (1 day)
        - 1w    (1 week)
        - 1mo   (1 calendar month)
        - 1y    (1 calendar year)

        Or combine them:
        "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        .. warning::
            This API is experimental and may change without it being considered a breaking change.

        Parameters
        ----------
        time_column
            Column used to group based on the time window.
            Often to type Date/Datetime
            This column must be sorted in ascending order. If not the output will not make sense.
        every
            interval of the window
        period
            length of the window, if None it is equal to 'every'
        offset
            offset of the window
        truncate
            truncate the time value to the window lower bound
        include_boundaries
            add the lower and upper bound of the window to the "_lower_bound" and "_upper_bound" columns
        closed
            Defines if the window interval is closed or not.
            Any of {"left", "right", "both" "none"}
        by
            Also group by this column/these columns

        """

        if period is None:
            period = every
        if offset is None:
            offset = "0ns"
        by = _prepare_groupby_inputs(by)
        lgb = self._ldf.groupby_dynamic(
            time_column, every, period, offset, truncate, include_boundaries, closed, by
        )
        return LazyGroupBy(lgb)

    def join(
        self,
        ldf: "LazyFrame",
        left_on: Optional[
            Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]
        ] = None,
        right_on: Optional[
            Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]
        ] = None,
        on: Optional[Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]] = None,
        how: str = "inner",
        suffix: str = "_right",
        allow_parallel: bool = True,
        force_parallel: bool = False,
        asof_by: Optional[Union[str, tp.List[str]]] = None,
        asof_by_left: Optional[Union[str, tp.List[str]]] = None,
        asof_by_right: Optional[Union[str, tp.List[str]]] = None,
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
        suffix
            Suffix to append to columns with a duplicate name.
        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to the join in parallel.
        asof_by
            join on these columns before doing asof join
        asof_by_left
            join on these columns before doing asof join
        asof_by_right
            join on these columns before doing asof join

        # Asof joins
        This is similar to a left-join except that we match on nearest key rather than equal keys.
        The keys must be sorted to perform an asof join

        """
        if how == "cross":
            return wrap_ldf(
                self._ldf.join(
                    ldf._ldf,
                    [],
                    [],
                    allow_parallel,
                    force_parallel,
                    how,
                    suffix,
                    [],
                    [],
                )
            )

        left_on_: Optional[tp.List[Union[str, pli.Expr]]]
        if isinstance(left_on, (str, pli.Expr)):
            left_on_ = [left_on]
        else:
            left_on_ = left_on

        right_on_: Optional[tp.List[Union[str, pli.Expr]]]
        if isinstance(right_on, (str, pli.Expr)):
            right_on_ = [right_on]
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
                column = pli.col(column)
            new_left_on.append(column._pyexpr)
        new_right_on = []
        for column in right_on_:
            if isinstance(column, str):
                column = pli.col(column)
            new_right_on.append(column._pyexpr)

        # set asof_by

        left_asof_by_: Union[tp.List[str], None]
        if isinstance(asof_by_left, str):
            left_asof_by_ = [asof_by_left]
        else:
            left_asof_by_ = asof_by_left

        right_asof_by_: Union[tp.List[str], None]
        if isinstance(asof_by_right, (str, pli.Expr)):
            right_asof_by_ = [asof_by_right]
        else:
            right_asof_by_ = asof_by_right

        if isinstance(asof_by, str):
            left_asof_by_ = [asof_by]
            right_asof_by_ = [asof_by]
        elif isinstance(asof_by, list):
            left_asof_by_ = asof_by
            right_asof_by_ = asof_by

        if left_asof_by_ is None:
            left_asof_by_ = []
        if right_asof_by_ is None:
            right_asof_by_ = []

        return wrap_ldf(
            self._ldf.join(
                ldf._ldf,
                new_left_on,
                new_right_on,
                allow_parallel,
                force_parallel,
                how,
                suffix,
                left_asof_by_,
                right_asof_by_,
            )
        )

    def with_columns(
        self, exprs: Union[tp.List["pli.Expr"], "pli.Expr"]
    ) -> "LazyFrame":
        """
        Add or overwrite multiple columns in a DataFrame.

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns.
        """
        if isinstance(exprs, pli.Expr):
            return self.with_column(exprs)

        pyexprs = []

        for e in exprs:
            if isinstance(e, pli.Expr):
                pyexprs.append(e._pyexpr)
            elif isinstance(e, pli.Series):
                pyexprs.append(pli.lit(e)._pyexpr)

        return wrap_ldf(self._ldf.with_columns(pyexprs))

    def with_column(self, expr: "pli.Expr") -> "LazyFrame":
        """
        Add or overwrite column in a DataFrame.

        Parameters
        ----------
        expr
            Expression that evaluates to column.
        """
        return self.with_columns([expr])

    def drop(self, columns: Union[str, tp.List[str]]) -> "LazyFrame":
        """
        Remove one or multiple columns from a DataFrame.

        Parameters
        ----------
        columns
            - Name of the column that should be removed.
            - List of column names.

        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_ldf(self._ldf.drop_columns(columns))

    def with_column_renamed(self, existing_name: str, new_name: str) -> "LazyFrame":
        """
        Rename a column in the DataFrame
        """
        return wrap_ldf(self._ldf.with_column_renamed(existing_name, new_name))

    def rename(self, mapping: Dict[str, str]) -> "LazyFrame":
        """
        Rename column names. This does not preserve column order.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name.
        """
        existing = list(mapping.keys())
        new = list(mapping.values())
        return wrap_ldf(self._ldf.rename(existing, new))

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
        self, periods: int, fill_value: Union["pli.Expr", int, str, float]
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
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
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
        Gets the first `n` rows of the DataFrame. You probably don't want to use this!

        Consider using the `fetch` operation. The `fetch` operation will truly load the first `n`
        rows lazily.

        This operation instead loads all the rows and only applies the `head` at the end.

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

    def with_row_count(self, name: str = "row_nr") -> "LazyFrame":
        """
        Add a column at index 0 that counts the rows.

        Parameters
        ----------
        name
            Name of the column to add.
        """
        return wrap_ldf(self._ldf.with_row_count(name))

    def fill_null(self, fill_value: Union[int, str, "pli.Expr"]) -> "LazyFrame":
        """
        Fill missing values

        Parameters
        ----------
        fill_value
            Value to fill the missing values with
        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return wrap_ldf(self._ldf.fill_null(fill_value._pyexpr))

    def fill_nan(self, fill_value: Union[int, str, float, "pli.Expr"]) -> "LazyFrame":
        """
        Fill floating point NaN values.

        ..warning::

            NOTE that floating point NaN (No a Number) are not missing values!
            to replace missing values, use `fill_null`.


        Parameters
        ----------
        fill_value
            Value to fill the NaN values with
        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return wrap_ldf(self._ldf.fill_nan(fill_value._pyexpr))

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

    def quantile(self, quantile: float, interpolation: str = "nearest") -> "LazyFrame":
        """
        Aggregate the columns in the DataFrame to their quantile value.
        """
        return wrap_ldf(self._ldf.quantile(quantile, interpolation))

    def explode(
        self, columns: Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]]
    ) -> "LazyFrame":
        """
        Explode lists to long format.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [[1, 2], [1, 3], [4, 3], [5, 5, 5], [6], [2, 1, 2]],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬────────────┐
        │ letters ┆ nrs        │
        │ ---     ┆ ---        │
        │ str     ┆ list [i64] │
        ╞═════════╪════════════╡
        │ c       ┆ [1, 2]     │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ [1, 3]     │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ [4, 3]     │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ [5, 5, 5]  │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ [6]        │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ b       ┆ [2, 1, 2]  │
        └─────────┴────────────┘
        >>> df.explode("nrs")
        shape: (13, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ ...     ┆ ... │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 2   │
        └─────────┴─────┘

        """
        columns = pli._selection_to_pyexpr_list(columns)
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, None, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.lazy().drop_nulls().collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        This method only drops nulls row-wise if any single value of the row is null.

        Below are some example snippets that show how you could drop null values based on other
        conditions

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, None, None, None],
        ...         "b": [1, 2, None, 1],
        ...         "c": [1, None, None, 1],
        ...     }
        ... )
        >>> df
        shape: (4, 3)
        ┌──────┬──────┬──────┐
        │ a    ┆ b    ┆ c    │
        │ ---  ┆ ---  ┆ ---  │
        │ f64  ┆ i64  ┆ i64  │
        ╞══════╪══════╪══════╡
        │ null ┆ 1    ┆ 1    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ 2    ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ null ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ 1    ┆ 1    │
        └──────┴──────┴──────┘

        Drop a row only if all values are null:

        >>> df.filter(
        ...     ~pl.fold(
        ...         acc=True,
        ...         f=lambda acc, s: acc & s.is_null(),
        ...         exprs=pl.all(),
        ...     )
        ... )
        shape: (3, 3)
        ┌──────┬─────┬──────┐
        │ a    ┆ b   ┆ c    │
        │ ---  ┆ --- ┆ ---  │
        │ f64  ┆ i64 ┆ i64  │
        ╞══════╪═════╪══════╡
        │ null ┆ 1   ┆ 1    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ 2   ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ 1   ┆ 1    │
        └──────┴─────┴──────┘

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
        f: Callable[[pli.DataFrame], pli.DataFrame],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        no_optimizations: bool = False,
    ) -> "LazyFrame":
        """
        Apply a custom function. It is important that the function returns a Polars DataFrame.

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

    def interpolate(self) -> "LazyFrame":
        """
        Interpolate intermediate values. The interpolation method is linear.
        """
        return self.select(pli.col("*").interpolate())


class LazyGroupBy:
    """
    Created by `df.lazy().groupby("foo)"`
    """

    def __init__(self, lgb: "PyLazyGroupBy"):
        self.lgb = lgb

    def agg(self, aggs: Union[tp.List["pli.Expr"], "pli.Expr"]) -> "LazyFrame":
        """
        Describe the aggregation that need to be done on a group.

        Parameters
        ----------
        aggs
            Single/ Multiple aggregation expression(s).

        Examples
        --------

        >>> (
        ...     pl.scan_csv("data.csv")
        ...     .groupby("groups")
        ...     .agg(
        ...         [
        ...             pl.col("name").n_unique().alias("unique_names"),
        ...             pl.max("values"),
        ...         ]
        ...     )
        ... )  # doctest: +SKIP

        """
        aggs = pli._selection_to_pyexpr_list(aggs)
        return wrap_ldf(self.lgb.agg(aggs))

    def head(self, n: int = 5) -> "LazyFrame":
        """
        Return first n rows of each group.

        Parameters
        ----------
        n
            Number of values of the group to select

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        └─────────┴─────┘
        >>> df.groupby("letters").head(2).sort("letters")
        shape: (5, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        └─────────┴─────┘

        """
        return wrap_ldf(self.lgb.head(n))

    def tail(self, n: int = 5) -> "LazyFrame":
        """
        Return last n rows of each group.

        Parameters
        ----------
        n
            Number of values of the group to select

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ c       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        └─────────┴─────┘
        >>> df.groupby("letters").tail(2).sort("letters")
         shape: (5, 2)
        ┌─────────┬─────┐
        │ letters ┆ nrs │
        │ ---     ┆ --- │
        │ str     ┆ i64 │
        ╞═════════╪═════╡
        │ a       ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ a       ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ c       ┆ 4   │
        └─────────┴─────┘

        """
        return wrap_ldf(self.lgb.tail(n))

    def apply(self, f: Callable[[pli.DataFrame], pli.DataFrame]) -> "LazyFrame":
        """
        Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
        this as materializing the `DataFrame` is quite expensive.

        Parameters
        ----------
        f
            Function to apply over the `DataFrame`.
        """
        return wrap_ldf(self.lgb.apply(f))

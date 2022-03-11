"""
This module contains all expressions and classes needed for lazy computation/ query execution.
"""
import os
import shutil
import subprocess
import tempfile
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

try:
    from polars.polars import PyExpr, PyLazyFrame, PyLazyGroupBy

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars import internals as pli
from polars.datatypes import DataType, py_type_to_dtype
from polars.utils import _in_notebook, _prepare_row_count_args, _process_null_values

# Used to type any type or subclass of LazyFrame.
# Used to indicate when LazyFrame methods return the same type as self,
# including sub-classes.
LDF = TypeVar("LDF", bound="LazyFrame")

# We redefine the DF type variable from polars.internals.frame here in order to prevent
# circular import issues. The frame module needs this module to be defined at import
# time due to how the metaclass of DataFrame is defined.
DF = TypeVar("DF", bound="pli.DataFrame")


def wrap_ldf(ldf: "PyLazyFrame") -> "LazyFrame":
    return LazyFrame._from_pyldf(ldf)


def _prepare_groupby_inputs(
    by: Optional[Union[str, List[str], "pli.Expr", List["pli.Expr"]]],
) -> List["PyExpr"]:
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


class LazyFrame(Generic[DF]):
    """
    Representation of a Lazy computation graph/ query.
    """

    def __init__(self) -> None:
        self._ldf: PyLazyFrame

    @classmethod
    def _from_pyldf(cls: Type[LDF], ldf: "PyLazyFrame") -> LDF:
        self = cls.__new__(cls)
        self._ldf = ldf
        return self

    @property
    def _dataframe_class(self) -> Type[DF]:
        """
        Return the associated DataFrame which is the equivalent of this LazyFrame object.

        This class is used when a LazyFrame object is casted to a non-lazy representation
        by the invocation of `.collect()`, `.fetch()`, and so on. By default we specify
        the regular `polars.internals.frame.DataFrame` class here, but any subclass of
        DataFrame that wishes to preserve its type when converted to LazyFrame and back
        (with `.lazy().collect()` for instance) must overwrite this class variable
        before setting DataFrame._lazyframe_class.

        This property is dynamically overwritten when DataFrame is sub-classed. See
        `polars.internals.frame.DataFrameMetaClass.__init__` for implementation details.
        """
        return pli.DataFrame  # type: ignore

    @classmethod
    def scan_csv(
        cls: Type[LDF],
        file: str,
        has_header: bool = True,
        sep: str = ",",
        comment_char: Optional[str] = None,
        quote_char: Optional[str] = r'"',
        skip_rows: int = 0,
        dtypes: Optional[Dict[str, Type[DataType]]] = None,
        null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
        ignore_errors: bool = False,
        cache: bool = True,
        with_column_names: Optional[Callable[[List[str]], List[str]]] = None,
        infer_schema_length: Optional[int] = 100,
        n_rows: Optional[int] = None,
        encoding: str = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
    ) -> LDF:
        """
        See Also: `pl.scan_csv`
        """
        dtype_list: Optional[List[Tuple[str, Type[DataType]]]] = None
        if dtypes is not None:
            dtype_list = []
            for k, v in dtypes.items():
                dtype_list.append((k, py_type_to_dtype(v)))
        processed_null_values = _process_null_values(null_values)

        self = cls.__new__(cls)
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
            rechunk,
            skip_rows_after_header,
            encoding,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
        return self

    @classmethod
    def scan_parquet(
        cls: Type[LDF],
        file: str,
        n_rows: Optional[int] = None,
        cache: bool = True,
        parallel: bool = True,
        rechunk: bool = True,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
    ) -> LDF:
        """
        See Also: `pl.scan_parquet`
        """

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_parquet(
            file,
            n_rows,
            cache,
            parallel,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
        return self

    @classmethod
    def scan_ipc(
        cls: Type[LDF],
        file: str,
        n_rows: Optional[int] = None,
        cache: bool = True,
        rechunk: bool = True,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
    ) -> LDF:
        """
        See Also: `pl.scan_ipc`
        """

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_ipc(
            file,
            n_rows,
            cache,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
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

        Examples
        --------

        >>> def cast_str_to_int(data, col_name):
        ...     return data.with_column(pl.col(col_name).cast(pl.Int64))
        ...
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]}).lazy()
        >>> df.pipe(cast_str_to_int, col_name="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 10  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 20  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 30  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 4   ┆ 40  │
        └─────┴─────┘

        """
        return func(self, *args, **kwargs)

    def _repr_html_(self) -> str:
        try:
            dot = self._ldf.to_dot(optimized=False)
            svg = subprocess.check_output(
                ["dot", "-Nshape=box", "-Tsvg"], input=f"{dot}".encode()
            )
            return f"<h4>NAIVE QUERY PLAN</h4><p>run <b>LazyFrame.show_graph()</b> to see the optimized version</p>{svg.decode()}"
        except Exception:
            insert = self.describe_plan().replace("\n", "<p></p>")

            return f"""<i>naive plan: (run <b>LazyFrame.describe_optimized_plan()</b> to see the optimized plan)</i>
    <p></p>
    <div>{insert}</div>"""

    def __str__(self) -> str:
        return f"""naive plan: (run LazyFrame.describe_optimized_plan() to see the optimized plan)

{self.describe_plan()}"""

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
        slice_pushdown: bool = True,
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
            slice_pushdown=slice_pushdown,
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
            Return dot syntax. This cannot be combined with `show`
        figsize
            Passed to matlotlib if `show` == True.
        """
        if raw_output:
            show = False

        if show and _in_notebook():
            try:
                from IPython.display import SVG, display

                dot = self._ldf.to_dot(optimized)
                svg = subprocess.check_output(
                    ["dot", "-Nshape=box", "-Tsvg"], input=f"{dot}".encode()
                )
                return display(SVG(svg))
            except Exception:
                raise ImportError(
                    "Graphviz dot binary should be on your PATH and matplotlib should be installed to show graph."
                )
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

    def inspect(self: LDF, fmt: str = "{}") -> LDF:
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
        self: LDF,
        by: Union[str, "pli.Expr", List[str], List["pli.Expr"]],
        reverse: Union[bool, List[bool]] = False,
        nulls_last: bool = False,
    ) -> LDF:
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
            Sort in descending order.
        nulls_last
            Place null values last. Can only be used if sorted by a single column.
        """
        if nulls_last and not isinstance(by, str):
            raise ValueError(
                "nulls_last can only be combined with 'by' argument of type str"
            )
        if type(by) is str:
            return self._from_pyldf(self._ldf.sort(by, reverse, nulls_last))
        if type(reverse) is bool:
            reverse = [reverse]

        by = pli.selection_to_pyexpr_list(by)
        return self._from_pyldf(self._ldf.sort_by_exprs(by, reverse))

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> DF:
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
        slice_pushdown
            Slice pushdown optimization.

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
            slice_pushdown,
        )
        return self._dataframe_class._from_pydf(ldf.collect())

    def fetch(
        self,
        n_rows: int = 500,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> DF:
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
        slice_pushdown
            Slice pushdown opitmizaiton

        Returns
        -------
        DataFrame
        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            string_cache,
            slice_pushdown,
        )
        return self._dataframe_class._from_pydf(ldf.fetch(n_rows))

    @property
    def columns(self) -> List[str]:
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

    def cache(self: LDF) -> LDF:
        """
        Cache the result once the execution of the physical plan hits this node.
        """
        return self._from_pyldf(self._ldf.cache())

    def filter(self: LDF, predicate: Union["pli.Expr", str]) -> LDF:
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
        return self._from_pyldf(self._ldf.filter(predicate._pyexpr))

    def select(
        self: LDF,
        exprs: Union[
            str, "pli.Expr", Sequence[str], Sequence["pli.Expr"], "pli.Series"
        ],
    ) -> LDF:
        """
        Select columns from this DataFrame.

        Parameters
        ----------
        exprs
            Column or columns to select.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> df.select("foo").collect()
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        ├╌╌╌╌╌┤
        │ 3   │
        └─────┘

        """
        exprs = pli.selection_to_pyexpr_list(exprs)
        return self._from_pyldf(self._ldf.select(exprs))

    def groupby(
        self,
        by: Union[str, List[str], "pli.Expr", List["pli.Expr"]],
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

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        # does NOT work:
        # df.groupby("a")["b"].sum().collect()
        #                ^^^^ TypeError: 'LazyGroupBy' object is not subscriptable
        # instead, use .agg():
        >>> df.groupby("a").agg(pl.col("b").sum()).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ c   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 11  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ a   ┆ 4   │
        └─────┴─────┘

        """
        new_by = _prepare_groupby_inputs(by)
        lgb = self._ldf.groupby(new_by, maintain_order)
        return LazyGroupBy(lgb)

    def groupby_rolling(
        self,
        index_column: str,
        period: str,
        offset: Optional[str] = None,
        closed: str = "right",
    ) -> "LazyGroupBy":
        """
        Create rolling groups based on a time column (or index value of type Int32, Int64).

        Different from a rolling groupby the windows are now determined by the individual values and are not of constant
        intervals. For constant intervals use *groupby_dynamic*

        .. seealso::

            groupby_dynamic


        The `period` and `offset` arguments are created with
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
        - 1i    (1 index count)

        Or combine them:
        "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        In case of a groupby_rolling on an integer column, the windows are defined by:

        - "1i"      # length 1
        - "10i"     # length 10

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often to type Date/Datetime
            This column must be sorted in ascending order. If not the output will not make sense.

            In case of a rolling groupby on indices, dtype needs to be one of {Int32, Int64}. Note that
            Int32 gets temporarely cast to Int64, so if performance matters use an Int64 column.
        period
            length of the window
        offset
            offset of the window. Default is -period
        closed
            Defines if the window interval is closed or not.
            Any of {"left", "right", "both" "none"}

        Examples
        --------

        >>> dates = [
        ...     "2020-01-01 13:45:48",
        ...     "2020-01-01 16:42:13",
        ...     "2020-01-01 16:45:09",
        ...     "2020-01-02 18:12:48",
        ...     "2020-01-03 19:45:32",
        ...     "2020-01-08 23:16:43",
        ... ]
        >>> df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_column(
        ...     pl.col("dt").str.strptime(pl.Datetime)
        ... )
        >>> out = df.groupby_rolling(index_column="dt", period="2d").agg(
        ...     [
        ...         pl.sum("a").alias("sum_a"),
        ...         pl.min("a").alias("min_a"),
        ...         pl.max("a").alias("max_a"),
        ...     ]
        ... )
        >>> assert out["sum_a"].to_list() == [3, 10, 15, 24, 11, 1]
        >>> assert out["max_a"].to_list() == [3, 7, 7, 9, 9, 1]
        >>> assert out["min_a"].to_list() == [3, 3, 3, 3, 2, 1]
        >>> out
        shape: (6, 4)
        ┌─────────────────────┬───────┬───────┬───────┐
        │ dt                  ┆ a_sum ┆ a_max ┆ a_min │
        │ ---                 ┆ ---   ┆ ---   ┆ ---   │
        │ datetime[ms]        ┆ i64   ┆ i64   ┆ i64   │
        ╞═════════════════════╪═══════╪═══════╪═══════╡
        │ 2020-01-01 13:45:48 ┆ 3     ┆ 3     ┆ 3     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-01 16:42:13 ┆ 10    ┆ 7     ┆ 3     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-01 16:45:09 ┆ 15    ┆ 7     ┆ 3     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-02 18:12:48 ┆ 24    ┆ 9     ┆ 3     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-03 19:45:32 ┆ 11    ┆ 9     ┆ 2     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-08 23:16:43 ┆ 1     ┆ 1     ┆ 1     │
        └─────────────────────┴───────┴───────┴───────┘


        """

        if offset is None:
            offset = f"-{period}"

        lgb = self._ldf.groupby_rolling(
            index_column,
            period,
            offset,
            closed,
        )
        return LazyGroupBy(lgb)

    def groupby_dynamic(
        self,
        index_column: str,
        every: str,
        period: Optional[str] = None,
        offset: Optional[str] = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: str = "right",
        by: Optional[Union[str, List[str], "pli.Expr", List["pli.Expr"]]] = None,
    ) -> "LazyGroupBy":
        """
        Groups based on a time value (or index value of type Int32, Int64). Time windows are calculated and rows are assigned to windows.
        Different from a normal groupby is that a row can be member of multiple groups. The time/index window could
        be seen as a rolling window, with a window size determined by dates/times/values instead of slots in the DataFrame.

        .. seealso::

            groupby_rolling

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
        - 1i    (1 index count)

        Or combine them:
        "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        In case of a groupby_dynamic on an integer column, the windows are defined by:

        - "1i"      # length 1
        - "10i"     # length 10

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often to type Date/Datetime
            This column must be sorted in ascending order. If not the output will not make sense.

            In case of a dynamic groupby on indices, dtype needs to be one of {Int32, Int64}. Note that
            Int32 gets temporarely cast to Int64, so if performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it is equal to 'every'
        offset
            offset of the window if None and period is None it will be equal to negative `every`
        truncate
            truncate the time value to the window lower bound
        include_boundaries
            add the lower and upper bound of the window to the "_lower_bound" and "_upper_bound" columns
            this will impact performance because it's harder to parallelize
        closed
            Defines if the window interval is closed or not.
            Any of {"left", "right", "both" "none"}
        by
            Also group by this column/these columns

        """

        if offset is None:
            if period is None:
                offset = f"-{every}"
            else:
                offset = "0ns"
        if period is None:
            period = every
        by = _prepare_groupby_inputs(by)
        lgb = self._ldf.groupby_dynamic(
            index_column,
            every,
            period,
            offset,
            truncate,
            include_boundaries,
            closed,
            by,
        )
        return LazyGroupBy(lgb)

    def join_asof(
        self: LDF,
        ldf: "LazyFrame",
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        on: Optional[str] = None,
        by_left: Optional[Union[str, List[str]]] = None,
        by_right: Optional[Union[str, List[str]]] = None,
        by: Optional[Union[str, List[str]]] = None,
        strategy: str = "backward",
        suffix: str = "_right",
        tolerance: Optional[Union[str, int, float]] = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LDF:
        """
        Perform an asof join. This is similar to a left-join except that we
        match on nearest key rather than equal keys.

        Both DataFrames must be sorted by the key.

        For each row in the left DataFrame:

          - A "backward" search selects the last row in the right DataFrame whose
            'on' key is less than or equal to the left's key.

          - A "forward" search selects the first row in the right DataFrame whose
            'on' key is greater than or equal to the left's key.

        The default is "backward".

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
        by
            join on these columns before doing asof join
        by_left
            join on these columns before doing asof join
        by_right
            join on these columns before doing asof join
        strategy
            One of {'forward', 'backward'}
        suffix
            Suffix to append to columns with a duplicate name.
        tolerance
            Numeric tolerance. By setting this the join will only be done if the near keys are within this distance.
            If an asof join is done on columns of dtype "Date", "Datetime", "Duration" or "Time" you
            use the following string language:

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
                - 1i    (1 index count)

                Or combine them:
                "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to the join in parallel.
        """

        if isinstance(on, str):
            left_on = on
            right_on = on

        if left_on is None or right_on is None:
            raise ValueError("You should pass the column to join on as an argument.")

        by_left_: Union[List[str], None]
        if isinstance(by_left, str):
            by_left_ = [by_left]
        else:
            by_left_ = by_left

        by_right_: Union[List[str], None]
        if isinstance(by_right, (str, pli.Expr)):
            by_right_ = [by_right]
        else:
            by_right_ = by_right

        if isinstance(by, str):
            by_left_ = [by]
            by_right_ = [by]
        elif isinstance(by, list):
            by_left_ = by
            by_right_ = by

        tolerance_str: Optional[str] = None
        tolerance_num: Optional[Union[float, int]] = None
        if isinstance(tolerance, str):
            tolerance_str = tolerance
        else:
            tolerance_num = tolerance

        return self._from_pyldf(
            self._ldf.join_asof(
                ldf._ldf,
                pli.col(left_on)._pyexpr,
                pli.col(right_on)._pyexpr,
                by_left_,
                by_right_,
                allow_parallel,
                force_parallel,
                suffix,
                strategy,
                tolerance_num,
                tolerance_str,
            )
        )

    def join(
        self: LDF,
        ldf: "LazyFrame",
        left_on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        right_on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        how: str = "inner",
        suffix: str = "_right",
        allow_parallel: bool = True,
        force_parallel: bool = False,
        asof_by: Optional[Union[str, List[str]]] = None,
        asof_by_left: Optional[Union[str, List[str]]] = None,
        asof_by_right: Optional[Union[str, List[str]]] = None,
    ) -> LDF:
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

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> other_df = pl.DataFrame(
        ...     {
        ...         "apple": ["x", "y", "z"],
        ...         "ham": ["a", "b", "d"],
        ...     }
        ... ).lazy()
        >>> df.join(other_df, on="ham").collect()
        shape: (2, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘
        >>> df.join(other_df, on="ham", how="outer").collect()
        shape: (4, 4)
        ┌──────┬──────┬─────┬───────┐
        │ foo  ┆ bar  ┆ ham ┆ apple │
        │ ---  ┆ ---  ┆ --- ┆ ---   │
        │ i64  ┆ f64  ┆ str ┆ str   │
        ╞══════╪══════╪═════╪═══════╡
        │ 1    ┆ 6.0  ┆ a   ┆ x     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2    ┆ 7.0  ┆ b   ┆ y     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ null ┆ null ┆ d   ┆ z     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3    ┆ 8.0  ┆ c   ┆ null  │
        └──────┴──────┴─────┴───────┘

        """
        if how == "asof":
            warnings.warn(
                "using asof join via LazyFrame.join is deprecated, please use LazyFrame.join_asof"
            )
        if how == "cross":
            return self._from_pyldf(
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

        left_on_: Optional[List[Union[str, pli.Expr]]]
        if isinstance(left_on, (str, pli.Expr)):
            left_on_ = [left_on]
        else:
            left_on_ = left_on

        right_on_: Optional[List[Union[str, pli.Expr]]]
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

        left_asof_by_: Union[List[str], None]
        if isinstance(asof_by_left, str):
            left_asof_by_ = [asof_by_left]
        else:
            left_asof_by_ = asof_by_left

        right_asof_by_: Union[List[str], None]
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

        return self._from_pyldf(
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

    def with_columns(self: LDF, exprs: Union[List["pli.Expr"], "pli.Expr"]) -> LDF:
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

        return self._from_pyldf(self._ldf.with_columns(pyexprs))

    def with_column(self: LDF, expr: "pli.Expr") -> LDF:
        """
        Add or overwrite column in a DataFrame.

        Parameters
        ----------
        expr
            Expression that evaluates to column.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.with_column((pl.col("b") ** 2).alias("b_squared")).collect()  # added
        shape: (3, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ b_squared │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ i64 ┆ f64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ 2   ┆ 4.0       │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ 4   ┆ 16.0      │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ 5   ┆ 6   ┆ 36.0      │
        └─────┴─────┴───────────┘
        >>> df.with_column(pl.col("a") ** 2).collect()  # replaced
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ f64  ┆ i64 │
        ╞══════╪═════╡
        │ 1.0  ┆ 2   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 9.0  ┆ 4   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 25.0 ┆ 6   │
        └──────┴─────┘

        """
        return self.with_columns([expr])

    def drop(self: LDF, columns: Union[str, List[str]]) -> LDF:
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
        return self._from_pyldf(self._ldf.drop_columns(columns))

    def rename(self: LDF, mapping: Dict[str, str]) -> LDF:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name.
        """
        existing = list(mapping.keys())
        new = list(mapping.values())
        return self._from_pyldf(self._ldf.rename(existing, new))

    def reverse(self: LDF) -> LDF:
        """
        Reverse the DataFrame.
        """
        return self._from_pyldf(self._ldf.reverse())

    def shift(self: LDF, periods: int) -> LDF:
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.shift(periods=1).collect()
        shape: (3, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ null ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 1    ┆ 2    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3    ┆ 4    │
        └──────┴──────┘
        >>> df.shift(periods=-1).collect()
        shape: (3, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 3    ┆ 4    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 5    ┆ 6    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ null │
        └──────┴──────┘

        """
        return self._from_pyldf(self._ldf.shift(periods))

    def shift_and_fill(
        self: LDF,
        periods: int,
        fill_value: Union["pli.Expr", int, str, float],
    ) -> LDF:
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with the result of this expression.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.shift_and_fill(periods=1, fill_value=0).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 0   ┆ 0   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 4   │
        └─────┴─────┘
        >>> df.shift_and_fill(periods=-1, fill_value=0).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 5   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 0   ┆ 0   │
        └─────┴─────┘

        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return self._from_pyldf(self._ldf.shift_and_fill(periods, fill_value._pyexpr))

    def slice(self: LDF, offset: int, length: int) -> LDF:
        """
        Slice the DataFrame.

        Parameters
        ----------
        offset
            Start index.
        length
            Length of the slice.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>>
        >>> df.slice(1, 2).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ y   ┆ 3   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ z   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘

        """
        return self._from_pyldf(self._ldf.slice(offset, length))

    def limit(self: LDF, n: int) -> LDF:
        """
        Limit the DataFrame to the first `n` rows. Note if you don't want the rows to be scanned,
        use the `fetch` operation.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self.slice(0, n)

    def head(self: LDF, n: int) -> LDF:
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

    def tail(self: LDF, n: int) -> LDF:
        """
        Get the last `n` rows of the DataFrame.

        Parameters
        ----------
        n
            Number of rows.
        """
        return self._from_pyldf(self._ldf.tail(n))

    def last(self: LDF) -> LDF:
        """
        Get the last row of the DataFrame.
        """
        return self.tail(1)

    def first(self: LDF) -> LDF:
        """
        Get the first row of the DataFrame.
        """
        return self.slice(0, 1)

    def with_row_count(self: LDF, name: str = "row_nr", offset: int = 0) -> LDF:
        """
        Add a column at index 0 that counts the rows.

        ..warning::
            This can have a negative effect on query performance.
            This may for instance block predicate pushdown optimization.

        Parameters
        ----------
        name
            Name of the column to add.
        offset
            Start the row count at this offset

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.with_row_count().collect()
        shape: (3, 3)
        ┌────────┬─────┬─────┐
        │ row_nr ┆ a   ┆ b   │
        │ ---    ┆ --- ┆ --- │
        │ u32    ┆ i64 ┆ i64 │
        ╞════════╪═════╪═════╡
        │ 0      ┆ 1   ┆ 2   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1      ┆ 3   ┆ 4   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2      ┆ 5   ┆ 6   │
        └────────┴─────┴─────┘

        """
        return self._from_pyldf(self._ldf.with_row_count(name, offset))

    def fill_null(self: LDF, fill_value: Union[int, str, "pli.Expr"]) -> LDF:
        """
        Fill missing values with a literal or Expr.

        Parameters
        ----------
        fill_value
            Value to fill the missing values with.
        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return self._from_pyldf(self._ldf.fill_null(fill_value._pyexpr))

    def fill_nan(self: LDF, fill_value: Union[int, str, float, "pli.Expr"]) -> LDF:
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
        return self._from_pyldf(self._ldf.fill_nan(fill_value._pyexpr))

    def std(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their standard deviation value.
        """
        return self._from_pyldf(self._ldf.std())

    def var(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their variance value.
        """
        return self._from_pyldf(self._ldf.var())

    def max(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their maximum value.
        """
        return self._from_pyldf(self._ldf.max())

    def min(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their minimum value.
        """
        return self._from_pyldf(self._ldf.min())

    def sum(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their sum value.
        """
        return self._from_pyldf(self._ldf.sum())

    def mean(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their mean value.
        """
        return self._from_pyldf(self._ldf.mean())

    def median(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their median value.
        """
        return self._from_pyldf(self._ldf.median())

    def quantile(self: LDF, quantile: float, interpolation: str = "nearest") -> LDF:
        """
        Aggregate the columns in the DataFrame to their quantile value.
        """
        return self._from_pyldf(self._ldf.quantile(quantile, interpolation))

    def explode(
        self: LDF,
        columns: Union[str, List[str], "pli.Expr", List["pli.Expr"]],
    ) -> LDF:
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
        │ a       ┆ 6   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ b       ┆ 2   │
        └─────────┴─────┘

        """
        columns = pli.selection_to_pyexpr_list(columns)
        return self._from_pyldf(self._ldf.explode(columns))

    def distinct(
        self: LDF,
        maintain_order: bool = True,
        subset: Optional[Union[str, List[str]]] = None,
        keep: str = "first",
    ) -> LDF:
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame or subset.

        Parameters
        ----------
        maintain_order
            Keep the same order as the original DataFrame. This requires more work to compute.
        subset
            Subset to use to compare rows
        keep
            any of {"first", "last"}

        Returns
        -------
        DataFrame with unique rows
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return self._from_pyldf(self._ldf.distinct(maintain_order, subset, keep))

    def drop_nulls(self: LDF, subset: Optional[Union[List[str], str]] = None) -> LDF:
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
        return self._from_pyldf(self._ldf.drop_nulls(subset))

    def melt(
        self: LDF,
        id_vars: Optional[Union[str, List[str]]] = None,
        value_vars: Optional[Union[str, List[str]]] = None,
    ) -> LDF:
        """
        Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

        This function is useful to massage a DataFrame into a format where one or more columns are identifier variables
        (id_vars), while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis,
        leaving just two non-identifier columns, ‘variable’ and ‘value’.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables.
        value_vars
            Values to use as identifier variables.
            If `value_vars` is empty all columns that are not in `id_vars` will be used.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.melt(id_vars="a", value_vars=["b", "c"]).collect()
        shape: (6, 3)
        ┌─────┬──────────┬───────┐
        │ a   ┆ variable ┆ value │
        │ --- ┆ ---      ┆ ---   │
        │ str ┆ str      ┆ i64   │
        ╞═════╪══════════╪═══════╡
        │ x   ┆ b        ┆ 1     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ y   ┆ b        ┆ 3     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ z   ┆ b        ┆ 5     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ x   ┆ c        ┆ 2     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ y   ┆ c        ┆ 4     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ z   ┆ c        ┆ 6     │
        └─────┴──────────┴───────┘

        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        if value_vars is None:
            value_vars = []
        if id_vars is None:
            id_vars = []
        return self._from_pyldf(self._ldf.melt(id_vars, value_vars))

    def map(
        self: LDF,
        f: Callable[["pli.DataFrame"], "pli.DataFrame"],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        no_optimizations: bool = False,
    ) -> LDF:
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
        if no_optimizations:
            predicate_pushdown = False
            projection_pushdown = False
        return self._from_pyldf(
            self._ldf.map(f, predicate_pushdown, projection_pushdown)
        )

    def interpolate(self: LDF) -> LDF:
        """
        Interpolate intermediate values. The interpolation method is linear.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 9, 10],
        ...         "bar": [6, 7, 9, None],
        ...         "baz": [1, None, None, 9],
        ...     }
        ... ).lazy()
        >>> df.interpolate().collect()
        shape: (4, 3)
        ┌─────┬──────┬─────┐
        │ foo ┆ bar  ┆ baz │
        │ --- ┆ ---  ┆ --- │
        │ i64 ┆ i64  ┆ i64 │
        ╞═════╪══════╪═════╡
        │ 1   ┆ 6    ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 5   ┆ 7    ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 9   ┆ 9    ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 10  ┆ null ┆ 9   │
        └─────┴──────┴─────┘

        """
        return self.select(pli.col("*").interpolate())

    def unnest(self, names: Union[str, List[str]]) -> "LazyFrame":
        """
        Decompose a struct into its fields. The fields will be inserted in to the `DataFrame` on the
        location of the `struct` type.

        Parameters
        ----------
        names
           Names of the struct columns that will be decomposed by its fields
        """
        if isinstance(names, str):
            names = [names]
        return wrap_ldf(self._ldf.unnest(names))


class LazyGroupBy:
    """
    Created by `df.lazy().groupby("foo)"`
    """

    def __init__(self, lgb: "PyLazyGroupBy"):
        self.lgb = lgb

    def agg(self, aggs: Union[List["pli.Expr"], "pli.Expr"]) -> "LazyFrame":
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
        aggs = pli.selection_to_pyexpr_list(aggs)
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

    def apply(self, f: Callable[["pli.DataFrame"], "pli.DataFrame"]) -> "LazyFrame":
        """
        Apply a function over the groups as a new `DataFrame`. It is not recommended that you use
        this as materializing the `DataFrame` is quite expensive.

        Parameters
        ----------
        f
            Function to apply over the `DataFrame`.
        """
        return wrap_ldf(self.lgb.apply(f))

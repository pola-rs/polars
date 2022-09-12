from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import typing
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, overload
from warnings import warn

from polars import internals as pli
from polars.cfg import Config
from polars.datatypes import DataType, PolarsDataType, Schema, py_type_to_dtype
from polars.internals import selection_to_pyexpr_list
from polars.internals.lazyframe.groupby import LazyGroupBy
from polars.internals.slice import LazyPolarsSlice
from polars.utils import (
    _in_notebook,
    _prepare_row_count_args,
    _process_null_values,
    format_path,
)

try:
    from polars.polars import PyLazyFrame

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False


if TYPE_CHECKING:
    from polars.internals.type_aliases import (
        AsofJoinStrategy,
        ClosedWindow,
        CsvEncoding,
        FillNullStrategy,
        InterpolationMethod,
        JoinStrategy,
        ParallelStrategy,
        UniqueKeepStrategy,
    )


# Used to type any type or subclass of LazyFrame.
# Used to indicate when LazyFrame methods return the same type as self,
# including sub-classes.
LDF = TypeVar("LDF", bound="LazyFrame")


def wrap_ldf(ldf: PyLazyFrame) -> LazyFrame:
    return LazyFrame._from_pyldf(ldf)


class LazyFrame:
    """Representation of a Lazy computation graph/query."""

    _ldf: PyLazyFrame

    @classmethod
    def _from_pyldf(cls: type[LDF], ldf: PyLazyFrame) -> LDF:
        self = cls.__new__(cls)
        self._ldf = ldf
        return self

    @classmethod
    def _scan_csv(
        cls: type[LDF],
        file: str,
        has_header: bool = True,
        sep: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = r'"',
        skip_rows: int = 0,
        dtypes: dict[str, PolarsDataType] | None = None,
        null_values: str | list[str] | dict[str, str] | None = None,
        ignore_errors: bool = False,
        cache: bool = True,
        with_column_names: Callable[[list[str]], list[str]] | None = None,
        infer_schema_length: int | None = 100,
        n_rows: int | None = None,
        encoding: CsvEncoding = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        parse_dates: bool = False,
        eol_char: str = "\n",
    ) -> LDF:
        """
        Lazily read from a CSV file or multiple files via glob patterns.

        Use ``pl.scan_csv`` to dispatch to this method.

        See Also
        --------
        polars.io.scan_csv

        """
        dtype_list: list[tuple[str, PolarsDataType]] | None = None
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
            parse_dates,
            eol_char=eol_char,
        )
        return self

    @classmethod
    def _scan_parquet(
        cls: type[LDF],
        file: str,
        n_rows: int | None = None,
        cache: bool = True,
        parallel: ParallelStrategy = "auto",
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        storage_options: dict[str, object] | None = None,
        low_memory: bool = False,
    ) -> LDF:
        """
        Lazily read from a parquet file or multiple files via glob patterns.

        Use ``pl.scan_parquet`` to dispatch to this method.

        See Also
        --------
        polars.io.scan_parquet

        """
        # try fsspec scanner
        if not pli._is_local_file(file):
            scan = pli._scan_parquet_fsspec(file, storage_options)
            if n_rows:
                scan = scan.head(n_rows)
            if row_count_name is not None:
                scan = scan.with_row_count(row_count_name, row_count_offset)
            return scan  # type: ignore[return-value]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_parquet(
            file,
            n_rows,
            cache,
            parallel,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
            low_memory,
        )
        return self

    @classmethod
    def _scan_ipc(
        cls: type[LDF],
        file: str | Path,
        n_rows: int | None = None,
        cache: bool = True,
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        storage_options: dict[str, object] | None = None,
        memory_map: bool = True,
    ) -> LDF:
        """
        Lazily read from an Arrow IPC (Feather v2) file.

        Use ``pl.scan_ipc`` to dispatch to this method.

        See Also
        --------
        polars.io.scan_ipc

        """
        if isinstance(file, (str, Path)):
            file = format_path(file)

        # try fsspec scanner
        if not pli._is_local_file(file):
            scan = pli._scan_ipc_fsspec(file, storage_options)
            if n_rows:
                scan = scan.head(n_rows)
            if row_count_name is not None:
                scan = scan.with_row_count(row_count_name, row_count_offset)
            return scan  # type: ignore[return-value]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_ipc(
            file,
            n_rows,
            cache,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
            memory_map=memory_map,
        )
        return self

    @classmethod
    def _scan_ndjson(
        cls: type[LDF],
        file: str,
        infer_schema_length: int | None = None,
        batch_size: int | None = None,
        n_rows: int | None = None,
        low_memory: bool = False,
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
    ) -> LDF:
        """
        Lazily read from a newline delimited JSON file.

        Use ``pl.scan_ndjson`` to dispatch to this method.

        See Also
        --------
        polars.io.scan_ndjson

        """
        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_ndjson(
            file,
            infer_schema_length,
            batch_size,
            n_rows,
            low_memory,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
        return self

    @classmethod
    def from_json(cls, json: str) -> LazyFrame:
        """
        Read a logical plan from a JSON string to construct a LazyFrame.

        Parameters
        ----------
        json
            String in JSON format.

        See Also
        --------
        read_json

        """
        bytes = StringIO(json).getvalue().encode()
        file = BytesIO(bytes)
        return wrap_ldf(PyLazyFrame.read_json(file))

    @classmethod
    def read_json(
        cls,
        file: str | Path | IOBase,
    ) -> LazyFrame:
        """
        Read a logical plan from a JSON file to construct a LazyFrame.

        Parameters
        ----------
        file
            Path to a file or a file-like object.

        See Also
        --------
        LazyFrame.from_json, LazyFrame.write_json

        """
        if isinstance(file, StringIO):
            file = BytesIO(file.getvalue().encode())
        elif isinstance(file, (str, Path)):
            file = format_path(file)

        return wrap_ldf(PyLazyFrame.read_json(file))

    @classmethod
    def _scan_python_function(
        cls, schema: pa.schema | dict[str, type[DataType]], scan_fn: bytes
    ) -> LazyFrame:
        self = cls.__new__(cls)
        if isinstance(schema, dict):
            self._ldf = PyLazyFrame.scan_from_python_function_pl_schema(
                [(name, dt) for name, dt in schema.items()], scan_fn
            )
        else:
            self._ldf = PyLazyFrame.scan_from_python_function_arrow_schema(
                list(schema), scan_fn
            )
        return self

    @property
    def columns(self) -> list[str]:
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

    @property
    def dtypes(self) -> list[type[DataType]]:
        """
        Get dtypes of columns in LazyFrame.

        Examples
        --------
        >>> lf = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> lf.dtypes
        [<class 'polars.datatypes.Int64'>, <class 'polars.datatypes.Float64'>, <class 'polars.datatypes.Utf8'>]

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        """  # noqa: E501
        return self._ldf.dtypes()

    @property
    def schema(self) -> Schema:
        """
        Get a dict[column name, DataType].

        Examples
        --------
        >>> lf = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> lf.schema
        {'foo': <class 'polars.datatypes.Int64'>, 'bar': <class 'polars.datatypes.Float64'>, 'ham': <class 'polars.datatypes.Utf8'>}

        """  # noqa: E501
        return self._ldf.schema()

    def __contains__(self: LDF, key: str) -> bool:
        return key in self.columns

    def __copy__(self: LDF) -> LDF:
        return self.clone()

    def __deepcopy__(self: LDF, memo: None = None) -> LDF:
        return self.clone()

    def __getitem__(self: LDF, item: int | range | slice) -> LazyFrame:
        if not isinstance(item, slice):
            raise TypeError(
                "'LazyFrame' object is not subscriptable (aside from slicing). Use"
                " 'select()' or 'filter()' instead."
            )
        return LazyPolarsSlice(self).apply(item)

    def __str__(self) -> str:
        return f"""\
naive plan: (run LazyFrame.describe_optimized_plan() to see the optimized plan)

{self.describe_plan()}\
"""

    def _repr_html_(self) -> str:
        try:
            dot = self._ldf.to_dot(optimized=False)
            svg = subprocess.check_output(
                ["dot", "-Nshape=box", "-Tsvg"], input=f"{dot}".encode()
            )
            return (
                "<h4>NAIVE QUERY PLAN</h4><p>run <b>LazyFrame.show_graph()</b> to see"
                f" the optimized version</p>{svg.decode()}"
            )
        except Exception:
            insert = self.describe_plan().replace("\n", "<p></p>")

            return f"""\
<i>naive plan: (run <b>LazyFrame.describe_optimized_plan()</b> to see the optimized plan)</i>
    <p></p>
    <div>{insert}</div>\
"""  # noqa: E501

    @overload
    def write_json(
        self,
        file: None = None,
        *,
        to_string: bool | None = ...,
    ) -> str:
        ...

    @overload
    def write_json(
        self,
        file: IOBase | str | Path,
        *,
        to_string: bool | None = ...,
    ) -> None:
        ...

    def write_json(
        self,
        file: IOBase | str | Path | None = None,
        *,
        to_string: bool | None = None,
    ) -> str | None:
        """
        Write the logical plan of this LazyFrame to a file or string in JSON format.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to ``None``
            (default), the output is returned as a string instead.
        to_string
            Deprecated argument. Ignore file argument and return a string.

        See Also
        --------
        LazyFrame.read_json

        """
        if to_string is not None:
            warn(
                "`to_string` argument for `LazyFrame.write_json` will be removed in a"
                " future version. Remove the argument and set `file=None` (default).",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            to_string = False

        if isinstance(file, (str, Path)):
            file = format_path(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if to_string or file is None or to_string_io:
            with BytesIO() as buf:
                self._ldf.write_json(buf)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._ldf.write_json(file)
        return None

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

    def describe_plan(self) -> str:
        """Create a string representation of the unoptimized query plan."""
        return self._ldf.describe_plan()

    def describe_optimized_plan(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
    ) -> str:
        """Create a string representation of the optimized query plan."""
        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown=slice_pushdown,
        )

        return ldf.describe_optimized_plan()

    def show_graph(
        self,
        optimized: bool = True,
        show: bool = True,
        output_path: str | None = None,
        raw_output: bool = False,
        figsize: tuple[float, float] = (16.0, 12.0),
    ) -> str | None:
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
            Passed to matplotlib if `show` == True.

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
            except Exception as exc:
                raise ImportError(
                    "Graphviz dot binary should be on your PATH and matplotlib should"
                    " be installed to show graph."
                ) from exc
        try:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Graphviz dot binary should be on your PATH and matplotlib should be"
                " installed to show graph."
            ) from None
        dot = self._ldf.to_dot(optimized)
        if raw_output:
            return dot
        with tempfile.TemporaryDirectory() as tmpdir_name:
            dot_path = os.path.join(tmpdir_name, "dot")
            with open(dot_path, "w", encoding="utf8") as f:
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
        Inspect a node in the computation graph.

        Print the value that this node in the computation graph evaluates to and passes
        on the value.

        Examples
        --------
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
        <polars.internals.lazyframe.frame.LazyFrame object at ...>

        """

        def inspect(s: pli.DataFrame) -> pli.DataFrame:
            print(fmt.format(s))
            return s

        return self.map(inspect, predicate_pushdown=True, projection_pushdown=True)

    def sort(
        self: LDF,
        by: str
        | pli.Expr
        | Sequence[str]
        | Sequence[pli.Expr]
        | Sequence[str | pli.Expr],
        reverse: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> LDF:
        """
        Sort the DataFrame.

        Sorting can be done by:

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
        if type(by) is str:
            return self._from_pyldf(self._ldf.sort(by, reverse, nulls_last))
        if type(reverse) is bool:
            reverse = [reverse]

        by = pli.selection_to_pyexpr_list(by)
        return self._from_pyldf(self._ldf.sort_by_exprs(by, reverse, nulls_last))

    def profile(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> tuple[pli.DataFrame, pli.DataFrame]:
        """
        Profile a LazyFrame.

        This will run the query and return a tuple
        containing the materialized DataFrame and a DataFrame that
        contains profiling information of each node that is executed.

        The units of the timings are microseconds.

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
        no_optimization
            Turn off (certain) optimizations.
        slice_pushdown
            Slice pushdown optimization.

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
            slice_pushdown,
        )
        df, timings = ldf.profile()
        return pli.wrap_df(df), pli.wrap_df(timings)

    def collect(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> pli.DataFrame:
        """
        Collect into a DataFrame.

        Note: use :func:`fetch` if you want to run your query on the first `n` rows
        only. This can be a huge time saver in debugging queries.

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
            This argument is deprecated. Please set the string cache globally.
            The argument will be ignored
        no_optimization
            Turn off (certain) optimizations.
        slice_pushdown
            Slice pushdown optimization.

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
            slice_pushdown,
        )
        return pli.wrap_df(ldf.collect())

    def fetch(
        self,
        n_rows: int = 500,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        string_cache: bool = False,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> pli.DataFrame:
        """
        Collect a small number of rows for debugging purposes.

        Fetch is like a :func:`collect` operation, but it overwrites the number of rows
        read by every scan operation. This is a utility that helps debug a query on a
        smaller number of rows.

        Note that the fetch does not guarantee the final number of rows in the
        DataFrame. Filter, join operations and a lower number of rows available in the
        scanned file influence the final number of rows.

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
            This argument is deprecated. Please set the string cache globally.
            The argument will be ignored
        no_optimization
            Turn off optimizations.
        slice_pushdown
            Slice pushdown optimization

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
            slice_pushdown,
        )
        return pli.wrap_df(ldf.fetch(n_rows))

    def lazy(self: LDF) -> LDF:
        """
        Return lazy representation, i.e. itself.

        Useful for writing code that expects either a :class:`DataFrame` or
        :class:`LazyFrame`.

        Returns
        -------
        LazyFrame

        """
        return self

    def cache(self: LDF) -> LDF:
        """Cache the result once the execution of the physical plan hits this node."""
        return self._from_pyldf(self._ldf.cache())

    def cleared(self) -> LazyFrame:
        """
        Create an empty copy of the current LazyFrame.

        The copy has an identical schema but no data.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... ).lazy()
        >>> df.cleared().fetch()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        """
        return pli.DataFrame(columns=self.schema).lazy()

    def clone(self: LDF) -> LDF:
        """
        Very cheap deepcopy/clone.

        See Also
        --------
        cleared : Create an empty copy of the current LazyFrame, with identical
            schema but no data.

        """
        return self._from_pyldf(self._ldf.clone())

    def filter(self: LDF, predicate: pli.Expr | str | pli.Series | list[bool]) -> LDF:
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
        if isinstance(predicate, list):
            predicate = pli.Series(predicate)

        return self._from_pyldf(
            self._ldf.filter(
                pli.expr_to_lit_or_expr(predicate, str_to_lit=False)._pyexpr
            )
        )

    def select(
        self: LDF,
        exprs: str | pli.Expr | pli.Series | Sequence[str | pli.Expr | pli.Series],
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
        self: LDF,
        by: str | list[str] | pli.Expr | list[pli.Expr],
        maintain_order: bool = False,
    ) -> LazyGroupBy[LDF]:
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more
            expensive than a default groupby.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()

        The following does NOT work:
        # df.groupby("a")["b"].sum().collect()
        #                ^^^^ TypeError: 'LazyGroupBy' object is not subscriptable
        instead, use .agg():
        >>> df.groupby(by="a", maintain_order=True).agg(pl.col("b").sum()).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 11  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ c   ┆ 6   │
        └─────┴─────┘

        """
        pyexprs_by = selection_to_pyexpr_list(by)
        lgb = self._ldf.groupby(pyexprs_by, maintain_order)
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    def groupby_rolling(
        self: LDF,
        index_column: str,
        period: str,
        offset: str | None = None,
        closed: ClosedWindow = "right",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ) -> LazyGroupBy[LDF]:
        """
        Create rolling groups based on a time column.

        Also works for index values of type Int32 or Int64.

        Different from a ``dynamic_groupby`` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals use
        *groupby_dynamic*

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
            This column must be sorted in ascending order. If not the output will not
            make sense.

            In case of a rolling groupby on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define whether the temporal window interval is closed or not.
        by
            Also group by this column/these columns

        See Also
        --------
        groupby_dynamic

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
        │ dt                  ┆ sum_a ┆ min_a ┆ max_a │
        │ ---                 ┆ ---   ┆ ---   ┆ ---   │
        │ datetime[μs]        ┆ i64   ┆ i64   ┆ i64   │
        ╞═════════════════════╪═══════╪═══════╪═══════╡
        │ 2020-01-01 13:45:48 ┆ 3     ┆ 3     ┆ 3     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-01 16:42:13 ┆ 10    ┆ 3     ┆ 7     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-01 16:45:09 ┆ 15    ┆ 3     ┆ 7     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-02 18:12:48 ┆ 24    ┆ 3     ┆ 9     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-03 19:45:32 ┆ 11    ┆ 2     ┆ 9     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2020-01-08 23:16:43 ┆ 1     ┆ 1     ┆ 1     │
        └─────────────────────┴───────┴───────┴───────┘

        """
        if offset is None:
            offset = f"-{period}"
        pyexprs_by = [] if by is None else selection_to_pyexpr_list(by)

        lgb = self._ldf.groupby_rolling(
            index_column, period, offset, closed, pyexprs_by
        )
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    def groupby_dynamic(
        self: LDF,
        index_column: str,
        every: str,
        period: str | None = None,
        offset: str | None = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: ClosedWindow = "left",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ) -> LazyGroupBy[LDF]:
        """
        Group based on a time value (or index value of type Int32, Int64).

        Time windows are calculated and rows are assigned to windows. Different from a
        normal groupby is that a row can be member of multiple groups. The time/index
        window could be seen as a rolling window, with a window size determined by
        dates/times/values instead of slots in the DataFrame.

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
            This column must be sorted in ascending order. If not the output will not
            make sense.

            In case of a dynamic groupby on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it is equal to 'every'
        offset
            offset of the window if None and period is None it will be equal to negative
            `every`
        truncate
            truncate the time value to the window lower bound
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_bound" and
            "_upper_bound" columns. This will impact performance because it's harder to
            parallelize
        closed : {'right', 'left', 'both', 'none'}
            Define whether the temporal window interval is closed or not.
        by
            Also group by this column/these columns

        See Also
        --------
        groupby_rolling

        """
        if offset is None:
            if period is None:
                offset = f"-{every}"
            else:
                offset = "0ns"
        if period is None:
            period = every
        pyexprs_by = [] if by is None else selection_to_pyexpr_list(by)
        lgb = self._ldf.groupby_dynamic(
            index_column,
            every,
            period,
            offset,
            truncate,
            include_boundaries,
            closed,
            pyexprs_by,
        )
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    def join_asof(
        self: LDF,
        other: LazyFrame,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | list[str] | None = None,
        by_right: str | list[str] | None = None,
        by: str | list[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
        tolerance: str | int | float | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LDF:
        """
        Perform an asof join.

        This is similar to a left-join except that we match on nearest key rather than
        equal keys.

        Both DataFrames must be sorted by the join_asof key.

        For each row in the left DataFrame:

          - A "backward" search selects the last row in the right DataFrame whose
            'on' key is less than or equal to the left's key.

          - A "forward" search selects the first row in the right DataFrame whose
            'on' key is greater than or equal to the left's key.

        The default is "backward".

        Parameters
        ----------
        other
            Lazy DataFrame to join with.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be
            None.
        by
            Join on these columns before doing asof join.
        by_left
            Join on these columns before doing asof join.
        by_right
            Join on these columns before doing asof join.
        strategy : {'backward', 'forward'}
            Join strategy.
        suffix
            Suffix to append to columns with a duplicate name.
        tolerance
            Numeric tolerance. By setting this the join will only be done if the near
            keys are within this distance. If an asof join is done on columns of dtype
            "Date", "Datetime", "Duration" or "Time" you use the following string
            language:

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
            Allow the physical plan to optionally evaluate the computation of both
            DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to
            the join in parallel.

        """
        if not isinstance(other, LazyFrame):
            raise ValueError(f"Expected a `LazyFrame` as join table, got {type(other)}")

        if isinstance(on, str):
            left_on = on
            right_on = on

        if left_on is None or right_on is None:
            raise ValueError("You should pass the column to join on as an argument.")

        by_left_: list[str] | None
        if isinstance(by_left, str):
            by_left_ = [by_left]
        else:
            by_left_ = by_left

        by_right_: list[str] | None
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

        tolerance_str: str | None = None
        tolerance_num: float | int | None = None
        if isinstance(tolerance, str):
            tolerance_str = tolerance
        else:
            tolerance_num = tolerance

        return self._from_pyldf(
            self._ldf.join_asof(
                other._ldf,
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
        other: LazyFrame,
        left_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        right_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        how: JoinStrategy = "inner",
        suffix: str = "_right",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LDF:
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        other
            Lazy DataFrame to join with.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be
            None.
        how : {'inner', 'left', 'outer', 'semi', 'anti', 'cross'}
            Join strategy.
        suffix
            Suffix to append to columns with a duplicate name.
        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both
            DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to
            the join in parallel.

        See Also
        --------
        join_asof

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
        if not isinstance(other, LazyFrame):
            raise ValueError(f"Expected a `LazyFrame` as join table, got {type(other)}")

        if how == "cross":
            return self._from_pyldf(
                self._ldf.join(
                    other._ldf, [], [], allow_parallel, force_parallel, how, suffix
                )
            )

        if on is not None:
            pyexprs = selection_to_pyexpr_list(on)
            pyexprs_left = pyexprs
            pyexprs_right = pyexprs
        elif left_on is not None and right_on is not None:
            pyexprs_left = selection_to_pyexpr_list(left_on)
            pyexprs_right = selection_to_pyexpr_list(right_on)
        else:
            raise ValueError("must specify `on` OR `left_on` and `right_on`")

        return self._from_pyldf(
            self._ldf.join(
                other._ldf,
                pyexprs_left,
                pyexprs_right,
                allow_parallel,
                force_parallel,
                how,
                suffix,
            )
        )

    def with_columns(
        self: LDF,
        exprs: pli.Expr | pli.Series | Sequence[pli.Expr | pli.Series] | None = None,
        **named_exprs: pli.Expr | pli.Series | str,
    ) -> LDF:
        """
        Add or overwrite multiple columns in a DataFrame.

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns.
        **named_exprs
            Named column Expressions, provided as kwargs.

        Examples
        --------
        >>> ldf = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... ).lazy()
        >>> ldf.with_columns(
        ...     [
        ...         (pl.col("a") ** 2).alias("a^2"),
        ...         (pl.col("b") / 2).alias("b/2"),
        ...         (pl.col("c").is_not()).alias("not c"),
        ...     ]
        ... ).collect()
        shape: (4, 6)
        ┌─────┬──────┬───────┬──────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ a^2  ┆ b/2  ┆ not c │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪══════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1.0  ┆ 0.25 ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  ┆ 4.0  ┆ 2.0  ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false ┆ 9.0  ┆ 5.0  ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  ┆ 16.0 ┆ 6.5  ┆ false │
        └─────┴──────┴───────┴──────┴──────┴───────┘

        >>> # Support for kwarg expressions is considered EXPERIMENTAL.
        >>> # Currently requires opt-in via `pl.Config` boolean flag:
        >>>
        >>> pl.Config.with_columns_kwargs = True
        >>> ldf.with_columns(
        ...     d=pl.col("a") * pl.col("b"),
        ...     e=pl.col("c").is_not(),
        ...     f="foo",
        ... ).collect()
        shape: (4, 6)
        ┌─────┬──────┬───────┬──────┬───────┬─────┐
        │ a   ┆ b    ┆ c     ┆ d    ┆ e     ┆ f   │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---   ┆ --- │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ bool  ┆ str │
        ╞═════╪══════╪═══════╪══════╪═══════╪═════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 0.5  ┆ false ┆ foo │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  ┆ 8.0  ┆ false ┆ foo │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false ┆ 30.0 ┆ true  ┆ foo │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  ┆ 52.0 ┆ false ┆ foo │
        └─────┴──────┴───────┴──────┴───────┴─────┘

        """
        if named_exprs and not Config.with_columns_kwargs:
            raise RuntimeError(
                "**kwargs support is experimental; requires opt-in via"
                " `pl.Config.with_columns_kwargs = True`"
            )
        elif exprs is None and not named_exprs:
            raise ValueError("Expected at least one of 'exprs' or **named_exprs")

        exprs = (
            []
            if exprs is None
            else ([exprs] if isinstance(exprs, pli.Expr) else list(exprs))
        )
        exprs.extend(
            (pli.lit(expr).alias(name) if isinstance(expr, str) else expr.alias(name))
            for name, expr in named_exprs.items()
        )
        pyexprs = []
        for e in exprs:
            if isinstance(e, pli.Expr):
                pyexprs.append(e._pyexpr)
            elif isinstance(e, pli.Series):
                pyexprs.append(pli.lit(e)._pyexpr)
            else:
                raise ValueError(f"Expected an expression, got {e}")

        return self._from_pyldf(self._ldf.with_columns(pyexprs))

    @typing.no_type_check
    def with_context(self, other: LDF | list[LDF]) -> LDF:
        """
        Add an external context to the computation graph.

        This allows expressions to also access columns from DataFrames
        that are not part of this one.

        Parameters
        ----------
        other
            One or multiple LazyFrames as external context

        """
        if not isinstance(other, list):
            other = [other]

        return self._from_pyldf(self._ldf.with_context([lf._ldf for lf in other]))

    def with_column(self: LDF, column: pli.Series | pli.Expr) -> LDF:
        """
        Add or overwrite column in a DataFrame.

        Parameters
        ----------
        column
            Expression that evaluates to column or a Series to use.

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
        if not isinstance(column, (pli.Expr, pli.Series)):
            raise TypeError(
                "`with_column` expects a single Expr or Series. "
                "Consider using `with_columns` if you need multiple columns."
            )
        return self.with_columns([column])

    def drop(self: LDF, columns: str | list[str]) -> LDF:
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

    def rename(self: LDF, mapping: dict[str, str]) -> LDF:
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
        """Reverse the DataFrame."""
        return self._from_pyldf(self._ldf.reverse())

    def shift(self: LDF, periods: int) -> LDF:
        """
        Shift the values by a given period.

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
        fill_value: pli.Expr | int | str | float,
    ) -> LDF:
        """
        Shift the values by a given period and fill the resulting null values.

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

    def slice(self: LDF, offset: int, length: int | None = None) -> LDF:
        """
        Get a slice of this DataFrame.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

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
        if length and length < 0:
            raise ValueError(
                f"Negative slice lengths ({length}) are invalid for LazyFrame"
            )
        return self._from_pyldf(self._ldf.slice(offset, length))

    def limit(self: LDF, n: int = 5) -> LDF:
        """
        Get the first `n` rows.

        Alias for :func:`LazyFrame.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        Notes
        -----
        Consider using the :func:`fetch` operation if you only want to test your
        query. The :func:`fetch` operation will load the first `n` rows at the scan
        level, whereas the :func:`head`/:func:`limit` are applied at the end.

        """
        return self.head(n)

    def head(self: LDF, n: int = 5) -> LDF:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Notes
        -----
        Consider using the :func:`fetch` operation if you only want to test your
        query. The :func:`fetch` operation will load the first `n` rows at the scan
        level, whereas the :func:`head`/:func:`limit` are applied at the end.

        """
        return self.slice(0, n)

    def tail(self: LDF, n: int = 5) -> LDF:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows.

        """
        return self._from_pyldf(self._ldf.tail(n))

    def last(self: LDF) -> LDF:
        """Get the last row of the DataFrame."""
        return self.tail(1)

    def first(self: LDF) -> LDF:
        """Get the first row of the DataFrame."""
        return self.slice(0, 1)

    def with_row_count(self: LDF, name: str = "row_nr", offset: int = 0) -> LDF:
        """
        Add a column at index 0 that counts the rows.

        Parameters
        ----------
        name
            Name of the column to add.
        offset
            Start the row count at this offset.

        Warnings
        --------
        This can have a negative effect on query performance.
        This may, for instance, block predicate pushdown optimization.

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

    def take_every(self: LDF, n: int) -> LDF:
        """
        Take every nth row in the LazyFrame and return as a new LazyFrame.

        Examples
        --------
        >>> s = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}).lazy()
        >>> s.take_every(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 7   │
        └─────┴─────┘

        """
        return self.select(pli.col("*").take_every(n))

    def fill_null(
        self: LDF,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> LDF:
        """
        Fill null values using the specified value or strategy.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the 'forward' or
            'backward' strategy.

        """
        return self.select(pli.all().fill_null(value, strategy, limit))

    def fill_nan(self: LDF, fill_value: int | str | float | pli.Expr | None) -> LDF:
        """
        Fill floating point NaN values.

        Parameters
        ----------
        fill_value
            Value to fill the NaN values with.

        Warnings
        --------
        Note that floating point NaN (Not a Number) are not missing values!
        To replace missing values, use :func:`fill_null` instead.

        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return self._from_pyldf(self._ldf.fill_nan(fill_value._pyexpr))

    def std(self: LDF, ddof: int = 1) -> LDF:
        """
        Aggregate the columns in the DataFrame to their standard deviation value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.std().collect()
        shape: (1, 2)
        ┌──────────┬─────┐
        │ a        ┆ b   │
        │ ---      ┆ --- │
        │ f64      ┆ f64 │
        ╞══════════╪═════╡
        │ 1.290994 ┆ 0.5 │
        └──────────┴─────┘

        """
        return self._from_pyldf(self._ldf.std(ddof))

    def var(self: LDF, ddof: int = 1) -> LDF:
        """
        Aggregate the columns in the DataFrame to their variance value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.var().collect()
        shape: (1, 2)
        ┌──────────┬──────┐
        │ a        ┆ b    │
        │ ---      ┆ ---  │
        │ f64      ┆ f64  │
        ╞══════════╪══════╡
        │ 1.666667 ┆ 0.25 │
        └──────────┴──────┘

        """
        return self._from_pyldf(self._ldf.var(ddof))

    def max(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.max().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.max())

    def min(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.min().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 1   │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.min())

    def sum(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their sum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.sum().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 10  ┆ 5   │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.sum())

    def mean(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their mean value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.mean().collect()
        shape: (1, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ f64 ┆ f64  │
        ╞═════╪══════╡
        │ 2.5 ┆ 1.25 │
        └─────┴──────┘

        """
        return self._from_pyldf(self._ldf.mean())

    def median(self: LDF) -> LDF:
        """
        Aggregate the columns in the DataFrame to their median value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.median().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 2.5 ┆ 1.0 │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.median())

    def quantile(
        self: LDF, quantile: float, interpolation: InterpolationMethod = "nearest"
    ) -> LDF:
        """
        Aggregate the columns in the DataFrame to their quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 1]}).lazy()
        >>> df.quantile(0.7).collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 3.0 ┆ 1.0 │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.quantile(quantile, interpolation))

    def explode(
        self: LDF,
        columns: str | list[str] | pli.Expr | list[pli.Expr],
    ) -> LDF:
        """
        Explode lists to long format.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["a", "a", "b", "c"],
        ...         "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]],
        ...     }
        ... ).lazy()
        >>> df.explode("numbers").collect()
        shape: (8, 2)
        ┌─────────┬─────────┐
        │ letters ┆ numbers │
        │ ---     ┆ ---     │
        │ str     ┆ i64     │
        ╞═════════╪═════════╡
        │ a       ┆ 1       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ 2       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ 3       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ b       ┆ 4       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ b       ┆ 5       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ 6       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ 7       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ 8       │
        └─────────┴─────────┘

        """
        columns = pli.selection_to_pyexpr_list(columns)
        return self._from_pyldf(self._ldf.explode(columns))

    def unique(
        self: LDF,
        maintain_order: bool = True,
        subset: str | list[str] | None = None,
        keep: UniqueKeepStrategy = "first",
    ) -> LDF:
        """
        Drop duplicate rows from this DataFrame.

        Note that this fails if there is a column of type `List` in the DataFrame or
        subset.

        Parameters
        ----------
        maintain_order
            Keep the same order as the original DataFrame. This requires more work to
            compute.
        subset
            Subset to use to compare rows.
        keep : {'first', 'last'}
            Which of the duplicate rows to keep.

        Returns
        -------
        DataFrame with unique rows

        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return self._from_pyldf(self._ldf.unique(maintain_order, subset, keep))

    def drop_nulls(self: LDF, subset: list[str] | str | None = None) -> LDF:
        """
        Drop rows with null values from this LazyFrame.

        Parameters
        ----------
        subset
            Subset of column(s) on which ``drop_nulls`` will be applied.

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

        Below are some example snippets that show how you could drop null values based
        on other conditions:

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
        id_vars: str | list[str] | None = None,
        value_vars: str | list[str] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> LDF:
        """
        Unpivot a DataFrame from wide to long format.

        Optionally leaves identifiers set.

        This function is useful to massage a DataFrame into a format where one or more
        columns are identifier variables (id_vars), while all other columns, considered
        measured variables (value_vars), are "unpivoted" to the row axis, leaving just
        two non-identifier columns, 'variable' and 'value'.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables.
        value_vars
            Values to use as identifier variables.
            If `value_vars` is empty all columns that are not in `id_vars` will be used.
        variable_name
            Name to give to the `value` column. Defaults to "variable"
        value_name
            Name to give to the `value` column. Defaults to "value"

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
        return self._from_pyldf(
            self._ldf.melt(id_vars, value_vars, value_name, variable_name)
        )

    def map(
        self: LDF,
        f: Callable[[pli.DataFrame], pli.DataFrame],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        slice_pushdown: bool = True,
        no_optimizations: bool = False,
        schema: None | Schema = None,
        validate_output_schema: bool = True,
    ) -> LDF:
        """
        Apply a custom function.

        It is important that the function returns a Polars DataFrame.

        Parameters
        ----------
        f
            Lambda/ function to apply.
        predicate_pushdown
            Allow predicate pushdown optimization to pass this node.
        projection_pushdown
            Allow projection pushdown optimization to pass this node.
        slice_pushdown
            Allow slice pushdown optimization to pass this node.
        no_optimizations
            Turn off all optimizations past this point.
        schema
            Output schema of the function, if set to ``None`` we assume that the schema
            will remain unchanged by the applied function.
        validate_output_schema
            It is paramount that polars' schema is correct. This flag will ensure that
            the output schema of this function will be checked with the expected schema.
            Setting this to ``False`` will not do this check, but may lead to hard to
            debug bugs.

        Warnings
        --------
        The ``schema`` of a `LazyFrame` must always be correct. It is up to the caller
        of this function to ensure that this invariant is upheld.

        It is important that the optimization flags are correct. If the custom function
        for instance does an aggregation of a column, ``predicate_pushdown`` should not
        be allowed, as this prunes rows and will influence your aggregation results.

        """
        if no_optimizations:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False
        return self._from_pyldf(
            self._ldf.map(
                f,
                predicate_pushdown,
                projection_pushdown,
                slice_pushdown,
                schema,
                validate_output_schema,
            )
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

    def unnest(self: LDF, names: str | list[str]) -> LDF:
        """
        Decompose a struct into its fields.

        The fields will be inserted into the `DataFrame` on the location of the
        `struct` type.

        Parameters
        ----------
        names
           Names of the struct columns that will be decomposed by its fields

        Examples
        --------
        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "before": ["foo", "bar"],
        ...             "t_a": [1, 2],
        ...             "t_b": ["a", "b"],
        ...             "t_c": [True, None],
        ...             "t_d": [[1, 2], [3]],
        ...             "after": ["baz", "womp"],
        ...         }
        ...     )
        ...     .lazy()
        ...     .select(
        ...         ["before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after"]
        ...     )
        ... )
        >>> df.fetch()
        shape: (2, 3)
        ┌────────┬─────────────────────┬───────┐
        │ before ┆ t_struct            ┆ after │
        │ ---    ┆ ---                 ┆ ---   │
        │ str    ┆ struct[4]           ┆ str   │
        ╞════════╪═════════════════════╪═══════╡
        │ foo    ┆ {1,"a",true,[1, 2]} ┆ baz   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ bar    ┆ {2,"b",null,[3]}    ┆ womp  │
        └────────┴─────────────────────┴───────┘
        >>> df.unnest("t_struct").fetch()
        shape: (2, 6)
        ┌────────┬─────┬─────┬──────┬───────────┬───────┐
        │ before ┆ t_a ┆ t_b ┆ t_c  ┆ t_d       ┆ after │
        │ ---    ┆ --- ┆ --- ┆ ---  ┆ ---       ┆ ---   │
        │ str    ┆ i64 ┆ str ┆ bool ┆ list[i64] ┆ str   │
        ╞════════╪═════╪═════╪══════╪═══════════╪═══════╡
        │ foo    ┆ 1   ┆ a   ┆ true ┆ [1, 2]    ┆ baz   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ bar    ┆ 2   ┆ b   ┆ null ┆ [3]       ┆ womp  │
        └────────┴─────┴─────┴──────┴───────────┴───────┘

        """
        if isinstance(names, str):
            names = [names]
        return self._from_pyldf(self._ldf.unnest(names))

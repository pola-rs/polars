from __future__ import annotations

import os
import subprocess
import sys
import typing
from datetime import date, datetime, time, timedelta
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    NoReturn,
    Sequence,
    TypeVar,
    overload,
)

from polars import internals as pli
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    N_INFER_DEFAULT,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    PolarsDataType,
    SchemaDict,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    py_type_to_dtype,
)
from polars.dependencies import pyarrow as pa
from polars.internals import selection_to_pyexpr_list
from polars.internals.lazyframe.groupby import LazyGroupBy
from polars.internals.slice import LazyPolarsSlice
from polars.internals.type_aliases import PythonLiteral
from polars.utils import (
    _in_notebook,
    _prepare_row_count_args,
    _process_null_values,
    _timedelta_to_pl_duration,
    deprecate_nonkeyword_arguments,
    normalise_filepath,
    redirect,
)

try:
    from polars.polars import PyLazyFrame

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import Concatenate, ParamSpec

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from polars.internals.type_aliases import (
        AsofJoinStrategy,
        ClosedInterval,
        CsvEncoding,
        FillNullStrategy,
        JoinStrategy,
        ParallelStrategy,
        PolarsExprType,
        RollingInterpolationMethod,
        StartBy,
        UniqueKeepStrategy,
    )

    T = TypeVar("T")
    P = ParamSpec("P")


def wrap_ldf(ldf: PyLazyFrame) -> LazyFrame:
    return LazyFrame._from_pyldf(ldf)


@redirect({"with_column": "with_columns"})
class LazyFrame:
    """
    Representation of a Lazy computation graph/query against a DataFrame.

    Notes
    -----
    LazyFrames are instantiated by calling :meth:`~DataFrame.lazy()` on an
    existing DataFrame; they are also created when calling the various "scan"
    :doc:`IO methods </reference/io>`, and are the preferred way to operate
    on data with polars.

    >>> ldf = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()

    """

    _ldf: PyLazyFrame
    _accessors: set[str] = set()

    @classmethod
    def _from_pyldf(cls, ldf: PyLazyFrame) -> Self:
        self = cls.__new__(cls)
        self._ldf = ldf
        return self

    @classmethod
    def _scan_csv(
        cls,
        file: str,
        has_header: bool = True,
        sep: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = r'"',
        skip_rows: int = 0,
        dtypes: SchemaDict | None = None,
        null_values: str | list[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        cache: bool = True,
        with_column_names: Callable[[list[str]], list[str]] | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        n_rows: int | None = None,
        encoding: CsvEncoding = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        parse_dates: bool = False,
        eol_char: str = "\n",
    ) -> Self:
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
            missing_utf8_is_empty_string,
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
        cls,
        file: str,
        n_rows: int | None = None,
        cache: bool = True,
        parallel: ParallelStrategy = "auto",
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        storage_options: dict[str, object] | None = None,
        low_memory: bool = False,
    ) -> Self:
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
            cloud_options=storage_options,
        )
        return self

    @classmethod
    def _scan_ipc(
        cls,
        file: str | Path,
        n_rows: int | None = None,
        cache: bool = True,
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        storage_options: dict[str, object] | None = None,
        memory_map: bool = True,
    ) -> Self:
        """
        Lazily read from an Arrow IPC (Feather v2) file.

        Use ``pl.scan_ipc`` to dispatch to this method.

        See Also
        --------
        polars.io.scan_ipc

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)

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
        cls,
        file: str,
        infer_schema_length: int | None = None,
        batch_size: int | None = None,
        n_rows: int | None = None,
        low_memory: bool = False,
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
    ) -> Self:
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
    def read_json(cls, file: str | Path | IOBase) -> LazyFrame:
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
            file = normalise_filepath(file)

        return wrap_ldf(PyLazyFrame.read_json(file))

    @classmethod
    def _scan_python_function(
        cls,
        schema: pa.schema | dict[str, PolarsDataType],
        scan_fn: bytes,
        pyarrow: bool = False,
    ) -> LazyFrame:
        self = cls.__new__(cls)
        if isinstance(schema, dict):
            self._ldf = PyLazyFrame.scan_from_python_function_pl_schema(
                [(name, dt) for name, dt in schema.items()], scan_fn, pyarrow
            )
        else:
            self._ldf = PyLazyFrame.scan_from_python_function_arrow_schema(
                list(schema), scan_fn, pyarrow
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
    def dtypes(self) -> list[PolarsDataType]:
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
        [Int64, Float64, Utf8]

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        """
        return self._ldf.dtypes()

    @property
    def schema(self) -> SchemaDict:
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
        {'foo': Int64, 'bar': Float64, 'ham': Utf8}

        """
        return self._ldf.schema()

    @property
    def width(self) -> int:
        """
        Get the width of the LazyFrame.

        Examples
        --------
        >>> lf = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]}).lazy()
        >>> lf.width
        2

        """
        return self._ldf.width()

    def __bool__(self) -> NoReturn:
        raise ValueError(
            "The truth value of a LazyFrame is ambiguous; consequently it "
            "cannot be used in boolean context with and/or/not operators. "
        )

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __copy__(self) -> Self:
        return self.clone()

    def __deepcopy__(self, memo: None = None) -> Self:
        return self.clone()

    def __getitem__(self, item: int | range | slice) -> LazyFrame:
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

    def __repr__(self) -> str:
        # don't expose internal/private classpath
        return f"<polars.{self.__class__.__name__} object at 0x{id(self):X}>"

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
"""

    @overload
    def write_json(self, file: None = ...) -> str:
        ...

    @overload
    def write_json(self, file: IOBase | str | Path) -> None:
        ...

    def write_json(self, file: IOBase | str | Path | None = None) -> str | None:
        """
        Write the logical plan of this LazyFrame to a file or string in JSON format.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to ``None``
            (default), the output is returned as a string instead.

        See Also
        --------
        LazyFrame.read_json

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... ).lazy()
        >>> df.write_json()
        '{"DataFrameScan":{"df":{"columns":[{"name":"foo","datatype":"Int64","values":[1,2,3]},{"name":"bar","datatype":"Int64","values":[6,7,8]}]},"schema":{"inner":{"foo":"Int64","bar":"Int64"}},"output_schema":null,"projection":null,"selection":null}}'

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if file is None or to_string_io:
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

    def pipe(
        self,
        func: Callable[Concatenate[LazyFrame, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """
        Offers a structured way to apply a sequence of user-defined functions (UDFs).

        Parameters
        ----------
        func
            Callable; will receive the frame as the first parameter,
            followed by any given args/kwargs.
        args
            Arguments to pass to the UDF.
        kwargs
            Keyword arguments to pass to the UDF.

        Examples
        --------
        >>> def cast_str_to_int(data, col_name):
        ...     return data.with_columns(pl.col(col_name).cast(pl.Int64))
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
        │ 2   ┆ 20  │
        │ 3   ┆ 30  │
        │ 4   ┆ 40  │
        └─────┴─────┘

        >>> df = pl.DataFrame({"b": [1, 2], "a": [3, 4]})
        >>> df
        shape: (2, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        >>> df.lazy().pipe(lambda tdf: tdf.select(sorted(tdf.columns))).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 1   │
        │ 4   ┆ 2   │
        └─────┴─────┘

        """
        return func(self, *args, **kwargs)

    def describe_plan(self, *, optimized: bool = False) -> str:
        """
        Create a string representation of the unoptimized query plan.

        Parameters
        ----------
        optimized
            Return an optimized query plan. Defaults to `False`.
            Use ``describe_optimized_plan`` to control
            the optimization flags.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> df.groupby("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).describe_plan()  # doctest: +SKIP

        """
        if optimized:
            return self._ldf.describe_optimized_plan()
        return self._ldf.describe_plan()

    def describe_optimized_plan(
        self,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        common_subplan_elimination: bool = True,
        streaming: bool = False,
    ) -> str:
        """Create a string representation of the optimized query plan."""
        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            common_subplan_elimination,
            streaming,
        )

        return ldf.describe_optimized_plan()

    def show_graph(
        self,
        optimized: bool = True,
        *,
        show: bool = True,
        output_path: str | None = None,
        raw_output: bool = False,
        figsize: tuple[float, float] = (16.0, 12.0),
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        common_subplan_elimination: bool = True,
        streaming: bool = False,
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
            Return dot syntax. This cannot be combined with `show` and/or `output_path`.
        figsize
            Passed to matplotlib if `show` == True.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        common_subplan_elimination
            Will try to cache branching subplans that occur on self-joins or unions.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> df.groupby("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).show_graph()  # doctest: +SKIP

        """
        _ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            common_subplan_elimination,
            streaming,
        )

        dot = _ldf.to_dot(optimized)

        if raw_output:
            # we do not show a graph, nor save a graph to disk
            return dot

        output_type = "svg" if _in_notebook() else "png"

        try:
            graph = subprocess.check_output(
                ["dot", "-Nshape=box", "-T" + output_type], input=f"{dot}".encode()
            )
        except (ImportError, FileNotFoundError):
            raise ImportError("Graphviz dot binary should be on your PATH") from None

        if output_path:
            with Path(output_path).open(mode="wb") as file:
                file.write(graph)

        if not show:
            return None

        if _in_notebook():
            from IPython.display import SVG, display

            return display(SVG(graph))
        else:
            try:
                import matplotlib.image as mpimg
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError(
                    "matplotlib should be installed to show graph."
                ) from None
            plt.figure(figsize=figsize)
            img = mpimg.imread(BytesIO(graph))
            plt.imshow(img)
            plt.show()
            return None

    def inspect(self, fmt: str = "{}") -> Self:
        """
        Inspect a node in the computation graph.

        Print the value that this node in the computation graph evaluates to and passes
        on the value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 1, -2, 3]}).lazy()
        >>> (
        ...     df.select(pl.col("foo").cumsum().alias("bar"))
        ...     .inspect()  # print the node before the filter
        ...     .filter(pl.col("bar") == pl.col("foo"))
        ... )  # doctest: +ELLIPSIS
        <polars.LazyFrame object at ...>

        """

        def inspect(s: pli.DataFrame) -> pli.DataFrame:
            print(fmt.format(s))
            return s

        return self.map(inspect, predicate_pushdown=True, projection_pushdown=True)

    def sort(
        self,
        by: (
            str
            | pli.Expr
            | Sequence[str]
            | Sequence[pli.Expr]
            | Sequence[str | pli.Expr]
        ),
        *,
        reverse: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, None],
        ...         "bar": [6.0, 7.0, 8.0, 9.0],
        ...         "ham": ["a", "b", "c", "d"],
        ...     }
        ... ).lazy()
        >>> df.sort("foo").collect()
        shape: (4, 3)
        ┌──────┬─────┬─────┐
        │ foo  ┆ bar ┆ ham │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ null ┆ 9.0 ┆ d   │
        │ 1    ┆ 6.0 ┆ a   │
        │ 2    ┆ 7.0 ┆ b   │
        │ 3    ┆ 8.0 ┆ c   │
        └──────┴─────┴─────┘
        >>> df.sort("foo", nulls_last=True).collect()
        shape: (4, 3)
        ┌──────┬─────┬─────┐
        │ foo  ┆ bar ┆ ham │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 1    ┆ 6.0 ┆ a   │
        │ 2    ┆ 7.0 ┆ b   │
        │ 3    ┆ 8.0 ┆ c   │
        │ null ┆ 9.0 ┆ d   │
        └──────┴─────┴─────┘
        >>> df.sort("foo", reverse=True).collect()
        shape: (4, 3)
        ┌──────┬─────┬─────┐
        │ foo  ┆ bar ┆ ham │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 3    ┆ 8.0 ┆ c   │
        │ 2    ┆ 7.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 9.0 ┆ d   │
        └──────┴─────┴─────┘

        **Sort by multiple columns.**
        For multiple columns we can also use expression syntax.

        >>> df.sort(
        ...     [pl.col("foo"), pl.col("bar") ** 2],
        ...     reverse=[True, False],
        ... ).collect()
        shape: (4, 3)
        ┌──────┬─────┬─────┐
        │ foo  ┆ bar ┆ ham │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 3    ┆ 8.0 ┆ c   │
        │ 2    ┆ 7.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 9.0 ┆ d   │
        └──────┴─────┴─────┘

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
        common_subplan_elimination: bool = True,
        show_plot: bool = False,
        truncate_nodes: int = 0,
        figsize: tuple[int, int] = (18, 8),
        streaming: bool = False,
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
        common_subplan_elimination
            Will try to cache branching subplans that occur on self-joins or unions.
        show_plot
            Show a gantt chart of the profiling result
        truncate_nodes
            Truncate the label lengths in the gantt chart to this number of
            characters.
        figsize
            matplotlib figsize of the profiling plot
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> df.groupby("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).profile()  # doctest: +SKIP
        (shape: (3, 3)
         ┌─────┬─────┬─────┐
         │ a   ┆ b   ┆ c   │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ i64 │
         ╞═════╪═════╪═════╡
         │ a   ┆ 4   ┆ 10  │
         │ b   ┆ 11  ┆ 10  │
         │ c   ┆ 6   ┆ 1   │
         └─────┴─────┴─────┘,
         shape: (3, 3)
         ┌────────────────────────┬───────┬──────┐
         │ node                   ┆ start ┆ end  │
         │ ---                    ┆ ---   ┆ ---  │
         │ str                    ┆ u64   ┆ u64  │
         ╞════════════════════════╪═══════╪══════╡
         │ optimization           ┆ 0     ┆ 5    │
         │ groupby_partitioned(a) ┆ 5     ┆ 470  │
         │ sort(a)                ┆ 475   ┆ 1964 │
         └────────────────────────┴───────┴──────┘)

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
            common_subplan_elimination,
            streaming,
        )
        df, timings = ldf.profile()
        (df, timings) = pli.wrap_df(df), pli.wrap_df(timings)

        if show_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, figsize=figsize)

                max_val = timings["end"][-1]
                timings_ = timings.reverse()

                if max_val > 1e9:
                    unit = "s"
                    timings_ = timings_.with_columns(
                        pli.col(["start", "end"]) / 1_000_000
                    )
                elif max_val > 1e6:
                    unit = "ms"
                    timings_ = timings_.with_columns(pli.col(["start", "end"]) / 1000)
                else:
                    unit = "us"
                if truncate_nodes > 0:
                    timings_ = timings_.with_columns(
                        pli.col("node").str.slice(0, truncate_nodes) + "..."
                    )

                max_in_unit = timings_["end"][0]
                ax.barh(
                    timings_["node"],
                    width=timings_["end"] - timings_["start"],
                    left=timings_["start"],
                )

                plt.title("Profiling result")
                ax.set_xlabel(f"node duration in [{unit}], total {max_in_unit}{unit}")
                ax.set_ylabel("nodes")
                plt.show()

            except ImportError:
                raise ImportError(
                    "matplotlib should be installed to show profiling plot."
                ) from None

        return df, timings

    def collect(
        self,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
        common_subplan_elimination: bool = True,
        streaming: bool = False,
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
        no_optimization
            Turn off (certain) optimizations.
        slice_pushdown
            Slice pushdown optimization.
        common_subplan_elimination
            Will try to cache branching subplans that occur on self-joins or unions.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> df.groupby("a", maintain_order=True).agg(pl.all().sum()).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 4   ┆ 10  │
        │ b   ┆ 11  ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘

        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False
            common_subplan_elimination = False

        if streaming:
            common_subplan_elimination = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            common_subplan_elimination,
            streaming,
        )
        return pli.wrap_df(ldf.collect())

    def sink_parquet(
        self,
        path: str | Path,
        *,
        compression: str = "zstd",
        compression_level: int | None = None,
        statistics: bool = False,
        row_group_size: int | None = None,
        data_pagesize_limit: int | None = None,
        maintain_order: bool = True,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> pli.DataFrame:
        """
        Persists a LazyFrame at the provided path.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        compression : {'lz4', 'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'zstd'}
            Choose "zstd" for good compression performance.
            Choose "lz4" for fast compression/decompression.
            Choose "snappy" for more backwards compatibility guarantees
            when you deal with older parquet readers.
        compression_level
            The level of compression to use. Higher compression means smaller files on
            disk.

            - "gzip" : min-level: 0, max-level: 10.
            - "brotli" : min-level: 0, max-level: 11.
            - "zstd" : min-level: 1, max-level: 22.

        statistics
            Write statistics to the parquet headers. This requires extra compute.
        row_group_size
            Size of the row groups in number of rows.
            If None (default), the chunks of the `DataFrame` are
            used. Writing in smaller chunks may reduce memory pressure and improve
            writing speeds. If None and ``use_pyarrow=True``, the row group size
            will be the minimum of the DataFrame size and 64 * 1024 * 1024.
        data_pagesize_limit
            Size limit of individual data pages.
            If not set defaults to 1024 * 1024 bytes
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will  be slightly faster.
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

        Examples
        --------
        >>> ldf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> ldf.sink_parquet("out.parquet")  # doctest: +SKIP

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
            cse=False,
            streaming=True,
        )
        return ldf.sink_parquet(
            path=path,
            compression=compression,
            compression_level=compression_level,
            statistics=statistics,
            row_group_size=row_group_size,
            data_pagesize_limit=data_pagesize_limit,
            maintain_order=maintain_order,
        )

    def sink_ipc(
        self,
        path: str | Path,
        *,
        compression: str | None = "zstd",
        maintain_order: bool = True,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> pli.DataFrame:
        """
        Persists a LazyFrame at the provided path.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        compression : {'lz4', 'zstd'}
            Choose "zstd" for good compression performance.
            Choose "lz4" for fast compression/decompression.
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will  be slightly faster.
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

        Examples
        --------
        >>> ldf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> ldf.sink_ipc("out.arrow")  # doctest: +SKIP

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
            cse=False,
            streaming=True,
        )
        return ldf.sink_ipc(
            path=path,
            compression=compression,
            maintain_order=maintain_order,
        )

    def fetch(
        self,
        n_rows: int = 500,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
        common_subplan_elimination: bool = True,
        streaming: bool = False,
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
        no_optimization
            Turn off optimizations.
        slice_pushdown
            Slice pushdown optimization
        common_subplan_elimination
            Will try to cache branching subplans that occur on self-joins or unions.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> df.groupby("a", maintain_order=True).agg(pl.all().sum()).fetch(2)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 6   │
        │ b   ┆ 2   ┆ 5   │
        └─────┴─────┴─────┘

        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False
            common_subplan_elimination = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            common_subplan_elimination,
            streaming,
        )
        return pli.wrap_df(ldf.fetch(n_rows))

    def lazy(self) -> Self:
        """
        Return lazy representation, i.e. itself.

        Useful for writing code that expects either a :class:`DataFrame` or
        :class:`LazyFrame`.

        Returns
        -------
        LazyFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> df.lazy()  # doctest: +ELLIPSIS
        <polars.LazyFrame object at ...>

        """
        return self

    def cache(self) -> Self:
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
        return pli.DataFrame(schema=self.schema).lazy()

    def clone(self) -> Self:
        """
        Very cheap deepcopy/clone.

        See Also
        --------
        cleared : Create an empty copy of the current LazyFrame, with identical
            schema but no data.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... ).lazy()
        >>> df.clone()  # doctest: +ELLIPSIS
        <polars.LazyFrame object at ...>

        """
        return self._from_pyldf(self._ldf.clone())

    def filter(self, predicate: pli.Expr | str | pli.Series | list[bool]) -> Self:
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

        Filter on an OR condition:

        >>> lf.filter((pl.col("foo") == 1) | (pl.col("ham") == "c")).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 3   ┆ 8   ┆ c   │
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
        self,
        exprs: (
            str
            | PolarsExprType
            | PythonLiteral
            | pli.Series
            | Iterable[str | PolarsExprType | PythonLiteral | pli.Series | None]
            | None
        ) = None,
        *more_exprs: str | PolarsExprType | PythonLiteral | pli.Series | None,
        **named_exprs: str | PolarsExprType | PythonLiteral | pli.Series | None,
    ) -> Self:
        """
        Select columns from this DataFrame.

        Parameters
        ----------
        exprs
            Column or columns to select. Accepts expression input. Strings are parsed
            as column names, other non-expression inputs are parsed as literals.
        *more_exprs
            Additional columns to select, specified as positional arguments.
        **named_exprs
            Additional columns to select, specified as keyword arguments. The columns
            will be renamed to the keyword used.

        Examples
        --------
        Pass the name of a column to select that column.

        >>> ldf = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> ldf.select("foo").collect()
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        Multiple columns can be selected by passing a list of column names.

        >>> ldf.select(["foo", "bar"]).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 6   │
        │ 2   ┆ 7   │
        │ 3   ┆ 8   │
        └─────┴─────┘

        Multiple columns can also be selected using positional arguments instead of a
        list. Expressions are also accepted.

        >>> ldf.select(pl.col("foo"), pl.col("bar") + 1).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        └─────┴─────┘

        Use keyword arguments to easily name your expression inputs.

        >>> ldf.select(
        ...     threshold=pl.when(pl.col("foo") > 2).then(10).otherwise(0)
        ... ).collect()
        shape: (3, 1)
        ┌───────────┐
        │ threshold │
        │ ---       │
        │ i32       │
        ╞═══════════╡
        │ 0         │
        │ 0         │
        │ 10        │
        └───────────┘

        Expressions with multiple outputs can be automatically instantiated as Structs
        by enabling the experimental setting ``Config.set_auto_structify(True)``:

        >>> from polars.datatypes import INTEGER_DTYPES
        >>> with pl.Config() as cfg:
        ...     cfg.set_auto_structify(True)  # doctest: +IGNORE_RESULT
        ...     ldf.select(
        ...         is_odd=(pl.col(INTEGER_DTYPES) % 2).suffix("_is_odd"),
        ...     ).collect()
        ...
        shape: (3, 1)
        ┌───────────┐
        │ is_odd    │
        │ ---       │
        │ struct[2] │
        ╞═══════════╡
        │ {1,0}     │
        │ {0,1}     │
        │ {1,0}     │
        └───────────┘

        """
        if exprs is None and not named_exprs:
            raise ValueError("Expected at least one of 'exprs' or **named_exprs")

        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        exprs = pli.selection_to_pyexpr_list(exprs, structify=structify)
        exprs.extend(pli.selection_to_pyexpr_list(more_exprs, structify=structify))
        exprs.extend(
            pli.expr_to_lit_or_expr(
                expr, structify=structify, name=name, str_to_lit=False
            )._pyexpr
            for name, expr in named_exprs.items()
        )
        return self._from_pyldf(self._ldf.select(exprs))

    def groupby(
        self,
        by: str | pli.Expr | Sequence[str | pli.Expr],
        *,
        maintain_order: bool = False,
    ) -> LazyGroupBy[Self]:
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
        >>> ldf = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... ).lazy()
        >>> ldf.groupby(by="a", maintain_order=True).agg(pl.col("b").sum()).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 4   │
        │ b   ┆ 11  │
        │ c   ┆ 6   │
        └─────┴─────┘

        Note that the following syntax does NOT work:

        >>> ldf.groupby("a")["b"].sum().collect()
        Traceback (most recent call last):
        ...
        TypeError: 'LazyGroupBy' object is not subscriptable

        """
        pyexprs_by = selection_to_pyexpr_list(by)
        lgb = self._ldf.groupby(pyexprs_by, maintain_order)
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    def groupby_rolling(
        self,
        index_column: str,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
    ) -> LazyGroupBy[Self]:
        """
        Create rolling groups based on a time column.

        Also works for index values of type Int32 or Int64.

        Different from a ``dynamic_groupby`` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals
        use *groupby_dynamic*.

        The `period` and `offset` arguments are created either from a timedelta, or
        by using the following string language:

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
            Define which sides of the temporal interval are closed (inclusive).
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
        >>> df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
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
        │ 2020-01-01 16:42:13 ┆ 10    ┆ 3     ┆ 7     │
        │ 2020-01-01 16:45:09 ┆ 15    ┆ 3     ┆ 7     │
        │ 2020-01-02 18:12:48 ┆ 24    ┆ 3     ┆ 9     │
        │ 2020-01-03 19:45:32 ┆ 11    ┆ 2     ┆ 9     │
        │ 2020-01-08 23:16:43 ┆ 1     ┆ 1     ┆ 1     │
        └─────────────────────┴───────┴───────┴───────┘

        """
        if offset is None:
            offset = f"-{period}"

        pyexprs_by = [] if by is None else selection_to_pyexpr_list(by)
        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)

        lgb = self._ldf.groupby_rolling(
            index_column, period, offset, closed, pyexprs_by
        )
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    def groupby_dynamic(
        self,
        index_column: str,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        by: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        start_by: StartBy = "window",
    ) -> LazyGroupBy[Self]:
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

        .. warning::
            The index column must be sorted in ascending order.

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
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        start_by : {'window', 'datapoint', 'monday'}
            The strategy to determine the start of the first window by.

            * 'window': Truncate the start of the window with the 'every' argument.
            * 'datapoint': Start from the first encountered data point.
            * 'monday': Start the window on the monday before the first data point.

        See Also
        --------
        groupby_rolling

        Examples
        --------
        >>> from datetime import datetime
        >>> # create an example dataframe
        >>> df = pl.DataFrame(
        ...     {
        ...         "time": pl.date_range(
        ...             low=datetime(2021, 12, 16),
        ...             high=datetime(2021, 12, 16, 3),
        ...             interval="30m",
        ...         ),
        ...         "n": range(7),
        ...     }
        ... )
        >>> df
        shape: (7, 2)
        ┌─────────────────────┬─────┐
        │ time                ┆ n   │
        │ ---                 ┆ --- │
        │ datetime[μs]        ┆ i64 │
        ╞═════════════════════╪═════╡
        │ 2021-12-16 00:00:00 ┆ 0   │
        │ 2021-12-16 00:30:00 ┆ 1   │
        │ 2021-12-16 01:00:00 ┆ 2   │
        │ 2021-12-16 01:30:00 ┆ 3   │
        │ 2021-12-16 02:00:00 ┆ 4   │
        │ 2021-12-16 02:30:00 ┆ 5   │
        │ 2021-12-16 03:00:00 ┆ 6   │
        └─────────────────────┴─────┘

        Group by windows of 1 hour starting at 2021-12-16 00:00:00.

        >>> df.lazy().groupby_dynamic("time", every="1h", closed="right").agg(
        ...     [
        ...         pl.col("time").min().alias("time_min"),
        ...         pl.col("time").max().alias("time_max"),
        ...     ]
        ... ).collect()
        shape: (4, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ time                ┆ time_min            ┆ time_max            │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╡
        │ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 00:00:00 │
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 00:30:00 ┆ 2021-12-16 01:00:00 │
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 01:30:00 ┆ 2021-12-16 02:00:00 │
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 02:30:00 ┆ 2021-12-16 03:00:00 │
        └─────────────────────┴─────────────────────┴─────────────────────┘

        The window boundaries can also be added to the aggregation result

        >>> df.lazy().groupby_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg([pl.col("time").count().alias("time_count")]).collect()
        shape: (4, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u32        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ 1          │
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 2          │
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        └─────────────────────┴─────────────────────┴─────────────────────┴────────────┘

        When closed="left", should not include right end of interval
        [lower_bound, upper_bound)

        >>> df.lazy().groupby_dynamic("time", every="1h", closed="left").agg(
        ...     [
        ...         pl.col("time").count().alias("time_count"),
        ...         pl.col("time").alias("time_agg_list"),
        ...     ]
        ... ).collect()
        shape: (4, 3)
        ┌─────────────────────┬────────────┬─────────────────────────────────────┐
        │ time                ┆ time_count ┆ time_agg_list                       │
        │ ---                 ┆ ---        ┆ ---                                 │
        │ datetime[μs]        ┆ u32        ┆ list[datetime[μs]]                  │
        ╞═════════════════════╪════════════╪═════════════════════════════════════╡
        │ 2021-12-16 00:00:00 ┆ 2          ┆ [2021-12-16 00:00:00, 2021-12-16... │
        │ 2021-12-16 01:00:00 ┆ 2          ┆ [2021-12-16 01:00:00, 2021-12-16... │
        │ 2021-12-16 02:00:00 ┆ 2          ┆ [2021-12-16 02:00:00, 2021-12-16... │
        │ 2021-12-16 03:00:00 ┆ 1          ┆ [2021-12-16 03:00:00]               │
        └─────────────────────┴────────────┴─────────────────────────────────────┘

        When closed="both" the time values at the window boundaries belong to 2 groups.

        >>> df.lazy().groupby_dynamic("time", every="1h", closed="both").agg(
        ...     pl.col("time").count().alias("time_count")
        ... ).collect()
        shape: (5, 2)
        ┌─────────────────────┬────────────┐
        │ time                ┆ time_count │
        │ ---                 ┆ ---        │
        │ datetime[μs]        ┆ u32        │
        ╞═════════════════════╪════════════╡
        │ 2021-12-15 23:00:00 ┆ 1          │
        │ 2021-12-16 00:00:00 ┆ 3          │
        │ 2021-12-16 01:00:00 ┆ 3          │
        │ 2021-12-16 02:00:00 ┆ 3          │
        │ 2021-12-16 03:00:00 ┆ 1          │
        └─────────────────────┴────────────┘

        Dynamic groupbys can also be combined with grouping on normal keys

        >>> df = pl.DataFrame(
        ...     {
        ...         "time": pl.date_range(
        ...             low=datetime(2021, 12, 16),
        ...             high=datetime(2021, 12, 16, 3),
        ...             interval="30m",
        ...         ),
        ...         "groups": ["a", "a", "a", "b", "b", "a", "a"],
        ...     }
        ... )
        >>> df
        shape: (7, 2)
        ┌─────────────────────┬────────┐
        │ time                ┆ groups │
        │ ---                 ┆ ---    │
        │ datetime[μs]        ┆ str    │
        ╞═════════════════════╪════════╡
        │ 2021-12-16 00:00:00 ┆ a      │
        │ 2021-12-16 00:30:00 ┆ a      │
        │ 2021-12-16 01:00:00 ┆ a      │
        │ 2021-12-16 01:30:00 ┆ b      │
        │ 2021-12-16 02:00:00 ┆ b      │
        │ 2021-12-16 02:30:00 ┆ a      │
        │ 2021-12-16 03:00:00 ┆ a      │
        └─────────────────────┴────────┘
        >>> (
        ...     df.lazy().groupby_dynamic(
        ...         "time",
        ...         every="1h",
        ...         closed="both",
        ...         by="groups",
        ...         include_boundaries=True,
        ...     )
        ... ).agg([pl.col("time").count().alias("time_count")]).collect()
        shape: (7, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ str    ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u32        │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ a      ┆ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ 1          │
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 3          │
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 1          │
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        │ a      ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 04:00:00 ┆ 2021-12-16 03:00:00 ┆ 1          │
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        │ b      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 1          │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴────────────┘

        Dynamic groupby on an index column

        >>> df = pl.DataFrame(
        ...     {
        ...         "idx": pl.arange(0, 6, eager=True),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> (
        ...     df.lazy().groupby_dynamic(
        ...         "idx",
        ...         every="2i",
        ...         period="3i",
        ...         include_boundaries=True,
        ...         closed="right",
        ...     )
        ... ).agg(pl.col("A").alias("A_agg_list")).collect()
        shape: (3, 4)
        ┌─────────────────┬─────────────────┬─────┬─────────────────┐
        │ _lower_boundary ┆ _upper_boundary ┆ idx ┆ A_agg_list      │
        │ ---             ┆ ---             ┆ --- ┆ ---             │
        │ i64             ┆ i64             ┆ i64 ┆ list[str]       │
        ╞═════════════════╪═════════════════╪═════╪═════════════════╡
        │ 0               ┆ 3               ┆ 0   ┆ ["A", "B", "B"] │
        │ 2               ┆ 5               ┆ 2   ┆ ["B", "B", "C"] │
        │ 4               ┆ 7               ┆ 4   ┆ ["C"]           │
        └─────────────────┴─────────────────┴─────┴─────────────────┘

        """  # noqa: W505
        if offset is None:
            offset = f"-{every}" if period is None else "0ns"

        if period is None:
            period = every

        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)
        every = _timedelta_to_pl_duration(every)

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
            start_by,
        )
        return LazyGroupBy(lgb, lazyframe_class=self.__class__)

    @deprecate_nonkeyword_arguments()
    def join_asof(
        self,
        other: LazyFrame,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
        tolerance: str | int | float | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> Self:
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

        Examples
        --------
        >>> from datetime import datetime
        >>> gdp = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             datetime(2016, 1, 1),
        ...             datetime(2017, 1, 1),
        ...             datetime(2018, 1, 1),
        ...             datetime(2019, 1, 1),
        ...         ],  # note record date: Jan 1st (sorted!)
        ...         "gdp": [4164, 4411, 4566, 4696],
        ...     }
        ... ).lazy()
        >>> population = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             datetime(2016, 5, 12),
        ...             datetime(2017, 5, 12),
        ...             datetime(2018, 5, 12),
        ...             datetime(2019, 5, 12),
        ...         ],  # note record date: May 12th (sorted!)
        ...         "population": [82.19, 82.66, 83.12, 83.52],
        ...     }
        ... ).lazy()
        >>> population.join_asof(
        ...     gdp, left_on="date", right_on="date", strategy="backward"
        ... ).collect()
        shape: (4, 3)
        ┌─────────────────────┬────────────┬──────┐
        │ date                ┆ population ┆ gdp  │
        │ ---                 ┆ ---        ┆ ---  │
        │ datetime[μs]        ┆ f64        ┆ i64  │
        ╞═════════════════════╪════════════╪══════╡
        │ 2016-05-12 00:00:00 ┆ 82.19      ┆ 4164 │
        │ 2017-05-12 00:00:00 ┆ 82.66      ┆ 4411 │
        │ 2018-05-12 00:00:00 ┆ 83.12      ┆ 4566 │
        │ 2019-05-12 00:00:00 ┆ 83.52      ┆ 4696 │
        └─────────────────────┴────────────┴──────┘

        """
        if not isinstance(other, LazyFrame):
            raise TypeError(
                f"Expected 'other' join table to be a LazyFrame, not a {type(other).__name__}"
            )

        if isinstance(on, str):
            left_on = on
            right_on = on

        if left_on is None or right_on is None:
            raise ValueError("You should pass the column to join on as an argument.")

        by_left_: Sequence[str] | None
        by_left_ = [by_left] if isinstance(by_left, str) else by_left

        by_right_: Sequence[str] | None
        by_right_ = [by_right] if isinstance(by_right, (str, pli.Expr)) else by_right

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

    @deprecate_nonkeyword_arguments(
        message=(
            "All arguments of LazyFrame.join except for 'other', 'on', and 'how' will be keyword-only in the next breaking release."
            " Use keyword arguments to silence this warning."
        )
    )
    def join(
        self,
        other: LazyFrame,
        left_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        right_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        how: JoinStrategy = "inner",
        suffix: str = "_right",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> Self:
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
        │ 2    ┆ 7.0  ┆ b   ┆ y     │
        │ null ┆ null ┆ d   ┆ z     │
        │ 3    ┆ 8.0  ┆ c   ┆ null  │
        └──────┴──────┴─────┴───────┘
        >>> df.join(other_df, on="ham", how="left").collect()
        shape: (3, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        │ 3   ┆ 8.0 ┆ c   ┆ null  │
        └─────┴─────┴─────┴───────┘
        >>> df.join(other_df, on="ham", how="semi").collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        └─────┴─────┴─────┘
        >>> df.join(other_df, on="ham", how="anti").collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        """
        if not isinstance(other, LazyFrame):
            raise TypeError(
                f"Expected 'other' join table to be a LazyFrame, not a {type(other).__name__}"
            )

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
        self,
        exprs: (
            str
            | PolarsExprType
            | PythonLiteral
            | pli.Series
            | Iterable[str | PolarsExprType | PythonLiteral | pli.Series | None]
            | None
        ) = None,
        *more_exprs: str | PolarsExprType | PythonLiteral | pli.Series | None,
        **named_exprs: str | PolarsExprType | PythonLiteral | pli.Series | None,
    ) -> Self:
        """
        Add columns to this DataFrame.

        Added columns will replace existing columns with the same name.

        Parameters
        ----------
        exprs
            Column or columns to add. Accepts expression input. Strings are parsed
            as column names, other non-expression inputs are parsed as literals.
        *more_exprs
            Additional columns to add, specified as positional arguments.
        **named_exprs
            Additional columns to add, specified as keyword arguments. The columns
            will be renamed to the keyword used.

        Returns
        -------
        A new LazyFrame with the columns added.

        Notes
        -----
        Creating a new LazyFrame using this method does not create a new copy of
        existing data.

        Examples
        --------
        Pass an expression to add it as a new column.

        >>> ldf = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... ).lazy()
        >>> ldf.with_columns((pl.col("a") ** 2).alias("a^2")).collect()
        shape: (4, 4)
        ┌─────┬──────┬───────┬──────┐
        │ a   ┆ b    ┆ c     ┆ a^2  │
        │ --- ┆ ---  ┆ ---   ┆ ---  │
        │ i64 ┆ f64  ┆ bool  ┆ f64  │
        ╞═════╪══════╪═══════╪══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1.0  │
        │ 2   ┆ 4.0  ┆ true  ┆ 4.0  │
        │ 3   ┆ 10.0 ┆ false ┆ 9.0  │
        │ 4   ┆ 13.0 ┆ true  ┆ 16.0 │
        └─────┴──────┴───────┴──────┘

        Added columns will replace existing columns with the same name.

        >>> ldf.with_columns(pl.col("a").cast(pl.Float64)).collect()
        shape: (4, 3)
        ┌─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     │
        │ --- ┆ ---  ┆ ---   │
        │ f64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╡
        │ 1.0 ┆ 0.5  ┆ true  │
        │ 2.0 ┆ 4.0  ┆ true  │
        │ 3.0 ┆ 10.0 ┆ false │
        │ 4.0 ┆ 13.0 ┆ true  │
        └─────┴──────┴───────┘

        Multiple columns can be added by passing a list of expressions.

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
        │ 2   ┆ 4.0  ┆ true  ┆ 4.0  ┆ 2.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 9.0  ┆ 5.0  ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 16.0 ┆ 6.5  ┆ false │
        └─────┴──────┴───────┴──────┴──────┴───────┘

        Multiple columns also can be added using positional arguments instead of a list.

        >>> ldf.with_columns(
        ...     (pl.col("a") ** 2).alias("a^2"),
        ...     (pl.col("b") / 2).alias("b/2"),
        ...     (pl.col("c").is_not()).alias("not c"),
        ... ).collect()
        shape: (4, 6)
        ┌─────┬──────┬───────┬──────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ a^2  ┆ b/2  ┆ not c │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪══════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1.0  ┆ 0.25 ┆ false │
        │ 2   ┆ 4.0  ┆ true  ┆ 4.0  ┆ 2.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 9.0  ┆ 5.0  ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 16.0 ┆ 6.5  ┆ false │
        └─────┴──────┴───────┴──────┴──────┴───────┘

        Use keyword arguments to easily name your expression inputs.

        >>> ldf.with_columns(
        ...     ab=pl.col("a") * pl.col("b"),
        ...     not_c=pl.col("c").is_not(),
        ... ).collect()
        shape: (4, 5)
        ┌─────┬──────┬───────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ ab   ┆ not_c │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 0.5  ┆ false │
        │ 2   ┆ 4.0  ┆ true  ┆ 8.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 30.0 ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 52.0 ┆ false │
        └─────┴──────┴───────┴──────┴───────┘

        Expressions with multiple outputs can be automatically instantiated as Structs
        by enabling the experimental setting ``Config.set_auto_structify(True)``:

        >>> with pl.Config() as cfg:
        ...     cfg.set_auto_structify(True)  # doctest: +IGNORE_RESULT
        ...     ldf.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().suffix("_diff"),
        ...     ).collect()
        ...
        shape: (4, 3)
        ┌─────┬──────┬─────────────┐
        │ a   ┆ b    ┆ diffs       │
        │ --- ┆ ---  ┆ ---         │
        │ i64 ┆ f64  ┆ struct[2]   │
        ╞═════╪══════╪═════════════╡
        │ 1   ┆ 0.5  ┆ {null,null} │
        │ 2   ┆ 4.0  ┆ {1,3.5}     │
        │ 3   ┆ 10.0 ┆ {1,6.0}     │
        │ 4   ┆ 13.0 ┆ {1,3.0}     │
        └─────┴──────┴─────────────┘

        """
        if exprs is None and not named_exprs:
            raise ValueError("Expected at least one of 'exprs' or **named_exprs")

        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        exprs = pli.selection_to_pyexpr_list(exprs, structify=structify)
        exprs.extend(pli.selection_to_pyexpr_list(more_exprs, structify=structify))
        exprs.extend(
            pli.expr_to_lit_or_expr(
                expr, structify=structify, name=name, str_to_lit=False
            )._pyexpr
            for name, expr in named_exprs.items()
        )

        return self._from_pyldf(self._ldf.with_columns(exprs))

    @typing.no_type_check
    def with_context(self, other: Self | list[Self]) -> Self:
        """
        Add an external context to the computation graph.

        This allows expressions to also access columns from DataFrames
        that are not part of this one.

        Parameters
        ----------
        other
            Lazy DataFrame to join with.

        Examples
        --------
        >>> df_a = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "c", None]}).lazy()
        >>> df_other = pl.DataFrame({"c": ["foo", "ham"]})
        >>> df_a.with_context(df_other.lazy()).select(
        ...     [pl.col("b") + pl.col("c").first()]
        ... ).collect()
        shape: (3, 1)
        ┌──────┐
        │ b    │
        │ ---  │
        │ str  │
        ╞══════╡
        │ afoo │
        │ cfoo │
        │ null │
        └──────┘

        Fill nulls with the median from another dataframe:

        >>> train_df = pl.DataFrame(
        ...     {"feature_0": [-1.0, 0, 1], "feature_1": [-1.0, 0, 1]}
        ... ).lazy()
        >>> test_df = pl.DataFrame(
        ...     {"feature_0": [-1.0, None, 1], "feature_1": [-1.0, 0, 1]}
        ... ).lazy()
        >>> test_df.with_context(train_df.select(pl.all().suffix("_train"))).select(
        ...     pl.col("feature_0").fill_null(pl.col("feature_0_train").median())
        ... ).collect()
        shape: (3, 1)
        ┌───────────┐
        │ feature_0 │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -1.0      │
        │ 0.0       │
        │ 1.0       │
        └───────────┘

        """
        if not isinstance(other, list):
            other = [other]

        return self._from_pyldf(self._ldf.with_context([lf._ldf for lf in other]))

    def drop(self, columns: str | list[str]) -> Self:
        """
        Remove one or multiple columns from a DataFrame.

        Parameters
        ----------
        columns
            - Name of the column that should be removed.
            - List of column names.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).lazy()
        >>> df.drop("ham").collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 6.0 │
        │ 2   ┆ 7.0 │
        │ 3   ┆ 8.0 │
        └─────┴─────┘

        """
        if isinstance(columns, str):
            columns = [columns]
        return self._from_pyldf(self._ldf.drop_columns(columns))

    def rename(self, mapping: dict[str, str]) -> Self:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name.

        Notes
        -----
        If names are swapped. E.g. 'A' points to 'B' and 'B' points to 'A', polars
        will block projection and predicate pushdowns at this node.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... ).lazy()
        >>> df.rename({"foo": "apple"}).collect()
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ apple ┆ bar ┆ ham │
        │ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64 ┆ str │
        ╞═══════╪═════╪═════╡
        │ 1     ┆ 6   ┆ a   │
        │ 2     ┆ 7   ┆ b   │
        │ 3     ┆ 8   ┆ c   │
        └───────┴─────┴─────┘

        """
        existing = list(mapping.keys())
        new = list(mapping.values())
        return self._from_pyldf(self._ldf.rename(existing, new))

    def reverse(self) -> Self:
        """
        Reverse the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "key": ["a", "b", "c"],
        ...         "val": [1, 2, 3],
        ...     }
        ... ).lazy()
        >>> df.reverse().collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ key ┆ val │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ c   ┆ 3   │
        │ b   ┆ 2   │
        │ a   ┆ 1   │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.reverse())

    def shift(self, periods: int) -> Self:
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
        │ 1    ┆ 2    │
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
        │ 5    ┆ 6    │
        │ null ┆ null │
        └──────┴──────┘

        """
        return self._from_pyldf(self._ldf.shift(periods))

    def shift_and_fill(
        self,
        periods: int,
        fill_value: pli.Expr | int | str | float,
    ) -> Self:
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
        │ 1   ┆ 2   │
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
        │ 5   ┆ 6   │
        │ 0   ┆ 0   │
        └─────┴─────┘

        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return self._from_pyldf(self._ldf.shift_and_fill(periods, fill_value._pyexpr))

    def slice(self, offset: int, length: int | None = None) -> Self:
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
        │ z   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘

        """
        if length and length < 0:
            raise ValueError(
                f"Negative slice lengths ({length}) are invalid for LazyFrame"
            )
        return self._from_pyldf(self._ldf.slice(offset, length))

    def limit(self, n: int = 5) -> Self:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... ).lazy()
        >>> df.limit().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        └─────┴─────┘
        >>> df.limit(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        └─────┴─────┘

        """
        return self.head(n)

    def head(self, n: int = 5) -> Self:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... ).lazy()
        >>> df.head().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        └─────┴─────┘
        >>> df.head(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        └─────┴─────┘

        """
        return self.slice(0, n)

    def tail(self, n: int = 5) -> Self:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... ).lazy()
        >>> df.tail().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        │ 6   ┆ 12  │
        └─────┴─────┘
        >>> df.tail(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 11  │
        │ 6   ┆ 12  │
        └─────┴─────┘

        """
        return self._from_pyldf(self._ldf.tail(n))

    def last(self) -> Self:
        """
        Get the last row of the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.last().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 6   │
        └─────┴─────┘

        """
        return self.tail(1)

    def first(self) -> Self:
        """
        Get the first row of the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... ).lazy()
        >>> df.first().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 2   │
        └─────┴─────┘

        """
        return self.slice(0, 1)

    def with_row_count(self, name: str = "row_nr", offset: int = 0) -> Self:
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
        │ 1      ┆ 3   ┆ 4   │
        │ 2      ┆ 5   ┆ 6   │
        └────────┴─────┴─────┘

        """
        return self._from_pyldf(self._ldf.with_row_count(name, offset))

    def take_every(self, n: int) -> Self:
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
        │ 3   ┆ 7   │
        └─────┴─────┘

        """
        return self.select(pli.col("*").take_every(n))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        matches_supertype: bool = True,
    ) -> Self:
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
        matches_supertype
            Fill all matching supertypes of the fill ``value`` literal.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 4],
        ...         "b": [0.5, 4, None, 13],
        ...     }
        ... ).lazy()
        >>> df.fill_null(99).collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 99  ┆ 99.0 │
        │ 4   ┆ 13.0 │
        └─────┴──────┘
        >>> df.fill_null(strategy="forward").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 2   ┆ 4.0  │
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        >>> df.fill_null(strategy="max").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 4   ┆ 13.0 │
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        >>> df.fill_null(strategy="zero").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 0   ┆ 0.0  │
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        """
        dtypes: Sequence[PolarsDataType]

        if value is not None:

            def infer_dtype(value: Any) -> PolarsDataType:
                return next(iter(self.select(value).schema.values()))

            if isinstance(value, pli.Expr):
                dtypes = [infer_dtype(value)]
            elif isinstance(value, bool):
                dtypes = [Boolean]
            elif matches_supertype and isinstance(value, (int, float)):
                dtypes = [
                    Int8,
                    Int16,
                    Int32,
                    Int64,
                    UInt8,
                    UInt16,
                    UInt32,
                    UInt64,
                    Float32,
                    Float64,
                ]
            elif isinstance(value, int):
                dtypes = [Int64]
            elif isinstance(value, float):
                dtypes = [Float64]
            elif isinstance(value, datetime):
                dtypes = [Datetime] + [Datetime(tu) for tu in DTYPE_TEMPORAL_UNITS]
            elif isinstance(value, timedelta):
                dtypes = [Duration] + [Duration(tu) for tu in DTYPE_TEMPORAL_UNITS]
            elif isinstance(value, date):
                dtypes = [Date]
            elif isinstance(value, time):
                dtypes = [Time]
            elif isinstance(value, str):
                dtypes = [Utf8, Categorical]
            else:
                # fallback; anything not explicitly handled above
                dtypes = [infer_dtype(pli.lit(value))]

            return self.with_columns(pli.col(dtypes).fill_null(value, strategy, limit))

        return self.select(pli.all().fill_null(value, strategy, limit))

    def fill_nan(self, fill_value: int | float | pli.Expr | None) -> Self:
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.5, 2, float("NaN"), 4],
        ...         "b": [0.5, 4, float("NaN"), 13],
        ...     }
        ... ).lazy()
        >>> df.fill_nan(99).collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ f64  ┆ f64  │
        ╞══════╪══════╡
        │ 1.5  ┆ 0.5  │
        │ 2.0  ┆ 4.0  │
        │ 99.0 ┆ 99.0 │
        │ 4.0  ┆ 13.0 │
        └──────┴──────┘

        """
        if not isinstance(fill_value, pli.Expr):
            fill_value = pli.lit(fill_value)
        return self._from_pyldf(self._ldf.fill_nan(fill_value._pyexpr))

    def std(self, ddof: int = 1) -> Self:
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
        >>> df.std(ddof=0).collect()
        shape: (1, 2)
        ┌──────────┬──────────┐
        │ a        ┆ b        │
        │ ---      ┆ ---      │
        │ f64      ┆ f64      │
        ╞══════════╪══════════╡
        │ 1.118034 ┆ 0.433013 │
        └──────────┴──────────┘

        """
        return self._from_pyldf(self._ldf.std(ddof))

    def var(self, ddof: int = 1) -> Self:
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
        >>> df.var(ddof=0).collect()
        shape: (1, 2)
        ┌──────┬────────┐
        │ a    ┆ b      │
        │ ---  ┆ ---    │
        │ f64  ┆ f64    │
        ╞══════╪════════╡
        │ 1.25 ┆ 0.1875 │
        └──────┴────────┘

        """
        return self._from_pyldf(self._ldf.var(ddof))

    def max(self) -> Self:
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

    def min(self) -> Self:
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

    def sum(self) -> Self:
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

    def mean(self) -> Self:
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

    def median(self) -> Self:
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
        self,
        quantile: float | pli.Expr,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> Self:
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
        quantile = pli.expr_to_lit_or_expr(quantile, str_to_lit=False)
        return self._from_pyldf(self._ldf.quantile(quantile._pyexpr, interpolation))

    def explode(
        self,
        columns: str | Sequence[str] | pli.Expr | Sequence[pli.Expr],
    ) -> Self:
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
        │ a       ┆ 2       │
        │ a       ┆ 3       │
        │ b       ┆ 4       │
        │ b       ┆ 5       │
        │ c       ┆ 6       │
        │ c       ┆ 7       │
        │ c       ┆ 8       │
        └─────────┴─────────┘

        """
        columns = pli.selection_to_pyexpr_list(columns)
        return self._from_pyldf(self._ldf.explode(columns))

    @deprecate_nonkeyword_arguments(
        message=(
            "All arguments of LazyFrame.unique except for 'subset' will be keyword-only in the next breaking release."
            " Use keyword arguments to silence this warning."
        )
    )
    def unique(
        self,
        maintain_order: bool = True,
        subset: str | Sequence[str] | None = None,
        keep: UniqueKeepStrategy = "first",
    ) -> Self:
        """
        Drop duplicate rows from this DataFrame.

        Parameters
        ----------
        maintain_order
            Keep the same order as the original DataFrame. This is more expensive to
            compute.
        subset
            Columns to consider for identifying duplicates. Defaults to using all
            columns.
        keep : {'first', 'last', 'none'}
            Which of the duplicate rows to keep.

        Returns
        -------
        DataFrame with unique rows.

        Warnings
        --------
        This method will fail if there is a column of type `List` in the DataFrame or
        subset.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 1],
        ...         "bar": ["a", "a", "a", "a"],
        ...         "ham": ["b", "b", "b", "b"],
        ...     }
        ... ).lazy()
        >>> df.unique().collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        │ 2   ┆ a   ┆ b   │
        │ 3   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> df.unique(subset=["bar", "ham"]).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> df.unique(keep="last").collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ a   ┆ b   │
        │ 3   ┆ a   ┆ b   │
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘

        """
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            elif not isinstance(subset, list):
                subset = list(subset)

        return self._from_pyldf(self._ldf.unique(maintain_order, subset, keep))

    def drop_nulls(self, subset: list[str] | str | None = None) -> Self:
        """
        Return a new LazyFrame where rows with null values are dropped.

        Parameters
        ----------
        subset
            Subset of column(s) for which null values are considered.

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
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        This method drops a row if any single value of the row is null.

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
        │ null ┆ 2    ┆ null │
        │ null ┆ null ┆ null │
        │ null ┆ 1    ┆ 1    │
        └──────┴──────┴──────┘

        Drop a row only if all values are null:

        >>> df.filter(~pl.all(pl.all().is_null()))
        shape: (3, 3)
        ┌──────┬─────┬──────┐
        │ a    ┆ b   ┆ c    │
        │ ---  ┆ --- ┆ ---  │
        │ f64  ┆ i64 ┆ i64  │
        ╞══════╪═════╪══════╡
        │ null ┆ 1   ┆ 1    │
        │ null ┆ 2   ┆ null │
        │ null ┆ 1   ┆ 1    │
        └──────┴─────┴──────┘

        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return self._from_pyldf(self._ldf.drop_nulls(subset))

    def melt(
        self,
        id_vars: str | list[str] | None = None,
        value_vars: str | list[str] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
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
            Name to give to the `variable` column. Defaults to "variable"
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
        │ y   ┆ b        ┆ 3     │
        │ z   ┆ b        ┆ 5     │
        │ x   ┆ c        ┆ 2     │
        │ y   ┆ c        ┆ 4     │
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

    @deprecate_nonkeyword_arguments()
    def map(
        self,
        f: Callable[[pli.DataFrame], pli.DataFrame],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        slice_pushdown: bool = True,
        no_optimizations: bool = False,
        schema: None | SchemaDict = None,
        validate_output_schema: bool = True,
        streamable: bool = False,
    ) -> Self:
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
        streamable
            Whether the function that is given is eligible ot running in the streaming
            engine. That means that the function must produce the same result if it
            is exectuted on batches as it would when executed on the full dataset.

        Warnings
        --------
        The ``schema`` of a `LazyFrame` must always be correct. It is up to the caller
        of this function to ensure that this invariant is upheld.

        It is important that the optimization flags are correct. If the custom function
        for instance does an aggregation of a column, ``predicate_pushdown`` should not
        be allowed, as this prunes rows and will influence your aggregation results.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()
        >>> df.map(lambda x: 2 * x).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 6   │
        │ 4   ┆ 8   │
        └─────┴─────┘

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
                streamable=streamable,
                schema=schema,
                validate_output=validate_output_schema,
            )
        )

    def interpolate(self) -> Self:
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
        │ 5   ┆ 7    ┆ 3   │
        │ 9   ┆ 9    ┆ 6   │
        │ 10  ┆ null ┆ 9   │
        └─────┴──────┴─────┘

        """
        return self.select(pli.col("*").interpolate())

    def unnest(self, names: str | list[str]) -> Self:
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
        │ bar    ┆ 2   ┆ b   ┆ null ┆ [3]       ┆ womp  │
        └────────┴─────┴─────┴──────┴───────────┴───────┘

        """
        if isinstance(names, str):
            names = [names]
        return self._from_pyldf(self._ldf.unnest(names))

    def merge_sorted(self, other: LazyFrame, key: str) -> Self:
        """
        Take two sorted DataFrames and merge them by the sorted key.

        The output of this operation will also be sorted.
        It is the callers responsibility that the frames are sorted
        by that key otherwise the output will not make sense.

        The schemas of both LazyFrames must be equal.

        Parameters
        ----------
        other
            Other DataFrame that must be merged
        key
            Key that is sorted.

        """
        return self._from_pyldf(self._ldf.merge_sorted(other._ldf, key))

    def update(
        self, other: LazyFrame, on: None | str | Sequence[str] = None, how: str = "left"
    ) -> LazyFrame:
        """
        Update the values in this `LazyFrame` with the non-null values in `other`.

        Notes
        -----
        This is syntactic sugar for a left/inner join + coalesce

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Parameters
        ----------
        other
            LazyFrame that will be used to update the values
        on
            Column names that will be joined on.
            If none given the row count is used.
        how : {'left', 'inner'}
            'Left' will keep the left table rows as is.
            'Inner' will remove rows that are not found in other

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4],
        ...         "B": [400, 500, 600, 700],
        ...     }
        ... )
        >>> df
        shape: (4, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 400 │
        │ 2   ┆ 500 │
        │ 3   ┆ 600 │
        │ 4   ┆ 700 │
        └─────┴─────┘
        >>> new_df = pl.DataFrame(
        ...     {
        ...         "B": [4, None, 6],
        ...         "C": [7, 8, 9],
        ...     }
        ... )
        >>> new_df
        shape: (3, 2)
        ┌──────┬─────┐
        │ B    ┆ C   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 4    ┆ 7   │
        │ null ┆ 8   │
        │ 6    ┆ 9   │
        └──────┴─────┘
        >>> df.update(new_df)
        shape: (4, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 500 │
        │ 3   ┆ 6   │
        │ 4   ┆ 700 │
        └─────┴─────┘

        """
        row_count_used = False
        if on is None:
            row_count_used = True
            row_count_name = "__POLARS_ROW_COUNT"
            self = self.with_row_count(row_count_name)
            other = other.with_row_count(row_count_name)
            on = row_count_name

        if isinstance(on, str):
            on = [on]

        union_names = set(self.columns) & set(other.columns)

        for name in on:
            if name not in union_names:
                raise ValueError(f"Join column {name} not found.")

        right_added_names = union_names - set(on)

        # no need to join if only join columns are in other
        if len(right_added_names) == 0:
            return self
        tmp_name = "__POLARS_RIGHT"

        result = (
            self.join(other.select(list(union_names)), on=on, how=how, suffix=tmp_name)  # type: ignore[arg-type]
            .with_columns(
                [
                    pli.coalesce([column_name + tmp_name, pli.col(column_name)]).alias(
                        column_name
                    )
                    for column_name in right_added_names
                ]
            )
            .drop([name + tmp_name for name in right_added_names])
        )
        if row_count_used:
            result = result.drop(row_count_name)
        return result

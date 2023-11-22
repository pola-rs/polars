from __future__ import annotations

import contextlib
import os
from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from functools import reduce
from io import BytesIO, StringIO
from operator import and_
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Iterable,
    Mapping,
    NoReturn,
    Sequence,
    TypeVar,
    overload,
)

import polars._reexport as pl
from polars import functions as F
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
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    py_type_to_dtype,
)
from polars.dependencies import dataframe_api_compat, subprocess
from polars.io._utils import _is_local_file, _is_supported_cloud
from polars.io.csv._utils import _check_arg_is_1byte
from polars.io.ipc.anonymous_scan import _scan_ipc_fsspec
from polars.io.parquet.anonymous_scan import _scan_parquet_fsspec
from polars.lazyframe.group_by import LazyGroupBy
from polars.selectors import _expand_selectors, expand_selector
from polars.slice import LazyPolarsSlice
from polars.utils._async import _AioDataFrameResult, _GeventDataFrameResult
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_as_list_of_expressions,
)
from polars.utils._wrap import wrap_df, wrap_expr
from polars.utils.convert import _negate_duration, _timedelta_to_pl_duration
from polars.utils.deprecation import (
    deprecate_function,
    deprecate_renamed_function,
    deprecate_renamed_parameter,
    deprecate_saturating,
    issue_deprecation_warning,
)
from polars.utils.various import (
    _in_notebook,
    _prepare_row_count_args,
    _process_null_values,
    is_bool_sequence,
    is_sequence,
    normalize_filepath,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyLazyFrame

if TYPE_CHECKING:
    import sys
    from io import IOBase
    from typing import Awaitable, Literal

    import pyarrow as pa

    from polars import DataFrame, Expr
    from polars.dependencies import numpy as np
    from polars.type_aliases import (
        AsofJoinStrategy,
        ClosedInterval,
        ColumnNameOrSelector,
        CsvEncoding,
        CsvQuoteStyle,
        FillNullStrategy,
        FrameInitTypes,
        IntoExpr,
        IntoExprColumn,
        JoinStrategy,
        JoinValidation,
        Label,
        Orientation,
        ParallelStrategy,
        PolarsDataType,
        RollingInterpolationMethod,
        SchemaDefinition,
        SchemaDict,
        StartBy,
        UniqueKeepStrategy,
    )

    if sys.version_info >= (3, 10):
        from typing import Concatenate, ParamSpec
    else:
        from typing_extensions import Concatenate, ParamSpec

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    T = TypeVar("T")
    P = ParamSpec("P")


class LazyFrame:
    """
    Representation of a Lazy computation graph/query against a DataFrame.

    This allows for whole-query optimisation in addition to parallelism, and
    is the preferred (and highest-performance) mode of operation for polars.

    Parameters
    ----------
    data : dict, Sequence, ndarray, Series, or pandas.DataFrame
        Two-dimensional data in various forms; dict input must contain Sequences,
        Generators, or a `range`. Sequence may contain Series or other Sequences.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the schema param will be overridden.
        underlying data, the names given here will overwrite them.

        The number of entries in the schema should match the underlying data
        dimensions, unless a sequence of dictionaries is being passed, in which case
        a *partial* schema can be declared to prevent specific fields from being loaded.
    orient : {'col', 'row'}, default None
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is inferred by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.
    infer_schema_length : int, default None
        Maximum number of rows to read for schema inference; only applies if the input
        data is a sequence or generator of rows; other input is read as-is.
    nan_to_null : bool, default False
        If the data comes from one or more numpy arrays, can optionally convert input
        data np.nan values to null instead. This is a no-op for all other input data.

    Notes
    -----
    Initialising `LazyFrame(...)` directly is equivalent to `DataFrame(...).lazy()`.

    Examples
    --------
    Constructing a LazyFrame directly from a dictionary:

    >>> data = {"a": [1, 2], "b": [3, 4]}
    >>> lf = pl.LazyFrame(data)
    >>> lf.collect()
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    Notice that the dtypes are automatically inferred as polars Int64:

    >>> lf.dtypes
    [Int64, Int64]

    To specify a more detailed/specific frame schema you can supply the `schema`
    parameter with a dictionary of (name,dtype) pairs...

    >>> data = {"col1": [0, 2], "col2": [3, 7]}
    >>> lf2 = pl.LazyFrame(data, schema={"col1": pl.Float32, "col2": pl.Int64})
    >>> lf2.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 0.0  ┆ 3    │
    │ 2.0  ┆ 7    │
    └──────┴──────┘

    ...a sequence of (name,dtype) pairs...

    >>> data = {"col1": [1, 2], "col2": [3, 4]}
    >>> lf3 = pl.LazyFrame(data, schema=[("col1", pl.Float32), ("col2", pl.Int64)])
    >>> lf3.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    ...or a list of typed Series.

    >>> data = [
    ...     pl.Series("col1", [1, 2], dtype=pl.Float32),
    ...     pl.Series("col2", [3, 4], dtype=pl.Int64),
    ... ]
    >>> lf4 = pl.LazyFrame(data)
    >>> lf4.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    Constructing a LazyFrame from a numpy ndarray, specifying column names:

    >>> import numpy as np
    >>> data = np.array([(1, 2), (3, 4)], dtype=np.int64)
    >>> lf5 = pl.LazyFrame(data, schema=["a", "b"], orient="col")
    >>> lf5.collect()
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    Constructing a LazyFrame from a list of lists, row orientation inferred:

    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> lf6 = pl.LazyFrame(data, schema=["a", "b", "c"])
    >>> lf6.collect()
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    │ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┘

    """

    _ldf: PyLazyFrame
    _accessors: ClassVar[set[str]] = set()

    def __init__(
        self,
        data: FrameInitTypes | None = None,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        nan_to_null: bool = False,
    ):
        from polars.dataframe import DataFrame

        self._ldf = (
            DataFrame(
                data=data,
                schema=schema,
                schema_overrides=schema_overrides,
                orient=orient,
                infer_schema_length=infer_schema_length,
                nan_to_null=nan_to_null,
            )
            .lazy()
            ._ldf
        )

    @classmethod
    def _from_pyldf(cls, ldf: PyLazyFrame) -> Self:
        self = cls.__new__(cls)
        self._ldf = ldf
        return self

    def __getstate__(self) -> bytes:
        return self._ldf.__getstate__()

    def __setstate__(self, state: bytes) -> None:
        self._ldf = LazyFrame()._ldf  # Initialize with a dummy
        self._ldf.__setstate__(state)

    @classmethod
    def _scan_csv(
        cls,
        source: str | list[str] | list[Path],
        *,
        has_header: bool = True,
        separator: str = ",",
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        dtypes: SchemaDict | None = None,
        schema: SchemaDict | None = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
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
        try_parse_dates: bool = False,
        eol_char: str = "\n",
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = True,
    ) -> Self:
        """
        Lazily read from a CSV file or multiple files via glob patterns.

        Use `pl.scan_csv` to dispatch to this method.

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

        if isinstance(source, list):
            sources = source
            source = None  # type: ignore[assignment]
        else:
            sources = []  # type: ignore[assignment]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_csv(
            source,
            sources,
            separator,
            has_header,
            ignore_errors,
            skip_rows,
            n_rows,
            cache,
            dtype_list,
            low_memory,
            comment_prefix,
            quote_char,
            processed_null_values,
            missing_utf8_is_empty_string,
            infer_schema_length,
            with_column_names,
            rechunk,
            skip_rows_after_header,
            encoding,
            _prepare_row_count_args(row_count_name, row_count_offset),
            try_parse_dates,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
            schema=schema,
        )
        return self

    @classmethod
    def _scan_parquet(
        cls,
        source: str | list[str] | list[Path],
        *,
        n_rows: int | None = None,
        cache: bool = True,
        parallel: ParallelStrategy = "auto",
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        storage_options: dict[str, object] | None = None,
        low_memory: bool = False,
        use_statistics: bool = True,
        hive_partitioning: bool = True,
        retries: int = 0,
    ) -> Self:
        """
        Lazily read from a parquet file or multiple files via glob patterns.

        Use `pl.scan_parquet` to dispatch to this method.

        See Also
        --------
        polars.io.scan_parquet

        """
        if isinstance(source, list):
            sources = source
            source = None  # type: ignore[assignment]
            can_use_fsspec = False
        else:
            can_use_fsspec = True
            sources = []  # type: ignore[assignment]

        # try fsspec scanner
        if (
            can_use_fsspec
            and not _is_local_file(source)  # type: ignore[arg-type]
            and not _is_supported_cloud(source)  # type: ignore[arg-type]
        ):
            scan = _scan_parquet_fsspec(source, storage_options)  # type: ignore[arg-type]
            if n_rows:
                scan = scan.head(n_rows)
            if row_count_name is not None:
                scan = scan.with_row_count(row_count_name, row_count_offset)
            return scan  # type: ignore[return-value]

        if storage_options is not None:
            storage_options = list(storage_options.items())  #  type: ignore[assignment]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_parquet(
            source,
            sources,
            n_rows,
            cache,
            parallel,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
            low_memory,
            cloud_options=storage_options,
            use_statistics=use_statistics,
            hive_partitioning=hive_partitioning,
            retries=retries,
        )
        return self

    @classmethod
    def _scan_ipc(
        cls,
        source: str | Path | list[str] | list[Path],
        *,
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

        Use `pl.scan_ipc` to dispatch to this method.

        See Also
        --------
        polars.io.scan_ipc

        """
        if isinstance(source, (str, Path)):
            can_use_fsspec = True
            source = normalize_filepath(source)
            sources = []
        else:
            can_use_fsspec = False
            sources = [normalize_filepath(source) for source in source]
            source = None  # type: ignore[assignment]

        # try fsspec scanner
        if can_use_fsspec and not _is_local_file(source):  # type: ignore[arg-type]
            scan = _scan_ipc_fsspec(source, storage_options)  # type: ignore[arg-type]
            if n_rows:
                scan = scan.head(n_rows)
            if row_count_name is not None:
                scan = scan.with_row_count(row_count_name, row_count_offset)
            return scan  # type: ignore[return-value]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_ipc(
            source,
            sources,
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
        source: str | Path | list[str] | list[Path],
        *,
        infer_schema_length: int | None = None,
        schema: SchemaDefinition | None = None,
        batch_size: int | None = None,
        n_rows: int | None = None,
        low_memory: bool = False,
        rechunk: bool = True,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
    ) -> Self:
        """
        Lazily read from a newline delimited JSON file.

        Use `pl.scan_ndjson` to dispatch to this method.

        See Also
        --------
        polars.io.scan_ndjson

        """
        if isinstance(source, (str, Path)):
            source = normalize_filepath(source)
            sources = []
        else:
            sources = [normalize_filepath(source) for source in source]
            source = None  # type: ignore[assignment]

        self = cls.__new__(cls)
        self._ldf = PyLazyFrame.new_from_ndjson(
            source,
            sources,
            infer_schema_length,
            schema,
            batch_size,
            n_rows,
            low_memory,
            rechunk,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
        return self

    @classmethod
    def _scan_python_function(
        cls,
        schema: pa.schema | Mapping[str, PolarsDataType],
        scan_fn: Any,
        *,
        pyarrow: bool = False,
    ) -> Self:
        self = cls.__new__(cls)
        if isinstance(schema, Mapping):
            self._ldf = PyLazyFrame.scan_from_python_function_pl_schema(
                list(schema.items()), scan_fn, pyarrow
            )
        else:
            self._ldf = PyLazyFrame.scan_from_python_function_arrow_schema(
                list(schema), scan_fn, pyarrow
            )
        return self

    @classmethod
    @deprecate_function(
        "Convert the JSON string to `StringIO` and then use `LazyFrame.deserialize`.",
        version="0.18.12",
    )
    def from_json(cls, json: str) -> Self:
        """
        Read a logical plan from a JSON string to construct a LazyFrame.

        .. deprecated:: 0.18.12
            This method is deprecated. Convert the JSON string to `StringIO`
            and then use `LazyFrame.deserialize`.

        Parameters
        ----------
        json
            String in JSON format.

        See Also
        --------
        deserialize

        """
        return cls.deserialize(StringIO(json))

    @classmethod
    @deprecate_renamed_function("deserialize", version="0.18.12")
    @deprecate_renamed_parameter("file", "source", version="0.18.12")
    def read_json(cls, source: str | Path | IOBase) -> Self:
        """
        Read a logical plan from a JSON file to construct a LazyFrame.

        .. deprecated:: 0.18.12
            This class method has been renamed to `deserialize`.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).

        See Also
        --------
        deserialize

        """
        return cls.deserialize(source)

    @classmethod
    def deserialize(cls, source: str | Path | IOBase) -> Self:
        """
        Read a logical plan from a JSON file to construct a LazyFrame.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).

        See Also
        --------
        LazyFrame.serialize

        Examples
        --------
        >>> import io
        >>> lf = pl.LazyFrame({"a": [1, 2, 3]}).sum()
        >>> json = lf.serialize()
        >>> pl.LazyFrame.deserialize(io.StringIO(json)).collect()
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘

        """
        if isinstance(source, StringIO):
            source = BytesIO(source.getvalue().encode())
        elif isinstance(source, (str, Path)):
            source = normalize_filepath(source)

        return cls._from_pyldf(PyLazyFrame.deserialize(source))

    @property
    def columns(self) -> list[str]:
        """
        Get column names.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).select("foo", "bar")
        >>> lf.columns
        ['foo', 'bar']

        """
        return self._ldf.columns()

    @property
    def dtypes(self) -> list[PolarsDataType]:
        """
        Get dtypes of columns in LazyFrame.

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.dtypes
        [Int64, Float64, Utf8]

        """
        return self._ldf.dtypes()

    @property
    def schema(self) -> SchemaDict:
        """
        Get a dict[column name, DataType].

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.schema
        OrderedDict({'foo': Int64, 'bar': Float64, 'ham': Utf8})

        """
        return OrderedDict(self._ldf.schema())

    def __dataframe_consortium_standard__(
        self, *, api_version: str | None = None
    ) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of polars.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        return dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
            self, api_version=api_version
        )

    @property
    def width(self) -> int:
        """
        Get the width of the LazyFrame.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4, 5, 6],
        ...     }
        ... )
        >>> lf.width
        2

        """
        return self._ldf.width()

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "the truth value of a LazyFrame is ambiguous"
            "\n\nLazyFrames cannot be used in boolean context with and/or/not operators."
        )

    def _comparison_error(self, operator: str) -> NoReturn:
        raise TypeError(
            f'"{operator!r}" comparison not supported for LazyFrame objects'
        )

    def __eq__(self, other: Any) -> NoReturn:
        self._comparison_error("==")

    def __ne__(self, other: Any) -> NoReturn:
        self._comparison_error("!=")

    def __gt__(self, other: Any) -> NoReturn:
        self._comparison_error(">")

    def __lt__(self, other: Any) -> NoReturn:
        self._comparison_error("<")

    def __ge__(self, other: Any) -> NoReturn:
        self._comparison_error(">=")

    def __le__(self, other: Any) -> NoReturn:
        self._comparison_error("<=")

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __copy__(self) -> Self:
        return self.clone()

    def __deepcopy__(self, memo: None = None) -> Self:
        return self.clone()

    def __getitem__(self, item: int | range | slice) -> LazyFrame:
        if not isinstance(item, slice):
            raise TypeError(
                "'LazyFrame' object is not subscriptable (aside from slicing)"
                "\n\nUse `select()` or `filter()` instead."
            )
        return LazyPolarsSlice(self).apply(item)

    def __str__(self) -> str:
        return f"""\
naive plan: (run LazyFrame.explain(optimized=True) to see the optimized plan)

{self.explain(optimized=False)}\
"""

    def __repr__(self) -> str:
        # don't expose internal/private classpath
        width = self.width
        cols_str = "{} col{}".format(width, "" if width == 1 else "s")
        schema_max_2 = (
            item for i, item in enumerate(self.schema.items()) if i in (0, width - 1)
        )
        schema_str = (", " if width == 2 else " … ").join(
            (f'"{k}": {v}' for k, v in schema_max_2)
        )
        return f"<{self.__class__.__name__} [{cols_str}, {{{schema_str}}}] at 0x{id(self):X}>"

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
            insert = self.explain(optimized=False).replace("\n", "<p></p>")

            return f"""\
<i>naive plan: (run <b>LazyFrame.explain(optimized=True)</b> to see the optimized plan)</i>
    <p></p>
    <div>{insert}</div>\
"""

    @overload
    def serialize(self, file: None = ...) -> str:
        ...

    @overload
    def serialize(self, file: IOBase | str | Path) -> None:
        ...

    def serialize(self, file: IOBase | str | Path | None = None) -> str | None:
        """
        Serialize the logical plan of this LazyFrame to a file or string in JSON format.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to `None`
            (default), the output is returned as a string instead.

        See Also
        --------
        LazyFrame.deserialize

        Examples
        --------
        Serialize the logical plan into a JSON string.

        >>> lf = pl.LazyFrame({"a": [1, 2, 3]}).sum()
        >>> json = lf.serialize()
        >>> json
        '{"Projection":{"expr":[{"Agg":{"Sum":{"Column":"a"}}}],"input":{"DataFrameScan":{"df":{"columns":[{"name":"a","datatype":"Int64","bit_settings":"","values":[1,2,3]}]},"schema":{"inner":{"a":"Int64"}},"output_schema":null,"projection":null,"selection":null}},"schema":{"inner":{"a":"Int64"}},"options":{"run_parallel":true,"duplicate_check":true}}}'

        The logical plan can later be deserialized back into a LazyFrame.

        >>> import io
        >>> pl.LazyFrame.deserialize(io.StringIO(json)).collect()
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘

        """
        if isinstance(file, (str, Path)):
            file = normalize_filepath(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if file is None or to_string_io:
            with BytesIO() as buf:
                self._ldf.serialize(buf)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._ldf.serialize(file)
        return None

    @overload
    def write_json(self, file: None = ...) -> str:
        ...

    @overload
    def write_json(self, file: IOBase | str | Path) -> None:
        ...

    @deprecate_renamed_function("serialize", version="0.18.12")
    def write_json(self, file: IOBase | str | Path | None = None) -> str | None:
        """
        Serialize the logical plan of this LazyFrame to a file or string in JSON format.

        .. deprecated:: 0.18.12
            This method has been renamed to :func:`LazyFrame.serialize`.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to `None`
            (default), the output is returned as a string instead.
        """
        return self.serialize(file)

    def pipe(
        self,
        function: Callable[Concatenate[LazyFrame, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """
        Offers a structured way to apply a sequence of user-defined functions (UDFs).

        Parameters
        ----------
        function
            Callable; will receive the frame as the first parameter,
            followed by any given args/kwargs.
        *args
            Arguments to pass to the UDF.
        **kwargs
            Keyword arguments to pass to the UDF.

        Examples
        --------
        >>> def cast_str_to_int(data, col_name):
        ...     return data.with_columns(pl.col(col_name).cast(pl.Int64))
        ...
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": ["10", "20", "30", "40"],
        ...     }
        ... )
        >>> lf.pipe(cast_str_to_int, col_name="b").collect()
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

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "b": [1, 2],
        ...         "a": [3, 4],
        ...     }
        ... )
        >>> lf.collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        >>> lf.pipe(lambda tdf: tdf.select(sorted(tdf.columns))).collect()
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
        return function(self, *args, **kwargs)

    @deprecate_renamed_parameter(
        "common_subplan_elimination", "comm_subplan_elim", version="0.18.9"
    )
    def explain(
        self,
        *,
        optimized: bool = True,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        streaming: bool = False,
    ) -> str:
        """
        Create a string representation of the query plan.

        Different optimizations can be turned on or off.

        Parameters
        ----------
        optimized
            Return an optimized query plan. Defaults to `True`.
            If this is set to `True` the subsequent
            optimization flags control which optimizations
            run.
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
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).explain()  # doctest: +SKIP
        """
        if optimized:
            ldf = self._ldf.optimization_toggle(
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
            return ldf.describe_optimized_plan()
        return self._ldf.describe_plan()

    @deprecate_renamed_parameter(
        "common_subplan_elimination", "comm_subplan_elim", version="0.18.9"
    )
    def show_graph(
        self,
        *,
        optimized: bool = True,
        show: bool = True,
        output_path: str | Path | None = None,
        raw_output: bool = False,
        figsize: tuple[float, float] = (16.0, 12.0),
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
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
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).show_graph()  # doctest: +SKIP

        """
        _ldf = self._ldf.optimization_toggle(
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
            Path(output_path).write_bytes(graph)

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
                raise ModuleNotFoundError(
                    "matplotlib should be installed to show graph"
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
        >>> lf = pl.LazyFrame({"foo": [1, 1, -2, 3]})
        >>> (
        ...     lf.with_columns(pl.col("foo").cum_sum().alias("bar"))
        ...     .inspect()  # print the node before the filter
        ...     .filter(pl.col("bar") == pl.col("foo"))
        ... )  # doctest: +ELLIPSIS
        <LazyFrame [2 cols, {"foo": Int64, "bar": Int64}] at ...>

        """

        def inspect(s: DataFrame) -> DataFrame:
            print(fmt.format(s))
            return s

        return self.map_batches(
            inspect, predicate_pushdown=True, projection_pushdown=True
        )

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
        maintain_order: bool = False,
    ) -> Self:
        """
        Sort the DataFrame by the given columns.

        Parameters
        ----------
        by
            Column(s) to sort by. Accepts expression input. Strings are parsed as column
            names.
        *more_by
            Additional columns to sort by, specified as positional arguments.
        descending
            Sort in descending order. When sorting by multiple columns, can be specified
            per column by passing a sequence of booleans.
        nulls_last
            Place null values last.
        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [6.0, 5.0, 4.0],
        ...         "c": ["a", "c", "b"],
        ...     }
        ... )
        >>> lf.sort("a").collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ null ┆ 4.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        │ 2    ┆ 5.0 ┆ c   │
        └──────┴─────┴─────┘

        Sorting by expressions is also supported.

        >>> lf.sort(pl.col("a") + pl.col("b") * 2, nulls_last=True).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 2    ┆ 5.0 ┆ c   │
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 4.0 ┆ b   │
        └──────┴─────┴─────┘

        Sort by multiple columns by passing a list of columns.

        >>> lf.sort(["c", "a"], descending=True).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 2    ┆ 5.0 ┆ c   │
        │ null ┆ 4.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        └──────┴─────┴─────┘

        Or use positional arguments to sort by multiple columns in the same way.

        >>> lf.sort("c", "a", descending=[False, True]).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 4.0 ┆ b   │
        │ 2    ┆ 5.0 ┆ c   │
        └──────┴─────┴─────┘

        """
        # Fast path for sorting by a single existing column
        if isinstance(by, str) and not more_by:
            return self._from_pyldf(
                self._ldf.sort(by, descending, nulls_last, maintain_order)
            )

        by = parse_as_list_of_expressions(by, *more_by)

        if isinstance(descending, bool):
            descending = [descending]
        elif len(by) != len(descending):
            raise ValueError(
                f"the length of `descending` ({len(descending)}) does not match the length of `by` ({len(by)})"
            )
        return self._from_pyldf(
            self._ldf.sort_by_exprs(by, descending, nulls_last, maintain_order)
        )

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
        maintain_order: bool = False,
    ) -> Self:
        """
        Return the `k` largest elements.

        If 'descending=True` the smallest elements will be given.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) included in sort order. Accepts expression input.
            Strings are parsed as column names.
        descending
            Return the 'k' smallest. Top-k by multiple columns can be specified
            per column by passing a sequence of booleans.
        nulls_last
            Place null values last.
        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might
            be worse since this requires a stable search.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 largest values in column b.

        >>> lf.top_k(4, by="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 3   │
        │ a   ┆ 2   │
        │ b   ┆ 2   │
        │ b   ┆ 1   │
        └─────┴─────┘

        Get the rows which contain the 4 largest values when sorting on column b and a.

        >>> lf.top_k(4, by=["b", "a"]).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 3   │
        │ b   ┆ 2   │
        │ a   ┆ 2   │
        │ c   ┆ 1   │
        └─────┴─────┘

        """
        by = parse_as_list_of_expressions(by)
        if isinstance(descending, bool):
            descending = [descending]
        elif len(by) != len(descending):
            raise ValueError(
                f"the length of `descending` ({len(descending)}) does not match the length of `by` ({len(by)})"
            )
        return self._from_pyldf(
            self._ldf.top_k(k, by, descending, nulls_last, maintain_order)
        )

    def bottom_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
        maintain_order: bool = False,
    ) -> Self:
        """
        Return the `k` smallest elements.

        If 'descending=True` the largest elements will be given.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) included in sort order. Accepts expression input.
            Strings are parsed as column names.
        descending
            Return the 'k' smallest. Top-k by multiple columns can be specified
            per column by passing a sequence of booleans.
        nulls_last
            Place null values last.
        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.

        See Also
        --------
        top_k

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 smallest values in column b.

        >>> lf.bottom_k(4, by="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 1   │
        │ a   ┆ 1   │
        │ c   ┆ 1   │
        │ a   ┆ 2   │
        └─────┴─────┘

        Get the rows which contain the 4 smallest values when sorting on column a and b.

        >>> lf.bottom_k(4, by=["a", "b"]).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        │ a   ┆ 2   │
        │ b   ┆ 1   │
        │ b   ┆ 2   │
        └─────┴─────┘

        """
        by = parse_as_list_of_expressions(by)
        if isinstance(descending, bool):
            descending = [descending]
        return self._from_pyldf(
            self._ldf.bottom_k(k, by, descending, nulls_last, maintain_order)
        )

    @deprecate_renamed_parameter(
        "common_subplan_elimination", "comm_subplan_elim", version="0.18.9"
    )
    def profile(
        self,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        show_plot: bool = False,
        truncate_nodes: int = 0,
        figsize: tuple[int, int] = (18, 8),
        streaming: bool = False,
    ) -> tuple[DataFrame, DataFrame]:
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
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        show_plot
            Show a gantt chart of the profiling result
        truncate_nodes
            Truncate the label lengths in the gantt chart to this number of
            characters.
        figsize
            matplotlib figsize of the profiling plot
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
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
         ┌─────────────────────────┬───────┬──────┐
         │ node                    ┆ start ┆ end  │
         │ ---                     ┆ ---   ┆ ---  │
         │ str                     ┆ u64   ┆ u64  │
         ╞═════════════════════════╪═══════╪══════╡
         │ optimization            ┆ 0     ┆ 5    │
         │ group_by_partitioned(a) ┆ 5     ┆ 470  │
         │ sort(a)                 ┆ 475   ┆ 1964 │
         └─────────────────────────┴───────┴──────┘)

        """
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            comm_subplan_elim = False
            comm_subexpr_elim = False

        ldf = self._ldf.optimization_toggle(
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
        df, timings = ldf.profile()
        (df, timings) = wrap_df(df), wrap_df(timings)

        if show_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, figsize=figsize)

                max_val = timings["end"][-1]
                timings_ = timings.reverse()

                if max_val > 1e9:
                    unit = "s"
                    timings_ = timings_.with_columns(
                        F.col(["start", "end"]) / 1_000_000
                    )
                elif max_val > 1e6:
                    unit = "ms"
                    timings_ = timings_.with_columns(F.col(["start", "end"]) / 1000)
                else:
                    unit = "us"
                if truncate_nodes > 0:
                    timings_ = timings_.with_columns(
                        F.col("node").str.slice(0, truncate_nodes) + "..."
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
                raise ModuleNotFoundError(
                    "matplotlib should be installed to show profiling plot"
                ) from None

        return df, timings

    @deprecate_renamed_parameter(
        "common_subplan_elimination", "comm_subplan_elim", version="0.18.9"
    )
    def collect(
        self,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        no_optimization: bool = False,
        streaming: bool = False,
        _eager: bool = False,
    ) -> DataFrame:
        """
        Materialize this LazyFrame into a DataFrame.

        By default, all query optimizations are enabled. Individual optimizations may
        be disabled by setting the corresponding parameter to `False`.

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
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        no_optimization
            Turn off (certain) optimizations.
        streaming
            Process the query in batches to handle larger-than-memory data.
            If set to `False` (default), the entire query is processed in a single
            batch.

            .. warning::
                This functionality is currently in an alpha state.

            .. note::
                Use :func:`explain` to see if Polars can process the query in streaming
                mode.

        Returns
        -------
        DataFrame

        See Also
        --------
        fetch: Run the query on the first `n` rows only for debugging purposes.
        explain : Print the query plan that is evaluated with collect.
        profile : Collect the LazyFrame and time each node in the computation graph.
        polars.collect_all : Collect multiple LazyFrames at the same time.
        polars.Config.set_streaming_chunk_size : Set the size of streaming batches.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a").agg(pl.all().sum()).collect()  # doctest: +SKIP
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

        Collect in streaming mode

        >>> lf.group_by("a").agg(pl.all().sum()).collect(
        ...     streaming=True
        ... )  # doctest: +SKIP
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
        if no_optimization or _eager:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False
            comm_subplan_elim = False
            comm_subexpr_elim = False

        if streaming:
            comm_subplan_elim = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim,
            comm_subexpr_elim,
            streaming,
            _eager,
        )
        return wrap_df(ldf.collect())

    @overload
    def collect_async(
        self,
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
    ) -> _GeventDataFrameResult[DataFrame]:
        ...

    @overload
    def collect_async(
        self,
        *,
        gevent: Literal[False] = False,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        streaming: bool = True,
    ) -> Awaitable[DataFrame]:
        ...

    def collect_async(
        self,
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
    ) -> Awaitable[DataFrame] | _GeventDataFrameResult[DataFrame]:
        """
        Collect DataFrame asynchronously in thread pool.

        Collects into a DataFrame (like :func:`collect`), but instead of returning
        DataFrame directly, they are scheduled to be collected inside thread pool,
        while this method returns almost instantly.

        May be useful if you use gevent or asyncio and want to release control to other
        greenlets/tasks while LazyFrames are being collected.

        Parameters
        ----------
        gevent
            Return wrapper to `gevent.event.AsyncResult` instead of Awaitable
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
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

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
        polars.collect_all : Collect multiple LazyFrames at the same time.
        polars.collect_all_async: Collect multiple LazyFrames at the same time lazily.

        Returns
        -------
        If `gevent=False` (default) then returns awaitable.

        If `gevent=True` then returns wrapper that has
        `.get(block=True, timeout=None)` method.

        Examples
        --------
        >>> import asyncio
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> async def main():
        ...     return await (
        ...         lf.group_by("a", maintain_order=True)
        ...         .agg(pl.all().sum())
        ...         .collect_async()
        ...     )
        ...
        >>> asyncio.run(main())
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
            comm_subplan_elim = False
            comm_subexpr_elim = False

        if streaming:
            comm_subplan_elim = False

        ldf = self._ldf.optimization_toggle(
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

        result = _GeventDataFrameResult() if gevent else _AioDataFrameResult()
        ldf.collect_with_callback(result._callback)  # type: ignore[attr-defined]
        return result  # type: ignore[return-value]

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
        slice_pushdown: bool = True,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Evaluate the query in streaming mode and write to a Parquet file.

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
            writing speeds.
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
        slice_pushdown
            Slice pushdown optimization.
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_parquet("out.parquet")  # doctest: +SKIP

        """
        lf = self._set_sink_optimizations(
            type_coercion=type_coercion,
            predicate_pushdown=predicate_pushdown,
            projection_pushdown=projection_pushdown,
            simplify_expression=simplify_expression,
            slice_pushdown=slice_pushdown,
            no_optimization=no_optimization,
        )

        return lf.sink_parquet(
            path=normalize_filepath(path),
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
        slice_pushdown: bool = True,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Evaluate the query in streaming mode and write to an IPC file.

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
        slice_pushdown
            Slice pushdown optimization.
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_ipc("out.arrow")  # doctest: +SKIP

        """
        lf = self._set_sink_optimizations(
            type_coercion=type_coercion,
            predicate_pushdown=predicate_pushdown,
            projection_pushdown=projection_pushdown,
            simplify_expression=simplify_expression,
            slice_pushdown=slice_pushdown,
            no_optimization=no_optimization,
        )

        return lf.sink_ipc(
            path=path,
            compression=compression,
            maintain_order=maintain_order,
        )

    @deprecate_renamed_parameter("quote", "quote_char", version="0.19.8")
    @deprecate_renamed_parameter("has_header", "include_header", version="0.19.13")
    def sink_csv(
        self,
        path: str | Path,
        *,
        include_bom: bool = False,
        include_header: bool = True,
        separator: str = ",",
        line_terminator: str = "\n",
        quote_char: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_precision: int | None = None,
        null_value: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        maintain_order: bool = True,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        no_optimization: bool = False,
    ) -> DataFrame:
        """
        Evaluate the query in streaming mode and write to a CSV file.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        include_bom
            Whether to include UTF-8 BOM in the CSV output.
        include_header
            Whether to include header in the CSV output.
        separator
            Separate CSV fields with this symbol.
        line_terminator
            String used to end each row.
        quote_char
            Byte to use as quoting character.
        batch_size
            Number of rows that will be processed per thread.
        datetime_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate. If no format specified, the default fractional-second
            precision is inferred from the maximum timeunit found in the frame's
            Datetime cols (if any).
        date_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        time_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        float_precision
            Number of decimal places to write, applied to both `Float32` and
            `Float64` datatypes.
        null_value
            A string representing null values (defaulting to the empty string).
        quote_style : {'necessary', 'always', 'non_numeric', 'never'}
            Determines the quoting strategy used.

            - necessary (default): This puts quotes around fields only when necessary.
              They are necessary when fields contain a quote,
              delimiter or record terminator.
              Quotes are also necessary when writing an empty record
              (which is indistinguishable from a record with one empty field).
              This is the default.
            - always: This puts quotes around every field. Always.
            - never: This never puts quotes around fields, even if that results in
              invalid CSV data (e.g.: by not quoting strings containing the
              separator).
            - non_numeric: This puts quotes around all fields that are non-numeric.
              Namely, when writing a field that does not parse as a valid float
              or integer, then quotes will be used even if they aren`t strictly
              necessary.
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
        slice_pushdown
            Slice pushdown optimization.
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_csv("out.csv")  # doctest: +SKIP

        """
        _check_arg_is_1byte("separator", separator, can_be_empty=False)
        _check_arg_is_1byte("quote_char", quote_char, can_be_empty=False)
        if not null_value:
            null_value = None

        lf = self._set_sink_optimizations(
            type_coercion=type_coercion,
            predicate_pushdown=predicate_pushdown,
            projection_pushdown=projection_pushdown,
            simplify_expression=simplify_expression,
            slice_pushdown=slice_pushdown,
            no_optimization=no_optimization,
        )

        return lf.sink_csv(
            path=path,
            include_bom=include_bom,
            include_header=include_header,
            separator=ord(separator),
            line_terminator=line_terminator,
            quote_char=ord(quote_char),
            batch_size=batch_size,
            datetime_format=datetime_format,
            date_format=date_format,
            time_format=time_format,
            float_precision=float_precision,
            null_value=null_value,
            quote_style=quote_style,
            maintain_order=maintain_order,
        )

    def sink_ndjson(
        self,
        path: str | Path,
        *,
        maintain_order: bool = True,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
    ) -> DataFrame:
        """
        Persists a LazyFrame at the provided path.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
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
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_json("out.json")  # doctest: +SKIP

        """
        lf = self._set_sink_optimizations(
            type_coercion=type_coercion,
            predicate_pushdown=predicate_pushdown,
            projection_pushdown=projection_pushdown,
            simplify_expression=simplify_expression,
            no_optimization=no_optimization,
            slice_pushdown=slice_pushdown,
        )

        return lf.sink_json(path=path, maintain_order=maintain_order)

    def _set_sink_optimizations(
        self,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        no_optimization: bool = False,
    ) -> PyLazyFrame:
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False

        return self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim=False,
            comm_subexpr_elim=False,
            streaming=True,
            _eager=False,
        )

    @deprecate_renamed_parameter(
        "common_subplan_elimination", "comm_subplan_elim", version="0.18.9"
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
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        streaming: bool = False,
    ) -> DataFrame:
        """
        Collect a small number of rows for debugging purposes.

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
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Notes
        -----
        This is similar to a :func:`collect` operation, but it overwrites the number of
        rows read by *every* scan operation. Be aware that `fetch` does not guarantee
        the final number of rows in the DataFrame. Filters, join operations and fewer
        rows being available in the scanned data will all influence the final number
        of rows (joins are especially susceptible to this, and may return no data
        at all if `n_rows` is too small as the join keys may not be present).

        Warnings
        --------
        This is strictly a utility function that can help to debug queries using a
        smaller number of rows, and should *not* be used in production code.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).fetch(2)
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
            comm_subplan_elim = False
            comm_subexpr_elim = False

        lf = self._ldf.optimization_toggle(
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
        return wrap_df(lf.fetch(n_rows))

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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.lazy()  # doctest: +ELLIPSIS
        <LazyFrame [3 cols, {"a": Int64 … "c": Boolean}] at ...>

        """
        return self

    def cache(self) -> Self:
        """Cache the result once the execution of the physical plan hits this node."""
        return self._from_pyldf(self._ldf.cache())

    def cast(
        self,
        dtypes: Mapping[ColumnNameOrSelector, PolarsDataType] | PolarsDataType,
        *,
        strict: bool = True,
    ) -> Self:
        """
        Cast LazyFrame column(s) to the specified dtype(s).

        Parameters
        ----------
        dtypes
            Mapping of column names (or selector) to dtypes, or a single dtype
            to which all columns will be cast.
        strict
            Throw an error if a cast could not be done (for instance, due to an
            overflow).

        Examples
        --------
        >>> from datetime import date
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        ...     }
        ... )

        Cast specific frame columns to the specified dtypes:

        >>> lf.cast({"foo": pl.Float32, "bar": pl.UInt8}).collect()
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ foo ┆ bar ┆ ham        │
        │ --- ┆ --- ┆ ---        │
        │ f32 ┆ u8  ┆ date       │
        ╞═════╪═════╪════════════╡
        │ 1.0 ┆ 6   ┆ 2020-01-02 │
        │ 2.0 ┆ 7   ┆ 2021-03-04 │
        │ 3.0 ┆ 8   ┆ 2022-05-06 │
        └─────┴─────┴────────────┘

        Cast all frame columns to the specified dtype:

        >>> lf.cast(pl.Utf8).collect().to_dict(as_series=False)
        {'foo': ['1', '2', '3'],
         'bar': ['6.0', '7.0', '8.0'],
         'ham': ['2020-01-02', '2021-03-04', '2022-05-06']}

        Use selectors to define the columns being cast:

        >>> import polars.selectors as cs
        >>> lf.cast({cs.numeric(): pl.UInt32, cs.temporal(): pl.Utf8}).collect()
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ foo ┆ bar ┆ ham        │
        │ --- ┆ --- ┆ ---        │
        │ u32 ┆ u32 ┆ str        │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 6   ┆ 2020-01-02 │
        │ 2   ┆ 7   ┆ 2021-03-04 │
        │ 3   ┆ 8   ┆ 2022-05-06 │
        └─────┴─────┴────────────┘

        """
        if not isinstance(dtypes, Mapping):
            return self._from_pyldf(self._ldf.cast_all(dtypes, strict))

        cast_map = {}
        for c, dtype in dtypes.items():
            dtype = py_type_to_dtype(dtype)
            cast_map.update(
                {c: dtype}
                if isinstance(c, str)
                else {x: dtype for x in expand_selector(self, c)}
            )

        return self._from_pyldf(self._ldf.cast(cast_map, strict))

    def clear(self, n: int = 0) -> LazyFrame:
        """
        Create an empty copy of the current LazyFrame, with zero to 'n' rows.

        Returns a copy with an identical schema but no data.

        Parameters
        ----------
        n
            Number of (empty) rows to return in the cleared frame.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.clear().fetch()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        >>> lf.clear(2).fetch()
        shape: (2, 3)
        ┌──────┬──────┬──────┐
        │ a    ┆ b    ┆ c    │
        │ ---  ┆ ---  ┆ ---  │
        │ i64  ┆ f64  ┆ bool │
        ╞══════╪══════╪══════╡
        │ null ┆ null ┆ null │
        │ null ┆ null ┆ null │
        └──────┴──────┴──────┘

        """
        return pl.DataFrame(schema=self.schema).clear(n).lazy()

    def clone(self) -> Self:
        """
        Create a copy of this LazyFrame.

        This is a cheap operation that does not copy data.

        See Also
        --------
        clear : Create an empty copy of the current LazyFrame, with identical
            schema but no data.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.clone()  # doctest: +ELLIPSIS
        <LazyFrame [3 cols, {"a": Int64 … "c": Boolean}] at ...>

        """
        return self._from_pyldf(self._ldf.clone())

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> Self:
        """
        Filter the rows in the LazyFrame based on a predicate expression.

        The original order of the remaining rows is preserved.

        Parameters
        ----------
        predicates
            Expression that evaluates to a boolean Series.
        constraints
            Column filters. Use name=value to filter column name by the supplied value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )

        Filter on one condition:

        >>> lf.filter(pl.col("foo") > 1).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
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

        Provide multiple filters using `*args` syntax:

        >>> lf.filter(
        ...     pl.col("foo") == 1,
        ...     pl.col("ham") == "a",
        ... ).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Provide multiple filters using `**kwargs` syntax:

        >>> lf.filter(foo=1, ham="a").collect()
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
        all_predicates: list[pl.Expr] = []
        boolean_masks = []

        # no-op; immediately matches all rows
        if len(predicates) == 1 and predicates[0] is True and not constraints:
            return self.clone()

        # note: identify masks separately from predicates
        for p in predicates:
            if p is False:  # immediately disallows all rows
                return self.clear()  # type: ignore[return-value]
            elif p is True:
                continue  # no-op; matches all rows
            elif is_bool_sequence(p, include_series=True):
                boolean_masks.append(pl.Series(p, dtype=Boolean))
            elif (
                (is_seq := is_sequence(p))
                and any(not isinstance(x, pl.Expr) for x in p)
            ) or (
                not is_seq
                and not isinstance(p, pl.Expr)
                and not (isinstance(p, str) and p in self.columns)
            ):
                err = (
                    f"Series(…, dtype={p.dtype})"
                    if isinstance(p, pl.Series)
                    else f"{p!r}"
                )
                raise ValueError(f"invalid predicate for `filter`: {err}")
            else:
                all_predicates.extend(
                    wrap_expr(x) for x in parse_as_list_of_expressions(p)
                )

        # identify deprecated usage of 'predicate' parameter
        if "predicate" in constraints:
            is_mask = False
            if isinstance(p := constraints["predicate"], pl.Expr) or (
                is_mask := is_bool_sequence(p)
            ):
                p = constraints.pop("predicate")
                issue_deprecation_warning(
                    "`filter` no longer takes a 'predicate' parameter.\n"
                    "To silence this warning you should omit the keyword and pass "
                    "as a positional argument instead.",
                    version="0.19.9",
                )
                if is_mask:
                    boolean_masks.append(pl.Series(p, dtype=Boolean))
                else:
                    all_predicates.append(p)  # type: ignore[arg-type]

        # unpack equality constraints from kwargs
        all_predicates.extend(
            F.col(name).eq(value) for name, value in constraints.items()
        )
        if not (all_predicates or boolean_masks):
            raise ValueError("No predicates or constraints provided to `filter`.")

        # if multiple predicates, combine as 'horizontal' expression
        combined_predicate = (
            (
                F.all_horizontal(*all_predicates)
                if len(all_predicates) > 1
                else all_predicates[0]
            )._pyexpr
            if all_predicates
            else None
        )

        # apply reduced boolean mask first, if applicable, then predicates
        ldf = (
            self._ldf.filter(F.lit(reduce(and_, boolean_masks))._pyexpr)
            if boolean_masks
            else self._ldf
        )
        return self._from_pyldf(
            ldf if combined_predicate is None else ldf.filter(combined_predicate)
        )

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """
        Select columns from this LazyFrame.

        Parameters
        ----------
        *exprs
            Column(s) to select, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names,
            other non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to select, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Examples
        --------
        Pass the name of a column to select that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.select("foo").collect()
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

        >>> lf.select(["foo", "bar"]).collect()
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

        >>> lf.select(pl.col("foo"), pl.col("bar") + 1).collect()
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

        >>> lf.select(
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
        by enabling the setting `Config.set_auto_structify(True)`:

        >>> with pl.Config(auto_structify=True):
        ...     lf.select(
        ...         is_odd=(pl.col(pl.INTEGER_DTYPES) % 2).name.suffix("_is_odd"),
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
        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        pyexprs = parse_as_list_of_expressions(
            *exprs, **named_exprs, __structify=structify
        )
        return self._from_pyldf(self._ldf.select(pyexprs))

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """
        Select columns from this LazyFrame.

        This will run all expression sequentially instead of in parallel.
        Use this when the work per expression is cheap.

        Parameters
        ----------
        *exprs
            Column(s) to select, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names,
            other non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to select, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        See Also
        --------
        select

        """
        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        pyexprs = parse_as_list_of_expressions(
            *exprs, **named_exprs, __structify=structify
        )
        return self._from_pyldf(self._ldf.select_seq(pyexprs))

    def group_by(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        maintain_order: bool = False,
    ) -> LazyGroupBy:
        """
        Start a group by operation.

        Parameters
        ----------
        by
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        *more_by
            Additional columns to group by, specified as positional arguments.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default group by.
            Setting this to `True` blocks the possibility
            to run on the streaming engine.

        Examples
        --------
        Group by one column and call `agg` to compute the grouped sum of another
        column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a").agg(pl.col("b").sum()).collect()  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 2   │
        │ b   ┆ 5   │
        │ c   ┆ 3   │
        └─────┴─────┘

        Set `maintain_order=True` to ensure the order of the groups is consistent with
        the input.

        >>> lf.group_by("a", maintain_order=True).agg(pl.col("c")).collect()
        shape: (3, 2)
        ┌─────┬───────────┐
        │ a   ┆ c         │
        │ --- ┆ ---       │
        │ str ┆ list[i64] │
        ╞═════╪═══════════╡
        │ a   ┆ [5, 3]    │
        │ b   ┆ [4, 2]    │
        │ c   ┆ [1]       │
        └─────┴───────────┘

        Group by multiple columns by passing a list of column names.

        >>> lf.group_by(["a", "b"]).agg(pl.max("c")).collect()  # doctest: +SKIP
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘

        Or use positional arguments to group by multiple columns in the same way.
        Expressions are also accepted.

        >>> lf.group_by("a", pl.col("b") // 2).agg(
        ...     pl.col("c").mean()
        ... ).collect()  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ f64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 0   ┆ 4.0 │
        │ b   ┆ 1   ┆ 3.0 │
        │ c   ┆ 1   ┆ 1.0 │
        └─────┴─────┴─────┘

        """
        exprs = parse_as_list_of_expressions(by, *more_by)
        lgb = self._ldf.group_by(exprs, maintain_order)
        return LazyGroupBy(lgb)

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        check_sorted: bool = True,
    ) -> LazyGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        Different from a `dynamic_group_by` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals
        use :func:`LazyFrame.group_by_dynamic`.

        If you have a time series `<t_0, t_1, ..., t_n>`, then by default the
        windows created will be

            * (t_0 - period, t_0]
            * (t_1 - period, t_1]
            * ...
            * (t_n - period, t_n]

        whereas if you pass a non-default `offset`, then the windows will be

            * (t_0 + offset, t_0 + offset + period]
            * (t_1 + offset, t_1 + offset + period]
            * ...
            * (t_n + offset, t_n + offset + period]

        The `period` and `offset` arguments are created either from a timedelta, or
        by using the following string language:

        - 1ns   (1 nanosecond)
        - 1us   (1 microsecond)
        - 1ms   (1 millisecond)
        - 1s    (1 second)
        - 1m    (1 minute)
        - 1h    (1 hour)
        - 1d    (1 calendar day)
        - 1w    (1 calendar week)
        - 1mo   (1 calendar month)
        - 1q    (1 calendar quarter)
        - 1y    (1 calendar year)
        - 1i    (1 index count)

        Or combine them:
        "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        In case of a rolling operation on an integer column, the windows are defined by:

        - "1i"      # length 1
        - "10i"     # length 10

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window - must be non-negative
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        check_sorted
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the by groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        See Also
        --------
        group_by_dynamic

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
        >>> df = pl.LazyFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
        ...     pl.col("dt").str.strptime(pl.Datetime).set_sorted()
        ... )
        >>> out = (
        ...     df.rolling(index_column="dt", period="2d")
        ...     .agg(
        ...         pl.sum("a").alias("sum_a"),
        ...         pl.min("a").alias("min_a"),
        ...         pl.max("a").alias("max_a"),
        ...     )
        ...     .collect()
        ... )
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
        period = deprecate_saturating(period)
        offset = deprecate_saturating(offset)
        index_column = parse_as_expression(index_column)
        if offset is None:
            offset = _negate_duration(_timedelta_to_pl_duration(period))

        pyexprs_by = parse_as_list_of_expressions(by) if by is not None else []
        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)

        lgb = self._ldf.rolling(
            index_column, period, offset, closed, pyexprs_by, check_sorted
        )
        return LazyGroupBy(lgb)

    def group_by_dynamic(
        self,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        truncate: bool | None = None,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        label: Label = "left",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        start_by: StartBy = "window",
        check_sorted: bool = True,
    ) -> LazyGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        Time windows are calculated and rows are assigned to windows. Different from a
        normal group by is that a row can be member of multiple groups.
        By default, the windows look like:

        - [start, start + period)
        - [start + every, start + every + period)
        - [start + 2*every, start + 2*every + period)
        - ...

        where `start` is determined by `start_by`, `offset`, and `every` (see parameter
        descriptions below).

        .. warning::
            The index column must be sorted in ascending order. If `by` is passed, then
            the index column must be sorted in ascending order within each group.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a dynamic group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it will equal 'every'
        offset
            offset of the window, only takes effect if `start_by` is `'window'`.
            Defaults to negative `every`.
        truncate
            truncate the time value to the window lower bound

            .. deprecated:: 0.19.4
                Use `label` instead.
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_boundary" and
            "_upper_boundary" columns. This will impact performance because it's harder to
            parallelize
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        label : {'left', 'right', 'datapoint'}
            Define which label to use for the window:

            - 'left': lower boundary of the window
            - 'right': upper boundary of the window
            - 'datapoint': the first value of the index column in the given window.
              If you don't need the label to be at one of the boundaries, choose this
              option for maximum performance
        by
            Also group by this column/these columns
        start_by : {'window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
            The strategy to determine the start of the first window by.

            * 'window': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * 'datapoint': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `'w'`):

              * 'monday': Start the window on the Monday before the first data point.
              * 'tuesday': Start the window on the Tuesday before the first data point.
              * ...
              * 'sunday': Start the window on the Sunday before the first data point.
        check_sorted
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the by groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        See Also
        --------
        rolling

        Notes
        -----
        1) If you're coming from pandas, then

           .. code-block:: python

               # polars
               df.group_by_dynamic("ts", every="1d").agg(pl.col("value").sum())

           is equivalent to

           .. code-block:: python

               # pandas
               df.set_index("ts").resample("D")["value"].sum().reset_index()

           though note that, unlike pandas, polars doesn't add extra rows for empty
           windows. If you need `index_column` to be evenly spaced, then please combine
           with :func:`DataFrame.upsample`.

        2) The `every`, `period` and `offset` arguments are created with
           the following string language:

           - 1ns   (1 nanosecond)
           - 1us   (1 microsecond)
           - 1ms   (1 millisecond)
           - 1s    (1 second)
           - 1m    (1 minute)
           - 1h    (1 hour)
           - 1d    (1 calendar day)
           - 1w    (1 calendar week)
           - 1mo   (1 calendar month)
           - 1q    (1 calendar quarter)
           - 1y    (1 calendar year)
           - 1i    (1 index count)

           Or combine them:
           "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

           By "calendar day", we mean the corresponding time on the next day (which may
           not be 24 hours, due to daylight savings). Similarly for "calendar week",
           "calendar month", "calendar quarter", and "calendar year".

           In case of a group_by_dynamic on an integer column, the windows are defined by:

           - "1i"      # length 1
           - "10i"     # length 10

        Examples
        --------
        >>> from datetime import datetime
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "time": pl.datetime_range(
        ...             start=datetime(2021, 12, 16),
        ...             end=datetime(2021, 12, 16, 3),
        ...             interval="30m",
        ...             eager=True,
        ...         ),
        ...         "n": range(7),
        ...     }
        ... )
        >>> lf.collect()
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

        >>> lf.group_by_dynamic("time", every="1h", closed="right").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (4, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-15 23:00:00 ┆ [0]       │
        │ 2021-12-16 00:00:00 ┆ [1, 2]    │
        │ 2021-12-16 01:00:00 ┆ [3, 4]    │
        │ 2021-12-16 02:00:00 ┆ [5, 6]    │
        └─────────────────────┴───────────┘

        The window boundaries can also be added to the aggregation result

        >>> lf.group_by_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg(pl.col("n").mean()).collect()
        shape: (4, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬─────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ n   │
        │ ---                 ┆ ---                 ┆ ---                 ┆ --- │
        │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ f64 │
        ╞═════════════════════╪═════════════════════╪═════════════════════╪═════╡
        │ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ 0.0 │
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 1.5 │
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 3.5 │
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 5.5 │
        └─────────────────────┴─────────────────────┴─────────────────────┴─────┘

        When closed="left", the window excludes the right end of interval:
        [lower_bound, upper_bound)

        >>> lf.group_by_dynamic("time", every="1h", closed="left").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (4, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-16 00:00:00 ┆ [0, 1]    │
        │ 2021-12-16 01:00:00 ┆ [2, 3]    │
        │ 2021-12-16 02:00:00 ┆ [4, 5]    │
        │ 2021-12-16 03:00:00 ┆ [6]       │
        └─────────────────────┴───────────┘

        When closed="both" the time values at the window boundaries belong to 2 groups.

        >>> lf.group_by_dynamic("time", every="1h", closed="both").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (5, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-15 23:00:00 ┆ [0]       │
        │ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ 2021-12-16 01:00:00 ┆ [2, 3, 4] │
        │ 2021-12-16 02:00:00 ┆ [4, 5, 6] │
        │ 2021-12-16 03:00:00 ┆ [6]       │
        └─────────────────────┴───────────┘

        Dynamic group bys can also be combined with grouping on normal keys

        >>> lf = lf.with_columns(groups=pl.Series(["a", "a", "a", "b", "b", "a", "a"]))
        >>> lf.collect()
        shape: (7, 3)
        ┌─────────────────────┬─────┬────────┐
        │ time                ┆ n   ┆ groups │
        │ ---                 ┆ --- ┆ ---    │
        │ datetime[μs]        ┆ i64 ┆ str    │
        ╞═════════════════════╪═════╪════════╡
        │ 2021-12-16 00:00:00 ┆ 0   ┆ a      │
        │ 2021-12-16 00:30:00 ┆ 1   ┆ a      │
        │ 2021-12-16 01:00:00 ┆ 2   ┆ a      │
        │ 2021-12-16 01:30:00 ┆ 3   ┆ b      │
        │ 2021-12-16 02:00:00 ┆ 4   ┆ b      │
        │ 2021-12-16 02:30:00 ┆ 5   ┆ a      │
        │ 2021-12-16 03:00:00 ┆ 6   ┆ a      │
        └─────────────────────┴─────┴────────┘
        >>> lf.group_by_dynamic(
        ...     "time",
        ...     every="1h",
        ...     closed="both",
        ...     by="groups",
        ...     include_boundaries=True,
        ... ).agg(pl.col("n")).collect()
        shape: (7, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬───────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ n         │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---       │
        │ str    ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ list[i64] │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪═══════════╡
        │ a      ┆ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ [0]       │
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [2]       │
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [5, 6]    │
        │ a      ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 04:00:00 ┆ 2021-12-16 03:00:00 ┆ [6]       │
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [3, 4]    │
        │ b      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [4]       │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴───────────┘

        Dynamic group by on an index column

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "idx": pl.int_range(0, 6, eager=True),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> lf.group_by_dynamic(
        ...     "idx",
        ...     every="2i",
        ...     period="3i",
        ...     include_boundaries=True,
        ...     closed="right",
        ... ).agg(pl.col("A").alias("A_agg_list")).collect()
        shape: (4, 4)
        ┌─────────────────┬─────────────────┬─────┬─────────────────┐
        │ _lower_boundary ┆ _upper_boundary ┆ idx ┆ A_agg_list      │
        │ ---             ┆ ---             ┆ --- ┆ ---             │
        │ i64             ┆ i64             ┆ i64 ┆ list[str]       │
        ╞═════════════════╪═════════════════╪═════╪═════════════════╡
        │ -2              ┆ 1               ┆ -2  ┆ ["A", "A"]      │
        │ 0               ┆ 3               ┆ 0   ┆ ["A", "B", "B"] │
        │ 2               ┆ 5               ┆ 2   ┆ ["B", "B", "C"] │
        │ 4               ┆ 7               ┆ 4   ┆ ["C"]           │
        └─────────────────┴─────────────────┴─────┴─────────────────┘

        """  # noqa: W505
        every = deprecate_saturating(every)
        period = deprecate_saturating(period)
        offset = deprecate_saturating(offset)
        if truncate is not None:
            if truncate:
                label = "left"
            else:
                label = "datapoint"
            issue_deprecation_warning(
                f"`truncate` is deprecated and will be removed in a future version."
                f" Please replace `truncate={truncate}` with `label='{label}'` to silence this warning.",
                version="0.19.4",
            )

        index_column = parse_as_expression(index_column)
        if offset is None:
            offset = _negate_duration(_timedelta_to_pl_duration(every))

        if period is None:
            period = every

        period = _timedelta_to_pl_duration(period)
        offset = _timedelta_to_pl_duration(offset)
        every = _timedelta_to_pl_duration(every)

        pyexprs_by = parse_as_list_of_expressions(by) if by is not None else []
        lgb = self._ldf.group_by_dynamic(
            index_column,
            every,
            period,
            offset,
            label,
            include_boundaries,
            closed,
            pyexprs_by,
            start_by,
            check_sorted,
        )
        return LazyGroupBy(lgb)

    def join_asof(
        self,
        other: LazyFrame,
        *,
        left_on: str | None | Expr = None,
        right_on: str | None | Expr = None,
        on: str | None | Expr = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
        tolerance: str | int | float | timedelta | None = None,
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

            A "nearest" search selects the last row in the right DataFrame whose value
            is nearest to the left's key. String keys are not currently supported for a
            nearest search.

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
        strategy : {'backward', 'forward', 'nearest'}
            Join strategy.
        suffix
            Suffix to append to columns with a duplicate name.
        tolerance
            Numeric tolerance. By setting this the join will only be done if the near
            keys are within this distance. If an asof join is done on columns of dtype
            "Date", "Datetime", "Duration" or "Time", use either a datetime.timedelta
            object or the following string language:

                - 1ns   (1 nanosecond)
                - 1us   (1 microsecond)
                - 1ms   (1 millisecond)
                - 1s    (1 second)
                - 1m    (1 minute)
                - 1h    (1 hour)
                - 1d    (1 calendar day)
                - 1w    (1 calendar week)
                - 1mo   (1 calendar month)
                - 1q    (1 calendar quarter)
                - 1y    (1 calendar year)
                - 1i    (1 index count)

                Or combine them:
                "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

                By "calendar day", we mean the corresponding time on the next day
                (which may not be 24 hours, due to daylight savings). Similarly for
                "calendar week", "calendar month", "calendar quarter", and
                "calendar year".

        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both
            DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to
            the join in parallel.

        Examples
        --------
        >>> from datetime import datetime
        >>> gdp = pl.LazyFrame(
        ...     {
        ...         "date": [
        ...             datetime(2016, 1, 1),
        ...             datetime(2017, 1, 1),
        ...             datetime(2018, 1, 1),
        ...             datetime(2019, 1, 1),
        ...         ],  # note record date: Jan 1st (sorted!)
        ...         "gdp": [4164, 4411, 4566, 4696],
        ...     }
        ... ).set_sorted("date")
        >>> population = pl.LazyFrame(
        ...     {
        ...         "date": [
        ...             datetime(2016, 5, 12),
        ...             datetime(2017, 5, 12),
        ...             datetime(2018, 5, 12),
        ...             datetime(2019, 5, 12),
        ...         ],  # note record date: May 12th (sorted!)
        ...         "population": [82.19, 82.66, 83.12, 83.52],
        ...     }
        ... ).set_sorted("date")
        >>> population.join_asof(gdp, on="date", strategy="backward").collect()
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
        tolerance = deprecate_saturating(tolerance)
        if not isinstance(other, LazyFrame):
            raise TypeError(
                f"expected `other` join table to be a LazyFrame, not a {type(other).__name__!r}"
            )

        if isinstance(on, (str, pl.Expr)):
            left_on = on
            right_on = on

        if left_on is None or right_on is None:
            raise ValueError("you should pass the column to join on as an argument")

        if by is not None:
            by_left_ = [by] if isinstance(by, str) else by
            by_right_ = by_left_
        elif (by_left is not None) and (by_right is not None):
            by_left_ = [by_left] if isinstance(by_left, str) else by_left
            by_right_ = [by_right] if isinstance(by_right, str) else by_right
        else:
            # no by
            by_left_ = None
            by_right_ = None

        tolerance_str: str | None = None
        tolerance_num: float | int | None = None
        if isinstance(tolerance, str):
            tolerance_str = tolerance
        elif isinstance(tolerance, timedelta):
            tolerance_str = _timedelta_to_pl_duration(tolerance)
        else:
            tolerance_num = tolerance

        if not isinstance(left_on, pl.Expr):
            left_on = F.col(left_on)
        if not isinstance(right_on, pl.Expr):
            right_on = F.col(right_on)

        return self._from_pyldf(
            self._ldf.join_asof(
                other._ldf,
                left_on._pyexpr,
                right_on._pyexpr,
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
        self,
        other: LazyFrame,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> Self:
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        other
            Lazy DataFrame to join with.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be
            None.
        how : {'inner', 'left', 'outer', 'semi', 'anti', 'cross'}
            Join strategy.

            .. note::
                A left join preserves the row order of the left DataFrame.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        suffix
            Suffix to append to columns with a duplicate name.
        validate: {'m:m', 'm:1', '1:m', '1:1'}
            Checks if join is of specified type.

                * *many_to_many*
                    “m:m”: default, does not result in checks
                * *one_to_one*
                    “1:1”: check if join keys are unique in both left and right datasets
                * *one_to_many*
                    “1:m”: check if join keys are unique in left dataset
                * *many_to_one*
                    “m:1”: check if join keys are unique in right dataset

            .. note::

                - This is currently not supported the streaming engine.
                - This is only supported when joined by single columns.
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> other_lf = pl.LazyFrame(
        ...     {
        ...         "apple": ["x", "y", "z"],
        ...         "ham": ["a", "b", "d"],
        ...     }
        ... )
        >>> lf.join(other_lf, on="ham").collect()
        shape: (2, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘
        >>> lf.join(other_lf, on="ham", how="outer").collect()
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
        >>> lf.join(other_lf, on="ham", how="left").collect()
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
        >>> lf.join(other_lf, on="ham", how="semi").collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        └─────┴─────┴─────┘
        >>> lf.join(other_lf, on="ham", how="anti").collect()
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
                f"expected `other` join table to be a LazyFrame, not a {type(other).__name__!r}"
            )

        if how == "cross":
            return self._from_pyldf(
                self._ldf.join(
                    other._ldf,
                    [],
                    [],
                    allow_parallel,
                    force_parallel,
                    how,
                    suffix,
                    validate,
                )
            )

        if on is not None:
            pyexprs = parse_as_list_of_expressions(on)
            pyexprs_left = pyexprs
            pyexprs_right = pyexprs
        elif left_on is not None and right_on is not None:
            pyexprs_left = parse_as_list_of_expressions(left_on)
            pyexprs_right = parse_as_list_of_expressions(right_on)
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
                validate,
            )
        )

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        """
        Add columns to this DataFrame.

        Added columns will replace existing columns with the same name.

        Parameters
        ----------
        *exprs
            Column(s) to add, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names, other
            non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to add, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        LazyFrame
            A new LazyFrame with the columns added.

        Notes
        -----
        Creating a new LazyFrame using this method does not create a new copy of
        existing data.

        Examples
        --------
        Pass an expression to add it as a new column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> lf.with_columns((pl.col("a") ** 2).alias("a^2")).collect()
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

        >>> lf.with_columns(pl.col("a").cast(pl.Float64)).collect()
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

        >>> lf.with_columns(
        ...     [
        ...         (pl.col("a") ** 2).alias("a^2"),
        ...         (pl.col("b") / 2).alias("b/2"),
        ...         (pl.col("c").not_()).alias("not c"),
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

        >>> lf.with_columns(
        ...     (pl.col("a") ** 2).alias("a^2"),
        ...     (pl.col("b") / 2).alias("b/2"),
        ...     (pl.col("c").not_()).alias("not c"),
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

        >>> lf.with_columns(
        ...     ab=pl.col("a") * pl.col("b"),
        ...     not_c=pl.col("c").not_(),
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
        by enabling the setting `Config.set_auto_structify(True)`:

        >>> with pl.Config(auto_structify=True):
        ...     lf.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().name.suffix("_diff"),
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
        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        pyexprs = parse_as_list_of_expressions(
            *exprs, **named_exprs, __structify=structify
        )
        return self._from_pyldf(self._ldf.with_columns(pyexprs))

    def with_columns_seq(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        """
        Add columns to this DataFrame.

        Added columns will replace existing columns with the same name.

        This will run all expression sequentially instead of in parallel.
        Use this when the work per expression is cheap.

        Parameters
        ----------
        *exprs
            Column(s) to add, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names, other
            non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to add, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        LazyFrame
            A new LazyFrame with the columns added.

        See Also
        --------
        with_columns

        """
        structify = bool(int(os.environ.get("POLARS_AUTO_STRUCTIFY", 0)))

        pyexprs = parse_as_list_of_expressions(
            *exprs, **named_exprs, __structify=structify
        )
        return self._from_pyldf(self._ldf.with_columns_seq(pyexprs))

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
        >>> lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "c", None]})
        >>> lf_other = pl.LazyFrame({"c": ["foo", "ham"]})
        >>> lf.with_context(lf_other).select(
        ...     pl.col("b") + pl.col("c").first()
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

        Fill nulls with the median from another DataFrame:

        >>> train_lf = pl.LazyFrame(
        ...     {"feature_0": [-1.0, 0, 1], "feature_1": [-1.0, 0, 1]}
        ... )
        >>> test_lf = pl.LazyFrame(
        ...     {"feature_0": [-1.0, None, 1], "feature_1": [-1.0, 0, 1]}
        ... )
        >>> test_lf.with_context(
        ...     train_lf.select(pl.all().name.suffix("_train"))
        ... ).select(
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

    def drop(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        """
        Remove columns from the DataFrame.

        Parameters
        ----------
        columns
            Name of the column(s) that should be removed from the DataFrame.
        *more_columns
            Additional columns to drop, specified as positional arguments.

        Examples
        --------
        Drop a single column by passing the name of that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.drop("ham").collect()
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

        Drop multiple columns by passing a selector.

        >>> import polars.selectors as cs
        >>> lf.drop(cs.numeric()).collect()
        shape: (3, 1)
        ┌─────┐
        │ ham │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        │ b   │
        │ c   │
        └─────┘

        Use positional arguments to drop multiple columns.

        >>> lf.drop("foo", "ham").collect()
        shape: (3, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 6.0 │
        │ 7.0 │
        │ 8.0 │
        └─────┘

        """
        drop_cols = _expand_selectors(self, columns, *more_columns)
        return self._from_pyldf(self._ldf.drop(drop_cols))

    def rename(self, mapping: dict[str, str]) -> Self:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name.

        Notes
        -----
        If existing names are swapped (e.g. 'A' points to 'B' and 'B' points to 'A'),
        polars will block projection and predicate pushdowns at this node.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.rename({"foo": "apple"}).collect()
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "key": ["a", "b", "c"],
        ...         "val": [1, 2, 3],
        ...     }
        ... )
        >>> lf.reverse().collect()
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

    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift(
        self, n: int | IntoExprColumn = 1, *, fill_value: IntoExpr | None = None
    ) -> Self:
        """
        Shift values by the given number of indices.

        Parameters
        ----------
        n
            Number of indices to shift forward. If a negative value is passed, values
            are shifted in the opposite direction instead.
        fill_value
            Fill the resulting null values with this value. Accepts expression input.
            Non-expression inputs are parsed as literals.

        Notes
        -----
        This method is similar to the `LAG` operation in SQL when the value for `n`
        is positive. With a negative value for `n`, it is similar to `LEAD`.

        Examples
        --------
        By default, values are shifted forward by one index.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> lf.shift().collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ null ┆ null │
        │ 1    ┆ 5    │
        │ 2    ┆ 6    │
        │ 3    ┆ 7    │
        └──────┴──────┘

        Pass a negative value to shift in the opposite direction instead.

        >>> lf.shift(-2).collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 3    ┆ 7    │
        │ 4    ┆ 8    │
        │ null ┆ null │
        │ null ┆ null │
        └──────┴──────┘

        Specify `fill_value` to fill the resulting null values.

        >>> lf.shift(-2, fill_value=100).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 7   │
        │ 4   ┆ 8   │
        │ 100 ┆ 100 │
        │ 100 ┆ 100 │
        └─────┴─────┘

        """
        if fill_value is not None:
            fill_value = parse_as_expression(fill_value, str_as_lit=True)
        n = parse_as_expression(n)
        return self._from_pyldf(self._ldf.shift(n, fill_value))

    def slice(self, offset: int, length: int | None = None) -> Self:
        """
        Get a slice of this DataFrame.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to `None`, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> lf.slice(1, 2).collect()
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
                f"negative slice lengths ({length!r}) are invalid for LazyFrame"
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.limit().collect()
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
        >>> lf.limit(2).collect()
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.head().collect()
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
        >>> lf.head(2).collect()
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
            Number of rows to return.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.tail().collect()
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
        >>> lf.tail(2).collect()
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.last().collect()
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.first().collect()
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

    def approx_n_unique(self) -> Self:
        """
        Approximate count of unique values.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.approx_n_unique().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘

        """
        return self.select(F.all().approx_n_unique())

    @deprecate_renamed_function("approx_n_unique", version="0.18.12")
    def approx_unique(self) -> Self:
        """
        Approximate count of unique values.

        .. deprecated:: 0.18.12
            This method has been renamed to :func:`LazyFrame.approx_n_unique`.

        """
        return self.approx_n_unique()

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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.with_row_count().collect()
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

    def gather_every(self, n: int) -> Self:
        """
        Take every nth row in the LazyFrame and return as a new LazyFrame.

        Parameters
        ----------
        n
            Gather every *n*-th row.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> lf.gather_every(2).collect()
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
        return self.select(F.col("*").gather_every(n))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        *,
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
            Fill all matching supertypes of the fill `value` literal.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, None, 4],
        ...         "b": [0.5, 4, None, 13],
        ...     }
        ... )
        >>> lf.fill_null(99).collect()
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
        >>> lf.fill_null(strategy="forward").collect()
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

        >>> lf.fill_null(strategy="max").collect()
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

        >>> lf.fill_null(strategy="zero").collect()
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

            if isinstance(value, pl.Expr):
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
                dtypes = [Datetime] + [Datetime(u) for u in DTYPE_TEMPORAL_UNITS]
            elif isinstance(value, timedelta):
                dtypes = [Duration] + [Duration(u) for u in DTYPE_TEMPORAL_UNITS]
            elif isinstance(value, date):
                dtypes = [Date]
            elif isinstance(value, time):
                dtypes = [Time]
            elif isinstance(value, str):
                dtypes = [Utf8, Categorical]
            else:
                # fallback; anything not explicitly handled above
                dtypes = [infer_dtype(F.lit(value))]

            return self.with_columns(F.col(dtypes).fill_null(value, strategy, limit))

        return self.select(F.all().fill_null(value, strategy, limit))

    def fill_nan(self, value: int | float | Expr | None) -> Self:
        """
        Fill floating point NaN values.

        Parameters
        ----------
        value
            Value to fill the NaN values with.

        Warnings
        --------
        Note that floating point NaN (Not a Number) are not missing values!
        To replace missing values, use :func:`fill_null` instead.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1.5, 2, float("nan"), 4],
        ...         "b": [0.5, 4, float("nan"), 13],
        ...     }
        ... )
        >>> lf.fill_nan(99).collect()
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
        if not isinstance(value, pl.Expr):
            value = F.lit(value)
        return self._from_pyldf(self._ldf.fill_nan(value._pyexpr))

    def std(self, ddof: int = 1) -> Self:
        """
        Aggregate the columns in the LazyFrame to their standard deviation value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.std().collect()
        shape: (1, 2)
        ┌──────────┬─────┐
        │ a        ┆ b   │
        │ ---      ┆ --- │
        │ f64      ┆ f64 │
        ╞══════════╪═════╡
        │ 1.290994 ┆ 0.5 │
        └──────────┴─────┘
        >>> lf.std(ddof=0).collect()
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
        Aggregate the columns in the LazyFrame to their variance value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.var().collect()
        shape: (1, 2)
        ┌──────────┬──────┐
        │ a        ┆ b    │
        │ ---      ┆ ---  │
        │ f64      ┆ f64  │
        ╞══════════╪══════╡
        │ 1.666667 ┆ 0.25 │
        └──────────┴──────┘
        >>> lf.var(ddof=0).collect()
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
        Aggregate the columns in the LazyFrame to their maximum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.max().collect()
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
        Aggregate the columns in the LazyFrame to their minimum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.min().collect()
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
        Aggregate the columns in the LazyFrame to their sum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.sum().collect()
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
        Aggregate the columns in the LazyFrame to their mean value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.mean().collect()
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
        Aggregate the columns in the LazyFrame to their median value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.median().collect()
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

    def null_count(self) -> Self:
        """
        Aggregate the columns in the LazyFrame as the sum of their null value count.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, None, 3],
        ...         "bar": [6, 7, None],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.null_count().collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 0   │
        └─────┴─────┴─────┘

        """
        return self._from_pyldf(self._ldf.null_count())

    def quantile(
        self,
        quantile: float | Expr,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> Self:
        """
        Aggregate the columns in the LazyFrame to their quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.quantile(0.7).collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 3.0 ┆ 1.0 │
        └─────┴─────┘

        """
        quantile = parse_as_expression(quantile)
        return self._from_pyldf(self._ldf.quantile(quantile, interpolation))

    def explode(
        self,
        columns: str | Expr | Sequence[str | Expr],
        *more_columns: str | Expr,
    ) -> Self:
        """
        Explode the DataFrame to long format by exploding the given columns.

        Parameters
        ----------
        columns
            Column names, expressions, or a selector defining them. The underlying
            columns being exploded must be of List or Utf8 datatype.
        *more_columns
            Additional names of columns to explode, specified as positional arguments.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "letters": ["a", "a", "b", "c"],
        ...         "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]],
        ...     }
        ... )
        >>> lf.explode("numbers").collect()
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
        columns = parse_as_list_of_expressions(
            *_expand_selectors(self, columns, *more_columns)
        )
        return self._from_pyldf(self._ldf.explode(columns))

    def unique(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        """
        Drop duplicate rows from this DataFrame.

        Parameters
        ----------
        subset
            Column name(s) or selector(s), to consider when identifying
            duplicate rows. If set to `None` (default), use all columns.
        keep : {'first', 'last', 'any', 'none'}
            Which of the duplicate rows to keep.

            * 'any': Does not give any guarantee of which row is kept.
                     This allows more optimizations.
            * 'none': Don't keep duplicate rows.
            * 'first': Keep first unique row.
            * 'last': Keep last unique row.
        maintain_order
            Keep the same order as the original DataFrame. This is more expensive to
            compute.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

        Returns
        -------
        LazyFrame
            LazyFrame with unique rows.

        Warnings
        --------
        This method will fail if there is a column of type `List` in the DataFrame or
        subset.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3, 1],
        ...         "bar": ["a", "a", "a", "a"],
        ...         "ham": ["b", "b", "b", "b"],
        ...     }
        ... )
        >>> lf.unique(maintain_order=True).collect()
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
        >>> lf.unique(subset=["bar", "ham"], maintain_order=True).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> lf.unique(keep="last", maintain_order=True).collect()
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
            subset = _expand_selectors(self, subset)
        return self._from_pyldf(self._ldf.unique(maintain_order, subset, keep))

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        """
        Drop all rows that contain null values.

        The original order of the remaining rows is preserved.

        Parameters
        ----------
        subset
            Column name(s) for which null values are considered.
            If set to `None` (default), use all columns.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, None, 8],
        ...         "ham": ["a", "b", None],
        ...     }
        ... )

        The default behavior of this method is to drop rows where any single
        value of the row is null.

        >>> lf.drop_nulls().collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        This behaviour can be constrained to consider only a subset of columns, as
        defined by name or with a selector. For example, dropping rows if there is
        a null in any of the integer columns:

        >>> import polars.selectors as cs
        >>> lf.drop_nulls(subset=cs.integer()).collect()
        shape: (2, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ a    │
        │ 3   ┆ 8   ┆ null │
        └─────┴─────┴──────┘

        This method drops a row if any single value of the row is null.

        Below are some example snippets that show how you could drop null
        values based on other conditions:

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, None, None, None],
        ...         "b": [1, 2, None, 1],
        ...         "c": [1, None, None, 1],
        ...     }
        ... )
        >>> lf.collect()
        shape: (4, 3)
        ┌──────┬──────┬──────┐
        │ a    ┆ b    ┆ c    │
        │ ---  ┆ ---  ┆ ---  │
        │ f32  ┆ i64  ┆ i64  │
        ╞══════╪══════╪══════╡
        │ null ┆ 1    ┆ 1    │
        │ null ┆ 2    ┆ null │
        │ null ┆ null ┆ null │
        │ null ┆ 1    ┆ 1    │
        └──────┴──────┴──────┘

        Drop a row only if all values are null:

        >>> lf.filter(~pl.all_horizontal(pl.all().is_null())).collect()
        shape: (3, 3)
        ┌──────┬─────┬──────┐
        │ a    ┆ b   ┆ c    │
        │ ---  ┆ --- ┆ ---  │
        │ f32  ┆ i64 ┆ i64  │
        ╞══════╪═════╪══════╡
        │ null ┆ 1   ┆ 1    │
        │ null ┆ 2   ┆ null │
        │ null ┆ 1   ┆ 1    │
        └──────┴─────┴──────┘

        """
        if subset is not None:
            subset = _expand_selectors(self, subset)
        return self._from_pyldf(self._ldf.drop_nulls(subset))

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
        *,
        streamable: bool = True,
    ) -> Self:
        """
        Unpivot a DataFrame from wide to long format.

        Optionally leaves identifiers set.

        This function is useful to massage a DataFrame into a format where one or more
        columns are identifier variables (id_vars) while all other columns, considered
        measured variables (value_vars), are "unpivoted" to the row axis leaving just
        two non-identifier columns, 'variable' and 'value'.

        Parameters
        ----------
        id_vars
            Column(s) or selector(s) to use as identifier variables.
        value_vars
            Column(s) or selector(s) to use as values variables; if `value_vars`
            is empty all columns that are not in `id_vars` will be used.
        variable_name
            Name to give to the `variable` column. Defaults to "variable"
        value_name
            Name to give to the `value` column. Defaults to "value"
        streamable
            Allow this node to run in the streaming engine.
            If this runs in streaming, the output of the melt operation
            will not have a stable ordering.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> import polars.selectors as cs
        >>> lf.melt(id_vars="a", value_vars=cs.numeric()).collect()
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
        value_vars = [] if value_vars is None else _expand_selectors(self, value_vars)
        id_vars = [] if id_vars is None else _expand_selectors(self, id_vars)

        return self._from_pyldf(
            self._ldf.melt(id_vars, value_vars, value_name, variable_name, streamable)
        )

    def map_batches(
        self,
        function: Callable[[DataFrame], DataFrame],
        *,
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
        function
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
            Output schema of the function, if set to `None` we assume that the schema
            will remain unchanged by the applied function.
        validate_output_schema
            It is paramount that polars' schema is correct. This flag will ensure that
            the output schema of this function will be checked with the expected schema.
            Setting this to `False` will not do this check, but may lead to hard to
            debug bugs.
        streamable
            Whether the function that is given is eligible to be running with the
            streaming engine. That means that the function must produce the same result
            when it is executed in batches or when it is be executed on the full
            dataset.

        Warnings
        --------
        The `schema` of a `LazyFrame` must always be correct. It is up to the caller
        of this function to ensure that this invariant is upheld.

        It is important that the optimization flags are correct. If the custom function
        for instance does an aggregation of a column, `predicate_pushdown` should not
        be allowed, as this prunes rows and will influence your aggregation results.

        Examples
        --------
        >>> lf = (  # doctest: +SKIP
        ...     pl.LazyFrame(
        ...         {
        ...             "a": pl.int_range(-100_000, 0, eager=True),
        ...             "b": pl.int_range(0, 100_000, eager=True),
        ...         }
        ...     )
        ...     .map_batches(lambda x: 2 * x, streamable=True)
        ...     .collect(streaming=True)
        ... )
        shape: (100_000, 2)
        ┌─────────┬────────┐
        │ a       ┆ b      │
        │ ---     ┆ ---    │
        │ i64     ┆ i64    │
        ╞═════════╪════════╡
        │ -200000 ┆ 0      │
        │ -199998 ┆ 2      │
        │ -199996 ┆ 4      │
        │ -199994 ┆ 6      │
        │ …       ┆ …      │
        │ -8      ┆ 199992 │
        │ -6      ┆ 199994 │
        │ -4      ┆ 199996 │
        │ -2      ┆ 199998 │
        └─────────┴────────┘

        """
        if no_optimizations:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False

        return self._from_pyldf(
            self._ldf.map_batches(
                function,
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
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, None, 9, 10],
        ...         "bar": [6, 7, 9, None],
        ...         "baz": [1, None, None, 9],
        ...     }
        ... )
        >>> lf.interpolate().collect()
        shape: (4, 3)
        ┌──────┬──────┬──────────┐
        │ foo  ┆ bar  ┆ baz      │
        │ ---  ┆ ---  ┆ ---      │
        │ f64  ┆ f64  ┆ f64      │
        ╞══════╪══════╪══════════╡
        │ 1.0  ┆ 6.0  ┆ 1.0      │
        │ 5.0  ┆ 7.0  ┆ 3.666667 │
        │ 9.0  ┆ 9.0  ┆ 6.333333 │
        │ 10.0 ┆ null ┆ 9.0      │
        └──────┴──────┴──────────┘

        """
        return self.select(F.col("*").interpolate())

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        """
        Decompose struct columns into separate columns for each of their fields.

        The new columns will be inserted into the DataFrame at the location of the
        struct column.

        Parameters
        ----------
        columns
            Name of the struct column(s) that should be unnested.
        *more_columns
            Additional columns to unnest, specified as positional arguments.

        Examples
        --------
        >>> df = pl.LazyFrame(
        ...     {
        ...         "before": ["foo", "bar"],
        ...         "t_a": [1, 2],
        ...         "t_b": ["a", "b"],
        ...         "t_c": [True, None],
        ...         "t_d": [[1, 2], [3]],
        ...         "after": ["baz", "womp"],
        ...     }
        ... ).select("before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after")
        >>> df.collect()
        shape: (2, 3)
        ┌────────┬─────────────────────┬───────┐
        │ before ┆ t_struct            ┆ after │
        │ ---    ┆ ---                 ┆ ---   │
        │ str    ┆ struct[4]           ┆ str   │
        ╞════════╪═════════════════════╪═══════╡
        │ foo    ┆ {1,"a",true,[1, 2]} ┆ baz   │
        │ bar    ┆ {2,"b",null,[3]}    ┆ womp  │
        └────────┴─────────────────────┴───────┘
        >>> df.unnest("t_struct").collect()
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
        columns = _expand_selectors(self, columns, *more_columns)
        return self._from_pyldf(self._ldf.unnest(columns))

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

        Examples
        --------
        >>> df0 = pl.LazyFrame(
        ...     {"name": ["steve", "elise", "bob"], "age": [42, 44, 18]}
        ... ).sort("age")
        >>> df0.collect()
        shape: (3, 2)
        ┌───────┬─────┐
        │ name  ┆ age │
        │ ---   ┆ --- │
        │ str   ┆ i64 │
        ╞═══════╪═════╡
        │ bob   ┆ 18  │
        │ steve ┆ 42  │
        │ elise ┆ 44  │
        └───────┴─────┘
        >>> df1 = pl.LazyFrame(
        ...     {"name": ["anna", "megan", "steve", "thomas"], "age": [21, 33, 42, 20]}
        ... ).sort("age")
        >>> df1.collect()
        shape: (4, 2)
        ┌────────┬─────┐
        │ name   ┆ age │
        │ ---    ┆ --- │
        │ str    ┆ i64 │
        ╞════════╪═════╡
        │ thomas ┆ 20  │
        │ anna   ┆ 21  │
        │ megan  ┆ 33  │
        │ steve  ┆ 42  │
        └────────┴─────┘
        >>> df0.merge_sorted(df1, key="age").collect()
        shape: (7, 2)
        ┌────────┬─────┐
        │ name   ┆ age │
        │ ---    ┆ --- │
        │ str    ┆ i64 │
        ╞════════╪═════╡
        │ bob    ┆ 18  │
        │ thomas ┆ 20  │
        │ anna   ┆ 21  │
        │ megan  ┆ 33  │
        │ steve  ┆ 42  │
        │ steve  ┆ 42  │
        │ elise  ┆ 44  │
        └────────┴─────┘
        """
        return self._from_pyldf(self._ldf.merge_sorted(other._ldf, key))

    def set_sorted(
        self,
        column: str | Iterable[str],
        *more_columns: str,
        descending: bool = False,
    ) -> Self:
        """
        Indicate that one or multiple columns are sorted.

        Parameters
        ----------
        column
            Columns that are sorted
        more_columns
            Additional columns that are sorted, specified as positional arguments.
        descending
            Whether the columns are sorted in descending order.
        """
        columns = parse_as_list_of_expressions(column, *more_columns)

        return self.with_columns(
            [wrap_expr(e).set_sorted(descending=descending) for e in columns]
        )

    def update(
        self,
        other: LazyFrame,
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "outer"] = "left",
        include_nulls: bool | None = False,
    ) -> Self:
        """
        Update the values in this `LazyFrame` with the non-null values in `other`.

        Parameters
        ----------
        other
            LazyFrame that will be used to update the values
        on
            Column names that will be joined on; if given `None` the implicit row
            index is used as a join key instead.
        left_on
           Join column(s) of the left DataFrame.
        right_on
           Join column(s) of the right DataFrame.
        how : {'left', 'inner', 'outer'}
            * 'left' will keep all rows from the left table; rows may be duplicated
              if multiple rows in the right frame match the left row's key.
            * 'inner' keeps only those rows where the key exists in both frames.
            * 'outer' will update existing rows where the key matches while also
              adding any new rows contained in the given frame.
        include_nulls
            If True, null values from the right DataFrame will be used to update the
            left DataFrame.

        Notes
        -----
        This is syntactic sugar for a left/inner join, with an optional coalesce when
        `include_nulls = False`.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "A": [1, 2, 3, 4],
        ...         "B": [400, 500, 600, 700],
        ...     }
        ... )
        >>> lf.collect()
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
        >>> new_lf = pl.LazyFrame(
        ...     {
        ...         "B": [-66, None, -99],
        ...         "C": [5, 3, 1],
        ...     }
        ... )

        Update `df` values with the non-null values in `new_df`, by row index:

        >>> lf.update(new_lf).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -66 │
        │ 2   ┆ 500 │
        │ 3   ┆ -99 │
        │ 4   ┆ 700 │
        └─────┴─────┘

        Update `df` values with the non-null values in `new_df`, by row index,
        but only keeping those rows that are common to both frames:

        >>> lf.update(new_lf, how="inner").collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -66 │
        │ 2   ┆ 500 │
        │ 3   ┆ -99 │
        └─────┴─────┘

        Update `df` values with the non-null values in `new_df`, using an outer join
        strategy that defines explicit join columns in each frame:

        >>> lf.update(new_lf, left_on=["A"], right_on=["C"], how="outer").collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -99 │
        │ 2   ┆ 500 │
        │ 3   ┆ 600 │
        │ 4   ┆ 700 │
        │ 5   ┆ -66 │
        └─────┴─────┘

        Update `df` values including null values in `new_df`, using an outer join
        strategy that defines explicit join columns in each frame:

        >>> lf.update(
        ...     new_lf, left_on="A", right_on="C", how="outer", include_nulls=True
        ... ).collect()
        shape: (5, 2)
        ┌─────┬──────┐
        │ A   ┆ B    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 1   ┆ -99  │
        │ 2   ┆ 500  │
        │ 3   ┆ null │
        │ 4   ┆ 700  │
        │ 5   ┆ -66  │
        └─────┴──────┘

        """
        if how not in ("left", "inner", "outer"):
            raise ValueError(
                f"`how` must be one of {{'left', 'inner', 'outer'}}; found {how!r}"
            )

        row_count_used = False
        if on is None:
            if left_on is None and right_on is None:
                # no keys provided--use row count
                row_count_used = True
                row_count_name = "__POLARS_ROW_COUNT"
                self = self.with_row_count(row_count_name)
                other = other.with_row_count(row_count_name)
                left_on = right_on = [row_count_name]
            else:
                # one of left or right is missing, raise error
                if left_on is None:
                    raise ValueError("missing join columns for left frame")
                if right_on is None:
                    raise ValueError("missing join columns for right frame")
        else:
            # move on into left/right_on to simplify logic
            left_on = right_on = on

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        left_names = self.columns
        for name in left_on:
            if name not in left_names:
                raise ValueError(f"left join column {name!r} not found")
        right_names = other.columns
        for name in right_on:
            if name not in right_names:
                raise ValueError(f"right join column {name!r} not found")

        # no need to join if *only* join columns are in other (inner/left update only)
        if how != "outer" and len(other.columns) == len(right_on):
            if row_count_used:
                return self.drop(row_count_name)
            return self

        # only use non-idx right columns present in left frame
        right_other = set(other.columns).intersection(self.columns) - set(right_on)

        # When include_nulls is True, we need to distinguish records after the join that
        # were originally null in the right frame, as opposed to records that were null
        # because the key was missing from the right frame.
        # Add a validity column to track whether row was matched or not.
        if include_nulls:
            validity = ("__POLARS_VALIDITY",)
            other = other.with_columns(F.lit(True).alias(validity[0]))
        else:
            validity = ()  # type: ignore[assignment]

        tmp_name = "__POLARS_RIGHT"
        drop_columns = [*(f"{name}{tmp_name}" for name in right_other), *validity]
        result = (
            self.join(
                other.select(*right_on, *right_other, *validity),
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffix=tmp_name,
            )
            .with_columns(
                (
                    # use left value only when right value failed to join
                    F.when(F.col(validity).is_null())
                    .then(F.col(name))
                    .otherwise(F.col(f"{name}{tmp_name}"))
                    if include_nulls
                    else F.coalesce([f"{name}{tmp_name}", F.col(name)])
                ).alias(name)
                for name in right_other
            )
            .drop(drop_columns)
        )
        if row_count_used:
            result = result.drop(row_count_name)

        return self._from_pyldf(result._ldf)

    @deprecate_renamed_function("group_by", version="0.19.0")
    def groupby(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        maintain_order: bool = False,
    ) -> LazyGroupBy:
        """
        Start a group by operation.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`LazyFrame.group_by`.

        Parameters
        ----------
        by
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        *more_by
            Additional columns to group by, specified as positional arguments.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default group by.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

        """
        return self.group_by(by, *more_by, maintain_order=maintain_order)

    @deprecate_renamed_function("rolling", version="0.19.0")
    def groupby_rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        check_sorted: bool = True,
    ) -> LazyGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`LazyFrame.rolling`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window - must be non-negative
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        check_sorted
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the by groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        """
        return self.rolling(
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            by=by,
            check_sorted=check_sorted,
        )

    @deprecate_renamed_function("rolling", version="0.19.9")
    def group_by_rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        check_sorted: bool = True,
    ) -> LazyGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.9
            This method has been renamed to :func:`LazyFrame.rolling`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window - must be non-negative
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        check_sorted
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the by groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        """
        return self.rolling(
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            by=by,
            check_sorted=check_sorted,
        )

    @deprecate_renamed_function("group_by_dynamic", version="0.19.0")
    def groupby_dynamic(
        self,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        start_by: StartBy = "window",
        check_sorted: bool = True,
    ) -> LazyGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`LazyFrame.group_by_dynamic`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a dynamic group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it will equal 'every'
        offset
            offset of the window, only takes effect if `start_by` is `'window'`.
            Defaults to negative `every`.
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
        start_by : {'window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
            The strategy to determine the start of the first window by.

            * 'window': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * 'datapoint': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `'w'`):

              * 'monday': Start the window on the Monday before the first data point.
              * 'tuesday': Start the window on the Tuesday before the first data point.
              * ...
              * 'sunday': Start the window on the Sunday before the first data point.
        check_sorted
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the by groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        """  # noqa: W505
        return self.group_by_dynamic(
            index_column,
            every=every,
            period=period,
            offset=offset,
            truncate=truncate,
            include_boundaries=include_boundaries,
            closed=closed,
            by=by,
            start_by=start_by,
            check_sorted=check_sorted,
        )

    @deprecate_renamed_function("map_batches", version="0.19.0")
    def map(
        self,
        function: Callable[[DataFrame], DataFrame],
        *,
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

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`LazyFrame.map_batches`.

        Parameters
        ----------
        function
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
            Output schema of the function, if set to `None` we assume that the schema
            will remain unchanged by the applied function.
        validate_output_schema
            It is paramount that polars' schema is correct. This flag will ensure that
            the output schema of this function will be checked with the expected schema.
            Setting this to `False` will not do this check, but may lead to hard to
            debug bugs.
        streamable
            Whether the function that is given is eligible to be running with the
            streaming engine. That means that the function must produce the same result
            when it is executed in batches or when it is be executed on the full
            dataset.

        """
        return self.map_batches(
            function,
            predicate_pushdown=predicate_pushdown,
            projection_pushdown=projection_pushdown,
            slice_pushdown=slice_pushdown,
            no_optimizations=no_optimizations,
            schema=schema,
            validate_output_schema=validate_output_schema,
            streamable=streamable,
        )

    @deprecate_function("Use `shift` instead.", version="0.19.12")
    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift_and_fill(
        self,
        fill_value: Expr | int | str | float,
        *,
        n: int = 1,
    ) -> Self:
        """
        Shift values by the given number of places and fill the resulting null values.

        .. deprecated:: 0.19.12
            Use :func:`shift` instead.

        Parameters
        ----------
        fill_value
            fill None values with the result of this expression.
        n
            Number of places to shift (may be negative).

        """
        return self.shift(n, fill_value=fill_value)

    @deprecate_renamed_function("gather_every", version="0.19.14")
    def take_every(self, n: int) -> Self:
        """
        Take every nth row in the LazyFrame and return as a new LazyFrame.

        .. deprecated:: 0.19.0
            This method has been renamed to :meth:`gather_every`.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        """
        return self.gather_every(n)

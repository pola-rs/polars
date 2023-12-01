"""Module containing logic related to eager DataFrames."""
from __future__ import annotations

import contextlib
import os
import random
from collections import OrderedDict, defaultdict
from collections.abc import Sized
from io import BytesIO, StringIO, TextIOWrapper
from operator import itemgetter
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import polars._reexport as pl
from polars import functions as F
from polars.dataframe._html import NotebookFormatter
from polars.dataframe.group_by import DynamicGroupBy, GroupBy, RollingGroupBy
from polars.datatypes import (
    INTEGER_DTYPES,
    N_INFER_DEFAULT,
    NUMERIC_DTYPES,
    Boolean,
    Float64,
    Object,
    Utf8,
    py_type_to_dtype,
)
from polars.dependencies import (
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
    _check_for_pyarrow,
    dataframe_api_compat,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import NoRowsReturnedError, TooManyRowsReturnedError
from polars.functions import col, lit
from polars.io._utils import _is_glob_pattern, _is_local_file
from polars.io.csv._utils import _check_arg_is_1byte
from polars.io.spreadsheet._write_utils import (
    _unpack_multi_column_dict,
    _xl_apply_conditional_formats,
    _xl_inject_sparklines,
    _xl_setup_table_columns,
    _xl_setup_table_options,
    _xl_setup_workbook,
    _xl_unique_table_name,
    _XLFormatCache,
)
from polars.selectors import _expand_selector_dicts, _expand_selectors
from polars.slice import PolarsSlice
from polars.utils._construction import (
    _post_apply_columns,
    arrow_to_pydf,
    dict_to_pydf,
    iterable_to_pydf,
    numpy_to_idxs,
    numpy_to_pydf,
    pandas_to_pydf,
    sequence_to_pydf,
    series_to_pydf,
)
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr, wrap_ldf, wrap_s
from polars.utils.convert import _timedelta_to_pl_duration
from polars.utils.deprecation import (
    deprecate_function,
    deprecate_nonkeyword_arguments,
    deprecate_renamed_function,
    deprecate_renamed_parameter,
    deprecate_saturating,
    issue_deprecation_warning,
)
from polars.utils.various import (
    _prepare_row_count_args,
    _process_null_values,
    can_create_dicts_with_pyarrow,
    handle_projection_columns,
    is_bool_sequence,
    is_int_sequence,
    is_str_sequence,
    normalize_filepath,
    parse_percentiles,
    parse_version,
    range_to_slice,
    scale_bytes,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame
    from polars.polars import dtype_str_repr as _dtype_str_repr

if TYPE_CHECKING:
    import sys
    from datetime import timedelta
    from io import IOBase
    from typing import Literal

    import deltalake
    from xlsxwriter import Workbook

    from polars import Expr, LazyFrame, Series
    from polars.interchange.dataframe import PolarsDataFrame
    from polars.type_aliases import (
        AsofJoinStrategy,
        AvroCompression,
        ClosedInterval,
        ColumnFormatDict,
        ColumnNameOrSelector,
        ColumnTotalsDefinition,
        ColumnWidthsDefinition,
        ComparisonOperator,
        ConditionalFormatDict,
        CsvEncoding,
        CsvQuoteStyle,
        DbWriteEngine,
        DbWriteMode,
        FillNullStrategy,
        FrameInitTypes,
        IndexOrder,
        IntoExpr,
        IntoExprColumn,
        IpcCompression,
        JoinStrategy,
        JoinValidation,
        Label,
        NullStrategy,
        OneOrMoreDataTypes,
        Orientation,
        ParallelStrategy,
        ParquetCompression,
        PivotAgg,
        PolarsDataType,
        RollingInterpolationMethod,
        RowTotalsDefinition,
        SchemaDefinition,
        SchemaDict,
        SelectorType,
        SizeUnit,
        StartBy,
        UniqueKeepStrategy,
        UnstackDirection,
    )

    if sys.version_info >= (3, 10):
        from typing import Concatenate, ParamSpec, TypeAlias
    else:
        from typing_extensions import Concatenate, ParamSpec, TypeAlias

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    # these aliases are used to annotate DataFrame.__getitem__()
    # MultiRowSelector indexes into the vertical axis and
    # MultiColSelector indexes into the horizontal axis
    # NOTE: wrapping these as strings is necessary for Python <3.10

    MultiRowSelector: TypeAlias = Union[slice, range, "list[int]", "Series"]
    MultiColSelector: TypeAlias = Union[
        slice, range, "list[int]", "list[str]", "list[bool]", "Series"
    ]

    T = TypeVar("T")
    P = ParamSpec("P")


class DataFrame:
    """
    Two-dimensional data structure representing data as a table with rows and columns.

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

    Examples
    --------
    Constructing a DataFrame from a dictionary:

    >>> data = {"a": [1, 2], "b": [3, 4]}
    >>> df = pl.DataFrame(data)
    >>> df
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

    >>> df.dtypes
    [Int64, Int64]

    To specify a more detailed/specific frame schema you can supply the `schema`
    parameter with a dictionary of (name,dtype) pairs...

    >>> data = {"col1": [0, 2], "col2": [3, 7]}
    >>> df2 = pl.DataFrame(data, schema={"col1": pl.Float32, "col2": pl.Int64})
    >>> df2
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
    >>> df3 = pl.DataFrame(data, schema=[("col1", pl.Float32), ("col2", pl.Int64)])
    >>> df3
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
    >>> df4 = pl.DataFrame(data)
    >>> df4
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    Constructing a DataFrame from a numpy ndarray, specifying column names:

    >>> import numpy as np
    >>> data = np.array([(1, 2), (3, 4)], dtype=np.int64)
    >>> df5 = pl.DataFrame(data, schema=["a", "b"], orient="col")
    >>> df5
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    Constructing a DataFrame from a list of lists, row orientation inferred:

    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> df6 = pl.DataFrame(data, schema=["a", "b", "c"])
    >>> df6
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    │ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┘

    Notes
    -----
    Some methods internally convert the DataFrame into a LazyFrame before collecting
    the results back into a DataFrame. This can lead to unexpected behavior when using
    a subclassed DataFrame. For example,

    >>> class MyDataFrame(pl.DataFrame):
    ...     pass
    ...
    >>> isinstance(MyDataFrame().lazy().collect(), MyDataFrame)
    False

    """

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
        if data is None:
            self._df = dict_to_pydf(
                {}, schema=schema, schema_overrides=schema_overrides
            )

        elif isinstance(data, dict):
            self._df = dict_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                nan_to_null=nan_to_null,
            )

        elif isinstance(data, (list, tuple, Sequence)):
            self._df = sequence_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                orient=orient,
                infer_schema_length=infer_schema_length,
            )
        elif isinstance(data, pl.Series):
            self._df = series_to_pydf(
                data, schema=schema, schema_overrides=schema_overrides
            )

        elif _check_for_numpy(data) and isinstance(data, np.ndarray):
            self._df = numpy_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                orient=orient,
                nan_to_null=nan_to_null,
            )

        elif _check_for_pyarrow(data) and isinstance(data, pa.Table):
            self._df = arrow_to_pydf(
                data, schema=schema, schema_overrides=schema_overrides
            )

        elif _check_for_pandas(data) and isinstance(data, pd.DataFrame):
            self._df = pandas_to_pydf(
                data, schema=schema, schema_overrides=schema_overrides
            )

        elif not isinstance(data, Sized) and isinstance(data, (Generator, Iterable)):
            self._df = iterable_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                orient=orient,
                infer_schema_length=infer_schema_length,
            )
        else:
            raise TypeError(
                f"DataFrame constructor called with unsupported type {type(data).__name__!r}"
                " for the `data` parameter"
            )

    @classmethod
    def _from_pydf(cls, py_df: PyDataFrame) -> Self:
        """Construct Polars DataFrame from FFI PyDataFrame object."""
        df = cls.__new__(cls)
        df._df = py_df
        return df

    @classmethod
    def _from_dicts(
        cls,
        data: Sequence[dict[str, Any]],
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
    ) -> Self:
        pydf = PyDataFrame.read_dicts(
            data, infer_schema_length, schema, schema_overrides
        )
        if schema or schema_overrides:
            pydf = _post_apply_columns(
                pydf, list(schema or pydf.columns()), schema_overrides=schema_overrides
            )
        return cls._from_pydf(pydf)

    @classmethod
    def _from_dict(
        cls,
        data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series],
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
    ) -> Self:
        """
        Construct a DataFrame from a dictionary of sequences.

        Parameters
        ----------
        data : dict of sequences
          Two-dimensional data represented as a dictionary. dict must contain
          Sequences.
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
          any dtypes inferred from the columns param will be overridden.

        """
        return cls._from_pydf(
            dict_to_pydf(data, schema=schema, schema_overrides=schema_overrides)
        )

    @classmethod
    def _from_records(
        cls,
        data: Sequence[Any],
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        orient: Orientation | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
    ) -> Self:
        """
        Construct a DataFrame from a sequence of sequences.

        Parameters
        ----------
        data : Sequence of sequences
            Two-dimensional data represented as a sequence of sequences.
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
            any dtypes inferred from the columns param will be overridden.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows. If None,
            the orientation is inferred by matching the columns and data dimensions. If
            this does not yield conclusive results, column orientation is used.
        infer_schema_length
            How many rows to scan to determine the column type.

        """
        return cls._from_pydf(
            sequence_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                orient=orient,
                infer_schema_length=infer_schema_length,
            )
        )

    @classmethod
    def _from_numpy(
        cls,
        data: np.ndarray[Any, Any],
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        orient: Orientation | None = None,
    ) -> Self:
        """
        Construct a DataFrame from a numpy ndarray.

        Parameters
        ----------
        data : numpy ndarray
            Two-dimensional data represented as a numpy ndarray.
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
            any dtypes inferred from the columns param will be overridden.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows. If None,
            the orientation is inferred by matching the columns and data dimensions. If
            this does not yield conclusive results, column orientation is used.

        """
        return cls._from_pydf(
            numpy_to_pydf(
                data, schema=schema, schema_overrides=schema_overrides, orient=orient
            )
        )

    @classmethod
    def _from_arrow(
        cls,
        data: pa.Table,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        rechunk: bool = True,
    ) -> Self:
        """
        Construct a DataFrame from an Arrow table.

        This operation will be zero copy for the most part. Types that are not
        supported by Polars may be cast to the closest supported type.

        Parameters
        ----------
        data : arrow table, array, or sequence of sequences
            Data representing an Arrow Table or Array.
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
            any dtypes inferred from the columns param will be overridden.
        rechunk : bool, default True
            Make sure that all data is in contiguous memory.

        """
        return cls._from_pydf(
            arrow_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                rechunk=rechunk,
            )
        )

    @classmethod
    def _from_pandas(
        cls,
        data: pd.DataFrame,
        schema: SchemaDefinition | None = None,
        *,
        schema_overrides: SchemaDict | None = None,
        rechunk: bool = True,
        nan_to_null: bool = True,
        include_index: bool = False,
    ) -> Self:
        """
        Construct a Polars DataFrame from a pandas DataFrame.

        Parameters
        ----------
        data : pandas DataFrame
            Two-dimensional data represented as a pandas DataFrame.
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
            any dtypes inferred from the columns param will be overridden.
        rechunk : bool, default True
            Make sure that all data is in contiguous memory.
        nan_to_null : bool, default True
            If the data contains NaN values they will be converted to null/None.
        include_index : bool, default False
            Load any non-default pandas indexes as columns.

        """
        return cls._from_pydf(
            pandas_to_pydf(
                data,
                schema=schema,
                schema_overrides=schema_overrides,
                rechunk=rechunk,
                nan_to_null=nan_to_null,
                include_index=include_index,
            )
        )

    @classmethod
    def _read_csv(
        cls,
        source: str | Path | BinaryIO | bytes,
        *,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        separator: str = ",",
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        dtypes: None | (SchemaDict | Sequence[PolarsDataType]) = None,
        schema: None | SchemaDict = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        try_parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        batch_size: int = 8192,
        n_rows: int | None = None,
        encoding: CsvEncoding = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
    ) -> DataFrame:
        """
        Read a CSV file into a DataFrame.

        Use `pl.read_csv` to dispatch to this method.

        See Also
        --------
        polars.io.read_csv

        """
        self = cls.__new__(cls)

        path: str | None
        if isinstance(source, (str, Path)):
            path = normalize_filepath(source)
        else:
            path = None
            if isinstance(source, BytesIO):
                source = source.getvalue()
            if isinstance(source, StringIO):
                source = source.getvalue().encode()

        dtype_list: Sequence[tuple[str, PolarsDataType]] | None = None
        dtype_slice: Sequence[PolarsDataType] | None = None
        if dtypes is not None:
            if isinstance(dtypes, dict):
                dtype_list = []
                for k, v in dtypes.items():
                    dtype_list.append((k, py_type_to_dtype(v)))
            elif isinstance(dtypes, Sequence):
                dtype_slice = dtypes
            else:
                raise TypeError(
                    f"`dtypes` should be of type list or dict, got {type(dtypes).__name__!r}"
                )

        processed_null_values = _process_null_values(null_values)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(source, str) and _is_glob_pattern(source):
            dtypes_dict = None
            if dtype_list is not None:
                dtypes_dict = dict(dtype_list)
            if dtype_slice is not None:
                raise ValueError(
                    "cannot use glob patterns and unnamed dtypes as `dtypes` argument"
                    "\n\nUse `dtypes`: Mapping[str, Type[DataType]]"
                )
            from polars import scan_csv

            scan = scan_csv(
                source,
                has_header=has_header,
                separator=separator,
                comment_prefix=comment_prefix,
                quote_char=quote_char,
                skip_rows=skip_rows,
                dtypes=dtypes_dict,
                schema=schema,
                null_values=null_values,
                missing_utf8_is_empty_string=missing_utf8_is_empty_string,
                ignore_errors=ignore_errors,
                infer_schema_length=infer_schema_length,
                n_rows=n_rows,
                low_memory=low_memory,
                rechunk=rechunk,
                skip_rows_after_header=skip_rows_after_header,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                eol_char=eol_char,
                raise_if_empty=raise_if_empty,
                truncate_ragged_lines=truncate_ragged_lines,
            )
            if columns is None:
                return scan.collect()
            elif is_str_sequence(columns, allow_str=False):
                return scan.select(columns).collect()
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns` argument"
                    "\n\nUse columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)

        self._df = PyDataFrame.read_csv(
            source,
            infer_schema_length,
            batch_size,
            has_header,
            ignore_errors,
            n_rows,
            skip_rows,
            projection,
            separator,
            rechunk,
            columns,
            encoding,
            n_threads,
            path,
            dtype_list,
            dtype_slice,
            low_memory,
            comment_prefix,
            quote_char,
            processed_null_values,
            missing_utf8_is_empty_string,
            try_parse_dates,
            skip_rows_after_header,
            _prepare_row_count_args(row_count_name, row_count_offset),
            sample_size=sample_size,
            eol_char=eol_char,
            raise_if_empty=raise_if_empty,
            truncate_ragged_lines=truncate_ragged_lines,
            schema=schema,
        )
        return self

    @classmethod
    def _read_parquet(
        cls,
        source: str | Path | BinaryIO | bytes,
        *,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
        parallel: ParallelStrategy = "auto",
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        low_memory: bool = False,
        use_statistics: bool = True,
        rechunk: bool = True,
    ) -> DataFrame:
        """
        Read into a DataFrame from a parquet file.

        Use `pl.read_parquet` to dispatch to this method.

        See Also
        --------
        polars.io.read_parquet

        """
        if isinstance(source, (str, Path)):
            source = normalize_filepath(source)
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(source, str) and _is_glob_pattern(source):
            from polars import scan_parquet

            scan = scan_parquet(
                source,
                n_rows=n_rows,
                rechunk=True,
                parallel=parallel,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                low_memory=low_memory,
            )

            if columns is None:
                return scan.collect()
            elif is_str_sequence(columns, allow_str=False):
                return scan.select(columns).collect()
            else:
                raise TypeError(
                    "cannot use glob patterns and integer based projection as `columns` argument"
                    "\n\nUse columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_parquet(
            source,
            columns,
            projection,
            n_rows,
            parallel,
            _prepare_row_count_args(row_count_name, row_count_offset),
            low_memory=low_memory,
            use_statistics=use_statistics,
            rechunk=rechunk,
        )
        return self

    @classmethod
    def _read_avro(
        cls,
        source: str | Path | BinaryIO | bytes,
        *,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
    ) -> Self:
        """
        Read into a DataFrame from Apache Avro format.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).
        columns
            Columns.
        n_rows
            Stop reading from Apache Avro file after reading `n_rows`.

        """
        if isinstance(source, (str, Path)):
            source = normalize_filepath(source)
        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_avro(source, columns, projection, n_rows)
        return self

    @classmethod
    def _read_ipc(
        cls,
        source: str | Path | BinaryIO | bytes,
        *,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        rechunk: bool = True,
        memory_map: bool = True,
    ) -> Self:
        """
        Read into a DataFrame from Arrow IPC file format.

        See "File or Random Access format" on https://arrow.apache.org/docs/python/ipc.html.
        Arrow IPC files are also known as Feather (v2) files.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a
            list of column names.
        n_rows
            Stop reading from IPC file after reading `n_rows`.
        row_count_name
            Row count name.
        row_count_offset
            Row count offset.
        rechunk
            Make sure that all data is contiguous.
        memory_map
            Memory map the file

        """
        if isinstance(source, (str, Path)):
            source = normalize_filepath(source)
        if isinstance(columns, str):
            columns = [columns]

        if (
            isinstance(source, str)
            and _is_glob_pattern(source)
            and _is_local_file(source)
        ):
            from polars import scan_ipc

            scan = scan_ipc(
                source,
                n_rows=n_rows,
                rechunk=rechunk,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                memory_map=memory_map,
            )
            if columns is None:
                df = scan.collect()
            elif is_str_sequence(columns, allow_str=False):
                df = scan.select(columns).collect()
            else:
                raise TypeError(
                    "cannot use glob patterns and integer based projection as `columns` argument"
                    "\n\nUse columns: List[str]"
                )
            return cls._from_pydf(df._df)

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ipc(
            source,
            columns,
            projection,
            n_rows,
            _prepare_row_count_args(row_count_name, row_count_offset),
            memory_map=memory_map,
        )
        return self

    @classmethod
    def _read_ipc_stream(
        cls,
        source: str | Path | BinaryIO | bytes,
        *,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        rechunk: bool = True,
    ) -> Self:
        """
        Read into a DataFrame from Arrow IPC record batch stream format.

        See "Streaming format" on https://arrow.apache.org/docs/python/ipc.html.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a
            list of column names.
        n_rows
            Stop reading from IPC stream after reading `n_rows`.
        row_count_name
            Row count name.
        row_count_offset
            Row count offset.
        rechunk
            Make sure that all data is contiguous.

        """
        if isinstance(source, (str, Path)):
            source = normalize_filepath(source)
        if isinstance(columns, str):
            columns = [columns]

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ipc_stream(
            source,
            columns,
            projection,
            n_rows,
            _prepare_row_count_args(row_count_name, row_count_offset),
            rechunk,
        )
        return self

    @classmethod
    def _read_json(
        cls,
        source: str | Path | IOBase | bytes,
        *,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDefinition | None = None,
    ) -> Self:
        """
        Read into a DataFrame from a JSON file.

        Use `pl.read_json` to dispatch to this method.

        See Also
        --------
        polars.io.read_json

        """
        if isinstance(source, StringIO):
            source = BytesIO(source.getvalue().encode())
        elif isinstance(source, (str, Path)):
            source = normalize_filepath(source)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_json(
            source,
            infer_schema_length=infer_schema_length,
            schema=schema,
            schema_overrides=schema_overrides,
        )
        return self

    @classmethod
    def _read_ndjson(
        cls,
        source: str | Path | IOBase | bytes,
        *,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDefinition | None = None,
        ignore_errors: bool = False,
    ) -> Self:
        """
        Read into a DataFrame from a newline delimited JSON file.

        Use `pl.read_ndjson` to dispatch to this method.

        See Also
        --------
        polars.io.read_ndjson

        """
        if isinstance(source, StringIO):
            source = BytesIO(source.getvalue().encode())
        elif isinstance(source, (str, Path)):
            source = normalize_filepath(source)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ndjson(
            source,
            ignore_errors=ignore_errors,
            schema=schema,
            schema_overrides=schema_overrides,
        )
        return self

    def _replace(self, column: str, new_column: Series) -> Self:
        """Replace a column by a new Series (in place)."""
        self._df.replace(column, new_column._s)
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> df.shape
        (5, 1)

        """
        return self._df.shape()

    @property
    def height(self) -> int:
        """
        Get the height of the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> df.height
        5

        """
        return self._df.height()

    @property
    def width(self) -> int:
        """
        Get the width of the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> df.width
        1

        """
        return self._df.width()

    @property
    def columns(self) -> list[str]:
        """
        Get or set column names.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.columns
        ['foo', 'bar', 'ham']

        Set column names:

        >>> df.columns = ["apple", "banana", "orange"]
        >>> df
        shape: (3, 3)
        ┌───────┬────────┬────────┐
        │ apple ┆ banana ┆ orange │
        │ ---   ┆ ---    ┆ ---    │
        │ i64   ┆ i64    ┆ str    │
        ╞═══════╪════════╪════════╡
        │ 1     ┆ 6      ┆ a      │
        │ 2     ┆ 7      ┆ b      │
        │ 3     ┆ 8      ┆ c      │
        └───────┴────────┴────────┘

        """
        return self._df.columns()

    @columns.setter
    def columns(self, names: Sequence[str]) -> None:
        """
        Change the column names of the `DataFrame`.

        Parameters
        ----------
        names
            A list with new names for the `DataFrame`.
            The length of the list should be equal to the width of the `DataFrame`.

        """
        self._df.set_column_names(names)

    @property
    def dtypes(self) -> list[PolarsDataType]:
        """
        Get the datatypes of the columns of this DataFrame.

        The datatypes can also be found in column headers when printing the DataFrame.

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.dtypes
        [Int64, Float64, Utf8]
        >>> df
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        """
        return self._df.dtypes()

    @property
    def flags(self) -> dict[str, dict[str, bool]]:
        """
        Get flags that are set on the columns of this DataFrame.

        Returns
        -------
        dict
            Mapping from column names to column flags.
        """
        return {name: self[name].flags for name in self.columns}

    @property
    def schema(self) -> SchemaDict:
        """
        Get a dict[column name, DataType].

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.schema
        OrderedDict({'foo': Int64, 'bar': Float64, 'ham': Utf8})

        """
        return OrderedDict(zip(self.columns, self.dtypes))

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        """
        Numpy __array__ interface protocol.

        Ensures that `np.asarray(pl.DataFrame(..))` works as expected, see
        https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method.
        """
        if dtype:
            return self.to_numpy().__array__(dtype)
        else:
            return self.to_numpy().__array__()

    def __dataframe__(
        self,
        nan_as_null: bool = False,  # noqa: FBT001
        allow_copy: bool = True,  # noqa: FBT001
    ) -> PolarsDataFrame:
        """
        Convert to a dataframe object implementing the dataframe interchange protocol.

        Parameters
        ----------
        nan_as_null
            Overwrite null values in the data with `NaN`.

            .. warning::
                This functionality has not been implemented and the parameter will be
                removed in a future version.
                Setting this to `True` will raise a `NotImplementedError`.
        allow_copy
            Allow memory to be copied to perform the conversion. If set to `False`,
            causes conversions that are not zero-copy to fail.

        Notes
        -----
        Details on the Python dataframe interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        Examples
        --------
        Convert a Polars DataFrame to a generic dataframe object and access some
        properties.

        >>> df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["x", "y"]})
        >>> dfi = df.__dataframe__()
        >>> dfi.num_rows()
        2
        >>> dfi.get_column(1).dtype
        (<DtypeKind.FLOAT: 2>, 64, 'g', '=')

        """
        if nan_as_null:
            raise NotImplementedError(
                "functionality for `nan_as_null` has not been implemented and the"
                " parameter will be removed in a future version"
                "\n\nUse the default `nan_as_null=False`."
            )

        from polars.interchange.dataframe import PolarsDataFrame

        return PolarsDataFrame(self, allow_copy=allow_copy)

    def __dataframe_consortium_standard__(
        self, *, api_version: str | None = None
    ) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of polars.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        return dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
            self.lazy(), api_version=api_version
        )

    def _comp(self, other: Any, op: ComparisonOperator) -> DataFrame:
        """Compare a DataFrame with another object."""
        if isinstance(other, DataFrame):
            return self._compare_to_other_df(other, op)
        else:
            return self._compare_to_non_df(other, op)

    def _compare_to_other_df(
        self,
        other: DataFrame,
        op: ComparisonOperator,
    ) -> DataFrame:
        """Compare a DataFrame with another DataFrame."""
        if self.columns != other.columns:
            raise ValueError("DataFrame columns do not match")
        if self.shape != other.shape:
            raise ValueError("DataFrame dimensions do not match")

        suffix = "__POLARS_CMP_OTHER"
        other_renamed = other.select(F.all().name.suffix(suffix))
        combined = F.concat([self, other_renamed], how="horizontal")

        if op == "eq":
            expr = [F.col(n) == F.col(f"{n}{suffix}") for n in self.columns]
        elif op == "neq":
            expr = [F.col(n) != F.col(f"{n}{suffix}") for n in self.columns]
        elif op == "gt":
            expr = [F.col(n) > F.col(f"{n}{suffix}") for n in self.columns]
        elif op == "lt":
            expr = [F.col(n) < F.col(f"{n}{suffix}") for n in self.columns]
        elif op == "gt_eq":
            expr = [F.col(n) >= F.col(f"{n}{suffix}") for n in self.columns]
        elif op == "lt_eq":
            expr = [F.col(n) <= F.col(f"{n}{suffix}") for n in self.columns]
        else:
            raise ValueError(f"unexpected comparison operator {op!r}")

        return combined.select(expr)

    def _compare_to_non_df(
        self,
        other: Any,
        op: ComparisonOperator,
    ) -> DataFrame:
        """Compare a DataFrame with a non-DataFrame object."""
        if op == "eq":
            return self.select(F.all() == other)
        elif op == "neq":
            return self.select(F.all() != other)
        elif op == "gt":
            return self.select(F.all() > other)
        elif op == "lt":
            return self.select(F.all() < other)
        elif op == "gt_eq":
            return self.select(F.all() >= other)
        elif op == "lt_eq":
            return self.select(F.all() <= other)
        else:
            raise ValueError(f"unexpected comparison operator {op!r}")

    def _div(self, other: Any, *, floordiv: bool) -> DataFrame:
        if isinstance(other, pl.Series):
            if floordiv:
                return self.select(F.all() // lit(other))
            return self.select(F.all() / lit(other))

        elif not isinstance(other, DataFrame):
            s = _prepare_other_arg(other, length=len(self))
            other = DataFrame([s.alias(f"n{i}") for i in range(len(self.columns))])

        orig_dtypes = other.dtypes
        other = self._cast_all_from_to(other, INTEGER_DTYPES, Float64)
        df = self._from_pydf(self._df.div_df(other._df))

        df = (
            df
            if not floordiv
            else df.with_columns([s.floor() for s in df if s.dtype.is_float()])
        )
        if floordiv:
            int_casts = [
                col(column).cast(tp)
                for i, (column, tp) in enumerate(self.schema.items())
                if tp.is_integer() and orig_dtypes[i].is_integer()
            ]
            if int_casts:
                return df.with_columns(int_casts)
        return df

    def _cast_all_from_to(
        self, df: DataFrame, from_: frozenset[PolarsDataType], to: PolarsDataType
    ) -> DataFrame:
        casts = [s.cast(to).alias(s.name) for s in df if s.dtype in from_]
        return df.with_columns(casts) if casts else df

    def __floordiv__(self, other: DataFrame | Series | int | float) -> DataFrame:
        return self._div(other, floordiv=True)

    def __truediv__(self, other: DataFrame | Series | int | float) -> DataFrame:
        return self._div(other, floordiv=False)

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "the truth value of a DataFrame is ambiguous"
            "\n\nHint: to check if a DataFrame contains any values, use `is_empty()`."
        )

    def __eq__(self, other: Any) -> DataFrame:  # type: ignore[override]
        return self._comp(other, "eq")

    def __ne__(self, other: Any) -> DataFrame:  # type: ignore[override]
        return self._comp(other, "neq")

    def __gt__(self, other: Any) -> DataFrame:
        return self._comp(other, "gt")

    def __lt__(self, other: Any) -> DataFrame:
        return self._comp(other, "lt")

    def __ge__(self, other: Any) -> DataFrame:
        return self._comp(other, "gt_eq")

    def __le__(self, other: Any) -> DataFrame:
        return self._comp(other, "lt_eq")

    def __getstate__(self) -> list[Series]:
        return self.get_columns()

    def __setstate__(self, state: list[Series]) -> None:
        self._df = DataFrame(state)._df

    def __mul__(self, other: DataFrame | Series | int | float) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.mul_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.mul(other._s))

    def __rmul__(self, other: DataFrame | Series | int | float) -> Self:
        return self * other

    def __add__(
        self, other: DataFrame | Series | int | float | bool | str
    ) -> DataFrame:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.add_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.add(other._s))

    def __radd__(  # type: ignore[misc]
        self, other: DataFrame | Series | int | float | bool | str
    ) -> DataFrame:
        if isinstance(other, str):
            return self.select((lit(other) + F.col("*")).name.keep())
        return self + other

    def __sub__(self, other: DataFrame | Series | int | float) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.sub_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.sub(other._s))

    def __mod__(self, other: DataFrame | Series | int | float) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.rem_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.rem(other._s))

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __iter__(self) -> Iterator[Series]:
        return self.iter_columns()

    def __reversed__(self) -> Iterator[Series]:
        return reversed(self.get_columns())

    def _pos_idx(self, idx: int, dim: int) -> int:
        if idx >= 0:
            return idx
        else:
            return self.shape[dim] + idx

    def _take_with_series(self, s: Series) -> DataFrame:
        return self._from_pydf(self._df.take_with_series(s._s))

    @overload
    def __getitem__(self, item: str) -> Series:
        ...

    @overload
    def __getitem__(
        self,
        item: (
            int
            | np.ndarray[Any, Any]
            | MultiColSelector
            | tuple[int, MultiColSelector]
            | tuple[MultiRowSelector, MultiColSelector]
        ),
    ) -> Self:
        ...

    @overload
    def __getitem__(self, item: tuple[int, int | str]) -> Any:
        ...

    @overload
    def __getitem__(self, item: tuple[MultiRowSelector, int | str]) -> Series:
        ...

    def __getitem__(
        self,
        item: (
            str
            | int
            | np.ndarray[Any, Any]
            | MultiColSelector
            | tuple[int, MultiColSelector]
            | tuple[MultiRowSelector, MultiColSelector]
            | tuple[MultiRowSelector, int | str]
            | tuple[int, int | str]
        ),
    ) -> DataFrame | Series:
        """Get item. Does quite a lot. Read the comments."""
        # fail on ['col1', 'col2', ..., 'coln']
        if (
            isinstance(item, tuple)
            and len(item) > 1
            and all(isinstance(x, str) for x in item)
        ):
            raise KeyError(item)

        # select rows and columns at once
        # every 2d selection, i.e. tuple is row column order, just like numpy
        if isinstance(item, tuple) and len(item) == 2:
            row_selection, col_selection = item

            # df[[], :]
            if isinstance(row_selection, Sequence):
                if len(row_selection) == 0:
                    # handle empty list by falling through to slice
                    row_selection = slice(0)

            # df[:, unknown]
            if isinstance(row_selection, slice):
                # multiple slices
                # df[:, :]
                if isinstance(col_selection, slice):
                    # slice can be
                    # by index
                    #   [1:8]
                    # or by column name
                    #   ["foo":"bar"]
                    # first we make sure that the slice is by index
                    start = col_selection.start
                    stop = col_selection.stop
                    if isinstance(col_selection.start, str):
                        start = self.get_column_index(col_selection.start)
                    if isinstance(col_selection.stop, str):
                        stop = self.get_column_index(col_selection.stop) + 1

                    col_selection = slice(start, stop, col_selection.step)

                    df = self.__getitem__(self.columns[col_selection])
                    return df[row_selection]

                # df[:, [True, False]]
                if is_bool_sequence(col_selection) or (
                    isinstance(col_selection, pl.Series)
                    and col_selection.dtype == Boolean
                ):
                    if len(col_selection) != self.width:
                        raise ValueError(
                            f"expected {self.width} values when selecting columns by"
                            f" boolean mask, got {len(col_selection)}"
                        )
                    series_list = []
                    for i, val in enumerate(col_selection):
                        if val:
                            series_list.append(self.to_series(i))

                    df = self.__class__(series_list)
                    return df[row_selection]

            # df[2, :] (select row as df)
            if isinstance(row_selection, int):
                if isinstance(col_selection, (slice, list)) or (
                    _check_for_numpy(col_selection)
                    and isinstance(col_selection, np.ndarray)
                ):
                    df = self[:, col_selection]
                    return df.slice(row_selection, 1)

            # df[:, "a"]
            if isinstance(col_selection, str):
                series = self.get_column(col_selection)
                return series[row_selection]

            # df[:, 1]
            if isinstance(col_selection, int):
                if (col_selection >= 0 and col_selection >= self.width) or (
                    col_selection < 0 and col_selection < -self.width
                ):
                    raise IndexError(f"column index {col_selection!r} is out of bounds")
                series = self.to_series(col_selection)
                return series[row_selection]

            if isinstance(col_selection, list):
                # df[:, [1, 2]]
                if is_int_sequence(col_selection):
                    for i in col_selection:
                        if (i >= 0 and i >= self.width) or (i < 0 and i < -self.width):
                            raise IndexError(
                                f"column index {col_selection!r} is out of bounds"
                            )
                    series_list = [self.to_series(i) for i in col_selection]
                    df = self.__class__(series_list)
                    return df[row_selection]

            df = self.__getitem__(col_selection)
            return df.__getitem__(row_selection)

        # select single column
        # df["foo"]
        if isinstance(item, str):
            return self.get_column(item)

        # df[idx]
        if isinstance(item, int):
            return self.slice(self._pos_idx(item, dim=0), 1)

        # df[range(n)]
        if isinstance(item, range):
            return self[range_to_slice(item)]

        # df[:]
        if isinstance(item, slice):
            return PolarsSlice(self).apply(item)

        # select rows by numpy mask or index
        # df[np.array([1, 2, 3])]
        # df[np.array([True, False, True])]
        if _check_for_numpy(item) and isinstance(item, np.ndarray):
            if item.ndim != 1:
                raise TypeError("multi-dimensional NumPy arrays not supported as index")
            if item.dtype.kind in ("i", "u"):
                # Numpy array with signed or unsigned integers.
                return self._take_with_series(numpy_to_idxs(item, self.shape[0]))
            if isinstance(item[0], str):
                return self._from_pydf(self._df.select(item))

        if is_str_sequence(item, allow_str=False):
            # select multiple columns
            # df[["foo", "bar"]]
            return self._from_pydf(self._df.select(item))
        elif is_int_sequence(item):
            item = pl.Series("", item)  # fall through to next if isinstance

        if isinstance(item, pl.Series):
            dtype = item.dtype
            if dtype == Utf8:
                return self._from_pydf(self._df.select(item))
            elif dtype.is_integer():
                return self._take_with_series(item._pos_idxs(self.shape[0]))

        # if no data has been returned, the operation is not supported
        raise TypeError(
            f"cannot use `__getitem__` on DataFrame with item {item!r}"
            f" of type {type(item).__name__!r}"
        )

    def __setitem__(
        self,
        key: str | Sequence[int] | Sequence[str] | tuple[Any, str | int],
        value: Any,
    ) -> None:  # pragma: no cover
        # df["foo"] = series
        if isinstance(key, str):
            raise TypeError(
                "DataFrame object does not support `Series` assignment by index"
                "\n\nUse `DataFrame.with_columns`."
            )

        # df[["C", "D"]]
        elif isinstance(key, list):
            # TODO: Use python sequence constructors
            value = np.array(value)
            if value.ndim != 2:
                raise ValueError("can only set multiple columns with 2D matrix")
            if value.shape[1] != len(key):
                raise ValueError(
                    "matrix columns should be equal to list used to determine column names"
                )

            # TODO: we can parallelize this by calling from_numpy
            columns = []
            for i, name in enumerate(key):
                columns.append(pl.Series(name, value[:, i]))
            self._df = self.with_columns(columns)._df

        # df[a, b]
        elif isinstance(key, tuple):
            row_selection, col_selection = key

            if (
                isinstance(row_selection, pl.Series) and row_selection.dtype == Boolean
            ) or is_bool_sequence(row_selection):
                raise TypeError(
                    "not allowed to set DataFrame by boolean mask in the row position"
                    "\n\nConsider using `DataFrame.with_columns`."
                )

            # get series column selection
            if isinstance(col_selection, str):
                s = self.__getitem__(col_selection)
            elif isinstance(col_selection, int):
                s = self[:, col_selection]
            else:
                raise TypeError(f"unexpected column selection {col_selection!r}")

            # dispatch to __setitem__ of Series to do modification
            s[row_selection] = value

            # now find the location to place series
            # df[idx]
            if isinstance(col_selection, int):
                self.replace_column(col_selection, s)
            # df["foo"]
            elif isinstance(col_selection, str):
                self._replace(col_selection, s)
        else:
            raise TypeError(
                f"cannot use `__setitem__` on DataFrame"
                f" with key {key!r} of type {type(key).__name__!r}"
                f" and value {value!r} of type {type(value).__name__!r}"
            )

    def __len__(self) -> int:
        return self.height

    def __copy__(self) -> Self:
        return self.clone()

    def __deepcopy__(self, memo: None = None) -> Self:
        return self.clone()

    def _ipython_key_completions_(self) -> list[str]:
        return self.columns

    def _repr_html_(self, **kwargs: Any) -> str:
        """
        Format output data in HTML for display in Jupyter Notebooks.

        Output rows and columns can be modified by setting the following ENVIRONMENT
        variables:

        * POLARS_FMT_MAX_COLS: set the number of columns
        * POLARS_FMT_MAX_ROWS: set the number of rows

        """
        max_cols = int(os.environ.get("POLARS_FMT_MAX_COLS", default=75))
        if max_cols < 0:
            max_cols = self.shape[1]
        max_rows = int(os.environ.get("POLARS_FMT_MAX_ROWS", default=25))
        if max_rows < 0:
            max_rows = self.shape[0]

        from_series = kwargs.get("from_series", False)
        return "".join(
            NotebookFormatter(
                self,
                max_cols=max_cols,
                max_rows=max_rows,
                from_series=from_series,
            ).render()
        )

    def item(self, row: int | None = None, column: int | str | None = None) -> Any:
        """
        Return the DataFrame as a scalar, or return the element at the given row/column.

        Parameters
        ----------
        row
            Optional row index.
        column
            Optional column index or name.

        See Also
        --------
        row: Get the values of a single row, either by index or by predicate.

        Notes
        -----
        If row/col not provided, this is equivalent to `df[0,0]`, with a check that
        the shape is (1,1). With row/col, this is equivalent to `df[row,col]`.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df.select((pl.col("a") * pl.col("b")).sum()).item()
        32
        >>> df.item(1, 1)
        5
        >>> df.item(2, "b")
        6

        """
        if row is None and column is None:
            if self.shape != (1, 1):
                raise ValueError(
                    "can only call `.item()` if the dataframe is of shape (1, 1),"
                    " or if explicit row/col values are provided;"
                    f" frame has shape {self.shape!r}"
                )
            return self._df.select_at_idx(0).get_index(0)

        elif row is None or column is None:
            raise ValueError("cannot call `.item()` with only one of `row` or `column`")

        s = (
            self._df.select_at_idx(column)
            if isinstance(column, int)
            else self._df.get_column(column)
        )
        if s is None:
            raise IndexError(f"column index {column!r} is out of bounds")
        return s.get_index_signed(row)

    def to_arrow(self) -> pa.Table:
        """
        Collect the underlying arrow arrays in an Arrow Table.

        This operation is mostly zero copy.

        Data types that do copy:
            - CategoricalType

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3, 4, 5, 6], "bar": ["a", "b", "c", "d", "e", "f"]}
        ... )
        >>> df.to_arrow()
        pyarrow.Table
        foo: int64
        bar: large_string
        ----
        foo: [[1,2,3,4,5,6]]
        bar: [["a","b","c","d","e","f"]]

        """
        if self.shape[1]:  # all except 0x0 dataframe
            record_batches = self._df.to_arrow()
            return pa.Table.from_batches(record_batches)
        else:  # 0x0 dataframe, cannot infer schema from batches
            return pa.table({})

    @overload
    def to_dict(self, as_series: Literal[True] = ...) -> dict[str, Series]:
        ...

    @overload
    def to_dict(self, as_series: Literal[False]) -> dict[str, list[Any]]:
        ...

    @overload
    def to_dict(
        self,
        as_series: bool,  # noqa: FBT001
    ) -> dict[str, Series] | dict[str, list[Any]]:
        ...

    @deprecate_nonkeyword_arguments(version="0.19.13")
    def to_dict(
        self,
        as_series: bool = True,  # noqa: FBT001
    ) -> dict[str, Series] | dict[str, list[Any]]:
        """
        Convert DataFrame to a dictionary mapping column name to values.

        Parameters
        ----------
        as_series
            True -> Values are Series
            False -> Values are List[Any]

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...         "optional": [28, 300, None, 2, -30],
        ...     }
        ... )
        >>> df
        shape: (5, 5)
        ┌─────┬────────┬─────┬────────┬──────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ optional │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---      │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64      │
        ╞═════╪════════╪═════╪════════╪══════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 28       │
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 300      │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ null     │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2        │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ -30      │
        └─────┴────────┴─────┴────────┴──────────┘
        >>> df.to_dict(as_series=False)
        {'A': [1, 2, 3, 4, 5],
        'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'],
        'B': [5, 4, 3, 2, 1],
        'cars': ['beetle', 'audi', 'beetle', 'beetle', 'beetle'],
        'optional': [28, 300, None, 2, -30]}
        >>> df.to_dict(as_series=True)
        {'A': shape: (5,)
        Series: 'A' [i64]
        [
            1
            2
            3
            4
            5
        ], 'fruits': shape: (5,)
        Series: 'fruits' [str]
        [
            "banana"
            "banana"
            "apple"
            "apple"
            "banana"
        ], 'B': shape: (5,)
        Series: 'B' [i64]
        [
            5
            4
            3
            2
            1
        ], 'cars': shape: (5,)
        Series: 'cars' [str]
        [
            "beetle"
            "audi"
            "beetle"
            "beetle"
            "beetle"
        ], 'optional': shape: (5,)
        Series: 'optional' [i64]
        [
            28
            300
            null
            2
            -30
        ]}

        """
        if as_series:
            return {s.name: s for s in self}
        else:
            return {s.name: s.to_list() for s in self}

    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert every row to a dictionary of Python-native values.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.to_dicts()
        [{'foo': 1, 'bar': 4}, {'foo': 2, 'bar': 5}, {'foo': 3, 'bar': 6}]

        """
        return list(self.iter_rows(named=True))

    @deprecate_nonkeyword_arguments(version="0.19.3")
    def to_numpy(
        self,
        structured: bool = False,  # noqa: FBT001
        *,
        order: IndexOrder = "fortran",
    ) -> np.ndarray[Any, Any]:
        """
        Convert DataFrame to a 2D NumPy array.

        This operation clones data.

        Parameters
        ----------
        structured
            Optionally return a structured array, with field names and
            dtypes that correspond to the DataFrame schema.
        order
            The index order of the returned NumPy array, either C-like or
            Fortran-like. In general, using the Fortran-like index order is faster.
            However, the C-like order might be more appropriate to use for downstream
            applications to prevent cloning data, e.g. when reshaping into a
            one-dimensional array. Note that this option only takes effect if
            `structured` is set to `False` and the DataFrame dtypes allow for a
            global dtype for all columns.

        Notes
        -----
        If you're attempting to convert Utf8 to an array you'll need to install
        `pyarrow`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.5, 7.0, 8.5],
        ...         "ham": ["a", "b", "c"],
        ...     },
        ...     schema_overrides={"foo": pl.UInt8, "bar": pl.Float32},
        ... )

        Export to a standard 2D numpy array.

        >>> df.to_numpy()
        array([[1, 6.5, 'a'],
               [2, 7.0, 'b'],
               [3, 8.5, 'c']], dtype=object)

        Export to a structured array, which can better-preserve individual
        column data, such as name and dtype...

        >>> df.to_numpy(structured=True)
        array([(1, 6.5, 'a'), (2, 7. , 'b'), (3, 8.5, 'c')],
              dtype=[('foo', 'u1'), ('bar', '<f4'), ('ham', '<U1')])

        ...optionally zero-copying as a record array view:

        >>> import numpy as np
        >>> df.to_numpy(structured=True).view(np.recarray)
        rec.array([(1, 6.5, 'a'), (2, 7. , 'b'), (3, 8.5, 'c')],
                  dtype=[('foo', 'u1'), ('bar', '<f4'), ('ham', '<U1')])

        """
        if structured:
            # see: https://numpy.org/doc/stable/user/basics.rec.html
            arrays = []
            for c, tp in self.schema.items():
                s = self[c]
                a = s.to_numpy()
                arrays.append(
                    a.astype(str, copy=False)
                    if tp == Utf8 and not s.null_count()
                    else a
                )

            out = np.empty(
                len(self), dtype=list(zip(self.columns, (a.dtype for a in arrays)))
            )
            for idx, c in enumerate(self.columns):
                out[c] = arrays[idx]
        else:
            out = self._df.to_numpy(order)
            if out is None:
                return np.vstack(
                    [self.to_series(i).to_numpy() for i in range(self.width)]
                ).T

        return out

    def to_pandas(  # noqa: D417
        self,
        *args: Any,
        use_pyarrow_extension_array: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Cast to a pandas DataFrame.

        This requires that :mod:`pandas` and :mod:`pyarrow` are installed.
        This operation clones data, unless `use_pyarrow_extension_array=True`.

        Parameters
        ----------
        use_pyarrow_extension_array
            Use PyArrow backed-extension arrays instead of numpy arrays for each column
            of the pandas DataFrame; this allows zero copy operations and preservation
            of null values. Subsequent operations on the resulting pandas DataFrame may
            trigger conversion to NumPy arrays if that operation is not supported by
            pyarrow compute functions.
        **kwargs
            Arguments will be sent to :meth:`pyarrow.Table.to_pandas`.

        Returns
        -------
        :class:`pandas.DataFrame`

        Examples
        --------
        >>> import pandas
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> pandas_df1 = df1.to_pandas()
        >>> type(pandas_df1)
        <class 'pandas.core.frame.DataFrame'>
        >>> pandas_df1.dtypes
        foo     int64
        bar     int64
        ham    object
        dtype: object
        >>> df2 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, None],
        ...         "bar": [6, None, 8],
        ...         "ham": [None, "b", "c"],
        ...     }
        ... )
        >>> pandas_df2 = df2.to_pandas()
        >>> pandas_df2
           foo  bar   ham
        0  1.0  6.0  None
        1  2.0  NaN     b
        2  NaN  8.0     c
        >>> pandas_df2.dtypes
        foo    float64
        bar    float64
        ham     object
        dtype: object
        >>> pandas_df2_pa = df2.to_pandas(
        ...     use_pyarrow_extension_array=True
        ... )  # doctest: +SKIP
        >>> pandas_df2_pa  # doctest: +SKIP
            foo   bar   ham
        0     1     6  <NA>
        1     2  <NA>     b
        2  <NA>     8     c
        >>> pandas_df2_pa.dtypes  # doctest: +SKIP
        foo           int64[pyarrow]
        bar           int64[pyarrow]
        ham    large_string[pyarrow]
        dtype: object

        """
        if use_pyarrow_extension_array:
            if parse_version(pd.__version__) < parse_version("1.5"):
                raise ModuleNotFoundError(
                    f'pandas>=1.5.0 is required for `to_pandas("use_pyarrow_extension_array=True")`, found Pandas {pd.__version__!r}'
                )
            if not _PYARROW_AVAILABLE or parse_version(pa.__version__) < parse_version(
                "8"
            ):
                msg = "pyarrow>=8.0.0 is required for `to_pandas(use_pyarrow_extension_array=True)`"
                if _PYARROW_AVAILABLE:
                    msg += f", found pyarrow {pa.__version__!r}."
                raise ModuleNotFoundError(msg)

        record_batches = self._df.to_pandas()
        tbl = pa.Table.from_batches(record_batches)
        if use_pyarrow_extension_array:
            return tbl.to_pandas(
                self_destruct=True,
                split_blocks=True,
                types_mapper=lambda pa_dtype: pd.ArrowDtype(pa_dtype),
                **kwargs,
            )

        date_as_object = kwargs.pop("date_as_object", False)
        return tbl.to_pandas(date_as_object=date_as_object, **kwargs)

    def to_series(self, index: int = 0) -> Series:
        """
        Select column as Series at index location.

        Parameters
        ----------
        index
            Location of selection.

        See Also
        --------
        get_column

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.to_series(1)
        shape: (3,)
        Series: 'bar' [i64]
        [
                6
                7
                8
        ]

        """
        if not isinstance(index, int):
            raise TypeError(
                f"index value {index!r} should be an int, but is {type(index).__name__!r}"
            )

        if index < 0:
            index = len(self.columns) + index
        return wrap_s(self._df.select_at_idx(index))

    def to_init_repr(self, n: int = 1000) -> str:
        """
        Convert DataFrame to instantiatable string representation.

        Parameters
        ----------
        n
            Only use first n rows.

        See Also
        --------
        polars.Series.to_init_repr
        polars.from_repr

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     [
        ...         pl.Series("foo", [1, 2, 3], dtype=pl.UInt8),
        ...         pl.Series("bar", [6.0, 7.0, 8.0], dtype=pl.Float32),
        ...         pl.Series("ham", ["a", "b", "c"], dtype=pl.Categorical),
        ...     ]
        ... )
        >>> print(df.to_init_repr())
        pl.DataFrame(
            [
                pl.Series("foo", [1, 2, 3], dtype=pl.UInt8),
                pl.Series("bar", [6.0, 7.0, 8.0], dtype=pl.Float32),
                pl.Series("ham", ['a', 'b', 'c'], dtype=pl.Categorical),
            ]
        )

        >>> df_from_str_repr = eval(df.to_init_repr())
        >>> df_from_str_repr
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ f32 ┆ cat │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        """
        output = StringIO()
        output.write("pl.DataFrame(\n    [\n")

        for i in range(self.width):
            output.write("        ")
            output.write(self.to_series(i).to_init_repr(n))
            output.write(",\n")

        output.write("    ]\n)\n")

        return output.getvalue()

    @overload
    def write_json(
        self,
        file: None = ...,
        *,
        pretty: bool = ...,
        row_oriented: bool = ...,
    ) -> str:
        ...

    @overload
    def write_json(
        self,
        file: IOBase | str | Path,
        *,
        pretty: bool = ...,
        row_oriented: bool = ...,
    ) -> None:
        ...

    def write_json(
        self,
        file: IOBase | str | Path | None = None,
        *,
        pretty: bool = False,
        row_oriented: bool = False,
    ) -> str | None:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            File path or writeable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.
        pretty
            Pretty serialize json.
        row_oriented
            Write to row oriented json. This is slower, but more common.

        See Also
        --------
        DataFrame.write_ndjson

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... )
        >>> df.write_json()
        '{"columns":[{"name":"foo","datatype":"Int64","bit_settings":"","values":[1,2,3]},{"name":"bar","datatype":"Int64","bit_settings":"","values":[6,7,8]}]}'
        >>> df.write_json(row_oriented=True)
        '[{"foo":1,"bar":6},{"foo":2,"bar":7},{"foo":3,"bar":8}]'

        """
        if isinstance(file, (str, Path)):
            file = normalize_filepath(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if file is None or to_string_io:
            with BytesIO() as buf:
                self._df.write_json(buf, pretty, row_oriented)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._df.write_json(file, pretty, row_oriented)
        return None

    @overload
    def write_ndjson(self, file: None = None) -> str:
        ...

    @overload
    def write_ndjson(self, file: IOBase | str | Path) -> None:
        ...

    def write_ndjson(self, file: IOBase | str | Path | None = None) -> str | None:
        r"""
        Serialize to newline delimited JSON representation.

        Parameters
        ----------
        file
            File path or writeable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... )
        >>> df.write_ndjson()
        '{"foo":1,"bar":6}\n{"foo":2,"bar":7}\n{"foo":3,"bar":8}\n'

        """
        if isinstance(file, (str, Path)):
            file = normalize_filepath(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if file is None or to_string_io:
            with BytesIO() as buf:
                self._df.write_ndjson(buf)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._df.write_ndjson(file)
        return None

    @overload
    def write_csv(
        self,
        file: None = None,
        *,
        include_bom: bool = ...,
        include_header: bool = ...,
        separator: str = ...,
        line_terminator: str = ...,
        quote_char: str = ...,
        batch_size: int = ...,
        datetime_format: str | None = ...,
        date_format: str | None = ...,
        time_format: str | None = ...,
        float_precision: int | None = ...,
        null_value: str | None = ...,
        quote_style: CsvQuoteStyle | None = ...,
    ) -> str:
        ...

    @overload
    def write_csv(
        self,
        file: BytesIO | TextIOWrapper | str | Path,
        *,
        include_bom: bool = ...,
        include_header: bool = ...,
        separator: str = ...,
        line_terminator: str = ...,
        quote_char: str = ...,
        batch_size: int = ...,
        datetime_format: str | None = ...,
        date_format: str | None = ...,
        time_format: str | None = ...,
        float_precision: int | None = ...,
        null_value: str | None = ...,
        quote_style: CsvQuoteStyle | None = ...,
    ) -> None:
        ...

    @deprecate_renamed_parameter("quote", "quote_char", version="0.19.8")
    @deprecate_renamed_parameter("has_header", "include_header", version="0.19.13")
    def write_csv(
        self,
        file: BytesIO | TextIOWrapper | str | Path | None = None,
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
    ) -> str | None:
        """
        Write to comma-separated values (CSV) file.

        Parameters
        ----------
        file
            File path or writeable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.
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
              separator or record terminator.
              Quotes are also necessary when writing an empty record
              (which is indistinguishable from a record with one empty field).
              This is the default.
            - always: This puts quotes around every field. Always.
            - never: This never puts quotes around fields, even if that results in
              invalid CSV data (e.g.: by not quoting strings containing the separator).
            - non_numeric: This puts quotes around all fields that are non-numeric.
              Namely, when writing a field that does not parse as a valid float
              or integer, then quotes will be used even if they aren`t strictly
              necessary.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.csv"
        >>> df.write_csv(path, separator=",")

        """
        _check_arg_is_1byte("separator", separator, can_be_empty=False)
        _check_arg_is_1byte("quote_char", quote_char, can_be_empty=True)
        if not null_value:
            null_value = None

        should_return_buffer = False
        if file is None:
            buffer = file = BytesIO()
            should_return_buffer = True
        elif isinstance(file, (str, os.PathLike)):
            file = normalize_filepath(file)
        elif isinstance(file, TextIOWrapper):
            file = cast(TextIOWrapper, file.buffer)

        self._df.write_csv(
            file,
            include_bom,
            include_header,
            ord(separator),
            line_terminator,
            ord(quote_char),
            batch_size,
            datetime_format,
            date_format,
            time_format,
            float_precision,
            null_value,
            quote_style,
        )

        if should_return_buffer:
            return str(buffer.getvalue(), encoding="utf-8")

        return None

    def write_avro(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: AvroCompression = "uncompressed",
        name: str = "",
    ) -> None:
        """
        Write to Apache Avro file.

        Parameters
        ----------
        file
            File path or writeable file-like object to which the data will be written.
        compression : {'uncompressed', 'snappy', 'deflate'}
            Compression method. Defaults to "uncompressed".
        name
            Schema name. Defaults to empty string.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.avro"
        >>> df.write_avro(path)

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = normalize_filepath(file)
        if name is None:
            name = ""

        self._df.write_avro(file, compression, name)

    @deprecate_renamed_parameter("has_header", "include_header", version="0.19.13")
    def write_excel(
        self,
        workbook: Workbook | BytesIO | Path | str | None = None,
        worksheet: str | None = None,
        *,
        position: tuple[int, int] | str = "A1",
        table_style: str | dict[str, Any] | None = None,
        table_name: str | None = None,
        column_formats: ColumnFormatDict | None = None,
        dtype_formats: dict[OneOrMoreDataTypes, str] | None = None,
        conditional_formats: ConditionalFormatDict | None = None,
        header_format: dict[str, Any] | None = None,
        column_totals: ColumnTotalsDefinition | None = None,
        column_widths: ColumnWidthsDefinition | None = None,
        row_totals: RowTotalsDefinition | None = None,
        row_heights: dict[int | tuple[int, ...], int] | int | None = None,
        sparklines: dict[str, Sequence[str] | dict[str, Any]] | None = None,
        formulas: dict[str, str | dict[str, str]] | None = None,
        float_precision: int = 3,
        include_header: bool = True,
        autofilter: bool = True,
        autofit: bool = False,
        hidden_columns: Sequence[str] | SelectorType | None = None,
        hide_gridlines: bool = False,
        sheet_zoom: int | None = None,
        freeze_panes: (
            str
            | tuple[int, int]
            | tuple[str, int, int]
            | tuple[int, int, int, int]
            | None
        ) = None,
    ) -> Workbook:
        """
        Write frame data to a table in an Excel workbook/worksheet.

        Parameters
        ----------
        workbook : Workbook
            String name or path of the workbook to create, BytesIO object to write
            into, or an open `xlsxwriter.Workbook` object that has not been closed.
            If None, writes to a `dataframe.xlsx` workbook in the working directory.
        worksheet : str
            Name of target worksheet; if None, writes to "Sheet1" when creating a new
            workbook (note that writing to an existing workbook requires a valid
            existing -or new- worksheet name).
        position : {str, tuple}
            Table position in Excel notation (eg: "A1"), or a (row,col) integer tuple.
        table_style : {str, dict}
            A named Excel table style, such as "Table Style Medium 4", or a dictionary
            of `{"key":value,}` options containing one or more of the following keys:
            "style", "first_column", "last_column", "banded_columns, "banded_rows".
        table_name : str
            Name of the output table object in the worksheet; can then be referred to
            in the sheet by formulae/charts, or by subsequent `xlsxwriter` operations.
        column_formats : dict
            A `{colname(s):str,}` or `{selector:str,}` dictionary for applying an
            Excel format string to the given columns. Formats defined here (such as
            "dd/mm/yyyy", "0.00%", etc) will override any defined in `dtype_formats`.
        dtype_formats : dict
            A `{dtype:str,}` dictionary that sets the default Excel format for the
            given dtype. (This can be overridden on a per-column basis by the
            `column_formats` param). It is also valid to use dtype groups such as
            `pl.FLOAT_DTYPES` as the dtype/format key, to simplify setting uniform
            integer and float formats.
        conditional_formats : dict
            A dictionary of colname (or selector) keys to a format str, dict, or list
            that defines conditional formatting options for the specified columns.

            * If supplying a string typename, should be one of the valid `xlsxwriter`
              types such as "3_color_scale", "data_bar", etc.
            * If supplying a dictionary you can make use of any/all `xlsxwriter`
              supported options, including icon sets, formulae, etc.
            * Supplying multiple columns as a tuple/key will apply a single format
              across all columns - this is effective in creating a heatmap, as the
              min/max values will be determined across the entire range, not per-column.
            * Finally, you can also supply a list made up from the above options
              in order to apply *more* than one conditional format to the same range.
        header_format : dict
            A `{key:value,}` dictionary of `xlsxwriter` format options to apply
            to the table header row, such as `{"bold":True, "font_color":"#702963"}`.
        column_totals : {bool, list, dict}
            Add a column-total row to the exported table.

            * If True, all numeric columns will have an associated total using "sum".
            * If passing a string, it must be one of the valid total function names
              and all numeric columns will have an associated total using that function.
            * If passing a list of colnames, only those given will have a total.
            * For more control, pass a `{colname:funcname,}` dict.

            Valid total function names are "average", "count_nums", "count", "max",
            "min", "std_dev", "sum", and "var".
        column_widths : {dict, int}
            A `{colname:int,}` or `{selector:int,}` dict or a single integer that
            sets (or overrides if autofitting) table column widths, in integer pixel
            units. If given as an integer the same value is used for all table columns.
        row_totals : {dict, bool}
            Add a row-total column to the right-hand side of the exported table.

            * If True, a column called "total" will be added at the end of the table
              that applies a "sum" function row-wise across all numeric columns.
            * If passing a list/sequence of column names, only the matching columns
              will participate in the sum.
            * Can also pass a `{colname:columns,}` dictionary to create one or
              more total columns with distinct names, referencing different columns.
        row_heights : {dict, int}
            An int or `{row_index:int,}` dictionary that sets the height of the given
            rows (if providing a dictionary) or all rows (if providing an integer) that
            intersect with the table body (including any header and total row) in
            integer pixel units. Note that `row_index` starts at zero and will be
            the header row (unless `include_header` is False).
        sparklines : dict
            A `{colname:list,}` or `{colname:dict,}` dictionary defining one or more
            sparklines to be written into a new column in the table.

            * If passing a list of colnames (used as the source of the sparkline data)
              the default sparkline settings are used (eg: line chart with no markers).
            * For more control an `xlsxwriter`-compliant options dict can be supplied,
              in which case three additional polars-specific keys are available:
              "columns", "insert_before", and "insert_after". These allow you to define
              the source columns and position the sparkline(s) with respect to other
              table columns. If no position directive is given, sparklines are added to
              the end of the table (eg: to the far right) in the order they are given.
        formulas : dict
            A `{colname:formula,}` or `{colname:dict,}` dictionary defining one or
            more formulas to be written into a new column in the table. Note that you
            are strongly advised to use structured references in your formulae wherever
            possible to make it simple to reference columns by name.

            * If providing a string formula (such as "=[@colx]*[@coly]") the column will
              be added to the end of the table (eg: to the far right), after any default
              sparklines and before any row_totals.
            * For the most control supply an options dictionary with the following keys:
              "formula" (mandatory), one of "insert_before" or "insert_after", and
              optionally "return_dtype". The latter is used to appropriately format the
              output of the formula and allow it to participate in row/column totals.
        float_precision : int
            Default number of decimals displayed for floating point columns (note that
            this is purely a formatting directive; the actual values are not rounded).
        include_header : bool
            Indicate if the table should be created with a header row.
        autofilter : bool
            If the table has headers, provide autofilter capability.
        autofit : bool
            Calculate individual column widths from the data.
        hidden_columns : list
             A list or selector representing table columns to hide in the worksheet.
        hide_gridlines : bool
            Do not display any gridlines on the output worksheet.
        sheet_zoom : int
            Set the default zoom level of the output worksheet.
        freeze_panes : str | (str, int, int) | (int, int) | (int, int, int, int)
            Freeze workbook panes.

            * If (row, col) is supplied, panes are split at the top-left corner of the
              specified cell, which are 0-indexed. Thus, to freeze only the top row,
              supply (1, 0).
            * Alternatively, cell notation can be used to supply the cell. For example,
              "A2" indicates the split occurs at the top-left of cell A2, which is the
              equivalent of (1, 0).
            * If (row, col, top_row, top_col) are supplied, the panes are split based on
              the `row` and `col`, and the scrolling region is inititalized to begin at
              the `top_row` and `top_col`. Thus, to freeze only the top row and have the
              scrolling region begin at row 10, column D (5th col), supply (1, 0, 9, 4).
              Using cell notation for (row, col), supplying ("A2", 9, 4) is equivalent.

        Notes
        -----
        * A list of compatible `xlsxwriter` format property names can be found here:
          https://xlsxwriter.readthedocs.io/format.html#format-methods-and-format-properties

        * Conditional formatting dictionaries should provide xlsxwriter-compatible
          definitions; polars will take care of how they are applied on the worksheet
          with respect to the relative sheet/column position. For supported options,
          see: https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html

        * Similarly, sparkline option dictionaries should contain xlsxwriter-compatible
          key/values, as well as a mandatory polars "columns" key that defines the
          sparkline source data; these source columns should all be adjacent. Two other
          polars-specific keys are available to help define where the sparkline appears
          in the table: "insert_after", and "insert_before". The value associated with
          these keys should be the name of a column in the exported table.
          https://xlsxwriter.readthedocs.io/working_with_sparklines.html

        * Formula dictionaries *must* contain a key called "formula", and then optional
          "insert_after", "insert_before", and/or "return_dtype" keys. These additional
          keys allow the column to be injected into the table at a specific location,
          and/or to define the return type of the formula (eg: "Int64", "Float64", etc).
          Formulas that refer to table columns should use Excel's structured references
          syntax to ensure the formula is applied correctly and is table-relative.
          https://support.microsoft.com/en-us/office/using-structured-references-with-excel-tables-f5ed2452-2337-4f71-bed3-c8ae6d2b276e

        Examples
        --------
        Instantiate a basic DataFrame:

        >>> from random import uniform
        >>> from datetime import date
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "dtm": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        ...         "num": [uniform(-500, 500), uniform(-500, 500), uniform(-500, 500)],
        ...         "val": [10_000, 20_000, 30_000],
        ...     }
        ... )

        Export to "dataframe.xlsx" (the default workbook name, if not specified) in the
        working directory, add column totals ("sum" by default) on all numeric columns,
        then autofit:

        >>> df.write_excel(column_totals=True, autofit=True)  # doctest: +SKIP

        Write frame to a specific location on the sheet, set a named table style,
        apply US-style date formatting, increase default float precision, apply a
        non-default total function to a single column, autofit:

        >>> df.write_excel(  # doctest: +SKIP
        ...     position="B4",
        ...     table_style="Table Style Light 16",
        ...     dtype_formats={pl.Date: "mm/dd/yyyy"},
        ...     column_totals={"num": "average"},
        ...     float_precision=6,
        ...     autofit=True,
        ... )

        Write the same frame to a named worksheet twice, applying different styles
        and conditional formatting to each table, adding table titles using explicit
        xlsxwriter integration:

        >>> from xlsxwriter import Workbook
        >>> with Workbook("multi_frame.xlsx") as wb:  # doctest: +SKIP
        ...     # basic/default conditional formatting
        ...     df.write_excel(
        ...         workbook=wb,
        ...         worksheet="data",
        ...         position=(3, 1),  # specify position as (row,col) coordinates
        ...         conditional_formats={"num": "3_color_scale", "val": "data_bar"},
        ...         table_style="Table Style Medium 4",
        ...     )
        ...
        ...     # advanced conditional formatting, custom styles
        ...     df.write_excel(
        ...         workbook=wb,
        ...         worksheet="data",
        ...         position=(len(df) + 7, 1),
        ...         table_style={
        ...             "style": "Table Style Light 4",
        ...             "first_column": True,
        ...         },
        ...         conditional_formats={
        ...             "num": {
        ...                 "type": "3_color_scale",
        ...                 "min_color": "#76933c",
        ...                 "mid_color": "#c4d79b",
        ...                 "max_color": "#ebf1de",
        ...             },
        ...             "val": {
        ...                 "type": "data_bar",
        ...                 "data_bar_2010": True,
        ...                 "bar_color": "#9bbb59",
        ...                 "bar_negative_color_same": True,
        ...                 "bar_negative_border_color_same": True,
        ...             },
        ...         },
        ...         column_formats={"num": "#,##0.000;[White]-#,##0.000"},
        ...         column_widths={"val": 125},
        ...         autofit=True,
        ...     )
        ...
        ...     # add some table titles (with a custom format)
        ...     ws = wb.get_worksheet_by_name("data")
        ...     fmt_title = wb.add_format(
        ...         {
        ...             "font_color": "#4f6228",
        ...             "font_size": 12,
        ...             "italic": True,
        ...             "bold": True,
        ...         }
        ...     )
        ...     ws.write(2, 1, "Basic/default conditional formatting", fmt_title)
        ...     ws.write(len(df) + 6, 1, "Customised conditional formatting", fmt_title)
        ...

        Export a table containing two different types of sparklines. Use default
        options for the "trend" sparkline and customised options (and positioning)
        for the "+/-" win_loss sparkline, with non-default integer dtype formatting,
        column totals, a subtle two-tone heatmap and hidden worksheet gridlines:

        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["aaa", "bbb", "ccc", "ddd", "eee"],
        ...         "q1": [100, 55, -20, 0, 35],
        ...         "q2": [30, -10, 15, 60, 20],
        ...         "q3": [-50, 0, 40, 80, 80],
        ...         "q4": [75, 55, 25, -10, -55],
        ...     }
        ... )
        >>> df.write_excel(  # doctest: +SKIP
        ...     table_style="Table Style Light 2",
        ...     # apply accounting format to all flavours of integer
        ...     dtype_formats={pl.INTEGER_DTYPES: "#,##0_);(#,##0)"},
        ...     sparklines={
        ...         # default options; just provide source cols
        ...         "trend": ["q1", "q2", "q3", "q4"],
        ...         # customised sparkline type, with positioning directive
        ...         "+/-": {
        ...             "columns": ["q1", "q2", "q3", "q4"],
        ...             "insert_after": "id",
        ...             "type": "win_loss",
        ...         },
        ...     },
        ...     conditional_formats={
        ...         # create a unified multi-column heatmap
        ...         ("q1", "q2", "q3", "q4"): {
        ...             "type": "2_color_scale",
        ...             "min_color": "#95b3d7",
        ...             "max_color": "#ffffff",
        ...         },
        ...     },
        ...     column_totals=["q1", "q2", "q3", "q4"],
        ...     row_totals=True,
        ...     hide_gridlines=True,
        ... )

        Export a table containing an Excel formula-based column that calculates a
        standardised Z-score, showing use of structured references in conjunction
        with positioning directives, column totals, and custom formatting.

        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a123", "b345", "c567", "d789", "e101"],
        ...         "points": [99, 45, 50, 85, 35],
        ...     }
        ... )
        >>> df.write_excel(  # doctest: +SKIP
        ...     table_style={
        ...         "style": "Table Style Medium 15",
        ...         "first_column": True,
        ...     },
        ...     column_formats={
        ...         "id": {"font": "Consolas"},
        ...         "points": {"align": "center"},
        ...         "z-score": {"align": "center"},
        ...     },
        ...     column_totals="average",
        ...     formulas={
        ...         "z-score": {
        ...             # use structured references to refer to the table columns and 'totals' row
        ...             "formula": "=STANDARDIZE([@points], [[#Totals],[points]], STDEV([points]))",
        ...             "insert_after": "points",
        ...             "return_dtype": pl.Float64,
        ...         }
        ...     },
        ...     hide_gridlines=True,
        ...     sheet_zoom=125,
        ... )

        """  # noqa: W505
        try:
            import xlsxwriter
            from xlsxwriter.utility import xl_cell_to_rowcol
        except ImportError:
            raise ImportError(
                "Excel export requires xlsxwriter"
                "\n\nPlease run: pip install XlsxWriter"
            ) from None

        # setup workbook/worksheet
        wb, ws, can_close = _xl_setup_workbook(workbook, worksheet)
        df, is_empty = self, not len(self)

        # setup table format/columns
        fmt_cache = _XLFormatCache(wb)
        column_formats = column_formats or {}
        table_style, table_options = _xl_setup_table_options(table_style)
        table_name = table_name or _xl_unique_table_name(wb)
        table_columns, column_formats, df = _xl_setup_table_columns(  # type: ignore[assignment]
            df=df,
            format_cache=fmt_cache,
            column_formats=column_formats,
            column_totals=column_totals,
            dtype_formats=dtype_formats,
            header_format=header_format,
            float_precision=float_precision,
            row_totals=row_totals,
            sparklines=sparklines,
            formulas=formulas,
        )

        # normalise cell refs (eg: "B3" => (2,1)) and establish table start/finish,
        # accounting for potential presence/absence of headers and a totals row.
        table_start = (
            xl_cell_to_rowcol(position) if isinstance(position, str) else position
        )
        table_finish = (
            table_start[0]
            + len(df)
            + int(is_empty)
            - int(not include_header)
            + int(bool(column_totals)),
            table_start[1] + len(df.columns) - 1,
        )

        # write table structure and formats into the target sheet
        if not is_empty or include_header:
            ws.add_table(
                *table_start,
                *table_finish,
                {
                    "style": table_style,
                    "columns": table_columns,
                    "header_row": include_header,
                    "autofilter": autofilter,
                    "total_row": bool(column_totals) and not is_empty,
                    "name": table_name,
                    **table_options,
                },
            )

            # write data into the table range, column-wise
            if not is_empty:
                column_start = [table_start[0] + int(include_header), table_start[1]]
                for c in df.columns:
                    if c in self.columns:
                        ws.write_column(
                            *column_start,
                            data=df[c].to_list(),
                            cell_format=column_formats.get(c),
                        )
                    column_start[1] += 1

            # apply conditional formats
            if conditional_formats:
                _xl_apply_conditional_formats(
                    df=df,
                    ws=ws,
                    conditional_formats=conditional_formats,
                    table_start=table_start,
                    include_header=include_header,
                    format_cache=fmt_cache,
                )
        # additional column-level properties
        if hidden_columns is None:
            hidden_columns = ()
        hidden_columns = _expand_selectors(df, hidden_columns)
        if isinstance(column_widths, int):
            column_widths = {column: column_widths for column in df.columns}
        else:
            column_widths = _expand_selector_dicts(  # type: ignore[assignment]
                df, column_widths, expand_keys=True, expand_values=False
            )
        column_widths = _unpack_multi_column_dict(column_widths or {})  # type: ignore[assignment]

        for column in df.columns:
            col_idx, options = table_start[1] + df.get_column_index(column), {}
            if column in hidden_columns:
                options = {"hidden": True}
            if column in column_widths:  # type: ignore[operator]
                ws.set_column_pixels(
                    col_idx,
                    col_idx,
                    column_widths[column],  # type: ignore[index]
                    None,
                    options,
                )
            elif options:
                ws.set_column(col_idx, col_idx, None, None, options)

        # finally, inject any sparklines into the table
        for column, params in (sparklines or {}).items():
            _xl_inject_sparklines(
                ws,
                df,
                table_start,
                column,
                include_header=include_header,
                params=params,
            )

        # worksheet options
        if hide_gridlines:
            ws.hide_gridlines(2)
        if sheet_zoom:
            ws.set_zoom(sheet_zoom)
        if row_heights:
            if isinstance(row_heights, int):
                for idx in range(table_start[0], table_finish[0] + 1):
                    ws.set_row_pixels(idx, row_heights)
            elif isinstance(row_heights, dict):
                for idx, height in _unpack_multi_column_dict(row_heights).items():  # type: ignore[assignment]
                    ws.set_row_pixels(idx, height)

        # table/rows all written; apply (optional) autofit
        if autofit and not is_empty:
            xlv = xlsxwriter.__version__
            if parse_version(xlv) < parse_version("3.0.8"):
                raise ModuleNotFoundError(
                    f"`autofit=True` requires xlsxwriter 3.0.8 or higher, found {xlv}"
                )
            ws.autofit()

        if freeze_panes:
            if isinstance(freeze_panes, str):
                ws.freeze_panes(freeze_panes)
            else:
                ws.freeze_panes(*freeze_panes)

        if can_close:
            wb.close()
        return wb

    @overload
    def write_ipc(
        self,
        file: None,
        compression: IpcCompression = "uncompressed",
    ) -> BytesIO:
        ...

    @overload
    def write_ipc(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: IpcCompression = "uncompressed",
    ) -> None:
        ...

    def write_ipc(
        self,
        file: BinaryIO | BytesIO | str | Path | None,
        compression: IpcCompression = "uncompressed",
    ) -> BytesIO | None:
        """
        Write to Arrow IPC binary stream or Feather file.

        See "File or Random Access format" in https://arrow.apache.org/docs/python/ipc.html.

        Parameters
        ----------
        file
            Path or writeable file-like object to which the IPC data will be
            written. If set to `None`, the output is returned as a BytesIO object.
        compression : {'uncompressed', 'lz4', 'zstd'}
            Compression method. Defaults to "uncompressed".

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.arrow"
        >>> df.write_ipc(path)

        """
        return_bytes = file is None
        if return_bytes:
            file = BytesIO()
        elif isinstance(file, (str, Path)):
            file = normalize_filepath(file)

        if compression is None:
            compression = "uncompressed"

        self._df.write_ipc(file, compression)
        return file if return_bytes else None  # type: ignore[return-value]

    @overload
    def write_ipc_stream(
        self,
        file: None,
        compression: IpcCompression = "uncompressed",
    ) -> BytesIO:
        ...

    @overload
    def write_ipc_stream(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: IpcCompression = "uncompressed",
    ) -> None:
        ...

    def write_ipc_stream(
        self,
        file: BinaryIO | BytesIO | str | Path | None,
        compression: IpcCompression = "uncompressed",
    ) -> BytesIO | None:
        """
        Write to Arrow IPC record batch stream.

        See "Streaming format" in https://arrow.apache.org/docs/python/ipc.html.

        Parameters
        ----------
        file
            Path or writeable file-like object to which the IPC record batch data will
            be written. If set to `None`, the output is returned as a BytesIO object.
        compression : {'uncompressed', 'lz4', 'zstd'}
            Compression method. Defaults to "uncompressed".

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.arrow"
        >>> df.write_ipc_stream(path)

        """
        return_bytes = file is None
        if return_bytes:
            file = BytesIO()
        elif isinstance(file, (str, Path)):
            file = normalize_filepath(file)

        if compression is None:
            compression = "uncompressed"

        self._df.write_ipc_stream(file, compression)
        return file if return_bytes else None  # type: ignore[return-value]

    def write_parquet(
        self,
        file: str | Path | BytesIO,
        *,
        compression: ParquetCompression = "zstd",
        compression_level: int | None = None,
        statistics: bool = False,
        row_group_size: int | None = None,
        data_page_size: int | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Write to Apache Parquet file.

        Parameters
        ----------
        file
            File path or writeable file-like object to which the result will be written.
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
            Size of the row groups in number of rows. Defaults to 512^2 rows.
        data_page_size
            Size of the data page in bytes. Defaults to 1024^2 bytes.
        use_pyarrow
            Use C++ parquet implementation vs Rust parquet implementation.
            At the moment C++ supports more features.
        pyarrow_options
            Arguments passed to `pyarrow.parquet.write_table`.

            If you pass `partition_cols` here, the dataset will be written
            using `pyarrow.parquet.write_to_dataset`.
            The `partition_cols` parameter leads to write the dataset to a directory.
            Similar to Spark's partitioned datasets.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.parquet"
        >>> df.write_parquet(path)

        We can use pyarrow with use_pyarrow_write_to_dataset=True
        to write partitioned datasets. The following example will
        write the first row to ../watermark=1/*.parquet and the
        other rows to ../watermark=2/*.parquet.

        >>> df = pl.DataFrame({"a": [1, 2, 3], "watermark": [1, 2, 2]})
        >>> path: pathlib.Path = dirpath / "partitioned_object"
        >>> df.write_parquet(
        ...     path,
        ...     use_pyarrow=True,
        ...     pyarrow_options={"partition_cols": ["watermark"]},
        ... )

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            if pyarrow_options is not None and pyarrow_options.get("partition_cols"):
                file = normalize_filepath(file, check_not_directory=False)
            else:
                file = normalize_filepath(file)

        if use_pyarrow:
            tbl = self.to_arrow()
            data = {}

            for i, column in enumerate(tbl):
                # extract the name before casting
                name = f"column_{i}" if column._name is None else column._name

                data[name] = column

            tbl = pa.table(data)

            # do not remove this import!
            # needed below
            import pyarrow.parquet  # noqa: F401

            if pyarrow_options is None:
                pyarrow_options = {}
            pyarrow_options["compression"] = (
                None if compression == "uncompressed" else compression
            )
            pyarrow_options["compression_level"] = compression_level
            pyarrow_options["write_statistics"] = statistics
            pyarrow_options["row_group_size"] = row_group_size
            pyarrow_options["data_page_size"] = data_page_size

            if pyarrow_options.get("partition_cols"):
                pa.parquet.write_to_dataset(
                    table=tbl,
                    root_path=file,
                    **(pyarrow_options or {}),
                )
            else:
                pa.parquet.write_table(
                    table=tbl,
                    where=file,
                    **(pyarrow_options or {}),
                )

        else:
            self._df.write_parquet(
                file,
                compression,
                compression_level,
                statistics,
                row_group_size,
                data_page_size,
            )

    @deprecate_renamed_parameter("connection_uri", "connection", version="0.18.9")
    def write_database(
        self,
        table_name: str,
        connection: str,
        *,
        if_exists: DbWriteMode = "fail",
        engine: DbWriteEngine = "sqlalchemy",
    ) -> None:
        """
        Write a polars frame to a database.

        Parameters
        ----------
        table_name
            Schema-qualified name of the table to create or append to in the target
            SQL database. If your table name contains special characters, it should
            be quoted.
        connection
            Connection URI string, for example:

            * "postgresql://user:pass@server:port/database"
            * "sqlite:////path/to/database.db"
        if_exists : {'append', 'replace', 'fail'}
            The insert mode:

            * 'replace' will create a new database table, overwriting an existing one.
            * 'append' will append to an existing table.
            * 'fail' will fail if table already exists.
        engine : {'sqlalchemy', 'adbc'}
            Select the engine used for writing the data.
        """
        from polars.io.database import _open_adbc_connection

        def unpack_table_name(name: str) -> tuple[str | None, str]:
            """Unpack optionally qualified table name into schema/table pair."""
            from csv import reader as delimited_read

            table_ident = next(delimited_read([name], delimiter="."))
            if len(table_ident) > 2:
                raise ValueError(f"`table_name` appears to be invalid: {name!r}")
            elif len(table_ident) > 1:
                schema = table_ident[0]
                tbl = table_ident[1]
            else:
                schema = None
                tbl = table_ident[0]
            return schema, tbl

        if engine == "adbc":
            try:
                import adbc_driver_manager

                adbc_version = parse_version(
                    getattr(adbc_driver_manager, "__version__", "0.0")
                )
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "adbc_driver_manager not found"
                    "\n\nInstall Polars with: pip install adbc_driver_manager"
                ) from exc

            if if_exists == "fail":
                # if the table exists, 'create' will raise an error,
                # resulting in behaviour equivalent to 'fail'
                mode = "create"
            elif if_exists == "replace":
                if adbc_version < (0, 7):
                    adbc_str_version = ".".join(str(v) for v in adbc_version)
                    raise ModuleNotFoundError(
                        f"`if_exists = 'replace'` requires ADBC version >= 0.7, found {adbc_str_version}"
                    )
                mode = "replace"
            elif if_exists == "append":
                mode = "append"
            else:
                raise ValueError(
                    f"unexpected value for `if_exists`: {if_exists!r}"
                    f"\n\nChoose one of {{'fail', 'replace', 'append'}}"
                )

            with _open_adbc_connection(connection) as conn, conn.cursor() as cursor:
                db_schema, unpacked_table_name = unpack_table_name(table_name)
                if adbc_version >= (0, 7):
                    if "sqlite" in conn.adbc_get_info()["driver_name"].lower():
                        if if_exists == "replace":
                            # note: adbc doesn't (yet) support 'replace' for sqlite
                            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                            mode = "create"
                        catalog, db_schema = db_schema, None
                    else:
                        catalog = None

                    cursor.adbc_ingest(
                        unpacked_table_name,
                        data=self.to_arrow(),
                        mode=mode,
                        catalog_name=catalog,
                        db_schema_name=db_schema,
                    )
                elif db_schema is not None:
                    adbc_str_version = ".".join(str(v) for v in adbc_version)
                    raise ModuleNotFoundError(
                        # https://github.com/apache/arrow-adbc/issues/1000
                        # https://github.com/apache/arrow-adbc/issues/1109
                        f"use of schema-qualified table names requires ADBC version >= 0.8, found {adbc_str_version}"
                    )
                else:
                    cursor.adbc_ingest(unpacked_table_name, self.to_arrow(), mode)
                conn.commit()

        elif engine == "sqlalchemy":
            if parse_version(pd.__version__) < parse_version("1.5"):
                raise ModuleNotFoundError(
                    f"writing with engine 'sqlalchemy' requires pandas 1.5.x or higher, found {pd.__version__!r}"
                )
            try:
                from sqlalchemy import create_engine
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "sqlalchemy not found"
                    "\n\nInstall Polars with: pip install polars[sqlalchemy]"
                ) from exc

            # ensure conversion to pandas uses the pyarrow extension array option
            # so that we can make use of the sql/db export without copying data
            engine_sa = create_engine(connection)
            db_schema, table_name = unpack_table_name(table_name)

            self.to_pandas(use_pyarrow_extension_array=True).to_sql(
                name=table_name,
                schema=db_schema,
                con=engine_sa,
                if_exists=if_exists,
                index=False,
            )
        else:
            raise ValueError(f"engine {engine!r} is not supported")

    def write_delta(
        self,
        target: str | Path | deltalake.DeltaTable,
        *,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
        overwrite_schema: bool = False,
        storage_options: dict[str, str] | None = None,
        delta_write_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Write DataFrame as delta table.

        Parameters
        ----------
        target
            URI of a table or a DeltaTable object.
        mode : {'error', 'append', 'overwrite', 'ignore'}
            How to handle existing data.

            * If 'error', throw an error if the table already exists (default).
            * If 'append', will add new data.
            * If 'overwrite', will replace table with new data.
            * If 'ignore', will not write anything if table already exists.
        overwrite_schema
            If True, allows updating the schema of the table.
        storage_options
            Extra options for the storage backends supported by `deltalake`.
            For cloud storages, this may include configurations for authentication etc.

            * See a list of supported storage options for S3 `here <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants>`__.
            * See a list of supported storage options for GCS `here <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants>`__.
            * See a list of supported storage options for Azure `here <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.
        delta_write_options
            Additional keyword arguments while writing a Delta lake Table.
            See a list of supported write options `here <https://github.com/delta-io/delta-rs/blob/395d48b47ea638b70415899dc035cc895b220e55/python/deltalake/writer.py#L65>`__.

        Raises
        ------
        TypeError
            If the DataFrame contains unsupported data types.
        ArrowInvalidError
            If the DataFrame contains data types that could not be cast to their
            primitive type.

        Notes
        -----
        The Polars data types :class:`Null`, :class:`Categorical` and :class:`Time`
        are not supported by the delta protocol specification and will raise a
        TypeError.

        Some other data types are not supported but have an associated `primitive type
        <https://github.com/delta-io/delta/blob/master/PROTOCOL.md#primitive-types>`__
        to which they can be cast. This affects the following data types:

        - Unsigned integers
        - :class:`Datetime` types with millisecond or nanosecond precision or with
            time zone information
        - :class:`Utf8`, :class:`Binary`, and :class:`List` ('large' types)

        Polars columns are always nullable. To write data to a delta table with
        non-nullable columns, a custom pyarrow schema has to be passed to the
        `delta_write_options`. See the last example below.

        Examples
        --------
        Write a dataframe to the local filesystem as a Delta Lake table.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> table_path = "/path/to/delta-table/"
        >>> df.write_delta(table_path)  # doctest: +SKIP

        Append data to an existing Delta Lake table on the local filesystem.
        Note that this will fail if the schema of the new data does not match the
        schema of the existing table.

        >>> df.write_delta(table_path, mode="append")  # doctest: +SKIP

        Overwrite a Delta Lake table as a new version.
        If the schemas of the new and old data are the same, setting
        `overwrite_schema` is not required.

        >>> existing_table_path = "/path/to/delta-table/"
        >>> df.write_delta(
        ...     existing_table_path, mode="overwrite", overwrite_schema=True
        ... )  # doctest: +SKIP

        Write a dataframe as a Delta Lake table to a cloud object store like S3.

        >>> table_path = "s3://bucket/prefix/to/delta-table/"
        >>> df.write_delta(
        ...     table_path,
        ...     storage_options={
        ...         "AWS_REGION": "THE_AWS_REGION",
        ...         "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
        ...         "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
        ...     },
        ... )  # doctest: +SKIP

        Write DataFrame as a Delta Lake table with non-nullable columns.

        >>> import pyarrow as pa
        >>> existing_table_path = "/path/to/delta-table/"
        >>> df.write_delta(
        ...     existing_table_path,
        ...     delta_write_options={
        ...         "schema": pa.schema([pa.field("foo", pa.int64(), nullable=False)])
        ...     },
        ... )  # doctest: +SKIP

        """
        from polars.io.delta import (
            _check_for_unsupported_types,
            _check_if_delta_available,
            _convert_pa_schema_to_delta,
            _resolve_delta_lake_uri,
        )

        _check_if_delta_available()

        from deltalake.writer import write_deltalake

        if delta_write_options is None:
            delta_write_options = {}

        if isinstance(target, (str, Path)):
            target = _resolve_delta_lake_uri(str(target), strict=False)

        _check_for_unsupported_types(self.dtypes)

        data = self.to_arrow()

        schema = delta_write_options.pop("schema", None)
        if schema is None:
            schema = _convert_pa_schema_to_delta(data.schema)

        data = data.cast(schema)

        write_deltalake(
            table_or_uri=target,
            data=data,
            schema=schema,
            mode=mode,
            overwrite_schema=overwrite_schema,
            storage_options=storage_options,
            large_dtypes=True,
            **delta_write_options,
        )

    def estimated_size(self, unit: SizeUnit = "b") -> int | float:
        """
        Return an estimation of the total (heap) allocated size of the `DataFrame`.

        Estimated size is given in the specified unit (bytes by default).

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, [`StructArray`]'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        unchanged. However, this function will yield a smaller number. This is because
        this function returns the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.

        Parameters
        ----------
        unit : {'b', 'kb', 'mb', 'gb', 'tb'}
            Scale the returned size to the given unit.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": list(reversed(range(1_000_000))),
        ...         "y": [v / 1000 for v in range(1_000_000)],
        ...         "z": [str(v) for v in range(1_000_000)],
        ...     },
        ...     schema=[("x", pl.UInt32), ("y", pl.Float64), ("z", pl.Utf8)],
        ... )
        >>> df.estimated_size()
        25888898
        >>> df.estimated_size("mb")
        24.689577102661133

        """
        sz = self._df.estimated_size()
        return scale_bytes(sz, unit)

    def transpose(
        self,
        *,
        include_header: bool = False,
        header_name: str = "column",
        column_names: str | Iterable[str] | None = None,
    ) -> Self:
        """
        Transpose a DataFrame over the diagonal.

        Parameters
        ----------
        include_header
            If set, the column names will be added as first column.
        header_name
            If `include_header` is set, this determines the name of the column that will
            be inserted.
        column_names
            Optional iterable yielding strings or a string naming an existing column.
            These will name the value (non-header) columns in the transposed data.

        Notes
        -----
        This is a very expensive operation. Perhaps you can do it differently.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        >>> df.transpose(include_header=True)
        shape: (2, 4)
        ┌────────┬──────────┬──────────┬──────────┐
        │ column ┆ column_0 ┆ column_1 ┆ column_2 │
        │ ---    ┆ ---      ┆ ---      ┆ ---      │
        │ str    ┆ i64      ┆ i64      ┆ i64      │
        ╞════════╪══════════╪══════════╪══════════╡
        │ a      ┆ 1        ┆ 2        ┆ 3        │
        │ b      ┆ 1        ┆ 2        ┆ 3        │
        └────────┴──────────┴──────────┴──────────┘

        Replace the auto-generated column names with a list

        >>> df.transpose(include_header=False, column_names=["a", "b", "c"])
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 3   │
        │ 1   ┆ 2   ┆ 3   │
        └─────┴─────┴─────┘

        Include the header as a separate column

        >>> df.transpose(
        ...     include_header=True, header_name="foo", column_names=["a", "b", "c"]
        ... )
        shape: (2, 4)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 2   ┆ 3   │
        │ b   ┆ 1   ┆ 2   ┆ 3   │
        └─────┴─────┴─────┴─────┘

        Replace the auto-generated column with column names from a generator function

        >>> def name_generator():
        ...     base_name = "my_column_"
        ...     count = 0
        ...     while True:
        ...         yield f"{base_name}{count}"
        ...         count += 1
        ...
        >>> df.transpose(include_header=False, column_names=name_generator())
        shape: (2, 3)
        ┌─────────────┬─────────────┬─────────────┐
        │ my_column_0 ┆ my_column_1 ┆ my_column_2 │
        │ ---         ┆ ---         ┆ ---         │
        │ i64         ┆ i64         ┆ i64         │
        ╞═════════════╪═════════════╪═════════════╡
        │ 1           ┆ 2           ┆ 3           │
        │ 1           ┆ 2           ┆ 3           │
        └─────────────┴─────────────┴─────────────┘

        Use an existing column as the new column names

        >>> df = pl.DataFrame(dict(id=["a", "b", "c"], col1=[1, 3, 2], col2=[3, 4, 6]))
        >>> df.transpose(column_names="id")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 3   ┆ 2   │
        │ 3   ┆ 4   ┆ 6   │
        └─────┴─────┴─────┘
        >>> df.transpose(include_header=True, header_name="new_id", column_names="id")
        shape: (2, 4)
        ┌────────┬─────┬─────┬─────┐
        │ new_id ┆ a   ┆ b   ┆ c   │
        │ ---    ┆ --- ┆ --- ┆ --- │
        │ str    ┆ i64 ┆ i64 ┆ i64 │
        ╞════════╪═════╪═════╪═════╡
        │ col1   ┆ 1   ┆ 3   ┆ 2   │
        │ col2   ┆ 3   ┆ 4   ┆ 6   │
        └────────┴─────┴─────┴─────┘
        """
        keep_names_as = header_name if include_header else None
        if isinstance(column_names, Generator):
            column_names = [next(column_names) for _ in range(self.height)]
        return self._from_pydf(self._df.transpose(keep_names_as, column_names))

    def reverse(self) -> DataFrame:
        """
        Reverse the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "key": ["a", "b", "c"],
        ...         "val": [1, 2, 3],
        ...     }
        ... )
        >>> df.reverse()
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
        return self.select(F.col("*").reverse())

    def rename(self, mapping: dict[str, str]) -> DataFrame:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> df.rename({"foo": "apple"})
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
        return self.lazy().rename(mapping).collect(_eager=True)

    def insert_column(self, index: int, column: Series) -> Self:
        """
        Insert a Series at a certain column index.

        This operation is in place.

        Parameters
        ----------
        index
            Index at which to insert the new `Series` column.
        column
            `Series` to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series("baz", [97, 98, 99])
        >>> df.insert_column(1, s)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ baz ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 97  ┆ 4   │
        │ 2   ┆ 98  ┆ 5   │
        │ 3   ┆ 99  ┆ 6   │
        └─────┴─────┴─────┘

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> s = pl.Series("d", [-2.5, 15, 20.5, 0])
        >>> df.insert_column(3, s)
        shape: (4, 4)
        ┌─────┬──────┬───────┬──────┐
        │ a   ┆ b    ┆ c     ┆ d    │
        │ --- ┆ ---  ┆ ---   ┆ ---  │
        │ i64 ┆ f64  ┆ bool  ┆ f64  │
        ╞═════╪══════╪═══════╪══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ -2.5 │
        │ 2   ┆ 4.0  ┆ true  ┆ 15.0 │
        │ 3   ┆ 10.0 ┆ false ┆ 20.5 │
        │ 4   ┆ 13.0 ┆ true  ┆ 0.0  │
        └─────┴──────┴───────┴──────┘

        """
        if index < 0:
            index = len(self.columns) + index
        self._df.insert_column(index, column._s)
        return self

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
    ) -> DataFrame:
        """
        Filter the rows in the DataFrame based on a predicate expression.

        The original order of the remaining rows is preserved.

        Parameters
        ----------
        predicates
            Expression that evaluates to a boolean Series.
        constraints
            Column filters. Use name=value to filter column name by the supplied value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )

        Filter on one condition:

        >>> df.filter(pl.col("foo") > 1)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Filter on multiple conditions, combined with and/or operators:

        >>> df.filter((pl.col("foo") < 3) & (pl.col("ham") == "a"))
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        >>> df.filter((pl.col("foo") == 1) | (pl.col("ham") == "c"))
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Provide multiple filters using `*args` syntax:

        >>> df.filter(
        ...     pl.col("foo") <= 2,
        ...     ~pl.col("ham").is_in(["b", "c"]),
        ... )
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Provide multiple filters using `**kwargs` syntax:

        >>> df.filter(foo=2, ham="b")
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        return self.lazy().filter(*predicates, **constraints).collect(_eager=True)

    @overload
    def glimpse(
        self,
        *,
        max_items_per_column: int = ...,
        max_colname_length: int = ...,
        return_as_string: Literal[False],
    ) -> None:
        ...

    @overload
    def glimpse(
        self,
        *,
        max_items_per_column: int = ...,
        max_colname_length: int = ...,
        return_as_string: Literal[True],
    ) -> str:
        ...

    def glimpse(
        self,
        *,
        max_items_per_column: int = 10,
        max_colname_length: int = 50,
        return_as_string: bool = False,
    ) -> str | None:
        """
        Return a dense preview of the DataFrame.

        The formatting shows one line per column so that wide dataframes display
        cleanly. Each line shows the column name, the data type, and the first
        few values.

        Parameters
        ----------
        max_items_per_column
            Maximum number of items to show per column.
        max_colname_length
            Maximum length of the displayed column names; values that exceed this
            value are truncated with a trailing ellipsis.
        return_as_string
            If True, return the preview as a string instead of printing to stdout.

        See Also
        --------
        describe, head, tail

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, 2.8, 3.0],
        ...         "b": [4, 5, None],
        ...         "c": [True, False, True],
        ...         "d": [None, "b", "c"],
        ...         "e": ["usd", "eur", None],
        ...         "f": [date(2020, 1, 1), date(2021, 1, 2), date(2022, 1, 1)],
        ...     }
        ... )
        >>> df.glimpse()
        Rows: 3
        Columns: 6
        $ a  <f64> 1.0, 2.8, 3.0
        $ b  <i64> 4, 5, None
        $ c <bool> True, False, True
        $ d  <str> None, 'b', 'c'
        $ e  <str> 'usd', 'eur', None
        $ f <date> 2020-01-01, 2021-01-02, 2022-01-01

        """
        # always print at most this number of values (mainly ensures that
        # we do not cast long arrays to strings, which would be slow)
        max_n_values = min(max_items_per_column, self.height)
        schema = self.schema

        def _parse_column(col_name: str, dtype: PolarsDataType) -> tuple[str, str, str]:
            fn = repr if schema[col_name] == Utf8 else str
            values = self[:max_n_values][col_name].to_list()
            val_str = ", ".join(fn(v) for v in values)  # type: ignore[operator]
            if len(col_name) > max_colname_length:
                col_name = col_name[: (max_colname_length - 1)] + "…"
            return col_name, f"<{_dtype_str_repr(dtype)}>", val_str

        data = [_parse_column(s, dtype) for s, dtype in self.schema.items()]

        # determine column layout widths
        max_col_name = max((len(col_name) for col_name, _, _ in data))
        max_col_dtype = max((len(dtype_str) for _, dtype_str, _ in data))
        max_col_values = 100 - max_col_name - max_col_dtype

        # print header
        output = StringIO()
        output.write(f"Rows: {self.height}\nColumns: {self.width}\n")

        # print individual columns: one row per column
        for col_name, dtype_str, val_str in data:
            output.write(
                f"$ {col_name:<{max_col_name}}"
                f" {dtype_str:>{max_col_dtype}}"
                f" {val_str:<{min(len(val_str), max_col_values)}}\n"
            )

        s = output.getvalue()
        if return_as_string:
            return s

        print(s, end=None)
        return None

    def describe(
        self, percentiles: Sequence[float] | float | None = (0.25, 0.50, 0.75)
    ) -> Self:
        """
        Summary statistics for a DataFrame.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics.
            All values must be in the range `[0, 1]`.

        Notes
        -----
        The median is included by default as the 50% percentile.

        See Also
        --------
        glimpse

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, 2.8, 3.0],
        ...         "b": [4, 5, None],
        ...         "c": [True, False, True],
        ...         "d": [None, "b", "c"],
        ...         "e": ["usd", "eur", None],
        ...         "f": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        ...     }
        ... )
        >>> df.describe()
        shape: (9, 7)
        ┌────────────┬──────────┬──────────┬──────────┬──────┬──────┬────────────┐
        │ describe   ┆ a        ┆ b        ┆ c        ┆ d    ┆ e    ┆ f          │
        │ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---  ┆ ---        │
        │ str        ┆ f64      ┆ f64      ┆ f64      ┆ str  ┆ str  ┆ str        │
        ╞════════════╪══════════╪══════════╪══════════╪══════╪══════╪════════════╡
        │ count      ┆ 3.0      ┆ 3.0      ┆ 3.0      ┆ 3    ┆ 3    ┆ 3          │
        │ null_count ┆ 0.0      ┆ 1.0      ┆ 0.0      ┆ 1    ┆ 1    ┆ 0          │
        │ mean       ┆ 2.266667 ┆ 4.5      ┆ 0.666667 ┆ null ┆ null ┆ null       │
        │ std        ┆ 1.101514 ┆ 0.707107 ┆ 0.57735  ┆ null ┆ null ┆ null       │
        │ min        ┆ 1.0      ┆ 4.0      ┆ 0.0      ┆ b    ┆ eur  ┆ 2020-01-01 │
        │ 25%        ┆ 1.0      ┆ 4.0      ┆ null     ┆ null ┆ null ┆ null       │
        │ 50%        ┆ 2.8      ┆ 5.0      ┆ null     ┆ null ┆ null ┆ null       │
        │ 75%        ┆ 3.0      ┆ 5.0      ┆ null     ┆ null ┆ null ┆ null       │
        │ max        ┆ 3.0      ┆ 5.0      ┆ 1.0      ┆ c    ┆ usd  ┆ 2022-01-01 │
        └────────────┴──────────┴──────────┴──────────┴──────┴──────┴────────────┘

        """
        # determine metrics and optional/additional percentiles
        metrics = ["count", "null_count", "mean", "std", "min"]
        percentile_exprs = []
        for p in parse_percentiles(percentiles):
            percentile_exprs.append(F.all().quantile(p).name.prefix(f"{p}:"))
            metrics.append(f"{p:.0%}")
        metrics.append("max")

        # execute metrics in parallel
        df_metrics = self.select(
            F.all().count().name.prefix("count:"),
            F.all().null_count().name.prefix("null_count:"),
            F.all().mean().name.prefix("mean:"),
            F.all().std().name.prefix("std:"),
            F.all().min().name.prefix("min:"),
            *percentile_exprs,
            F.all().max().name.prefix("max:"),
        ).row(0)

        # reshape wide result
        n_cols = len(self.columns)
        described = [
            df_metrics[(n * n_cols) : (n + 1) * n_cols] for n in range(len(metrics))
        ]

        # cast by column type (numeric/bool -> float), (other -> string)
        summary = dict(zip(self.columns, list(zip(*described))))
        num_or_bool = NUMERIC_DTYPES | {Boolean}
        for c, tp in self.schema.items():
            summary[c] = [
                None
                if (v is None or isinstance(v, dict))
                else (float(v) if tp in num_or_bool else str(v))
                for v in summary[c]
            ]

        # return results as a frame
        df_summary = self.__class__(summary)
        df_summary.insert_column(0, pl.Series("describe", metrics))
        return df_summary

    def get_column_index(self, name: str) -> int:
        """
        Find the index of a column by name.

        Parameters
        ----------
        name
            Name of the column to find.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> df.get_column_index("ham")
        2

        """
        return self._df.get_column_index(name)

    def replace_column(self, index: int, column: Series) -> Self:
        """
        Replace a column at an index location.

        This operation is in place.

        Parameters
        ----------
        index
            Column index.
        column
            Series that will replace the column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> s = pl.Series("apple", [10, 20, 30])
        >>> df.replace_column(0, s)
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ apple ┆ bar ┆ ham │
        │ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64 ┆ str │
        ╞═══════╪═════╪═════╡
        │ 10    ┆ 6   ┆ a   │
        │ 20    ┆ 7   ┆ b   │
        │ 30    ┆ 8   ┆ c   │
        └───────┴─────┴─────┘
        """
        if index < 0:
            index = len(self.columns) + index
        self._df.replace_column(index, column._s)
        return self

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
    ) -> DataFrame:
        """
        Sort the dataframe by the given columns.

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

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [6.0, 5.0, 4.0],
        ...         "c": ["a", "c", "b"],
        ...     }
        ... )
        >>> df.sort("a")
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

        >>> df.sort(pl.col("a") + pl.col("b") * 2, nulls_last=True)
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

        >>> df.sort(["c", "a"], descending=True)
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

        >>> df.sort("c", "a", descending=[False, True])
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
        return (
            self.lazy()
            .sort(by, *more_by, descending=descending, nulls_last=nulls_last)
            .collect(_eager=True)
        )

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
        maintain_order: bool = False,
    ) -> DataFrame:
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
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 largest values in column b.

        >>> df.top_k(4, by="b")
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

        >>> df.top_k(4, by=["b", "a"])
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
        return (
            self.lazy()
            .top_k(
                k,
                by=by,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
            )
            .collect(
                projection_pushdown=False,
                predicate_pushdown=False,
                comm_subplan_elim=False,
                slice_pushdown=True,
            )
        )

    def bottom_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        descending: bool | Sequence[bool] = False,
        nulls_last: bool = False,
        maintain_order: bool = False,
    ) -> DataFrame:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 smallest values in column b.

        >>> df.bottom_k(4, by="b")
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

        >>> df.bottom_k(4, by=["a", "b"])
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
        return (
            self.lazy()
            .bottom_k(
                k,
                by=by,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
            )
            .collect(
                projection_pushdown=False,
                predicate_pushdown=False,
                comm_subplan_elim=False,
                slice_pushdown=True,
            )
        )

    def equals(self, other: DataFrame, *, null_equal: bool = True) -> bool:
        """
        Check whether the DataFrame is equal to another DataFrame.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.

        See Also
        --------
        assert_frame_equal

        Examples
        --------
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df2 = pl.DataFrame(
        ...     {
        ...         "foo": [3, 2, 1],
        ...         "bar": [8.0, 7.0, 6.0],
        ...         "ham": ["c", "b", "a"],
        ...     }
        ... )
        >>> df1.equals(df1)
        True
        >>> df1.equals(df2)
        False

        """
        return self._df.equals(other._df, null_equal)

    @deprecate_function(
        "DataFrame.replace is deprecated and will be removed in a future version. "
        "Please use\n"
        "    df = df.with_columns(new_column.alias(column_name))\n"
        "instead.",
        version="0.19.0",
    )
    def replace(self, column: str, new_column: Series) -> Self:
        """
        Replace a column by a new Series.

        Parameters
        ----------
        column
            Column to replace.
        new_column
            New column to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series([10, 20, 30])
        >>> df.replace("foo", s)  # works in-place!  # doctest: +SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 10  ┆ 4   │
        │ 20  ┆ 5   │
        │ 30  ┆ 6   │
        └─────┴─────┘

        """
        return self._replace(column, new_column)

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.slice(1, 2)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7.0 ┆ b   │
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        """
        if (length is not None) and length < 0:
            length = self.height - offset + length
        return self._from_pydf(self._df.slice(offset, length))

    def head(self, n: int = 5) -> Self:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the last `abs(n)`.

        See Also
        --------
        tail, glimpse, slice

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.head(3)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Pass a negative value to get all rows `except` the last `abs(n)`.

        >>> df.head(-3)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        if n < 0:
            n = max(0, self.height + n)
        return self._from_pydf(self._df.head(n))

    def tail(self, n: int = 5) -> Self:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the first `abs(n)`.

        See Also
        --------
        head, slice

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.tail(3)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        │ 4   ┆ 9   ┆ d   │
        │ 5   ┆ 10  ┆ e   │
        └─────┴─────┴─────┘

        Pass a negative value to get all rows `except` the first `abs(n)`.

        >>> df.tail(-3)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 4   ┆ 9   ┆ d   │
        │ 5   ┆ 10  ┆ e   │
        └─────┴─────┴─────┘

        """
        if n < 0:
            n = max(0, self.height + n)
        return self._from_pydf(self._df.tail(n))

    def limit(self, n: int = 5) -> Self:
        """
        Get the first `n` rows.

        Alias for :func:`DataFrame.head`.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the last `abs(n)`.

        See Also
        --------
        head

        """
        return self.head(n)

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> DataFrame:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, None, 8],
        ...         "ham": ["a", "b", None],
        ...     }
        ... )

        The default behavior of this method is to drop rows where any single
        value of the row is null.

        >>> df.drop_nulls()
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
        >>> df.drop_nulls(subset=cs.integer())
        shape: (2, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ a    │
        │ 3   ┆ 8   ┆ null │
        └─────┴─────┴──────┘

        Below are some additional examples that show how to drop null
        values based on other conditions.

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
        │ f32  ┆ i64  ┆ i64  │
        ╞══════╪══════╪══════╡
        │ null ┆ 1    ┆ 1    │
        │ null ┆ 2    ┆ null │
        │ null ┆ null ┆ null │
        │ null ┆ 1    ┆ 1    │
        └──────┴──────┴──────┘

        Drop a row only if all values are null:

        >>> df.filter(~pl.all_horizontal(pl.all().is_null()))
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

        Drop a column if all values are null:

        >>> df[[s.name for s in df if not (s.null_count() == df.height)]]
        shape: (4, 2)
        ┌──────┬──────┐
        │ b    ┆ c    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 1    ┆ 1    │
        │ 2    ┆ null │
        │ null ┆ null │
        │ 1    ┆ 1    │
        └──────┴──────┘

        """
        return self.lazy().drop_nulls(subset).collect(_eager=True)

    def pipe(
        self,
        function: Callable[Concatenate[DataFrame, P], T],
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

        Notes
        -----
        It is recommended to use LazyFrame when piping operations, in order
        to fully take advantage of query optimization and parallelization.
        See :meth:`df.lazy() <polars.DataFrame.lazy>`.

        Examples
        --------
        >>> def cast_str_to_int(data, col_name):
        ...     return data.with_columns(pl.col(col_name).cast(pl.Int64))
        ...
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]})
        >>> df.pipe(cast_str_to_int, col_name="b")
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
        >>> df.pipe(lambda tdf: tdf.select(sorted(tdf.columns)))
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

    def with_row_count(self, name: str = "row_nr", offset: int = 0) -> Self:
        """
        Add a column at index 0 that counts the rows.

        Parameters
        ----------
        name
            Name of the column to add.
        offset
            Start the row count at this offset. Default = 0

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.with_row_count()
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
        return self._from_pydf(self._df.with_row_count(name, offset))

    def group_by(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        maintain_order: bool = False,
    ) -> GroupBy:
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
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

            .. note::
                Within each group, the order of rows is always preserved, regardless
                of this argument.

        Returns
        -------
        GroupBy
            Object which can be used to perform aggregations.

        Examples
        --------
        Group by one column and call `agg` to compute the grouped sum of another
        column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.group_by("a").agg(pl.col("b").sum())  # doctest: +IGNORE_RESULT
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

        >>> df.group_by("a", maintain_order=True).agg(pl.col("c"))
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

        >>> df.group_by(["a", "b"]).agg(pl.max("c"))  # doctest: +IGNORE_RESULT
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

        >>> df.group_by("a", pl.col("b") // 2).agg(pl.col("c").mean())  # doctest: +SKIP
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

        The `GroupBy` object returned by this method is iterable, returning the name
        and data of each group.

        >>> for name, data in df.group_by("a"):  # doctest: +SKIP
        ...     print(name)
        ...     print(data)
        ...
        a
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘
        b
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘
        c
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘

        """
        return GroupBy(self, by, *more_by, maintain_order=maintain_order)

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: IntoExpr | Iterable[IntoExpr] | None = None,
        check_sorted: bool = True,
    ) -> RollingGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        Different from a `group_by_dynamic` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals use
        :func:`DataFrame.group_by_dynamic`.

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

        - **"1i"      # length 1**
        - **"10i"     # length 10**

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling operation on indices, dtype needs to be one of
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
        RollingGroupBy
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
        >>> df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
        ...     pl.col("dt").str.strptime(pl.Datetime).set_sorted()
        ... )
        >>> out = df.rolling(index_column="dt", period="2d").agg(
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
        period = deprecate_saturating(period)
        offset = deprecate_saturating(offset)
        return RollingGroupBy(
            self,
            index_column=index_column,
            period=period,
            offset=offset,
            closed=closed,
            by=by,
            check_sorted=check_sorted,
        )

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
    ) -> DynamicGroupBy:
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
        DynamicGroupBy
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
        >>> df = pl.DataFrame(
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

        >>> df.group_by_dynamic("time", every="1h", closed="right").agg(pl.col("n"))
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

        >>> df.group_by_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg(pl.col("n").mean())
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

        >>> df.group_by_dynamic("time", every="1h", closed="left").agg(pl.col("n"))
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

        >>> df.group_by_dynamic("time", every="1h", closed="both").agg(pl.col("n"))
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

        >>> df = df.with_columns(groups=pl.Series(["a", "a", "a", "b", "b", "a", "a"]))
        >>> df
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
        >>> df.group_by_dynamic(
        ...     "time",
        ...     every="1h",
        ...     closed="both",
        ...     by="groups",
        ...     include_boundaries=True,
        ... ).agg(pl.col("n"))
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

        >>> df = pl.DataFrame(
        ...     {
        ...         "idx": pl.int_range(0, 6, eager=True),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> (
        ...     df.group_by_dynamic(
        ...         "idx",
        ...         every="2i",
        ...         period="3i",
        ...         include_boundaries=True,
        ...         closed="right",
        ...     ).agg(pl.col("A").alias("A_agg_list"))
        ... )
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
        return DynamicGroupBy(
            self,
            index_column=index_column,
            every=every,
            period=period,
            offset=offset,
            truncate=truncate,
            label=label,
            include_boundaries=include_boundaries,
            closed=closed,
            by=by,
            start_by=start_by,
            check_sorted=check_sorted,
        )

    def upsample(
        self,
        time_column: str,
        *,
        every: str | timedelta,
        offset: str | timedelta | None = None,
        by: str | Sequence[str] | None = None,
        maintain_order: bool = False,
    ) -> Self:
        """
        Upsample a DataFrame at a regular frequency.

        The `every` and `offset` arguments are created with
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

        - "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds


        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        Parameters
        ----------
        time_column
            time column will be used to determine a date_range.
            Note that this column has to be sorted for the output to make sense.
        every
            interval will start 'every' duration
        offset
            change the start of the date_range by this offset.
        by
            First group by these columns and then upsample for every group
        maintain_order
            Keep the ordering predictable. This is slower.

        Returns
        -------
        DataFrame
            Result will be sorted by `time_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).

        Examples
        --------
        Upsample a DataFrame by a certain interval.

        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "time": [
        ...             datetime(2021, 2, 1),
        ...             datetime(2021, 4, 1),
        ...             datetime(2021, 5, 1),
        ...             datetime(2021, 6, 1),
        ...         ],
        ...         "groups": ["A", "B", "A", "B"],
        ...         "values": [0, 1, 2, 3],
        ...     }
        ... ).set_sorted("time")
        >>> df.upsample(
        ...     time_column="time", every="1mo", by="groups", maintain_order=True
        ... ).select(pl.all().forward_fill())
        shape: (7, 3)
        ┌─────────────────────┬────────┬────────┐
        │ time                ┆ groups ┆ values │
        │ ---                 ┆ ---    ┆ ---    │
        │ datetime[μs]        ┆ str    ┆ i64    │
        ╞═════════════════════╪════════╪════════╡
        │ 2021-02-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-03-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-04-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-05-01 00:00:00 ┆ A      ┆ 2      │
        │ 2021-04-01 00:00:00 ┆ B      ┆ 1      │
        │ 2021-05-01 00:00:00 ┆ B      ┆ 1      │
        │ 2021-06-01 00:00:00 ┆ B      ┆ 3      │
        └─────────────────────┴────────┴────────┘

        """
        every = deprecate_saturating(every)
        offset = deprecate_saturating(offset)
        if by is None:
            by = []
        if isinstance(by, str):
            by = [by]
        if offset is None:
            offset = "0ns"

        every = _timedelta_to_pl_duration(every)
        offset = _timedelta_to_pl_duration(offset)

        return self._from_pydf(
            self._df.upsample(by, time_column, every, offset, maintain_order)
        )

    def join_asof(
        self,
        other: DataFrame,
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
    ) -> DataFrame:
        """
        Perform an asof join.

        This is similar to a left-join except that we match on nearest key rather than
        equal keys.

        Both DataFrames must be sorted by the asof_join key.

        For each row in the left DataFrame:

          - A "backward" search selects the last row in the right DataFrame whose
            'on' key is less than or equal to the left's key.

          - A "forward" search selects the first row in the right DataFrame whose
            'on' key is greater than or equal to the left's key.

          - A "nearest" search selects the last row in the right DataFrame whose value
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
            join on these columns before doing asof join
        by_left
            join on these columns before doing asof join
        by_right
            join on these columns before doing asof join
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
        ... ).set_sorted("date")
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
        ... ).set_sorted("date")
        >>> population.join_asof(gdp, on="date", strategy="backward")
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
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"expected `other` join table to be a DataFrame, got {type(other).__name__!r}"
            )

        if on is not None:
            if not isinstance(on, (str, pl.Expr)):
                raise TypeError(
                    f"expected `on` to be str or Expr, got {type(on).__name__!r}"
                )
        else:
            if not isinstance(left_on, (str, pl.Expr)):
                raise TypeError(
                    f"expected `left_on` to be str or Expr, got {type(left_on).__name__!r}"
                )
            elif not isinstance(right_on, (str, pl.Expr)):
                raise TypeError(
                    f"expected `right_on` to be str or Expr, got {type(right_on).__name__!r}"
                )

        return (
            self.lazy()
            .join_asof(
                other.lazy(),
                left_on=left_on,
                right_on=right_on,
                on=on,
                by_left=by_left,
                by_right=by_right,
                by=by,
                strategy=strategy,
                suffix=suffix,
                tolerance=tolerance,
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
            )
            .collect(_eager=True)
        )

    def join(
        self,
        other: DataFrame,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
    ) -> DataFrame:
        """
        Join in SQL-like fashion.

        Parameters
        ----------
        other
            DataFrame to join with.
        on
            Name(s) of the join columns in both DataFrames.
        how : {'inner', 'left', 'outer', 'semi', 'anti', 'cross'}
            Join strategy.

            .. note::
                A left join preserves the row order of the left DataFrame.
        left_on
            Name(s) of the left join column(s).
        right_on
            Name(s) of the right join column(s).
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

        Returns
        -------
        DataFrame

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
        ... )
        >>> other_df = pl.DataFrame(
        ...     {
        ...         "apple": ["x", "y", "z"],
        ...         "ham": ["a", "b", "d"],
        ...     }
        ... )
        >>> df.join(other_df, on="ham")
        shape: (2, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘

        >>> df.join(other_df, on="ham", how="outer")
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

        >>> df.join(other_df, on="ham", how="left")
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

        >>> df.join(other_df, on="ham", how="semi")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        └─────┴─────┴─────┘

        >>> df.join(other_df, on="ham", how="anti")
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        Notes
        -----
        For joining on columns with categorical data, see `pl.StringCache()`.

        """
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"expected `other` join table to be a DataFrame, got {type(other).__name__!r}"
            )

        return (
            self.lazy()
            .join(
                other=other.lazy(),
                left_on=left_on,
                right_on=right_on,
                on=on,
                how=how,
                suffix=suffix,
                validate=validate,
            )
            .collect(_eager=True)
        )

    def map_rows(
        self,
        function: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        inference_size: int = 256,
    ) -> DataFrame:
        """
        Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

        The UDF will receive each row as a tuple of values: `udf(row)`.

        Implementing logic using a Python function is almost always *significantly*
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs typically cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        function
            Custom function or lambda.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema.

        Notes
        -----
        * The frame-level `apply` cannot track column names (as the UDF is a black-box
          that may arbitrarily drop, rearrange, transform, or add new columns); if you
          want to apply a UDF such that column names are preserved, you should use the
          expression-level `apply` syntax instead.

        * If your function is expensive and you don't want it to be called more than
          once for a given input, consider applying an `@lru_cache` decorator to it.
          If your data is suitable you may achieve *significant* speedups.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [-1, 5, 8]})

        Return a DataFrame by mapping each row to a tuple:

        >>> df.map_rows(lambda t: (t[0] * 2, t[1] * 3))
        shape: (3, 2)
        ┌──────────┬──────────┐
        │ column_0 ┆ column_1 │
        │ ---      ┆ ---      │
        │ i64      ┆ i64      │
        ╞══════════╪══════════╡
        │ 2        ┆ -3       │
        │ 4        ┆ 15       │
        │ 6        ┆ 24       │
        └──────────┴──────────┘

        However, it is much better to implement this with a native expression:

        >>> df.select(
        ...     pl.col("foo") * 2,
        ...     pl.col("bar") * 3,
        ... )  # doctest: +IGNORE_RESULT

        Return a DataFrame with a single column by mapping each row to a scalar:

        >>> df.map_rows(lambda t: (t[0] * 2 + t[1]))  # doctest: +SKIP
        shape: (3, 1)
        ┌───────┐
        │ apply │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 1     │
        │ 9     │
        │ 14    │
        └───────┘

        In this case it is better to use the following native expression:

        >>> df.select(pl.col("foo") * 2 + pl.col("bar"))  # doctest: +IGNORE_RESULT

        """
        # TODO: Enable warning for inefficient map
        # from polars.utils.udfs import warn_on_inefficient_map
        # warn_on_inefficient_map(function, columns=self.columns, map_target="frame)

        out, is_df = self._df.map_rows(function, return_dtype, inference_size)
        if is_df:
            return self._from_pydf(out)
        else:
            return wrap_s(out).to_frame()

    def hstack(
        self, columns: list[Series] | DataFrame, *, in_place: bool = False
    ) -> Self:
        """
        Return a new DataFrame grown horizontally by stacking multiple Series to it.

        Parameters
        ----------
        columns
            Series to stack.
        in_place
            Modify in place.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> x = pl.Series("apple", [10, 20, 30])
        >>> df.hstack([x])
        shape: (3, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ str ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6   ┆ a   ┆ 10    │
        │ 2   ┆ 7   ┆ b   ┆ 20    │
        │ 3   ┆ 8   ┆ c   ┆ 30    │
        └─────┴─────┴─────┴───────┘

        """
        if not isinstance(columns, list):
            columns = columns.get_columns()
        if in_place:
            self._df.hstack_mut([s._s for s in columns])
            return self
        else:
            return self._from_pydf(self._df.hstack([s._s for s in columns]))

    @deprecate_renamed_parameter("df", "other", version="0.18.8")
    def vstack(self, other: DataFrame, *, in_place: bool = False) -> Self:
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.

        Parameters
        ----------
        other
            DataFrame to stack.
        in_place
            Modify in place.

        See Also
        --------
        extend

        Examples
        --------
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2],
        ...         "bar": [6, 7],
        ...         "ham": ["a", "b"],
        ...     }
        ... )
        >>> df2 = pl.DataFrame(
        ...     {
        ...         "foo": [3, 4],
        ...         "bar": [8, 9],
        ...         "ham": ["c", "d"],
        ...     }
        ... )
        >>> df1.vstack(df2)
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        │ 4   ┆ 9   ┆ d   │
        └─────┴─────┴─────┘

        """
        if in_place:
            try:
                self._df.vstack_mut(other._df)
            except RuntimeError as exc:
                if str(exc) == "Already mutably borrowed":
                    self._df.vstack_mut(other._df.clone())
                    return self
                else:
                    raise
            else:
                return self

        return self._from_pydf(self._df.vstack(other._df))

    def extend(self, other: DataFrame) -> Self:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from `vstack` which adds the chunks from `other` to the chunks of
        this `DataFrame`, `extend` appends the data from `other` to the underlying
        memory locations and thus may cause a reallocation.

        If this does not cause a reallocation, the resulting data structure will not
        have any extra chunks and thus will yield faster queries.

        Prefer `extend` over `vstack` when you want to do a query after a single
        append. For instance, during online operations where you add `n` rows and rerun
        a query.

        Prefer `vstack` over `extend` when you want to append many times before
        doing a query. For instance, when you read in multiple files and want to store
        them in a single `DataFrame`. In the latter case, finish the sequence of
        `vstack` operations with a `rechunk`.

        Parameters
        ----------
        other
            DataFrame to vertically add.

        Warnings
        --------
        This method modifies the dataframe in-place. The dataframe is returned for
        convenience only.

        See Also
        --------
        vstack

        Examples
        --------
        >>> df1 = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df2 = pl.DataFrame({"foo": [10, 20, 30], "bar": [40, 50, 60]})
        >>> df1.extend(df2)
        shape: (6, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        │ 10  ┆ 40  │
        │ 20  ┆ 50  │
        │ 30  ┆ 60  │
        └─────┴─────┘

        """
        try:
            self._df.extend(other._df)
        except RuntimeError as exc:
            if str(exc) == "Already mutably borrowed":
                self._df.extend(other._df.clone())
            else:
                raise
        return self

    def drop(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> DataFrame:
        """
        Remove columns from the dataframe.

        Parameters
        ----------
        columns
            Names of the columns that should be removed from the dataframe, or
            a selector that determines the columns to drop.
        *more_columns
            Additional columns to drop, specified as positional arguments.

        Examples
        --------
        Drop a single column by passing the name of that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.drop("ham")
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

        Drop multiple columns by passing a list of column names.

        >>> df.drop(["bar", "ham"])
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

        Drop multiple columns by passing a selector.

        >>> import polars.selectors as cs
        >>> df.drop(cs.numeric())
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

        >>> df.drop("foo", "ham")
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
        return self.lazy().drop(columns, *more_columns).collect(_eager=True)

    def drop_in_place(self, name: str) -> Series:
        """
        Drop a single column in-place and return the dropped column.

        Parameters
        ----------
        name
            Name of the column to drop.

        Returns
        -------
        Series
            The dropped column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.drop_in_place("ham")
        shape: (3,)
        Series: 'ham' [str]
        [
            "a"
            "b"
            "c"
        ]

        """
        return wrap_s(self._df.drop_in_place(name))

    def cast(
        self,
        dtypes: Mapping[ColumnNameOrSelector, PolarsDataType] | PolarsDataType,
        *,
        strict: bool = True,
    ) -> DataFrame:
        """
        Cast DataFrame column(s) to the specified dtype(s).

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        ...     }
        ... )

        Cast specific frame columns to the specified dtypes:

        >>> df.cast({"foo": pl.Float32, "bar": pl.UInt8})
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

        >>> df.cast(pl.Utf8).to_dict(as_series=False)
        {'foo': ['1', '2', '3'],
         'bar': ['6.0', '7.0', '8.0'],
         'ham': ['2020-01-02', '2021-03-04', '2022-05-06']}

        Use selectors to define the columns being cast:

        >>> import polars.selectors as cs
        >>> df.cast({cs.numeric(): pl.UInt32, cs.temporal(): pl.Utf8})
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
        return self.lazy().cast(dtypes, strict=strict).collect(_eager=True)

    def clear(self, n: int = 0) -> Self:
        """
        Create an empty (n=0) or `n`-row null-filled (n>0) copy of the DataFrame.

        Returns a `n`-row null-filled DataFrame with an identical schema.
        `n` can be greater than the current number of rows in the DataFrame.

        Parameters
        ----------
        n
            Number of (null-filled) rows to return in the cleared frame.

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
        ... )
        >>> df.clear()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        >>> df.clear(n=2)
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
        # faster path
        if n == 0:
            return self._from_pydf(self._df.clear())
        if n > 0 or len(self) > 0:
            return self.__class__(
                {
                    nm: pl.Series(name=nm, dtype=tp).extend_constant(None, n)
                    for nm, tp in self.schema.items()
                }
            )
        return self.clone()

    def clone(self) -> Self:
        """
        Create a copy of this DataFrame.

        This is a cheap operation that does not copy data.

        See Also
        --------
        clear : Create an empty copy of the current DataFrame, with identical
            schema but no data.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.clone()
        shape: (4, 3)
        ┌─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     │
        │ --- ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  │
        │ 2   ┆ 4.0  ┆ true  │
        │ 3   ┆ 10.0 ┆ false │
        │ 4   ┆ 13.0 ┆ true  │
        └─────┴──────┴───────┘

        """
        return self._from_pydf(self._df.clone())

    def get_columns(self) -> list[Series]:
        """
        Get the DataFrame as a List of Series.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.get_columns()
        [shape: (3,)
        Series: 'foo' [i64]
        [
                1
                2
                3
        ], shape: (3,)
        Series: 'bar' [i64]
        [
                4
                5
                6
        ]]

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.get_columns()
        [shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
        ], shape: (4,)
        Series: 'b' [f64]
        [
            0.5
            4.0
            10.0
            13.0
        ], shape: (4,)
        Series: 'c' [bool]
        [
            true
            true
            false
            true
        ]]

        """
        return [wrap_s(s) for s in self._df.get_columns()]

    def get_column(self, name: str) -> Series:
        """
        Get a single column by name.

        Parameters
        ----------
        name : str
            Name of the column to retrieve.

        Returns
        -------
        Series

        See Also
        --------
        to_series

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.get_column("foo")
        shape: (3,)
        Series: 'foo' [i64]
        [
                1
                2
                3
        ]

        """
        return wrap_s(self._df.get_column(name))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        *,
        matches_supertype: bool = True,
    ) -> DataFrame:
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
            Fill all matching supertype of the fill `value`.

        Returns
        -------
        DataFrame
            DataFrame with None values replaced by the filling strategy.

        See Also
        --------
        fill_nan

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 4],
        ...         "b": [0.5, 4, None, 13],
        ...     }
        ... )
        >>> df.fill_null(99)
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
        >>> df.fill_null(strategy="forward")
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

        >>> df.fill_null(strategy="max")
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

        >>> df.fill_null(strategy="zero")
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
        return (
            self.lazy()
            .fill_null(value, strategy, limit, matches_supertype=matches_supertype)
            .collect(_eager=True)
        )

    def fill_nan(self, value: Expr | int | float | None) -> DataFrame:
        """
        Fill floating point NaN values by an Expression evaluation.

        Parameters
        ----------
        value
            Value with which to replace NaN values.

        Returns
        -------
        DataFrame
            DataFrame with NaN values replaced by the given value.

        Warnings
        --------
        Note that floating point NaNs (Not a Number) are not missing values!
        To replace missing values, use :func:`fill_null`.

        See Also
        --------
        fill_null

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.5, 2, float("nan"), 4],
        ...         "b": [0.5, 4, float("nan"), 13],
        ...     }
        ... )
        >>> df.fill_nan(99)
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
        return self.lazy().fill_nan(value).collect(_eager=True)

    def explode(
        self,
        columns: str | Expr | Sequence[str | Expr],
        *more_columns: str | Expr,
    ) -> DataFrame:
        """
        Explode the dataframe to long format by exploding the given columns.

        Parameters
        ----------
        columns
            Column names, expressions, or a selector defining them. The underlying
            columns being exploded must be of List or Utf8 datatype.
        *more_columns
            Additional names of columns to explode, specified as positional arguments.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["a", "a", "b", "c"],
        ...         "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]],
        ...     }
        ... )
        >>> df
        shape: (4, 2)
        ┌─────────┬───────────┐
        │ letters ┆ numbers   │
        │ ---     ┆ ---       │
        │ str     ┆ list[i64] │
        ╞═════════╪═══════════╡
        │ a       ┆ [1]       │
        │ a       ┆ [2, 3]    │
        │ b       ┆ [4, 5]    │
        │ c       ┆ [6, 7, 8] │
        └─────────┴───────────┘
        >>> df.explode("numbers")
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
        return self.lazy().explode(columns, *more_columns).collect(_eager=True)

    def pivot(
        self,
        values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None,
        aggregate_function: PivotAgg | Expr | None = None,
        *,
        maintain_order: bool = True,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> Self:
        """
        Create a spreadsheet-style pivot table as a DataFrame.

        Only available in eager mode. See "Examples" section below for how to do a
        "lazy pivot" if you know the unique column values in advance.

        Parameters
        ----------
        values
            Column values to aggregate. Can be multiple columns if the *columns*
            arguments contains multiple columns as well.
        index
            One or multiple keys to group by.
        columns
            Name of the column(s) whose values will be used as the header of the output
            DataFrame.
        aggregate_function
            Choose from:

            - None: no aggregation takes place, will raise error if multiple values are in group.
            - A predefined aggregate function string, one of
              {'first', 'sum', 'max', 'min', 'mean', 'median', 'last', 'count'}
            - An expression to do the aggregation.

        maintain_order
            Sort the grouped keys so that the output order is predictable.
        sort_columns
            Sort the transposed columns by name. Default is by order of discovery.
        separator
            Used as separator/delimiter in generated column names.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["one", "one", "two", "two", "one", "two"],
        ...         "bar": ["y", "y", "y", "x", "x", "x"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.pivot(values="baz", index="foo", columns="bar", aggregate_function="sum")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ y   ┆ x   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ one ┆ 3   ┆ 5   │
        │ two ┆ 3   ┆ 10  │
        └─────┴─────┴─────┘

        Pivot using selectors to determine the index/values/columns:

        >>> import polars.selectors as cs
        >>> df.pivot(
        ...     values=cs.numeric(),
        ...     index=cs.string(),
        ...     columns=cs.string(),
        ...     aggregate_function="sum",
        ...     sort_columns=True,
        ... ).sort(
        ...     by=cs.string(),
        ... )
        shape: (4, 6)
        ┌─────┬─────┬──────┬──────┬──────┬──────┐
        │ foo ┆ bar ┆ one  ┆ two  ┆ x    ┆ y    │
        │ --- ┆ --- ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
        │ str ┆ str ┆ i64  ┆ i64  ┆ i64  ┆ i64  │
        ╞═════╪═════╪══════╪══════╪══════╪══════╡
        │ one ┆ x   ┆ 5    ┆ null ┆ 5    ┆ null │
        │ one ┆ y   ┆ 3    ┆ null ┆ null ┆ 3    │
        │ two ┆ x   ┆ null ┆ 10   ┆ 10   ┆ null │
        │ two ┆ y   ┆ null ┆ 3    ┆ null ┆ 3    │
        └─────┴─────┴──────┴──────┴──────┴──────┘

        Run an expression as aggregation function

        >>> df = pl.DataFrame(
        ...     {
        ...         "col1": ["a", "a", "a", "b", "b", "b"],
        ...         "col2": ["x", "x", "x", "x", "y", "y"],
        ...         "col3": [6, 7, 3, 2, 5, 7],
        ...     }
        ... )
        >>> df.pivot(
        ...     index="col1",
        ...     columns="col2",
        ...     values="col3",
        ...     aggregate_function=pl.element().tanh().mean(),
        ... )
        shape: (2, 3)
        ┌──────┬──────────┬──────────┐
        │ col1 ┆ x        ┆ y        │
        │ ---  ┆ ---      ┆ ---      │
        │ str  ┆ f64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ a    ┆ 0.998347 ┆ null     │
        │ b    ┆ 0.964028 ┆ 0.999954 │
        └──────┴──────────┴──────────┘

        Note that `pivot` is only available in eager mode. If you know the unique
        column values in advance, you can use :meth:`polars.LazyFrame.groupby` to
        get the same result as above in lazy mode:

        >>> index = pl.col("col1")
        >>> columns = pl.col("col2")
        >>> values = pl.col("col3")
        >>> unique_column_values = ["x", "y"]
        >>> aggregate_function = lambda col: col.tanh().mean()
        >>> (
        ...     df.lazy()
        ...     .group_by(index)
        ...     .agg(
        ...         *[
        ...             aggregate_function(values.filter(columns == value)).alias(value)
        ...             for value in unique_column_values
        ...         ]
        ...     )
        ...     .collect()
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌──────┬──────────┬──────────┐
        │ col1 ┆ x        ┆ y        │
        │ ---  ┆ ---      ┆ ---      │
        │ str  ┆ f64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ a    ┆ 0.998347 ┆ null     │
        │ b    ┆ 0.964028 ┆ 0.999954 │
        └──────┴──────────┴──────────┘

        """  # noqa: W505
        values = _expand_selectors(self, values)
        index = _expand_selectors(self, index)
        columns = _expand_selectors(self, columns)

        if isinstance(aggregate_function, str):
            if aggregate_function == "first":
                aggregate_expr = F.element().first()._pyexpr
            elif aggregate_function == "sum":
                aggregate_expr = F.element().sum()._pyexpr
            elif aggregate_function == "max":
                aggregate_expr = F.element().max()._pyexpr
            elif aggregate_function == "min":
                aggregate_expr = F.element().min()._pyexpr
            elif aggregate_function == "mean":
                aggregate_expr = F.element().mean()._pyexpr
            elif aggregate_function == "median":
                aggregate_expr = F.element().median()._pyexpr
            elif aggregate_function == "last":
                aggregate_expr = F.element().last()._pyexpr
            elif aggregate_function == "count":
                aggregate_expr = F.count()._pyexpr
            else:
                raise ValueError(
                    f"invalid input for `aggregate_function` argument: {aggregate_function!r}"
                )
        elif aggregate_function is None:
            aggregate_expr = None
        else:
            aggregate_expr = aggregate_function._pyexpr

        return self._from_pydf(
            self._df.pivot_expr(
                values,
                index,
                columns,
                maintain_order,
                sort_columns,
                aggregate_expr,
                separator,
            )
        )

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> import polars.selectors as cs
        >>> df.melt(id_vars="a", value_vars=cs.numeric())
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

        return self._from_pydf(
            self._df.melt(id_vars, value_vars, value_name, variable_name)
        )

    def unstack(
        self,
        step: int,
        how: UnstackDirection = "vertical",
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        fill_values: list[Any] | None = None,
    ) -> DataFrame:
        """
        Unstack a long table to a wide form without doing an aggregation.

        This can be much faster than a pivot, because it can skip the grouping phase.

        Warnings
        --------
        This functionality is experimental and may be subject to changes
        without it being considered a breaking change.

        Parameters
        ----------
        step
            Number of rows in the unstacked frame.
        how : { 'vertical', 'horizontal' }
            Direction of the unstack.
        columns
            Column name(s) or selector(s) to include in the operation.
            If set to `None` (default), use all columns.
        fill_values
            Fill values that don't fit the new size with this value.

        Examples
        --------
        >>> from string import ascii_uppercase
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": list(ascii_uppercase[0:8]),
        ...         "y": pl.int_range(1, 9, eager=True),
        ...     }
        ... ).with_columns(
        ...     z=pl.int_ranges(pl.col("y"), pl.col("y") + 2, dtype=pl.UInt8),
        ... )
        >>> df
        shape: (8, 3)
        ┌─────┬─────┬──────────┐
        │ x   ┆ y   ┆ z        │
        │ --- ┆ --- ┆ ---      │
        │ str ┆ i64 ┆ list[u8] │
        ╞═════╪═════╪══════════╡
        │ A   ┆ 1   ┆ [1, 2]   │
        │ B   ┆ 2   ┆ [2, 3]   │
        │ C   ┆ 3   ┆ [3, 4]   │
        │ D   ┆ 4   ┆ [4, 5]   │
        │ E   ┆ 5   ┆ [5, 6]   │
        │ F   ┆ 6   ┆ [6, 7]   │
        │ G   ┆ 7   ┆ [7, 8]   │
        │ H   ┆ 8   ┆ [8, 9]   │
        └─────┴─────┴──────────┘
        >>> df.unstack(step=4, how="vertical")
        shape: (4, 6)
        ┌─────┬─────┬─────┬─────┬──────────┬──────────┐
        │ x_0 ┆ x_1 ┆ y_0 ┆ y_1 ┆ z_0      ┆ z_1      │
        │ --- ┆ --- ┆ --- ┆ --- ┆ ---      ┆ ---      │
        │ str ┆ str ┆ i64 ┆ i64 ┆ list[u8] ┆ list[u8] │
        ╞═════╪═════╪═════╪═════╪══════════╪══════════╡
        │ A   ┆ E   ┆ 1   ┆ 5   ┆ [1, 2]   ┆ [5, 6]   │
        │ B   ┆ F   ┆ 2   ┆ 6   ┆ [2, 3]   ┆ [6, 7]   │
        │ C   ┆ G   ┆ 3   ┆ 7   ┆ [3, 4]   ┆ [7, 8]   │
        │ D   ┆ H   ┆ 4   ┆ 8   ┆ [4, 5]   ┆ [8, 9]   │
        └─────┴─────┴─────┴─────┴──────────┴──────────┘
        >>> df.unstack(step=2, how="horizontal")
        shape: (4, 6)
        ┌─────┬─────┬─────┬─────┬──────────┬──────────┐
        │ x_0 ┆ x_1 ┆ y_0 ┆ y_1 ┆ z_0      ┆ z_1      │
        │ --- ┆ --- ┆ --- ┆ --- ┆ ---      ┆ ---      │
        │ str ┆ str ┆ i64 ┆ i64 ┆ list[u8] ┆ list[u8] │
        ╞═════╪═════╪═════╪═════╪══════════╪══════════╡
        │ A   ┆ B   ┆ 1   ┆ 2   ┆ [1, 2]   ┆ [2, 3]   │
        │ C   ┆ D   ┆ 3   ┆ 4   ┆ [3, 4]   ┆ [4, 5]   │
        │ E   ┆ F   ┆ 5   ┆ 6   ┆ [5, 6]   ┆ [6, 7]   │
        │ G   ┆ H   ┆ 7   ┆ 8   ┆ [7, 8]   ┆ [8, 9]   │
        └─────┴─────┴─────┴─────┴──────────┴──────────┘
        >>> import polars.selectors as cs
        >>> df.unstack(step=5, columns=cs.numeric(), fill_values=0)
        shape: (5, 2)
        ┌─────┬─────┐
        │ y_0 ┆ y_1 │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 6   │
        │ 2   ┆ 7   │
        │ 3   ┆ 8   │
        │ 4   ┆ 0   │
        │ 5   ┆ 0   │
        └─────┴─────┘

        """
        import math

        df = self.select(columns) if columns is not None else self

        height = df.height
        if how == "vertical":
            n_rows = step
            n_cols = math.ceil(height / n_rows)
        else:
            n_cols = step
            n_rows = math.ceil(height / n_cols)

        n_fill = n_cols * n_rows - height

        if n_fill:
            if not isinstance(fill_values, list):
                fill_values = [fill_values for _ in range(df.width)]

            df = df.select(
                [
                    s.extend_constant(next_fill, n_fill)
                    for s, next_fill in zip(df, fill_values)
                ]
            )

        if how == "horizontal":
            df = (
                df.with_columns(
                    (F.int_range(0, n_cols * n_rows, eager=True) % n_cols).alias(
                        "__sort_order"
                    ),
                )
                .sort("__sort_order")
                .drop("__sort_order")
            )

        zfill_val = math.floor(math.log10(n_cols)) + 1
        slices = [
            s.slice(slice_nbr * n_rows, n_rows).alias(
                s.name + "_" + str(slice_nbr).zfill(zfill_val)
            )
            for s in df
            for slice_nbr in range(n_cols)
        ]

        return DataFrame(slices)

    @overload
    def partition_by(
        self,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *more_by: str,
        maintain_order: bool = ...,
        include_key: bool = ...,
        as_dict: Literal[False] = ...,
    ) -> list[Self]:
        ...

    @overload
    def partition_by(
        self,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *more_by: str,
        maintain_order: bool = ...,
        include_key: bool = ...,
        as_dict: Literal[True],
    ) -> dict[Any, Self]:
        ...

    def partition_by(
        self,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
        maintain_order: bool = True,
        include_key: bool = True,
        as_dict: bool = False,
    ) -> list[Self] | dict[Any, Self]:
        """
        Group by the given columns and return the groups as separate dataframes.

        Parameters
        ----------
        by
            Column name(s) or selector(s) to group by.
        *more_by
            Additional names of columns to group by, specified as positional arguments.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default partition by operation.
        include_key
            Include the columns used to partition the DataFrame in the output.
        as_dict
            Return a dictionary instead of a list. The dictionary keys are the distinct
            group values that identify that group.

        Examples
        --------
        Pass a single column name to partition by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.partition_by("a")  # doctest: +IGNORE_RESULT
        [shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘]

        Partition by multiple columns by either passing a list of column names, or by
        specifying each column name as a positional argument.

        >>> df.partition_by("a", "b")  # doctest: +IGNORE_RESULT
        [shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘]

        Return the partitions as a dictionary by specifying `as_dict=True`.

        >>> import polars.selectors as cs
        >>> df.partition_by(cs.string(), as_dict=True)  # doctest: +IGNORE_RESULT
        {'a': shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        'b': shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        'c': shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘}

        """
        by = _expand_selectors(self, by, *more_by)
        partitions = [
            self._from_pydf(_df)
            for _df in self._df.partition_by(by, maintain_order, include_key)
        ]

        if as_dict:
            df = self._from_pydf(self._df)
            if include_key:
                if len(by) == 1:
                    names = [p[by[0]][0] for p in partitions]
                else:
                    names = [p.select(by).row(0) for p in partitions]
            else:
                if len(by) == 1:
                    names = df[by[0]].unique(maintain_order=True).to_list()
                else:
                    names = df.select(by).unique(maintain_order=True).rows()

            return dict(zip(names, partitions))

        return partitions

    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> DataFrame:
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

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> df.shift()
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

        >>> df.shift(-2)
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

        >>> df.shift(-2, fill_value=100)
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
        return self.lazy().shift(n, fill_value=fill_value).collect(_eager=True)

    def is_duplicated(self) -> Series:
        """
        Get a mask of all duplicated rows in this DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["x", "y", "z", "x"],
        ...     }
        ... )
        >>> df.is_duplicated()
        shape: (4,)
        Series: '' [bool]
        [
                true
                false
                false
                true
        ]

        This mask can be used to visualize the duplicated lines like this:

        >>> df.filter(df.is_duplicated())
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ x   │
        │ 1   ┆ x   │
        └─────┴─────┘
        """
        return wrap_s(self._df.is_duplicated())

    def is_unique(self) -> Series:
        """
        Get a mask of all unique rows in this DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["x", "y", "z", "x"],
        ...     }
        ... )
        >>> df.is_unique()
        shape: (4,)
        Series: '' [bool]
        [
                false
                true
                true
                false
        ]

        This mask can be used to visualize the unique lines like this:

        >>> df.filter(df.is_unique())
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 2   ┆ y   │
        │ 3   ┆ z   │
        └─────┴─────┘
        """
        return wrap_s(self._df.is_unique())

    def lazy(self) -> LazyFrame:
        """
        Start a lazy query from this point. This returns a `LazyFrame` object.

        Operations on a `LazyFrame` are not executed until this is requested by either
        calling:

        * :meth:`.fetch() <polars.LazyFrame.fetch>`
            (run on a small number of rows)
        * :meth:`.collect() <polars.LazyFrame.collect>`
            (run on all data)
        * :meth:`.describe_plan() <polars.LazyFrame.describe_plan>`
            (print unoptimized query plan)
        * :meth:`.describe_optimized_plan() <polars.LazyFrame.describe_optimized_plan>`
            (print optimized query plan)
        * :meth:`.show_graph() <polars.LazyFrame.show_graph>`
            (show (un)optimized query plan as graphviz graph)

        Lazy operations are advised because they allow for query optimization and more
        parallelization.

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
        <LazyFrame [3 cols, {"a": Int64 … "c": Boolean}] at ...>

        """
        return wrap_ldf(self._df.lazy())

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        """
        Select columns from this DataFrame.

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

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.select("foo")
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

        >>> df.select(["foo", "bar"])
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

        >>> df.select(pl.col("foo"), pl.col("bar") + 1)
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

        >>> df.select(threshold=pl.when(pl.col("foo") > 2).then(10).otherwise(0))
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
        ...     df.select(
        ...         is_odd=(pl.col(pl.INTEGER_DTYPES) % 2).name.suffix("_is_odd"),
        ...     )
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
        return self.lazy().select(*exprs, **named_exprs).collect(_eager=True)

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
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
        return self.lazy().select_seq(*exprs, **named_exprs).collect(_eager=True)

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrame:
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
        DataFrame
            A new DataFrame with the columns added.

        Notes
        -----
        Creating a new DataFrame using this method does not create a new copy of
        existing data.

        Examples
        --------
        Pass an expression to add it as a new column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.with_columns((pl.col("a") ** 2).alias("a^2"))
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

        >>> df.with_columns(pl.col("a").cast(pl.Float64))
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

        >>> df.with_columns(
        ...     [
        ...         (pl.col("a") ** 2).alias("a^2"),
        ...         (pl.col("b") / 2).alias("b/2"),
        ...         (pl.col("c").not_()).alias("not c"),
        ...     ]
        ... )
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

        >>> df.with_columns(
        ...     (pl.col("a") ** 2).alias("a^2"),
        ...     (pl.col("b") / 2).alias("b/2"),
        ...     (pl.col("c").not_()).alias("not c"),
        ... )
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

        >>> df.with_columns(
        ...     ab=pl.col("a") * pl.col("b"),
        ...     not_c=pl.col("c").not_(),
        ... )
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
        ...     df.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().name.suffix("_diff"),
        ...     )
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
        return self.lazy().with_columns(*exprs, **named_exprs).collect(_eager=True)

    def with_columns_seq(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DataFrame:
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
        return self.lazy().with_columns_seq(*exprs, **named_exprs).collect(_eager=True)

    @overload
    def n_chunks(self, strategy: Literal["first"] = ...) -> int:
        ...

    @overload
    def n_chunks(self, strategy: Literal["all"]) -> list[int]:
        ...

    def n_chunks(self, strategy: str = "first") -> int | list[int]:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame.

        Parameters
        ----------
        strategy : {'first', 'all'}
            Return the number of chunks of the 'first' column,
            or 'all' columns in this DataFrame.


        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.n_chunks()
        1
        >>> df.n_chunks(strategy="all")
        [1, 1, 1]

        """
        if strategy == "first":
            return self._df.n_chunks()
        elif strategy == "all":
            return [s.n_chunks() for s in self.__iter__()]
        else:
            raise ValueError(
                f"unexpected input for `strategy`: {strategy!r}"
                f"\n\nChoose one of {{'first', 'all'}}"
            )

    @overload
    def max(self, axis: Literal[0] = ...) -> Self:
        ...

    @overload
    def max(self, axis: Literal[1]) -> Series:
        ...

    @overload
    def max(self, axis: int = 0) -> Self | Series:
        ...

    def max(self, axis: int | None = None) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their maximum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`max_horizontal`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.max()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        """
        if axis is not None:
            issue_deprecation_warning(
                "The `axis` parameter for `DataFrame.max` is deprecated."
                " Use `DataFrame.max_horizontal()` to perform horizontal aggregation.",
                version="0.19.14",
            )
        else:
            axis = 0

        if axis == 0:
            return self._from_pydf(self._df.max())
        if axis == 1:
            return wrap_s(self._df.max_horizontal())
        raise ValueError("axis should be 0 or 1")

    def max_horizontal(self) -> Series:
        """
        Get the maximum value horizontally across columns.

        Returns
        -------
        Series
            A Series named `"max"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.max_horizontal()
        shape: (3,)
        Series: 'max' [f64]
        [
                4.0
                5.0
                6.0
        ]
        """
        return self.select(F.max_horizontal(F.all())).to_series()

    @overload
    def min(self, axis: Literal[0] | None = ...) -> Self:
        ...

    @overload
    def min(self, axis: Literal[1]) -> Series:
        ...

    @overload
    def min(self, axis: int) -> Self | Series:
        ...

    def min(self, axis: int | None = None) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their minimum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`min_horizontal`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.min()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        """
        if axis is not None:
            issue_deprecation_warning(
                "The `axis` parameter for `DataFrame.min` is deprecated."
                " Use `DataFrame.min_horizontal()` to perform horizontal aggregation.",
                version="0.19.14",
            )
        else:
            axis = 0

        if axis == 0:
            return self._from_pydf(self._df.min())
        if axis == 1:
            return wrap_s(self._df.min_horizontal())
        raise ValueError("axis should be 0 or 1")

    def min_horizontal(self) -> Series:
        """
        Get the minimum value horizontally across columns.

        Returns
        -------
        Series
            A Series named `"min"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.min_horizontal()
        shape: (3,)
        Series: 'min' [f64]
        [
                1.0
                2.0
                3.0
        ]
        """
        return self.select(F.min_horizontal(F.all())).to_series()

    @overload
    def sum(
        self,
        *,
        axis: Literal[0] = ...,
        null_strategy: NullStrategy = "ignore",
    ) -> Self:
        ...

    @overload
    def sum(
        self,
        *,
        axis: Literal[1],
        null_strategy: NullStrategy = "ignore",
    ) -> Series:
        ...

    @overload
    def sum(
        self,
        *,
        axis: int,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | Series:
        ...

    def sum(
        self,
        *,
        axis: int | None = None,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their sum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`sum_horizontal`.
        null_strategy : {'ignore', 'propagate'}
            This argument is only used if `axis == 1`.

            .. deprecated:: 0.19.14
                This argument will be removed in a future version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sum()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 6   ┆ 21  ┆ null │
        └─────┴─────┴──────┘
        """
        if axis is not None:
            issue_deprecation_warning(
                "The `axis` parameter for `DataFrame.sum` is deprecated."
                " Use `DataFrame.sum_horizontal()` to perform horizontal aggregation.",
                version="0.19.14",
            )
        else:
            axis = 0

        if axis == 0:
            return self._from_pydf(self._df.sum())
        if axis == 1:
            if null_strategy == "ignore":
                ignore_nulls = True
            elif null_strategy == "propagate":
                ignore_nulls = False
            else:
                raise ValueError(
                    f"`null_strategy` must be one of {{'ignore', 'propagate'}}, got {null_strategy}"
                )
            return self.sum_horizontal(ignore_nulls=ignore_nulls)
        raise ValueError("axis should be 0 or 1")

    def sum_horizontal(self, *, ignore_nulls: bool = True) -> Series:
        """
        Sum all values horizontally across columns.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).
            If set to `False`, any null value in the input will lead to a null output.

        Returns
        -------
        Series
            A Series named `"sum"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.sum_horizontal()
        shape: (3,)
        Series: 'sum' [f64]
        [
                5.0
                7.0
                9.0
        ]
        """
        return wrap_s(self._df.sum_horizontal(ignore_nulls)).alias("sum")

    @overload
    def mean(
        self,
        *,
        axis: Literal[0] = ...,
        null_strategy: NullStrategy = "ignore",
    ) -> Self:
        ...

    @overload
    def mean(
        self,
        *,
        axis: Literal[1],
        null_strategy: NullStrategy = "ignore",
    ) -> Series:
        ...

    @overload
    def mean(
        self,
        *,
        axis: int,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | Series:
        ...

    def mean(
        self,
        *,
        axis: int | None = None,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their mean value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`mean_horizontal`.
        null_strategy : {'ignore', 'propagate'}
            This argument is only used if `axis == 1`.

            .. deprecated:: 0.19.14
                This argument will be removed in a future version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...         "spam": [True, False, None],
        ...     }
        ... )
        >>> df.mean()
        shape: (1, 4)
        ┌─────┬─────┬──────┬──────┐
        │ foo ┆ bar ┆ ham  ┆ spam │
        │ --- ┆ --- ┆ ---  ┆ ---  │
        │ f64 ┆ f64 ┆ str  ┆ f64  │
        ╞═════╪═════╪══════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null ┆ 0.5  │
        └─────┴─────┴──────┴──────┘
        """
        if axis is not None:
            issue_deprecation_warning(
                "The `axis` parameter for `DataFrame.mean` is deprecated."
                " Use `DataFrame.mean_horizontal()` to perform horizontal aggregation.",
                version="0.19.14",
            )
        else:
            axis = 0

        if axis == 0:
            return self._from_pydf(self._df.mean())
        if axis == 1:
            if null_strategy == "ignore":
                ignore_nulls = True
            elif null_strategy == "propagate":
                ignore_nulls = False
            else:
                raise ValueError(
                    f"`null_strategy` must be one of {{'ignore', 'propagate'}}, got {null_strategy}"
                )
            return self.mean_horizontal(ignore_nulls=ignore_nulls)
        raise ValueError("axis should be 0 or 1")

    def mean_horizontal(self, *, ignore_nulls: bool = True) -> Series:
        """
        Take the mean of all values horizontally across columns.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).
            If set to `False`, any null value in the input will lead to a null output.

        Returns
        -------
        Series
            A Series named `"mean"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.mean_horizontal()
        shape: (3,)
        Series: 'mean' [f64]
        [
                2.5
                3.5
                4.5
        ]
        """
        return wrap_s(self._df.mean_horizontal(ignore_nulls)).alias("mean")

    def std(self, ddof: int = 1) -> Self:
        """
        Aggregate the columns of this DataFrame to their standard deviation value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.std()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1.0 ┆ 1.0 ┆ null │
        └─────┴─────┴──────┘
        >>> df.std(ddof=0)
        shape: (1, 3)
        ┌──────────┬──────────┬──────┐
        │ foo      ┆ bar      ┆ ham  │
        │ ---      ┆ ---      ┆ ---  │
        │ f64      ┆ f64      ┆ str  │
        ╞══════════╪══════════╪══════╡
        │ 0.816497 ┆ 0.816497 ┆ null │
        └──────────┴──────────┴──────┘

        """
        return self._from_pydf(self._df.std(ddof))

    def var(self, ddof: int = 1) -> Self:
        """
        Aggregate the columns of this DataFrame to their variance value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.var()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1.0 ┆ 1.0 ┆ null │
        └─────┴─────┴──────┘
        >>> df.var(ddof=0)
        shape: (1, 3)
        ┌──────────┬──────────┬──────┐
        │ foo      ┆ bar      ┆ ham  │
        │ ---      ┆ ---      ┆ ---  │
        │ f64      ┆ f64      ┆ str  │
        ╞══════════╪══════════╪══════╡
        │ 0.666667 ┆ 0.666667 ┆ null │
        └──────────┴──────────┴──────┘

        """
        return self._from_pydf(self._df.var(ddof))

    def median(self) -> Self:
        """
        Aggregate the columns of this DataFrame to their median value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.median()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null │
        └─────┴─────┴──────┘

        """
        return self._from_pydf(self._df.median())

    def product(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their product values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [0.5, 4, 10],
        ...         "c": [True, True, False],
        ...     }
        ... )

        >>> df.product()
        shape: (1, 3)
        ┌─────┬──────┬─────┐
        │ a   ┆ b    ┆ c   │
        │ --- ┆ ---  ┆ --- │
        │ i64 ┆ f64  ┆ i64 │
        ╞═════╪══════╪═════╡
        │ 6   ┆ 20.0 ┆ 0   │
        └─────┴──────┴─────┘

        """
        return self.select(F.all().product())

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> Self:
        """
        Aggregate the columns of this DataFrame to their quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.quantile(0.5, "nearest")
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null │
        └─────┴─────┴──────┘

        """
        return self._from_pydf(self._df.quantile(quantile, interpolation))

    def to_dummies(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ) -> Self:
        """
        Convert categorical variables into dummy/indicator variables.

        Parameters
        ----------
        columns
            Column name(s) or selector(s) that should be converted to dummy
            variables. If set to `None` (default), convert all columns.
        separator
            Separator/delimiter used when generating column names.
        drop_first
            Remove the first category from the variables being encoded.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2],
        ...         "bar": [3, 4],
        ...         "ham": ["a", "b"],
        ...     }
        ... )
        >>> df.to_dummies()
        shape: (2, 6)
        ┌───────┬───────┬───────┬───────┬───────┬───────┐
        │ foo_1 ┆ foo_2 ┆ bar_3 ┆ bar_4 ┆ ham_a ┆ ham_b │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
        │ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     │
        │ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     │
        └───────┴───────┴───────┴───────┴───────┴───────┘

        >>> df.to_dummies(drop_first=True)
        shape: (2, 3)
        ┌───────┬───────┬───────┐
        │ foo_2 ┆ bar_4 ┆ ham_b │
        │ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╡
        │ 0     ┆ 0     ┆ 0     │
        │ 1     ┆ 1     ┆ 1     │
        └───────┴───────┴───────┘

        >>> import polars.selectors as cs
        >>> df.to_dummies(cs.integer(), separator=":")
        shape: (2, 5)
        ┌───────┬───────┬───────┬───────┬─────┐
        │ foo:1 ┆ foo:2 ┆ bar:3 ┆ bar:4 ┆ ham │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ --- │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ str │
        ╞═══════╪═══════╪═══════╪═══════╪═════╡
        │ 1     ┆ 0     ┆ 1     ┆ 0     ┆ a   │
        │ 0     ┆ 1     ┆ 0     ┆ 1     ┆ b   │
        └───────┴───────┴───────┴───────┴─────┘

        >>> df.to_dummies(cs.integer(), drop_first=True, separator=":")
        shape: (2, 3)
        ┌───────┬───────┬─────┐
        │ foo:2 ┆ bar:4 ┆ ham │
        │ ---   ┆ ---   ┆ --- │
        │ u8    ┆ u8    ┆ str │
        ╞═══════╪═══════╪═════╡
        │ 0     ┆ 0     ┆ a   │
        │ 1     ┆ 1     ┆ b   │
        └───────┴───────┴─────┘

        """
        if columns is not None:
            columns = _expand_selectors(self, columns)
        return self._from_pydf(self._df.to_dummies(columns, separator, drop_first))

    def unique(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> DataFrame:
        """
        Drop duplicate rows from this dataframe.

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
        DataFrame
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
        ... )
        >>> df.unique(maintain_order=True)
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
        >>> df.unique(subset=["bar", "ham"], maintain_order=True)
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> df.unique(keep="last", maintain_order=True)
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
        return (
            self.lazy()
            .unique(subset=subset, keep=keep, maintain_order=maintain_order)
            .collect(_eager=True)
        )

    def n_unique(self, subset: str | Expr | Sequence[str | Expr] | None = None) -> int:
        """
        Return the number of unique rows, or the number of unique row-subsets.

        Parameters
        ----------
        subset
            One or more columns/expressions that define what to count;
            omit to return the count of unique rows.

        Notes
        -----
        This method operates at the `DataFrame` level; to operate on subsets at the
        expression level you can make use of struct-packing instead, for example:

        >>> expr_unique_subset = pl.struct(["a", "b"]).n_unique()

        If instead you want to count the number of unique values per-column, you can
        also use expression-level syntax to return a new frame containing that result:

        >>> df = pl.DataFrame([[1, 2, 3], [1, 2, 4]], schema=["a", "b", "c"])
        >>> df_nunique = df.select(pl.all().n_unique())

        In aggregate context there is also an equivalent method for returning the
        unique values per-group:

        >>> df_agg_nunique = df.group_by(by=["a"]).n_unique()

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 1.0, 2.0, 3.0, 3.0],
        ...         "c": [True, True, True, False, True, True],
        ...     }
        ... )
        >>> df.n_unique()
        5

        Simple columns subset.

        >>> df.n_unique(subset=["b", "c"])
        4

        Expression subset.

        >>> df.n_unique(
        ...     subset=[
        ...         (pl.col("a") // 2),
        ...         (pl.col("c") | (pl.col("b") >= 2)),
        ...     ],
        ... )
        3

        """
        if isinstance(subset, str):
            expr = F.col(subset)
        elif isinstance(subset, pl.Expr):
            expr = subset
        elif isinstance(subset, Sequence) and len(subset) == 1:
            expr = wrap_expr(parse_as_expression(subset[0]))
        else:
            struct_fields = F.all() if (subset is None) else subset
            expr = F.struct(struct_fields)  # type: ignore[call-overload]

        df = self.lazy().select(expr.n_unique()).collect(_eager=True)
        return 0 if df.is_empty() else df.row(0)[0]

    def approx_n_unique(self) -> DataFrame:
        """
        Approximate count of unique values.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> df.approx_n_unique()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘

        """
        return self.lazy().approx_n_unique().collect(_eager=True)

    @deprecate_renamed_function("approx_n_unique", version="0.18.12")
    def approx_unique(self) -> DataFrame:
        """
        Approximate count of unique values.

        .. deprecated:: 0.18.12
            This method has been renamed to :func:`DataFrame.approx_n_unique`.

        """
        return self.approx_n_unique()

    def rechunk(self) -> Self:
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable
        performance.
        """
        return self._from_pydf(self._df.rechunk())

    def null_count(self) -> Self:
        """
        Create a new DataFrame that shows the null counts per column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3],
        ...         "bar": [6, 7, None],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.null_count()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 0   │
        └─────┴─────┴─────┘

        """
        return self._from_pydf(self._df.null_count())

    def sample(
        self,
        n: int | Series | None = None,
        *,
        fraction: float | Series | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Self:
        """
        Sample from this DataFrame.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `fraction`. Defaults to 1 if
            `fraction` is None.
        fraction
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            If set to True, the order of the sampled rows will be shuffled. If
            set to False (default), the order of the returned rows will be
            neither stable nor fully random.
        seed
            Seed for the random number generator. If set to None (default), a
            random seed is generated for each sample operation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sample(n=2, seed=0)  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        if n is not None and fraction is not None:
            raise ValueError("cannot specify both `n` and `fraction`")

        if seed is None:
            seed = random.randint(0, 10000)

        if n is None and fraction is not None:
            if not isinstance(fraction, pl.Series):
                fraction = pl.Series("frac", [fraction])

            return self._from_pydf(
                self._df.sample_frac(fraction._s, with_replacement, shuffle, seed)
            )

        if n is None:
            n = 1

        if not isinstance(n, pl.Series):
            n = pl.Series("", [n])

        return self._from_pydf(self._df.sample_n(n._s, with_replacement, shuffle, seed))

    def fold(self, operation: Callable[[Series, Series], Series]) -> Series:
        """
        Apply a horizontal reduction on a DataFrame.

        This can be used to effectively determine aggregations on a row level, and can
        be applied to any DataType that can be supercasted (casted to a similar parent
        type).

        An example of the supercast rules when applying an arithmetic operation on two
        DataTypes are for instance:

        - Int8 + Utf8 = Utf8
        - Float32 + Int64 = Float32
        - Float32 + Float64 = Float64

        Examples
        --------
        A horizontal sum operation:

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [2, 1, 3],
        ...         "b": [1, 2, 3],
        ...         "c": [1.0, 2.0, 3.0],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 + s2)
        shape: (3,)
        Series: 'a' [f64]
        [
            4.0
            5.0
            9.0
        ]

        A horizontal minimum operation:

        >>> df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
        >>> df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2))
        shape: (3,)
        Series: 'a' [f64]
        [
            1.0
            1.0
            3.0
        ]

        A horizontal string concatenation:

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["foo", "bar", 2],
        ...         "b": [1, 2, 3],
        ...         "c": [1.0, 2.0, 3.0],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 + s2)
        shape: (3,)
        Series: 'a' [str]
        [
            "foo11.0"
            "bar22.0"
            null
        ]

        A horizontal boolean or, similar to a row-wise .any():

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [False, False, True],
        ...         "b": [False, True, False],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 | s2)
        shape: (3,)
        Series: 'a' [bool]
        [
                false
                true
                true
        ]

        Parameters
        ----------
        operation
            function that takes two `Series` and returns a `Series`.

        """
        acc = self.to_series(0)

        for i in range(1, self.width):
            acc = operation(acc, self.to_series(i))
        return acc

    @overload
    def row(
        self,
        index: int | None = ...,
        *,
        by_predicate: Expr | None = ...,
        named: Literal[False] = ...,
    ) -> tuple[Any, ...]:
        ...

    @overload
    def row(
        self,
        index: int | None = ...,
        *,
        by_predicate: Expr | None = ...,
        named: Literal[True],
    ) -> dict[str, Any]:
        ...

    def row(
        self,
        index: int | None = None,
        *,
        by_predicate: Expr | None = None,
        named: bool = False,
    ) -> tuple[Any, ...] | dict[str, Any]:
        """
        Get the values of a single row, either by index or by predicate.

        Parameters
        ----------
        index
            Row index.
        by_predicate
            Select the row according to a given expression/predicate.
        named
            Return a dictionary instead of a tuple. The dictionary is a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.

        Returns
        -------
        tuple (default) or dictionary of row values

        Notes
        -----
        The `index` and `by_predicate` params are mutually exclusive. Additionally,
        to ensure clarity, the `by_predicate` parameter must be supplied by keyword.

        When using `by_predicate` it is an error condition if anything other than
        one row is returned; more than one row raises `TooManyRowsReturnedError`, and
        zero rows will raise `NoRowsReturnedError` (both inherit from `RowsError`).

        Warnings
        --------
        You should NEVER use this method to iterate over a DataFrame; if you require
        row-iteration you should strongly prefer use of `iter_rows()` instead.

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        rows : Materialise all frame data as a list of rows (potentially expensive).
        item: Return dataframe element as a scalar.

        Examples
        --------
        Specify an index to return the row at the given index as a tuple.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.row(2)
        (3, 8, 'c')

        Specify `named=True` to get a dictionary instead with a mapping of column
        names to row values.

        >>> df.row(2, named=True)
        {'foo': 3, 'bar': 8, 'ham': 'c'}

        Use `by_predicate` to return the row that matches the given predicate.

        >>> df.row(by_predicate=(pl.col("ham") == "b"))
        (2, 7, 'b')

        """
        if index is not None and by_predicate is not None:
            raise ValueError(
                "cannot set both 'index' and 'by_predicate'; mutually exclusive"
            )
        elif isinstance(index, pl.Expr):
            raise TypeError(
                "expressions should be passed to the `by_predicate` parameter"
            )

        if index is not None:
            row = self._df.row_tuple(index)
            if named:
                return dict(zip(self.columns, row))
            else:
                return row

        elif by_predicate is not None:
            if not isinstance(by_predicate, pl.Expr):
                raise TypeError(
                    f"expected `by_predicate` to be an expression, got {type(by_predicate).__name__!r}"
                )
            rows = self.filter(by_predicate).rows()
            n_rows = len(rows)
            if n_rows > 1:
                raise TooManyRowsReturnedError(
                    f"predicate <{by_predicate!s}> returned {n_rows} rows"
                )
            elif n_rows == 0:
                raise NoRowsReturnedError(
                    f"predicate <{by_predicate!s}> returned no rows"
                )

            row = rows[0]
            if named:
                return dict(zip(self.columns, row))
            else:
                return row
        else:
            raise ValueError("one of `index` or `by_predicate` must be set")

    @overload
    def rows(self, *, named: Literal[False] = ...) -> list[tuple[Any, ...]]:
        ...

    @overload
    def rows(self, *, named: Literal[True]) -> list[dict[str, Any]]:
        ...

    def rows(
        self, *, named: bool = False
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """
        Returns all data in the DataFrame as a list of rows of python-native values.

        Parameters
        ----------
        named
            Return dictionaries instead of tuples. The dictionaries are a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Warnings
        --------
        Row-iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods.
        Where possible you should also consider using `iter_rows` instead to avoid
        materialising all the data at once.

        Returns
        -------
        list of tuples (default) or dictionaries of row values

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        rows_by_key : Materialises frame data as a key-indexed dictionary.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": ["a", "b", "b", "a"],
        ...         "y": [1, 2, 3, 4],
        ...         "z": [0, 3, 6, 9],
        ...     }
        ... )
        >>> df.rows()
        [('a', 1, 0), ('b', 2, 3), ('b', 3, 6), ('a', 4, 9)]
        >>> df.rows(named=True)
        [{'x': 'a', 'y': 1, 'z': 0},
         {'x': 'b', 'y': 2, 'z': 3},
         {'x': 'b', 'y': 3, 'z': 6},
         {'x': 'a', 'y': 4, 'z': 9}]

        """
        if named:
            # Load these into the local namespace for a minor performance boost
            dict_, zip_, columns = dict, zip, self.columns
            return [dict_(zip_(columns, row)) for row in self._df.row_tuples()]
        else:
            return self._df.row_tuples()

    def rows_by_key(
        self,
        key: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *,
        named: bool = False,
        include_key: bool = False,
        unique: bool = False,
    ) -> dict[Any, Iterable[Any]]:
        """
        Returns DataFrame data as a keyed dictionary of python-native values.

        Note that this method should not be used in place of native operations, due to
        the high cost of materialising all frame data out into a dictionary; it should
        be used only when you need to move the values out into a Python data structure
        or other object that cannot operate directly with Polars/Arrow.

        Parameters
        ----------
        key
            The column(s) to use as the key for the returned dictionary. If multiple
            columns are specified, the key will be a tuple of those values, otherwise
            it will be a string.
        named
            Return dictionary rows instead of tuples, mapping column name to row value.
        include_key
            Include key values inline with the associated data (by default the key
            values are omitted as a memory/performance optimisation, as they can be
            reoconstructed from the key).
        unique
            Indicate that the key is unique; this will result in a 1:1 mapping from
            key to a single associated row. Note that if the key is *not* actually
            unique the last row with the given key will be returned.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        See Also
        --------
        rows : Materialise all frame data as a list of rows (potentially expensive).
        iter_rows : Row iterator over frame data (does not materialise all rows).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "w": ["a", "b", "b", "a"],
        ...         "x": ["q", "q", "q", "k"],
        ...         "y": [1.0, 2.5, 3.0, 4.5],
        ...         "z": [9, 8, 7, 6],
        ...     }
        ... )

        Group rows by the given key column(s):

        >>> df.rows_by_key(key=["w"])
        defaultdict(<class 'list'>,
            {'a': [('q', 1.0, 9), ('k', 4.5, 6)],
             'b': [('q', 2.5, 8), ('q', 3.0, 7)]})

        Return the same row groupings as dictionaries:

        >>> df.rows_by_key(key=["w"], named=True)
        defaultdict(<class 'list'>,
            {'a': [{'x': 'q', 'y': 1.0, 'z': 9},
                   {'x': 'k', 'y': 4.5, 'z': 6}],
             'b': [{'x': 'q', 'y': 2.5, 'z': 8},
                   {'x': 'q', 'y': 3.0, 'z': 7}]})

        Return row groupings, assuming keys are unique:

        >>> df.rows_by_key(key=["z"], unique=True)
        {9: ('a', 'q', 1.0),
         8: ('b', 'q', 2.5),
         7: ('b', 'q', 3.0),
         6: ('a', 'k', 4.5)}

        Return row groupings as dictionaries, assuming keys are unique:

        >>> df.rows_by_key(key=["z"], named=True, unique=True)
        {9: {'w': 'a', 'x': 'q', 'y': 1.0},
         8: {'w': 'b', 'x': 'q', 'y': 2.5},
         7: {'w': 'b', 'x': 'q', 'y': 3.0},
         6: {'w': 'a', 'x': 'k', 'y': 4.5}}

        Return dictionary rows grouped by a compound key, including key values:

        >>> df.rows_by_key(key=["w", "x"], named=True, include_key=True)
        defaultdict(<class 'list'>,
            {('a', 'q'): [{'w': 'a', 'x': 'q', 'y': 1.0, 'z': 9}],
             ('b', 'q'): [{'w': 'b', 'x': 'q', 'y': 2.5, 'z': 8},
                          {'w': 'b', 'x': 'q', 'y': 3.0, 'z': 7}],
             ('a', 'k'): [{'w': 'a', 'x': 'k', 'y': 4.5, 'z': 6}]})

        """
        from polars.selectors import expand_selector, is_selector

        if is_selector(key):
            key_tuple = expand_selector(target=self, selector=key)
        elif not isinstance(key, str):
            key_tuple = tuple(key)  # type: ignore[arg-type]
        else:
            key_tuple = (key,)

        # establish index or name-based getters for the key and data values
        data_cols = [k for k in self.schema if k not in key_tuple]
        if named:
            get_data = itemgetter(*data_cols)
            get_key = itemgetter(*key_tuple)
        else:
            data_idxs, index_idxs = [], []
            for idx, c in enumerate(self.columns):
                if c in key_tuple:
                    index_idxs.append(idx)
                else:
                    data_idxs.append(idx)
            if not index_idxs:
                raise ValueError(f"no columns found for key: {key_tuple!r}")
            get_data = itemgetter(*data_idxs)  # type: ignore[assignment]
            get_key = itemgetter(*index_idxs)  # type: ignore[assignment]

        # if unique, we expect to write just one entry per key; otherwise, we're
        # returning a list of rows for each key, so append into a defaultdict.
        rows: dict[Any, Any] = {} if unique else defaultdict(list)

        # return named values (key -> dict | list of dicts), eg:
        # "{(key,): [{col:val, col:val, ...}],
        #   (key,): [{col:val, col:val, ...}],}"
        if named:
            if unique and include_key:
                rows = {get_key(row): row for row in self.iter_rows(named=True)}
            else:
                for d in self.iter_rows(named=True):
                    k = get_key(d)
                    if not include_key:
                        for ix in key_tuple:
                            del d[ix]  # type: ignore[arg-type]
                    if unique:
                        rows[k] = d
                    else:
                        rows[k].append(d)

        # return values (key -> tuple | list of tuples), eg:
        # "{(key,): [(val, val, ...)],
        #   (key,): [(val, val, ...)], ...}"
        elif unique:
            rows = (
                {get_key(row): row for row in self.iter_rows()}
                if include_key
                else {get_key(row): get_data(row) for row in self.iter_rows()}
            )
        elif include_key:
            for row in self.iter_rows(named=False):
                rows[get_key(row)].append(row)
        else:
            for row in self.iter_rows(named=False):
                rows[get_key(row)].append(get_data(row))

        return rows

    @overload
    def iter_rows(
        self, *, named: Literal[False] = ..., buffer_size: int = ...
    ) -> Iterator[tuple[Any, ...]]:
        ...

    @overload
    def iter_rows(
        self, *, named: Literal[True], buffer_size: int = ...
    ) -> Iterator[dict[str, Any]]:
        ...

    def iter_rows(
        self, *, named: bool = False, buffer_size: int = 500
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        """
        Returns an iterator over the DataFrame of rows of python-native values.

        Parameters
        ----------
        named
            Return dictionaries instead of tuples. The dictionaries are a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.
        buffer_size
            Determines the number of rows that are buffered internally while iterating
            over the data; you should only modify this in very specific cases where the
            default value is determined not to be a good fit to your access pattern, as
            the speedup from using the buffer is significant (~2-4x). Setting this
            value to zero disables row buffering (not recommended).

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Warnings
        --------
        Row iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods
        that deals with columnar data.

        Returns
        -------
        iterator of tuples (default) or dictionaries (if named) of python row values

        See Also
        --------
        rows : Materialises all frame data as a list of rows (potentially expensive).
        rows_by_key : Materialises frame data as a key-indexed dictionary.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> [row[0] for row in df.iter_rows()]
        [1, 3, 5]
        >>> [row["b"] for row in df.iter_rows(named=True)]
        [2, 4, 6]

        """
        # load into the local namespace for a (minor) performance boost in the hot loops
        columns, get_row, dict_, zip_ = self.columns, self.row, dict, zip
        has_object = Object in self.dtypes

        # note: buffering rows results in a 2-4x speedup over individual calls
        # to ".row(i)", so it should only be disabled in extremely specific cases.
        if buffer_size and not has_object:
            for offset in range(0, self.height, buffer_size):
                zerocopy_slice = self.slice(offset, buffer_size)
                if named and can_create_dicts_with_pyarrow(self.dtypes):
                    yield from zerocopy_slice.to_arrow().to_pylist()
                else:
                    rows_chunk = zerocopy_slice.rows(named=False)
                    if named:
                        for row in rows_chunk:
                            yield dict_(zip_(columns, row))
                    else:
                        yield from rows_chunk
        elif named:
            for i in range(self.height):
                yield dict_(zip_(columns, get_row(i)))
        else:
            for i in range(self.height):
                yield get_row(i)

    def iter_columns(self) -> Iterator[Series]:
        """
        Returns an iterator over the DataFrame's columns.

        Notes
        -----
        Consider whether you can use :func:`all` instead.
        If you can, it will be more efficient.

        Returns
        -------
        Iterator of Series.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> [s.name for s in df.iter_columns()]
        ['a', 'b']

        If you're using this to modify a dataframe's columns, e.g.

        >>> # Do NOT do this
        >>> pl.DataFrame(column * 2 for column in df.iter_columns())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 4   │
        │ 6   ┆ 8   │
        │ 10  ┆ 12  │
        └─────┴─────┘

        then consider whether you can use :func:`all` instead:

        >>> df.select(pl.all() * 2)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 4   │
        │ 6   ┆ 8   │
        │ 10  ┆ 12  │
        └─────┴─────┘

        """
        return (wrap_s(s) for s in self._df.get_columns())

    def iter_slices(self, n_rows: int = 10_000) -> Iterator[DataFrame]:
        r"""
        Returns a non-copying iterator of slices over the underlying DataFrame.

        Parameters
        ----------
        n_rows
            Determines the number of rows contained in each DataFrame slice.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     data={
        ...         "a": range(17_500),
        ...         "b": date(2023, 1, 1),
        ...         "c": "klmnoopqrstuvwxyz",
        ...     },
        ...     schema_overrides={"a": pl.Int32},
        ... )
        >>> for idx, frame in enumerate(df.iter_slices()):
        ...     print(f"{type(frame).__name__}:[{idx}]:{len(frame)}")
        ...
        DataFrame:[0]:10000
        DataFrame:[1]:7500

        Using `iter_slices` is an efficient way to chunk-iterate over DataFrames and
        any supported frame export/conversion types; for example, as RecordBatches:

        >>> for frame in df.iter_slices(n_rows=15_000):
        ...     record_batch = frame.to_arrow().to_batches()[0]
        ...     print(f"{record_batch.schema}\n<< {len(record_batch)}")
        ...
        a: int32
        b: date32[day]
        c: large_string
        << 15000
        a: int32
        b: date32[day]
        c: large_string
        << 2500

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        partition_by : Split into multiple DataFrames, partitioned by groups.

        """
        for offset in range(0, self.height, n_rows):
            yield self.slice(offset, n_rows)

    def shrink_to_fit(self, *, in_place: bool = False) -> Self:
        """
        Shrink DataFrame memory usage.

        Shrinks to fit the exact capacity needed to hold the data.

        """
        if in_place:
            self._df.shrink_to_fit()
            return self
        else:
            df = self.clone()
            df._df.shrink_to_fit()
            return df

    def gather_every(self, n: int) -> DataFrame:
        """
        Take every nth row in the DataFrame and return as a new DataFrame.

        Parameters
        ----------
        n
            Gather every *n*-th row.

        Examples
        --------
        >>> s = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        >>> s.gather_every(2)
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

    def hash_rows(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> Series:
        """
        Hash and combine the rows in this DataFrame.

        The hash value is of type `UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to 0.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

        Notes
        -----
        This implementation of :func:`hash_rows` does not guarantee stable results
        across different Polars versions. Its stability is only guaranteed within a
        single version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3, 4],
        ...         "ham": ["a", "b", None, "d"],
        ...     }
        ... )
        >>> df.hash_rows(seed=42)  # doctest: +IGNORE_RESULT
        shape: (4,)
        Series: '' [u64]
        [
            10783150408545073287
            1438741209321515184
            10047419486152048166
            2047317070637311557
        ]

        """
        k0 = seed
        k1 = seed_1 if seed_1 is not None else seed
        k2 = seed_2 if seed_2 is not None else seed
        k3 = seed_3 if seed_3 is not None else seed
        return wrap_s(self._df.hash_rows(k0, k1, k2, k3))

    def interpolate(self) -> DataFrame:
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
        ... )
        >>> df.interpolate()
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

    def is_empty(self) -> bool:
        """
        Check if the dataframe is empty.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.is_empty()
        False
        >>> df.filter(pl.col("foo") > 99).is_empty()
        True

        """
        return self.height == 0

    def to_struct(self, name: str) -> Series:
        """
        Convert a `DataFrame` to a `Series` of type `Struct`.

        Parameters
        ----------
        name
            Name for the struct Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": ["one", "two", "three", "four", "five"],
        ...     }
        ... )
        >>> df.to_struct("nums")
        shape: (5,)
        Series: 'nums' [struct[2]]
        [
            {1,"one"}
            {2,"two"}
            {3,"three"}
            {4,"four"}
            {5,"five"}
        ]

        """
        return wrap_s(self._df.to_struct(name))

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        """
        Decompose struct columns into separate columns for each of their fields.

        The new columns will be inserted into the dataframe at the location of the
        struct column.

        Parameters
        ----------
        columns
            Name of the struct column(s) that should be unnested.
        *more_columns
            Additional columns to unnest, specified as positional arguments.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "before": ["foo", "bar"],
        ...         "t_a": [1, 2],
        ...         "t_b": ["a", "b"],
        ...         "t_c": [True, None],
        ...         "t_d": [[1, 2], [3]],
        ...         "after": ["baz", "womp"],
        ...     }
        ... ).select("before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after")
        >>> df
        shape: (2, 3)
        ┌────────┬─────────────────────┬───────┐
        │ before ┆ t_struct            ┆ after │
        │ ---    ┆ ---                 ┆ ---   │
        │ str    ┆ struct[4]           ┆ str   │
        ╞════════╪═════════════════════╪═══════╡
        │ foo    ┆ {1,"a",true,[1, 2]} ┆ baz   │
        │ bar    ┆ {2,"b",null,[3]}    ┆ womp  │
        └────────┴─────────────────────┴───────┘
        >>> df.unnest("t_struct")
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
        return self._from_pydf(self._df.unnest(columns))

    def corr(self, **kwargs: Any) -> DataFrame:
        """
        Return pairwise Pearson product-moment correlation coefficients between columns.

        See numpy `corrcoef` for more information:
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html

        Notes
        -----
        This functionality requires numpy to be installed.

        Parameters
        ----------
        **kwargs
            Keyword arguments are passed to numpy `corrcoef`.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [3, 2, 1], "ham": [7, 8, 9]})
        >>> df.corr()
        shape: (3, 3)
        ┌──────┬──────┬──────┐
        │ foo  ┆ bar  ┆ ham  │
        │ ---  ┆ ---  ┆ ---  │
        │ f64  ┆ f64  ┆ f64  │
        ╞══════╪══════╪══════╡
        │ 1.0  ┆ -1.0 ┆ 1.0  │
        │ -1.0 ┆ 1.0  ┆ -1.0 │
        │ 1.0  ┆ -1.0 ┆ 1.0  │
        └──────┴──────┴──────┘

        """
        return DataFrame(np.corrcoef(self.to_numpy().T, **kwargs), schema=self.columns)

    def merge_sorted(self, other: DataFrame, key: str) -> DataFrame:
        """
        Take two sorted DataFrames and merge them by the sorted key.

        The output of this operation will also be sorted.
        It is the callers responsibility that the frames are sorted
        by that key otherwise the output will not make sense.

        The schemas of both DataFrames must be equal.

        Parameters
        ----------
        other
            Other DataFrame that must be merged
        key
            Key that is sorted.

        Examples
        --------
        >>> df0 = pl.DataFrame(
        ...     {"name": ["steve", "elise", "bob"], "age": [42, 44, 18]}
        ... ).sort("age")
        >>> df0
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
        >>> df1 = pl.DataFrame(
        ...     {"name": ["anna", "megan", "steve", "thomas"], "age": [21, 33, 42, 20]}
        ... ).sort("age")
        >>> df1
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
        >>> df0.merge_sorted(df1, key="age")
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
        return self.lazy().merge_sorted(other.lazy(), key).collect(_eager=True)

    def set_sorted(
        self,
        column: str | Iterable[str],
        *more_columns: str,
        descending: bool = False,
    ) -> DataFrame:
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
        return (
            self.lazy()
            .set_sorted(column, *more_columns, descending=descending)
            .collect(_eager=True)
        )

    def update(
        self,
        other: DataFrame,
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "outer"] = "left",
        include_nulls: bool | None = False,
    ) -> DataFrame:
        """
        Update the values in this `DataFrame` with the values in `other`.

        By default, null values in the right dataframe are ignored. Use
        `ignore_nulls=False` to overwrite values in this frame with null values in other
        frame.

        Notes
        -----
        This is syntactic sugar for a left/inner join, with an optional coalesce when
        `include_nulls = False`.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Parameters
        ----------
        other
            DataFrame that will be used to update the values
        on
            Column names that will be joined on.
            If none given the row count is used.
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
            If True, null values from the right dataframe will be used to update the
            left dataframe.

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
        ...         "B": [-66, None, -99],
        ...         "C": [5, 3, 1],
        ...     }
        ... )

        Update `df` values with the non-null values in `new_df`, by row index:

        >>> df.update(new_df)
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

        >>> df.update(new_df, how="inner")
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

        >>> df.update(new_df, left_on=["A"], right_on=["C"], how="outer")
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

        >>> df.update(
        ...     new_df, left_on="A", right_on="C", how="outer", include_nulls=True
        ... )
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
        return (
            self.lazy()
            .update(other.lazy(), on, left_on, right_on, how, include_nulls)
            .collect(_eager=True)
        )

    @deprecate_renamed_function("group_by", version="0.19.0")
    def groupby(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        maintain_order: bool = False,
    ) -> GroupBy:
        """
        Start a group by operation.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.group_by`.

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

            .. note::
                Within each group, the order of rows is always preserved, regardless
                of this argument.

        Returns
        -------
        GroupBy
            Object which can be used to perform aggregations.

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
    ) -> RollingGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.rolling`.

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
    ) -> RollingGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.9
            This method has been renamed to :func:`DataFrame.rolling`.

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
    ) -> DynamicGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.group_by_dynamic`.

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
        closed : {'left', 'right', 'both', 'none'}
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
        DynamicGroupBy
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

    @deprecate_renamed_function("map_rows", version="0.19.0")
    def apply(
        self,
        function: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        inference_size: int = 256,
    ) -> DataFrame:
        """
        Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.map_rows`.

        Parameters
        ----------
        function
            Custom function or lambda.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema

        """
        return self.map_rows(function, return_dtype, inference_size=inference_size)

    @deprecate_function("Use `shift` instead.", version="0.19.12")
    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift_and_fill(
        self,
        fill_value: int | str | float,
        *,
        n: int = 1,
    ) -> DataFrame:
        """
        Shift values by the given number of places and fill the resulting null values.

        .. deprecated:: 0.19.12
            Use :func:`shift` instead.

        Parameters
        ----------
        fill_value
            fill None values with this value.
        n
            Number of places to shift (may be negative).

        """
        return self.shift(n, fill_value=fill_value)

    @deprecate_renamed_function("gather_every", version="0.19.12")
    def take_every(self, n: int) -> DataFrame:
        """
        Take every nth row in the DataFrame and return as a new DataFrame.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`gather_every`.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        """
        return self.gather_every(n)

    @deprecate_renamed_function("get_column_index", version="0.19.14")
    def find_idx_by_name(self, name: str) -> int:
        """
        Find the index of a column by name.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`get_column_index`.

        Parameters
        ----------
        name
            Name of the column to find.
        """
        return self.get_column_index(name)

    @deprecate_renamed_function("insert_column", version="0.19.14")
    @deprecate_renamed_parameter("series", "column", version="0.19.14")
    def insert_at_idx(self, index: int, column: Series) -> Self:
        """
        Insert a Series at a certain column index. This operation is in place.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`insert_column`.

        Parameters
        ----------
        index
            Column to insert the new `Series` column.
        column
            `Series` to insert.
        """
        return self.insert_column(index, column)

    @deprecate_renamed_function("replace_column", version="0.19.14")
    @deprecate_renamed_parameter("series", "new_column", version="0.19.14")
    def replace_at_idx(self, index: int, new_column: Series) -> Self:
        """
        Replace a column at an index location.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`replace_column`.

        Parameters
        ----------
        index
            Column index.
        new_column
            Series that will replace the column.
        """
        return self.replace_column(index, new_column)

    @deprecate_renamed_function("equals", version="0.19.16")
    def frame_equal(self, other: DataFrame, *, null_equal: bool = True) -> bool:
        """
        Check whether the DataFrame is equal to another DataFrame.

        .. deprecated:: 0.19.16
            This method has been renamed to :func:`equals`.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.
        """
        return self.equals(other, null_equal=null_equal)


def _prepare_other_arg(other: Any, length: int | None = None) -> Series:
    # if not a series create singleton series such that it will broadcast
    value = other
    if not isinstance(other, pl.Series):
        if isinstance(other, str):
            pass
        elif isinstance(other, Sequence):
            raise TypeError("operation not supported")
        other = pl.Series("", [other])

    if length and length > 1:
        other = other.extend_constant(value=value, n=length - 1)

    return other

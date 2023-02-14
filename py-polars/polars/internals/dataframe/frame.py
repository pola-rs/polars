"""Module containing logic related to eager DataFrames."""
from __future__ import annotations

import math
import os
import random
import sys
import typing
from collections.abc import Sized
from datetime import timedelta
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Sequence,
    TypeVar,
    overload,
)

from polars import internals as pli
from polars._html import NotebookFormatter
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    N_INFER_DEFAULT,
    Boolean,
    Categorical,
    DataTypeClass,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    PolarsDataType,
    SchemaDefinition,
    SchemaDict,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    get_idx_type,
    py_type_to_dtype,
)
from polars.dependencies import (
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
    _check_for_pyarrow,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import NoRowsReturned, TooManyRowsReturned
from polars.internals.construction import (
    _post_apply_columns,
    arrow_to_pydf,
    dict_to_pydf,
    iterable_to_pydf,
    numpy_to_pydf,
    pandas_to_pydf,
    sequence_to_pydf,
    series_to_pydf,
)
from polars.internals.dataframe.groupby import DynamicGroupBy, GroupBy, RollingGroupBy
from polars.internals.slice import PolarsSlice
from polars.utils import (
    _prepare_row_count_args,
    _process_null_values,
    _timedelta_to_pl_duration,
    deprecate_nonkeyword_arguments,
    deprecated_alias,
    handle_projection_columns,
    is_bool_sequence,
    is_int_sequence,
    is_str_sequence,
    normalise_filepath,
    range_to_slice,
    redirect,
    scale_bytes,
)

try:
    from polars.polars import PyDataFrame

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec, TypeAlias
else:
    from typing_extensions import Concatenate, ParamSpec, TypeAlias

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from pyarrow.interchange.dataframe import _PyArrowDataFrame

    from polars.internals.type_aliases import (
        AsofJoinStrategy,
        AvroCompression,
        ClosedInterval,
        ComparisonOperator,
        CsvEncoding,
        FillNullStrategy,
        IpcCompression,
        JoinStrategy,
        NullStrategy,
        Orientation,
        ParallelStrategy,
        ParquetCompression,
        PivotAgg,
        PolarsExprType,
        PythonLiteral,
        RollingInterpolationMethod,
        SizeUnit,
        StartBy,
        UniqueKeepStrategy,
        UnstackDirection,
    )

    # these aliases are used to annotate DataFrame.__getitem__()
    # MultiRowSelector indexes into the vertical axis and
    # MultiColSelector indexes into the horizontal axis
    # NOTE: wrapping these as strings is necessary for Python <3.10

    MultiRowSelector: TypeAlias = "slice | range | list[int] | pli.Series"
    MultiColSelector: TypeAlias = (
        "slice | range | list[int] | list[str] | list[bool] | pli.Series"
    )

    T = TypeVar("T")
    P = ParamSpec("P")


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame._from_pydf(df)


@redirect(
    {
        "iterrows": "iter_rows",
        "with_column": "with_columns",
    }
)
class DataFrame:
    """
    Two-dimensional data structure representing data as a table with rows and columns.

    Parameters
    ----------
    data : dict, Sequence, ndarray, Series, or pandas.DataFrame
        Two-dimensional data in various forms. dict must contain Sequences.
        Sequence may contain Series or other Sequences.
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
        a _partial_ schema can be declared to prevent specific fields from being loaded.
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

    Notice that the dtype is automatically inferred as a polars Int64:

    >>> df.dtypes
    [Int64, Int64]

    To specify the frame schema you supply the `schema` parameter with a dictionary
    of (name,dtype) pairs...

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

    _accessors: set[str] = set()

    @deprecated_alias(columns="schema")
    def __init__(
        self,
        data: (
            Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | pli.Series]
            | Sequence[Any]
            | np.ndarray[Any, Any]
            | pa.Table
            | pd.DataFrame
            | pli.Series
            | None
        ) = None,
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
        elif isinstance(data, pli.Series):
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
            raise ValueError(
                f"DataFrame constructor called with unsupported type; got {type(data)}"
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
        infer_schema_length: int | None = N_INFER_DEFAULT,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDict | None = None,
    ) -> Self:
        pydf = PyDataFrame.read_dicts(data, infer_schema_length, schema)
        if schema or schema_overrides:
            pydf = _post_apply_columns(
                pydf, list(schema or pydf.columns()), schema_overrides=schema_overrides
            )
        return cls._from_pydf(pydf)

    @classmethod
    def _from_dict(
        cls,
        data: Mapping[
            str, Sequence[object] | Mapping[str, Sequence[object]] | pli.Series
        ],
        schema: SchemaDefinition | None = None,
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

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(
            dict_to_pydf(data, schema=schema, schema_overrides=schema_overrides)
        )

    @classmethod
    @deprecated_alias(columns="schema")
    def _from_records(
        cls,
        data: Sequence[Sequence[Any]],
        schema: SchemaDefinition | None = None,
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

        Returns
        -------
        DataFrame

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
    @deprecated_alias(columns="schema")
    def _from_numpy(
        cls,
        data: np.ndarray[Any, Any],
        schema: SchemaDefinition | None = None,
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

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(
            numpy_to_pydf(
                data, schema=schema, schema_overrides=schema_overrides, orient=orient
            )
        )

    @classmethod
    @deprecated_alias(columns="schema")
    def _from_arrow(
        cls,
        data: pa.Table,
        schema: SchemaDefinition | None = None,
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

        Returns
        -------
        DataFrame

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
    @deprecated_alias(columns="schema", nan_to_none="nan_to_null")
    def _from_pandas(
        cls,
        data: pd.DataFrame,
        schema: SchemaDefinition | None = None,
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

        Returns
        -------
        DataFrame

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
        file: str | Path | BinaryIO | bytes,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        sep: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = r'"',
        skip_rows: int = 0,
        dtypes: None | (SchemaDict | Sequence[PolarsDataType]) = None,
        null_values: str | list[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        parse_dates: bool = False,
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
    ) -> Self:
        """
        Read a CSV file into a DataFrame.

        Use ``pl.read_csv`` to dispatch to this method.

        See Also
        --------
        polars.io.read_csv

        """
        self = cls.__new__(cls)

        path: str | None
        if isinstance(file, (str, Path)):
            path = normalise_filepath(file)
        else:
            path = None
            if isinstance(file, BytesIO):
                file = file.getvalue()
            if isinstance(file, StringIO):
                file = file.getvalue().encode()

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
                raise ValueError("dtype arg should be list or dict")

        processed_null_values = _process_null_values(null_values)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(file, str) and "*" in file:
            dtypes_dict = None
            if dtype_list is not None:
                dtypes_dict = {name: dt for (name, dt) in dtype_list}
            if dtype_slice is not None:
                raise ValueError(
                    "cannot use glob patterns and unnamed dtypes as `dtypes` argument;"
                    " Use dtypes: Mapping[str, Type[DataType]"
                )
            from polars import scan_csv

            scan = scan_csv(
                file,
                has_header=has_header,
                sep=sep,
                comment_char=comment_char,
                quote_char=quote_char,
                skip_rows=skip_rows,
                dtypes=dtypes_dict,
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
            )
            if columns is None:
                return self._from_pydf(scan.collect()._df)
            elif is_str_sequence(columns, allow_str=False):
                return self._from_pydf(scan.select(columns).collect()._df)
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns`"
                    " argument; Use columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)

        self._df = PyDataFrame.read_csv(
            file,
            infer_schema_length,
            batch_size,
            has_header,
            ignore_errors,
            n_rows,
            skip_rows,
            projection,
            sep,
            rechunk,
            columns,
            encoding,
            n_threads,
            path,
            dtype_list,
            dtype_slice,
            low_memory,
            comment_char,
            quote_char,
            processed_null_values,
            missing_utf8_is_empty_string,
            parse_dates,
            skip_rows_after_header,
            _prepare_row_count_args(row_count_name, row_count_offset),
            sample_size=sample_size,
            eol_char=eol_char,
        )
        return self

    @classmethod
    def _read_parquet(
        cls,
        file: str | Path | BinaryIO,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
        parallel: ParallelStrategy = "auto",
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        low_memory: bool = False,
    ) -> Self:
        """
        Read into a DataFrame from a parquet file.

        Use ``pl.read_parquet`` to dispatch to this method.

        See Also
        --------
        polars.io.read_parquet

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(file, str) and "*" in file and pli._is_local_file(file):
            from polars import scan_parquet

            scan = scan_parquet(
                file,
                n_rows=n_rows,
                rechunk=True,
                parallel=parallel,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                low_memory=low_memory,
            )

            if columns is None:
                return cls._from_pydf(scan.collect()._df)
            elif is_str_sequence(columns, allow_str=False):
                return cls._from_pydf(scan.select(columns).collect()._df)
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns`"
                    " argument; Use columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_parquet(
            file,
            columns,
            projection,
            n_rows,
            parallel,
            _prepare_row_count_args(row_count_name, row_count_offset),
            low_memory=low_memory,
        )
        return self

    @classmethod
    def _read_avro(
        cls,
        file: str | Path | BinaryIO,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
    ) -> Self:
        """
        Read into a DataFrame from Apache Avro format.

        Parameters
        ----------
        file
            Path to a file or a file-like object.
        columns
            Columns.
        n_rows
            Stop reading from Apache Avro file after reading ``n_rows``.

        Returns
        -------
        DataFrame

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)
        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_avro(file, columns, projection, n_rows)
        return self

    @classmethod
    def _read_ipc(
        cls,
        file: str | Path | BinaryIO,
        columns: Sequence[int] | Sequence[str] | None = None,
        n_rows: int | None = None,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        rechunk: bool = True,
        memory_map: bool = True,
    ) -> DataFrame:
        """
        Read into a DataFrame from Arrow IPC stream format.

        Arrow IPC is also know as Feather (v2).

        Parameters
        ----------
        file
            Path to a file or a file-like object.
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a
            list of column names.
        n_rows
            Stop reading from IPC file after reading ``n_rows``.
        row_count_name
            Row count name.
        row_count_offset
            Row count offset.
        rechunk
            Make sure that all data is contiguous.
        memory_map
            Memory map the file

        Returns
        -------
        DataFrame

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(file, str) and "*" in file and pli._is_local_file(file):
            from polars import scan_ipc

            scan = scan_ipc(
                file,
                n_rows=n_rows,
                rechunk=rechunk,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                memory_map=memory_map,
            )
            if columns is None:
                return scan.collect()
            elif is_str_sequence(columns, allow_str=False):
                return scan.select(columns).collect()
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns`"
                    " argument; Use columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ipc(
            file,
            columns,
            projection,
            n_rows,
            _prepare_row_count_args(row_count_name, row_count_offset),
            memory_map=memory_map,
        )
        return self

    @classmethod
    def _read_json(cls, file: str | Path | IOBase) -> Self:
        """
        Read into a DataFrame from a JSON file.

        Use ``pl.read_json`` to dispatch to this method.

        See Also
        --------
        polars.io.read_json

        """
        if isinstance(file, StringIO):
            file = BytesIO(file.getvalue().encode())
        elif isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_json(file, False)
        return self

    @classmethod
    def _read_ndjson(cls, file: str | Path | IOBase) -> Self:
        """
        Read into a DataFrame from a newline delimited JSON file.

        Use ``pl.read_ndjson`` to dispatch to this method.

        See Also
        --------
        polars.io.read_ndjson

        """
        if isinstance(file, StringIO):
            file = BytesIO(file.getvalue().encode())
        elif isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ndjson(file)
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
    def columns(self, columns: Sequence[str]) -> None:
        """
        Change the column names of the `DataFrame`.

        Parameters
        ----------
        columns
            A list with new names for the `DataFrame`.
            The length of the list should be equal to the width of the `DataFrame`.

        """
        self._df.set_column_names(columns)

    @property
    def dtypes(self) -> list[PolarsDataType]:
        """
        Get the datatypes of the columns of this DataFrame.

        The datatypes can also be found in column headers when printing the DataFrame.

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

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        """
        return self._df.dtypes()

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
        {'foo': Int64, 'bar': Float64, 'ham': Utf8}

        """
        return dict(zip(self.columns, self.dtypes))

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame:
        """
        Convert to a dataframe object implementing the dataframe interchange protocol.

        Parameters
        ----------
        nan_as_null
            Overwrite null values in the data with ``NaN``.
        allow_copy
            Allow memory to be copied to perform the conversion. If set to False, causes
            conversions that are not zero-copy to fail.

        Notes
        -----
        Details on the dataframe interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        `nan_as_null` currently has no effect; once support for nullable extension
        dtypes is added, this value should be propagated to columns.

        """
        if not _PYARROW_AVAILABLE or int(pa.__version__.split(".")[0]) < 11:
            raise ImportError(
                "pyarrow>=11.0.0 is required for converting a Polars dataframe to a"
                " dataframe interchange object."
            )
        if not allow_copy and Categorical in self.schema.values():
            raise NotImplementedError(
                "Polars does not offer zero-copy conversion to Arrow for categorical"
                " columns. Set `allow_copy=True` or cast categorical columns to"
                " string first."
            )
        return self.to_arrow().__dataframe__(nan_as_null, allow_copy)

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
        other_renamed = other.select(pli.all().suffix(suffix))
        combined = pli.concat([self, other_renamed], how="horizontal")

        if op == "eq":
            expr = [pli.col(n) == pli.col(f"{n}{suffix}") for n in self.columns]
        elif op == "neq":
            expr = [pli.col(n) != pli.col(f"{n}{suffix}") for n in self.columns]
        elif op == "gt":
            expr = [pli.col(n) > pli.col(f"{n}{suffix}") for n in self.columns]
        elif op == "lt":
            expr = [pli.col(n) < pli.col(f"{n}{suffix}") for n in self.columns]
        elif op == "gt_eq":
            expr = [pli.col(n) >= pli.col(f"{n}{suffix}") for n in self.columns]
        elif op == "lt_eq":
            expr = [pli.col(n) <= pli.col(f"{n}{suffix}") for n in self.columns]
        else:
            raise ValueError(f"got unexpected comparison operator: {op}")

        return combined.select(expr)

    def _compare_to_non_df(
        self,
        other: Any,
        op: ComparisonOperator,
    ) -> Self:
        """Compare a DataFrame with a non-DataFrame object."""
        if op == "eq":
            return self.select(pli.all() == other)
        elif op == "neq":
            return self.select(pli.all() != other)
        elif op == "gt":
            return self.select(pli.all() > other)
        elif op == "lt":
            return self.select(pli.all() < other)
        elif op == "gt_eq":
            return self.select(pli.all() >= other)
        elif op == "lt_eq":
            return self.select(pli.all() <= other)
        else:
            raise ValueError(f"got unexpected comparison operator: {op}")

    def _div(self, other: Any, floordiv: bool) -> Self:
        if isinstance(other, pli.Series):
            other = other.to_frame()
        elif not isinstance(other, DataFrame):
            s = _prepare_other_arg(other, length=len(self))
            other = DataFrame([s.rename(f"n{i}") for i in range(len(self.columns))])

        orig_dtypes = other.dtypes
        other = self._cast_all_from_to(other, INTEGER_DTYPES, Float64)
        df = self._from_pydf(self._df.div_df(other._df))
        df = (
            df  # type: ignore[assignment]
            if not floordiv
            else df.with_columns([s.floor() for s in df if s.dtype() in FLOAT_DTYPES])
        )
        if floordiv:
            int_casts = [
                pli.col(col).cast(tp)
                for i, (col, tp) in enumerate(self.schema.items())
                if tp in INTEGER_DTYPES and orig_dtypes[i] in INTEGER_DTYPES
            ]
            if int_casts:
                return df.with_columns(int_casts)  # type: ignore[return-value]
        return df

    def _cast_all_from_to(
        self, df: DataFrame, from_: frozenset[PolarsDataType], to: PolarsDataType
    ) -> DataFrame:
        casts = [s.cast(to).alias(s.name) for s in df if s.dtype() in from_]
        return df.with_columns(casts) if casts else df

    def __floordiv__(self, other: DataFrame | pli.Series | int | float) -> Self:
        return self._div(other, floordiv=True)

    def __truediv__(self, other: DataFrame | pli.Series | int | float) -> Self:
        return self._div(other, floordiv=False)

    def __bool__(self) -> NoReturn:
        raise ValueError(
            "The truth value of a DataFrame is ambiguous. "
            "Hint: to check if a DataFrame contains any values, use 'is_empty()'"
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

    def __getstate__(self) -> list[pli.Series]:
        return self.get_columns()

    def __setstate__(self, state) -> None:  # type: ignore[no-untyped-def]
        self._df = DataFrame(state)._df

    def __mul__(self, other: DataFrame | pli.Series | int | float) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.mul_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.mul(other._s))

    def __rmul__(self, other: DataFrame | pli.Series | int | float) -> Self:
        return self * other

    def __add__(self, other: DataFrame | pli.Series | int | float | bool | str) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.add_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.add(other._s))

    def __radd__(
        self, other: DataFrame | pli.Series | int | float | bool | str
    ) -> Self:
        if isinstance(other, str):
            return self.select((pli.lit(other) + pli.col("*")).keep_name())
        return self + other

    def __sub__(self, other: DataFrame | pli.Series | int | float) -> Self:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.sub_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.sub(other._s))

    def __mod__(self, other: DataFrame | pli.Series | int | float) -> Self:
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

    def __iter__(self) -> Iterator[Any]:
        return self.get_columns().__iter__()

    def _pos_idx(self, idx: int, dim: int) -> int:
        if idx >= 0:
            return idx
        else:
            return self.shape[dim] + idx

    def _pos_idxs(
        self, idxs: np.ndarray[Any, Any] | pli.Series, dim: int
    ) -> pli.Series:
        # pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx).
        idx_type = get_idx_type()

        if isinstance(idxs, pli.Series):
            if idxs.dtype == idx_type:
                return idxs
            if idxs.dtype in {
                UInt8,
                UInt16,
                UInt64 if idx_type == UInt32 else UInt32,
                Int8,
                Int16,
                Int32,
                Int64,
            }:
                if idx_type == UInt32:
                    if idxs.dtype in {Int64, UInt64}:
                        if idxs.max() >= 2**32:  # type: ignore[operator]
                            raise ValueError(
                                "Index positions should be smaller than 2^32."
                            )
                    if idxs.dtype == Int64:
                        if idxs.min() < -(2**32):  # type: ignore[operator]
                            raise ValueError(
                                "Index positions should be bigger than -2^32 + 1."
                            )
                if idxs.dtype in {Int8, Int16, Int32, Int64}:
                    if idxs.min() < 0:  # type: ignore[operator]
                        if idx_type == UInt32:
                            if idxs.dtype in {Int8, Int16}:
                                idxs = idxs.cast(Int32)
                        else:
                            if idxs.dtype in {Int8, Int16, Int32}:
                                idxs = idxs.cast(Int64)

                        idxs = pli.select(
                            pli.when(pli.lit(idxs) < 0)
                            .then(self.shape[dim] + pli.lit(idxs))
                            .otherwise(pli.lit(idxs))
                        ).to_series()

                return idxs.cast(idx_type)

        if _check_for_numpy(idxs) and isinstance(idxs, np.ndarray):
            if idxs.ndim != 1:
                raise ValueError("Only 1D numpy array is supported as index.")
            if idxs.dtype.kind in ("i", "u"):
                # Numpy array with signed or unsigned integers.

                if idx_type == UInt32:
                    if idxs.dtype in {np.int64, np.uint64} and idxs.max() >= 2**32:
                        raise ValueError("Index positions should be smaller than 2^32.")
                    if idxs.dtype == np.int64 and idxs.min() < -(2**32):
                        raise ValueError(
                            "Index positions should be bigger than -2^32 + 1."
                        )
                if idxs.dtype.kind == "i" and idxs.min() < 0:
                    if idx_type == UInt32:
                        if idxs.dtype in (np.int8, np.int16):
                            idxs = idxs.astype(np.int32)
                    else:
                        if idxs.dtype in (np.int8, np.int16, np.int32):
                            idxs = idxs.astype(np.int64)

                    # Update negative indexes to absolute indexes.
                    idxs = np.where(idxs < 0, self.shape[dim] + idxs, idxs)

                return pli.Series("", idxs, dtype=idx_type)

        raise NotImplementedError("Unsupported idxs datatype.")

    @overload
    def __getitem__(self, item: str) -> pli.Series:
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
    def __getitem__(self, item: tuple[MultiRowSelector, int]) -> pli.Series:
        ...

    @overload
    def __getitem__(self, item: tuple[MultiRowSelector, str]) -> pli.Series:
        ...

    @overload
    def __getitem__(self, item: tuple[int, int]) -> Any:
        ...

    @overload
    def __getitem__(self, item: tuple[int, str]) -> Any:
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
            | tuple[MultiRowSelector, int]
            | tuple[MultiRowSelector, str]
            | tuple[int, int]
            | tuple[int, str]
        ),
    ) -> DataFrame | pli.Series:
        """Get item. Does quite a lot. Read the comments."""
        # select rows and columns at once
        # every 2d selection, i.e. tuple is row column order, just like numpy
        if isinstance(item, tuple) and len(item) == 2:
            row_selection, col_selection = item

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
                        start = self.find_idx_by_name(col_selection.start)
                    if isinstance(col_selection.stop, str):
                        stop = self.find_idx_by_name(col_selection.stop) + 1

                    col_selection = slice(start, stop, col_selection.step)

                    df = self.__getitem__(self.columns[col_selection])
                    return df[row_selection]

                # df[:, [True, False]]
                if is_bool_sequence(col_selection) or (
                    isinstance(col_selection, pli.Series)
                    and col_selection.dtype == Boolean
                ):
                    if len(col_selection) != self.width:
                        raise ValueError(
                            f"Expected {self.width} values when selecting columns by"
                            f" boolean mask. Got {len(col_selection)}."
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
                # df[2, "a"]
                if isinstance(col_selection, str):
                    return self[col_selection][row_selection]

            # column selection can be "a" and ["a", "b"]
            if isinstance(col_selection, str):
                col_selection = [col_selection]

            # df[:, 1]
            if isinstance(col_selection, int):
                series = self.to_series(col_selection)
                return series[row_selection]

            if isinstance(col_selection, list):
                # df[:, [1, 2]]
                if is_int_sequence(col_selection):
                    series_list = [self.to_series(i) for i in col_selection]
                    df = self.__class__(series_list)
                    return df[row_selection]

            df = self.__getitem__(col_selection)
            return df.__getitem__(row_selection)

        # select single column
        # df["foo"]
        if isinstance(item, str):
            return pli.wrap_s(self._df.column(item))

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
                raise ValueError("Only a 1D-Numpy array is supported as index.")
            if item.dtype.kind in ("i", "u"):
                # Numpy array with signed or unsigned integers.
                return self._from_pydf(
                    self._df.take_with_series(self._pos_idxs(item, dim=0)._s)
                )
            if isinstance(item[0], str):
                return self._from_pydf(self._df.select(item))

        if is_str_sequence(item, allow_str=False):
            # select multiple columns
            # df[["foo", "bar"]]
            return self._from_pydf(self._df.select(item))
        elif is_int_sequence(item):
            item = pli.Series("", item)  # fall through to next if isinstance

        if isinstance(item, pli.Series):
            dtype = item.dtype
            if dtype == Utf8:
                return self._from_pydf(self._df.select(item))
            if dtype == UInt32:
                return self._from_pydf(self._df.take_with_series(item._s))
            if dtype in {UInt8, UInt16, UInt64, Int8, Int16, Int32, Int64}:
                return self._from_pydf(
                    self._df.take_with_series(self._pos_idxs(item, dim=0)._s)
                )

        # if no data has been returned, the operation is not supported
        raise ValueError(
            f"Cannot __getitem__ on DataFrame with item: '{item}'"
            f" of type: '{type(item)}'."
        )

    def __setitem__(
        self,
        key: str | Sequence[int] | Sequence[str] | tuple[Any, str | int],
        value: Any,
    ) -> None:  # pragma: no cover
        # df["foo"] = series
        if isinstance(key, str):
            raise TypeError(
                "'DataFrame' object does not support 'Series' assignment by index. "
                "Use 'DataFrame.with_columns'"
            )

        # df[["C", "D"]]
        elif isinstance(key, list):
            # TODO: Use python sequence constructors
            value = np.array(value)
            if value.ndim != 2:
                raise ValueError("can only set multiple columns with 2D matrix")
            if value.shape[1] != len(key):
                raise ValueError(
                    "matrix columns should be equal to list use to determine column"
                    " names"
                )

            # todo! we can parallelize this by calling from_numpy
            columns = []
            for i, name in enumerate(key):
                columns.append(pli.Series(name, value[:, i]))
            self._df = self.with_columns(columns)._df

        # df[a, b]
        elif isinstance(key, tuple):
            row_selection, col_selection = key

            if (
                isinstance(row_selection, pli.Series) and row_selection.dtype == Boolean
            ) or is_bool_sequence(row_selection):
                raise ValueError(
                    "Not allowed to set 'DataFrame' by boolean mask in the "
                    "row position. Consider using 'DataFrame.with_columns'"
                )

            # get series column selection
            if isinstance(col_selection, str):
                s = self.__getitem__(col_selection)
            elif isinstance(col_selection, int):
                s = self[:, col_selection]
            else:
                raise ValueError(f"column selection not understood: {col_selection}")

            # dispatch to __setitem__ of Series to do modification
            s[row_selection] = value

            # now find the location to place series
            # df[idx]
            if isinstance(col_selection, int):
                self.replace_at_idx(col_selection, s)
            # df["foo"]
            elif isinstance(col_selection, str):
                self.replace(col_selection, s)
        else:
            raise ValueError(
                f"Cannot __setitem__ on DataFrame with key: '{key}' "
                f"of type: '{type(key)}' and value: '{value}' "
                f"of type: '{type(value)}'."
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
        return "\n".join(
            NotebookFormatter(
                self,
                max_cols=max_cols,
                max_rows=max_rows,
                from_series=from_series,
            ).render()
        )

    def item(self) -> Any:
        """
        Return the dataframe as a scalar.

        Equivalent to ``df[0,0]``, with a check that the shape is (1,1).

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> result = df.select((pl.col("a") * pl.col("b")).sum())
        >>> result
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 32  │
        └─────┘
        >>> result.item()
        32

        """
        if self.shape != (1, 1):
            raise ValueError(
                f"Can only call .item() if the dataframe is of shape (1,1), "
                f"dataframe is of shape {self.shape}"
            )
        return self[0, 0]

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
        record_batches = self._df.to_arrow()
        return pa.Table.from_batches(record_batches)

    @overload
    def to_dict(self, as_series: Literal[True] = ...) -> dict[str, pli.Series]:
        ...

    @overload
    def to_dict(self, as_series: Literal[False]) -> dict[str, list[Any]]:
        ...

    @overload
    def to_dict(
        self, as_series: bool = True
    ) -> dict[str, pli.Series] | dict[str, list[Any]]:
        ...

    def to_dict(
        self, as_series: bool = True
    ) -> dict[str, pli.Series] | dict[str, list[Any]]:
        """
        Convert DataFrame to a dictionary mapping column name to values.

        Parameters
        ----------
        as_series
            True -> Values are series
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
        Convert every row to a dictionary of python-native values.

        Notes
        -----
        If you have ``ns``-precision temporal values you should be aware that python
        natively only supports up to ``us``-precision; if this matters you should export
        to a different format.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.to_dicts()
        [{'foo': 1, 'bar': 4}, {'foo': 2, 'bar': 5}, {'foo': 3, 'bar': 6}]

        """
        return list(self.iter_rows(named=True))

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Convert DataFrame to a 2D NumPy array.

        This operation clones data.

        Notes
        -----
        If you're attempting to convert Utf8 to an array you'll need to install
        ``pyarrow``.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> numpy_array = df.to_numpy()
        >>> type(numpy_array)
        <class 'numpy.ndarray'>

        """
        out = self._df.to_numpy()
        if out is None:
            return np.vstack(
                [self.to_series(i).to_numpy() for i in range(self.width)]
            ).T
        else:
            return out

    def to_pandas(  # noqa: D417
        self, *args: Any, use_pyarrow_extension_array: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Cast to a pandas DataFrame.

        This requires that :mod:`pandas` and :mod:`pyarrow` are installed.
        This operation clones data, unless `use_pyarrow_extension_array=True`.

        Parameters
        ----------
        use_pyarrow_extension_array
            Use PyArrow backed-extension arrays instead of numpy arrays for each
            column of the pandas DataFrame. This allows zero copy operations and
            preservation of nulls values.
            Further operations on this pandas DataFrame, might trigger conversion
            to NumPy arrays if that operation is not supported by pyarrow compute
            functions.
        kwargs
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
            pandas_version_major, pandas_version_minor = (
                int(x) for x in pd.__version__.split(".")[0:2]
            )
            if pandas_version_major == 0 or (
                pandas_version_major == 1 and pandas_version_minor < 5
            ):
                raise ModuleNotFoundError(
                    f'"use_pyarrow_extension_array=True" requires Pandas 1.5.x or higher, found Pandas {pd.__version__}.'
                )

        record_batches = self._df.to_pandas()
        tbl = pa.Table.from_batches(record_batches)
        return (
            tbl.to_pandas(
                self_destruct=True,
                split_blocks=True,
                types_mapper=lambda pa_dtype: pd.ArrowDtype(pa_dtype),
                **kwargs,
            )
            if use_pyarrow_extension_array
            else tbl.to_pandas(**kwargs)
        )

    def to_series(self, index: int = 0) -> pli.Series:
        """
        Select column as Series at index location.

        Parameters
        ----------
        index
            Location of selection.

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
        if index < 0:
            index = len(self.columns) + index
        return pli.wrap_s(self._df.select_at_idx(index))

    @overload
    def write_json(
        self,
        file: None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
    ) -> str:
        ...

    @overload
    def write_json(
        self,
        file: IOBase | str | Path,
        pretty: bool = ...,
        row_oriented: bool = ...,
    ) -> None:
        ...

    def write_json(
        self,
        file: IOBase | str | Path | None = None,
        pretty: bool = False,
        row_oriented: bool = False,
    ) -> str | None:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to ``None``
            (default), the output is returned as a string instead.
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
        '{"columns":[{"name":"foo","datatype":"Int64","values":[1,2,3]},{"name":"bar","datatype":"Int64","values":[6,7,8]}]}'
        >>> df.write_json(row_oriented=True)
        '[{"foo":1,"bar":6},{"foo":2,"bar":7},{"foo":3,"bar":8}]'

        """
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)
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
            File path to which the result should be written. If set to ``None``
            (default), the output is returned as a string instead.

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
            file = normalise_filepath(file)
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
        has_header: bool = ...,
        sep: str = ...,
        quote: str = ...,
        batch_size: int = ...,
        datetime_format: str | None = ...,
        date_format: str | None = ...,
        time_format: str | None = ...,
        float_precision: int | None = ...,
        null_value: str | None = ...,
    ) -> str:
        ...

    @overload
    def write_csv(
        self,
        file: BytesIO | str | Path,
        has_header: bool = ...,
        sep: str = ...,
        quote: str = ...,
        batch_size: int = ...,
        datetime_format: str | None = ...,
        date_format: str | None = ...,
        time_format: str | None = ...,
        float_precision: int | None = ...,
        null_value: str | None = ...,
    ) -> None:
        ...

    def write_csv(
        self,
        file: BytesIO | str | Path | None = None,
        has_header: bool = True,
        sep: str = ",",
        quote: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_precision: int | None = None,
        null_value: str | None = None,
    ) -> str | None:
        """
        Write to comma-separated values (CSV) file.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to ``None``
            (default), the output is returned as a string instead.
        has_header
            Whether to include header in the CSV output.
        sep
            Separate CSV fields with this symbol.
        quote
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
            Number of decimal places to write, applied to both ``Float32`` and
            ``Float64`` datatypes.
        null_value
            A string representing null values (defaulting to the empty string).

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
        >>> df.write_csv(path, sep=",")

        """
        if len(sep) > 1:
            raise ValueError("only single byte separator is allowed")
        elif len(quote) > 1:
            raise ValueError("only single byte quote char is allowed")
        elif null_value == "":
            null_value = None

        if file is None:
            buffer = BytesIO()
            self._df.write_csv(
                buffer,
                has_header,
                ord(sep),
                ord(quote),
                batch_size,
                datetime_format,
                date_format,
                time_format,
                float_precision,
                null_value,
            )
            return str(buffer.getvalue(), encoding="utf-8")

        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        self._df.write_csv(
            file,
            has_header,
            ord(sep),
            ord(quote),
            batch_size,
            datetime_format,
            date_format,
            time_format,
            float_precision,
            null_value,
        )
        return None

    def write_avro(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: AvroCompression = "uncompressed",
    ) -> None:
        """
        Write to Apache Avro file.

        Parameters
        ----------
        file
            File path to which the file should be written.
        compression : {'uncompressed', 'snappy', 'deflate'}
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
        >>> path: pathlib.Path = dirpath / "new_file.avro"
        >>> df.write_avro(path)

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        self._df.write_avro(file, compression)

    def write_ipc(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: IpcCompression = "uncompressed",
    ) -> None:
        """
        Write to Arrow IPC binary stream or Feather file.

        Parameters
        ----------
        file
            File path to which the file should be written.
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
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        self._df.write_ipc(file, compression)

    def write_parquet(
        self,
        file: str | Path | BytesIO,
        *,
        compression: ParquetCompression = "zstd",
        compression_level: int | None = None,
        statistics: bool = False,
        row_group_size: int | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, object] | None = None,
    ) -> None:
        """
        Write to Apache Parquet file.

        Parameters
        ----------
        file
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
        use_pyarrow
            Use C++ parquet implementation vs Rust parquet implementation.
            At the moment C++ supports more features.
        pyarrow_options
            Arguments passed to ``pyarrow.parquet.write_table``.

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

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = normalise_filepath(file)

        if use_pyarrow:
            tbl = self.to_arrow()
            data = {}

            for i, column in enumerate(tbl):
                # extract the name before casting
                name = f"column_{i}" if column._name is None else column._name

                data[name] = column

            tbl = pa.table(data)

            # do not remove this
            # needed below
            import pyarrow.parquet  # noqa: F401

            pa.parquet.write_table(
                table=tbl,
                where=file,
                row_group_size=row_group_size,
                compression=None if compression == "uncompressed" else compression,
                compression_level=compression_level,
                write_statistics=statistics,
                **(pyarrow_options or {}),
            )
        else:
            self._df.write_parquet(
                file, compression, compression_level, statistics, row_group_size
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
        include_header: bool = False,
        header_name: str = "column",
        column_names: Iterator[str] | Sequence[str] | None = None,
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
            Optional generator/iterator that yields column names. Will be used to
            replace the columns in the DataFrame.

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

        """
        df = self._from_pydf(self._df.transpose(include_header, header_name))
        if column_names is not None:
            names = []
            n = df.width
            if include_header:
                names.append(header_name)
                n -= 1

            column_names = iter(column_names)
            for _ in range(n):
                names.append(next(column_names))
            df.columns = names
        return df

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
        return self.select(pli.col("*").reverse())

    def rename(self, mapping: dict[str, str]) -> Self:
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
        return self._from_pydf(
            self.lazy().rename(mapping).collect(no_optimization=True)._df
        )

    def insert_at_idx(self, index: int, series: pli.Series) -> Self:
        """
        Insert a Series at a certain column index. This operation is in place.

        Parameters
        ----------
        index
            Column to insert the new `Series` column.
        series
            `Series` to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series("baz", [97, 98, 99])
        >>> df.insert_at_idx(1, s)
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
        >>> df.insert_at_idx(3, s)
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
        self._df.insert_at_idx(index, series._s)
        return self

    def filter(
        self,
        predicate: (
            pli.Expr | str | pli.Series | list[bool] | np.ndarray[Any, Any] | bool
        ),
    ) -> DataFrame:
        """
        Filter the rows in the DataFrame based on a predicate expression.

        Parameters
        ----------
        predicate
            Expression that evaluates to a boolean Series.

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

        >>> df.filter(pl.col("foo") < 3)
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

        >>> df.filter((pl.col("foo") < 3) & (pl.col("ham") == "a"))
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Filter on an OR condition:

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

        """
        if _check_for_numpy(predicate) and isinstance(predicate, np.ndarray):
            predicate = pli.Series(predicate)

        return (
            self.lazy()
            .filter(predicate)  # type: ignore[arg-type]
            .collect(no_optimization=True)
        )

    @overload
    def glimpse(self, return_as_string: Literal[False]) -> None:
        ...

    @overload
    def glimpse(self, return_as_string: Literal[True]) -> str:
        ...

    def glimpse(self, return_as_string: bool = False) -> str | None:
        """
        Return a dense preview of the dataframe.

        The formatting is done one line per column, so wide dataframes show nicely.
        Each line will show the column name, the data type and the first few values.

        Parameters
        ----------
        return_as_string
            If True, return as string rather than printing to stdout.

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
        $ d  <str> None, b, c
        $ e  <str> usd, eur, None
        $ f <date> 2020-01-01, 2021-01-02, 2022-01-01

        """
        # always print at most this number of values, mainly used to ensure
        # we do not cast long arrays to strings which would be very slow
        max_num_values = min(10, self.height)

        def _parse_column(col_name: str, dtype: PolarsDataType) -> tuple[str, str, str]:
            dtype_str = (
                f"<{DataTypeClass._string_repr(dtype)}>"
                if isinstance(dtype, DataTypeClass)
                else f"<{dtype._string_repr()}>"
            )
            val = self[:max_num_values][col_name].to_list()
            val_str = ", ".join(map(str, val))
            return col_name, dtype_str, val_str

        data = [_parse_column(s, dtype) for s, dtype in self.schema.items()]

        # we make the first column as small as possible by taking the longest
        # column name
        max_col_name = max((len(col_name) for col_name, _, _ in data))

        # dtype string
        max_col_dtype = max((len(dtype_str) for _, dtype_str, _ in data))

        # limit the amount of data printed such that total width is fixed
        max_col_values = 100 - max_col_name - max_col_dtype

        output = StringIO()
        # print header
        output.write(f"Rows: {self.height}\nColumns: {self.width}\n")

        # print individual columns: one row per column
        for col_name, dtype_str, val_str in data:
            output.write(
                f"$ {col_name:<{max_col_name}}"
                f" {dtype_str:>{max_col_dtype}}"
                f" {val_str:<{min(len(val_str), max_col_values)}}\n"
            )

        s = output.getvalue()

        if not return_as_string:
            print(s, end=None)
            return None
        else:
            return s

    def describe(self) -> Self:
        """
        Summary statistics for a DataFrame.

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
        shape: (7, 7)
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
        │ max        ┆ 3.0      ┆ 5.0      ┆ 1.0      ┆ c    ┆ usd  ┆ 2022-01-01 │
        │ median     ┆ 2.8      ┆ 4.5      ┆ 1.0      ┆ null ┆ null ┆ null       │
        └────────────┴──────────┴──────────┴──────────┴──────┴──────┴────────────┘

        """

        def describe_cast(stat: Self) -> Self:
            columns = []
            for i, s in enumerate(self.columns):
                if self[s].is_numeric() or self[s].is_boolean():
                    columns.append(stat[:, i].cast(float))
                else:
                    # for dates, strings, etc, we cast to string so that all
                    # statistics can be shown
                    columns.append(stat[:, i].cast(str))
            return self.__class__(columns)

        summary = self._from_pydf(
            pli.concat(
                [
                    describe_cast(
                        self.__class__({c: [len(self)] for c in self.columns})
                    ),
                    describe_cast(self.null_count()),
                    describe_cast(self.mean()),
                    describe_cast(self.std()),
                    describe_cast(self.min()),
                    describe_cast(self.max()),
                    describe_cast(self.median()),
                ]
            )._df
        )
        summary.insert_at_idx(
            0,
            pli.Series(
                "describe",
                ["count", "null_count", "mean", "std", "min", "max", "median"],
            ),
        )
        return summary

    def find_idx_by_name(self, name: str) -> int:
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
        >>> df.find_idx_by_name("ham")
        2

        """
        return self._df.find_idx_by_name(name)

    def replace_at_idx(self, index: int, series: pli.Series) -> Self:
        """
        Replace a column at an index location.

        Parameters
        ----------
        index
            Column index.
        series
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
        >>> df.replace_at_idx(0, s)
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
        self._df.replace_at_idx(index, series._s)
        return self

    def sort(
        self,
        by: str | pli.Expr | Sequence[str] | Sequence[pli.Expr],
        *,
        reverse: bool | list[bool] = False,
        nulls_last: bool = False,
    ) -> Self:
        """
        Sort the DataFrame by column.

        Parameters
        ----------
        by
            By which column to sort. Only accepts string.
        reverse
            Reverse/descending sort.
        nulls_last
            Place null values last. Can only be used if sorted by a single column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sort("foo", reverse=True)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        │ 2   ┆ 7.0 ┆ b   │
        │ 1   ┆ 6.0 ┆ a   │
        └─────┴─────┴─────┘

        **Sort by multiple columns.**
        For multiple columns we can also use expression syntax.

        >>> df.sort(
        ...     [pl.col("foo"), pl.col("bar") ** 2],
        ...     reverse=[True, False],
        ... )
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        │ 2   ┆ 7.0 ┆ b   │
        │ 1   ┆ 6.0 ┆ a   │
        └─────┴─────┴─────┘

        """
        if not isinstance(by, str) and isinstance(by, (Sequence, pli.Expr)):
            df = (
                self.lazy()
                .sort(by, reverse=reverse, nulls_last=nulls_last)
                .collect(no_optimization=True)
            )
            return self._from_pydf(df._df)
        return self._from_pydf(self._df.sort(by, reverse, nulls_last))

    def frame_equal(self, other: DataFrame, null_equal: bool = True) -> bool:
        """
        Check if DataFrame is equal to other.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.

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
        >>> df1.frame_equal(df1)
        True
        >>> df1.frame_equal(df2)
        False

        """
        return self._df.frame_equal(other._df, null_equal)

    def replace(self, column: str, new_col: pli.Series) -> Self:
        """
        Replace a column by a new Series.

        Parameters
        ----------
        column
            Column to replace.
        new_col
            New column to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series([10, 20, 30])
        >>> df.replace("foo", s)  # works in-place!
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
        self._df.replace(column, new_col._s)
        return self

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

    def limit(self, n: int = 5) -> Self:
        """
        Get the first `n` rows.

        Alias for :func:`DataFrame.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3, 4, 5, 6], "bar": ["a", "b", "c", "d", "e", "f"]}
        ... )
        >>> df.limit(4)
        shape: (4, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ a   │
        │ 2   ┆ b   │
        │ 3   ┆ c   │
        │ 4   ┆ d   │
        └─────┴─────┘

        """
        return self.head(n)

    def head(self, n: int = 5) -> Self:
        """
        Get the first `n` rows (if negative, returns all rows except the last `n`).

        Parameters
        ----------
        n
            Number of rows to return.

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

        Negative values of ``head`` return all rows _except_ the last abs(n).

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
            n = len(self) + n
        return self._from_pydf(self._df.head(n))

    def tail(self, n: int = 5) -> Self:
        """
        Get the last `n` rows (if negative, returns all rows except the first `n`).

        Parameters
        ----------
        n
            Number of rows to return.

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

        Negative values of ``tail`` return all rows _except_ the first abs(n).

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
            n = len(self) + n
        return self._from_pydf(self._df.tail(n))

    def drop_nulls(self, subset: str | Sequence[str] | None = None) -> Self:
        """
        Return a new DataFrame where rows with null values are dropped.

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
        >>> df.drop_nulls()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        This method drops rows where any single value of the row is null.

        Below are some example snippets that show how you could drop null values based
        on other conditions

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
        if isinstance(subset, str):
            subset = [subset]
        return self._from_pydf(self._df.drop_nulls(subset))

    def pipe(
        self,
        func: Callable[Concatenate[DataFrame, P], T],
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
        return func(self, *args, **kwargs)

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

    def groupby(
        self,
        by: str | pli.Expr | Sequence[str | pli.Expr],
        *,
        maintain_order: bool = False,
    ) -> GroupBy[Self]:
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more
            expensive than a default groupby. Note that this only works in expression
            aggregations.

        Examples
        --------
        Below we group by column `"a"`, and we sum column `"b"`.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.groupby("a").agg(pl.col("b").sum()).sort(by="a")
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

        We can also loop over the grouped `DataFrame`

        >>> for sub_df in df.groupby("a"):
        ...     print(sub_df)  # doctest: +IGNORE_RESULT
        ...
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 5   │
        │ b   ┆ 4   ┆ 3   │
        │ b   ┆ 5   ┆ 2   │
        └─────┴─────┴─────┘
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘

        """
        # Explicitly handle case where user mistakenly calls `groupby("a", "b")`
        if not isinstance(maintain_order, bool):
            raise TypeError(
                f"invalid input for groupby arg `maintain_order`: {maintain_order}."
            )

        return GroupBy(self._df, by, self.__class__, maintain_order=maintain_order)

    def groupby_rolling(
        self,
        index_column: str,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        by: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
    ) -> RollingGroupBy[Self]:
        """
        Create rolling groups based on a time column.

        Also works for index values of type Int32 or Int64.

        Different from a ``dynamic_groupby`` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals use
        *groupby_dynamic*

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

        - **"1i"      # length 1**
        - **"10i"     # length 10**

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
        return RollingGroupBy(self, index_column, period, offset, closed, by)

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
    ) -> DynamicGroupBy[Self]:
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
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        start_by : {'window', 'datapoint', 'monday'}
            The strategy to determine the start of the first window by.

            - 'window': Truncate the start of the window with the 'every' argument.
            - 'datapoint': Start from the first encountered data point.
            - 'monday': Start the window on the monday before the first data point.

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

        >>> df.groupby_dynamic("time", every="1h", closed="right").agg(
        ...     [
        ...         pl.col("time").min().alias("time_min"),
        ...         pl.col("time").max().alias("time_max"),
        ...     ]
        ... )
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

        >>> df.groupby_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg([pl.col("time").count().alias("time_count")])
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

        >>> df.groupby_dynamic("time", every="1h", closed="left").agg(
        ...     [
        ...         pl.col("time").count().alias("time_count"),
        ...         pl.col("time").alias("time_agg_list"),
        ...     ]
        ... )
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

        >>> df.groupby_dynamic("time", every="1h", closed="both").agg(
        ...     [pl.col("time").count().alias("time_count")]
        ... )
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
        >>> df.groupby_dynamic(
        ...     "time",
        ...     every="1h",
        ...     closed="both",
        ...     by="groups",
        ...     include_boundaries=True,
        ... ).agg([pl.col("time").count().alias("time_count")])
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
        ...     df.groupby_dynamic(
        ...         "idx",
        ...         every="2i",
        ...         period="3i",
        ...         include_boundaries=True,
        ...         closed="right",
        ...     ).agg(pl.col("A").alias("A_agg_list"))
        ... )
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
        return DynamicGroupBy(
            self,
            index_column,
            every,
            period,
            offset,
            truncate,
            include_boundaries,
            closed,
            by,
            start_by,
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

        The `every` and `offset` arguments are created with
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
        ... )
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

    @deprecate_nonkeyword_arguments()
    def join_asof(
        self,
        other: DataFrame,
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
        ... )
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
        ... )
        >>> population.join_asof(
        ...     gdp, left_on="date", right_on="date", strategy="backward"
        ... )
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
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"Expected 'other' join table to be a DataFrame, not a {type(other).__name__}"
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
            .collect(no_optimization=True)
        )

    @deprecate_nonkeyword_arguments(
        message=(
            "All arguments of DataFrame.join except for 'other', 'on', and 'how' will be keyword-only in the next breaking release."
            " Use keyword arguments to silence this warning."
        )
    )
    def join(
        self,
        other: DataFrame,
        left_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        right_on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        on: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
        how: JoinStrategy = "inner",
        suffix: str = "_right",
    ) -> DataFrame:
        """
        Join in SQL-like fashion.

        Parameters
        ----------
        other
            DataFrame to join with.
        left_on
            Name(s) of the left join column(s).
        right_on
            Name(s) of the right join column(s).
        on
            Name(s) of the join columns in both DataFrames.
        how : {'inner', 'left', 'outer', 'semi', 'anti', 'cross'}
            Join strategy.
        suffix
            Suffix to append to columns with a duplicate name.

        Returns
        -------
            Joined DataFrame

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
        For joining on columns with categorical data, see ``pl.StringCache()``.

        """
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"Expected 'other' join table to be a DataFrame, not a {type(other).__name__}"
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
            )
            .collect(no_optimization=True)
        )

    @deprecate_nonkeyword_arguments(allowed_args=["self", "f", "return_dtype"])
    def apply(
        self,
        f: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        inference_size: int = 256,
    ) -> Self:
        """
        Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

        The UDF will receive each row as a tuple of values: ``udf(row)``.

        Implementing logic using a Python function is almost always _significantly_
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        f
            Custom function/ lambda function.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema

        Notes
        -----
        The frame-level ``apply`` cannot track column names (as the UDF is a black-box
        that may arbitrarily drop, rearrange, transform, or add new columns); if you
        want to apply a UDF such that column names are preserved, you should use the
        expression-level ``apply`` syntax instead.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [-1, 5, 8]})

        Return a DataFrame by mapping each row to a tuple:

        >>> df.apply(lambda t: (t[0] * 2, t[1] * 3))
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

        It is better to implement this with an expression:

        >>> df.select([pl.col("foo") * 2, pl.col("bar") * 3])  # doctest: +IGNORE_RESULT

        Return a Series by mapping each row to a scalar:

        >>> df.apply(lambda t: (t[0] * 2 + t[1]))
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

        In this case it is better to use the following expression:

        >>> df.select(pl.col("foo") * 2 + pl.col("bar"))  # doctest: +IGNORE_RESULT

        """
        out, is_df = self._df.apply(f, return_dtype, inference_size)
        if is_df:
            return self._from_pydf(out)
        else:
            return self._from_pydf(pli.wrap_s(out).to_frame()._df)

    def hstack(
        self,
        columns: list[pli.Series] | DataFrame,
        in_place: bool = False,
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

    def vstack(self, df: DataFrame, in_place: bool = False) -> Self:
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.

        Parameters
        ----------
        df
            DataFrame to stack.
        in_place
            Modify in place

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
            self._df.vstack_mut(df._df)
            return self
        else:
            return self._from_pydf(self._df.vstack(df._df))

    def extend(self, other: Self) -> Self:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from `vstack` which adds the chunks from `other` to the chunks of this
        `DataFrame` `extend` appends the data from `other` to the underlying memory
        locations and thus may cause a reallocation.

        If this does not cause a reallocation, the resulting data structure will not
        have any extra chunks and thus will yield faster queries.

        Prefer `extend` over `vstack` when you want to do a query after a single append.
        For instance during online operations where you add `n` rows and rerun a query.

        Prefer `vstack` over `extend` when you want to append many times before doing a
        query. For instance when you read in multiple files and when to store them in a
        single `DataFrame`. In the latter case, finish the sequence of `vstack`
        operations with a `rechunk`.

        Parameters
        ----------
        other
            DataFrame to vertically add.

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
        self._df.extend(other._df)
        return self

    def drop(self, columns: str | Sequence[str]) -> Self:
        """
        Remove column from DataFrame and return as new.

        Parameters
        ----------
        columns
            Column(s) to drop.

        Examples
        --------
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

        """
        if isinstance(columns, list):
            df = self.clone()

            for n in columns:
                df._df.drop_in_place(n)
            return df

        return self._from_pydf(self._df.drop(columns))

    def drop_in_place(self, name: str) -> pli.Series:
        """
        Drop in place.

        Parameters
        ----------
        name
            Column to drop.

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
        return pli.wrap_s(self._df.drop_in_place(name))

    def cleared(self) -> Self:
        """
        Create an empty copy of the current DataFrame.

        Returns a DataFrame with identical schema but no data.

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
        >>> df.cleared()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        """
        return self.head(0) if len(self) > 0 else self.clone()

    def clone(self) -> Self:
        """
        Cheap deepcopy/clone.

        See Also
        --------
        cleared : Create an empty copy of the current DataFrame, with identical
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

    def get_columns(self) -> list[pli.Series]:
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
        return [pli.wrap_s(s) for s in self._df.get_columns()]

    def get_column(self, name: str) -> pli.Series:
        """
        Get a single column as Series by name.

        Parameters
        ----------
        name : str
            Name of the column to retrieve.

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
        return self[name]

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
            Fill all matching supertype of the fill ``value``.

        Returns
        -------
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
        return self._from_pydf(
            self.lazy()
            .fill_null(value, strategy, limit, matches_supertype)
            .collect(no_optimization=True)
            ._df
        )

    def fill_nan(self, fill_value: pli.Expr | int | float | None) -> DataFrame:
        """
        Fill floating point NaN values by an Expression evaluation.

        Parameters
        ----------
        fill_value
            Value to fill NaN with.

        Returns
        -------
            DataFrame with NaN replaced with fill_value

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
        ...         "a": [1.5, 2, float("NaN"), 4],
        ...         "b": [0.5, 4, float("NaN"), 13],
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
        return self.lazy().fill_nan(fill_value).collect(no_optimization=True)

    def explode(
        self,
        columns: str | Sequence[str] | pli.Expr | Sequence[pli.Expr],
    ) -> DataFrame:
        """
        Explode `DataFrame` to long format by exploding a column with Lists.

        Parameters
        ----------
        columns
            Column of LargeList type.

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
        return self.lazy().explode(columns).collect(no_optimization=True)

    @deprecate_nonkeyword_arguments(
        allowed_args=["self", "values", "index", "columns", "aggregate_fn"]
    )
    def pivot(
        self,
        values: Sequence[str] | str,
        index: Sequence[str] | str,
        columns: Sequence[str] | str,
        aggregate_fn: PivotAgg | pli.Expr = "first",
        maintain_order: bool = True,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> Self:
        """
        Create a spreadsheet-style pivot table as a DataFrame.

        Parameters
        ----------
        values
            Column values to aggregate. Can be multiple columns if the *columns*
            arguments contains multiple columns as well
        index
            One or multiple keys to group by
        columns
            Columns whose values will be used as the header of the output DataFrame
        aggregate_fn : {'first', 'sum', 'max', 'min', 'mean', 'median', 'last', 'count'}
            A predefined aggregate function str or an expression.
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
        ...         "foo": ["one", "one", "one", "two", "two", "two"],
        ...         "bar": ["A", "B", "C", "A", "B", "C"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.pivot(values="baz", index="foo", columns="bar")
        shape: (2, 4)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ A   ┆ B   ┆ C   │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╡
        │ one ┆ 1   ┆ 2   ┆ 3   │
        │ two ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        """
        if isinstance(values, str):
            values = [values]
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(aggregate_fn, str):
            if aggregate_fn == "first":
                aggregate_fn = pli.element().first()
            elif aggregate_fn == "sum":
                aggregate_fn = pli.element().sum()
            elif aggregate_fn == "max":
                aggregate_fn = pli.element().max()
            elif aggregate_fn == "min":
                aggregate_fn = pli.element().min()
            elif aggregate_fn == "mean":
                aggregate_fn = pli.element().mean()
            elif aggregate_fn == "median":
                aggregate_fn = pli.element().median()
            elif aggregate_fn == "last":
                aggregate_fn = pli.element().last()
            elif aggregate_fn == "count":
                aggregate_fn = pli.count()
            else:
                raise ValueError(
                    f"Argument aggregate fn: '{aggregate_fn}' " f"was not expected."
                )

        return self._from_pydf(
            self._df.pivot_expr(
                values,
                index,
                columns,
                aggregate_fn._pyexpr,
                maintain_order,
                sort_columns,
                separator,
            )
        )

    def melt(
        self,
        id_vars: Sequence[str] | str | None = None,
        value_vars: Sequence[str] | str | None = None,
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
        ... )
        >>> df.melt(id_vars="a", value_vars=["b", "c"])
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
        return self._from_pydf(
            self._df.melt(id_vars, value_vars, value_name, variable_name)
        )

    def unstack(
        self,
        step: int,
        how: UnstackDirection = "vertical",
        columns: str | Sequence[str] | None = None,
        fill_values: list[Any] | None = None,
    ) -> Self:
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
            Column to include in the operation.
        fill_values
            Fill values that don't fit the new size with this value.

        Examples
        --------
        >>> from string import ascii_uppercase
        >>> df = pl.DataFrame(
        ...     {
        ...         "col1": list(ascii_uppercase[0:9]),
        ...         "col2": pl.arange(0, 9, eager=True),
        ...     }
        ... )
        >>> df
        shape: (9, 2)
        ┌──────┬──────┐
        │ col1 ┆ col2 │
        │ ---  ┆ ---  │
        │ str  ┆ i64  │
        ╞══════╪══════╡
        │ A    ┆ 0    │
        │ B    ┆ 1    │
        │ C    ┆ 2    │
        │ D    ┆ 3    │
        │ ...  ┆ ...  │
        │ F    ┆ 5    │
        │ G    ┆ 6    │
        │ H    ┆ 7    │
        │ I    ┆ 8    │
        └──────┴──────┘
        >>> df.unstack(step=3, how="vertical")
        shape: (3, 6)
        ┌────────┬────────┬────────┬────────┬────────┬────────┐
        │ col1_0 ┆ col1_1 ┆ col1_2 ┆ col2_0 ┆ col2_1 ┆ col2_2 │
        │ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    │
        │ str    ┆ str    ┆ str    ┆ i64    ┆ i64    ┆ i64    │
        ╞════════╪════════╪════════╪════════╪════════╪════════╡
        │ A      ┆ D      ┆ G      ┆ 0      ┆ 3      ┆ 6      │
        │ B      ┆ E      ┆ H      ┆ 1      ┆ 4      ┆ 7      │
        │ C      ┆ F      ┆ I      ┆ 2      ┆ 5      ┆ 8      │
        └────────┴────────┴────────┴────────┴────────┴────────┘
        >>> df.unstack(step=3, how="horizontal")
        shape: (3, 6)
        ┌────────┬────────┬────────┬────────┬────────┬────────┐
        │ col1_0 ┆ col1_1 ┆ col1_2 ┆ col2_0 ┆ col2_1 ┆ col2_2 │
        │ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    │
        │ str    ┆ str    ┆ str    ┆ i64    ┆ i64    ┆ i64    │
        ╞════════╪════════╪════════╪════════╪════════╪════════╡
        │ A      ┆ B      ┆ C      ┆ 0      ┆ 1      ┆ 2      │
        │ D      ┆ E      ┆ F      ┆ 3      ┆ 4      ┆ 5      │
        │ G      ┆ H      ┆ I      ┆ 6      ┆ 7      ┆ 8      │
        └────────┴────────┴────────┴────────┴────────┴────────┘

        """
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
                fill_values = [fill_values for _ in range(0, df.width)]

            df = df.select(
                [
                    s.extend_constant(next_fill, n_fill)
                    for s, next_fill in zip(df, fill_values)
                ]
            )

        if how == "horizontal":
            df = (
                df.with_columns(  # type: ignore[assignment]
                    (pli.arange(0, n_cols * n_rows, eager=True) % n_cols).alias(
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
            for slice_nbr in range(0, n_cols)
        ]

        return self._from_pydf(DataFrame(slices)._df)

    @overload
    def partition_by(
        self,
        groups: str | Sequence[str],
        *,
        maintain_order: bool = ...,
        as_dict: Literal[False] = ...,
    ) -> list[Self]:
        ...

    @overload
    def partition_by(
        self,
        groups: str | Sequence[str],
        *,
        maintain_order: bool = ...,
        as_dict: Literal[True],
    ) -> dict[Any, Self]:
        ...

    @overload
    def partition_by(
        self,
        groups: str | Sequence[str],
        *,
        maintain_order: bool,
        as_dict: bool,
    ) -> list[Self] | dict[Any, Self]:
        ...

    def partition_by(
        self,
        groups: str | Sequence[str],
        *,
        maintain_order: bool = True,
        as_dict: bool = False,
    ) -> list[Self] | dict[Any, Self]:
        """
        Split into multiple DataFrames partitioned by groups.

        Parameters
        ----------
        groups
            Groups to partition by.
        maintain_order
            Keep predictable output order. This is slower as it requires an extra sort
            operation.
        as_dict
            If True, return the partitions in a dictionary keyed by the distinct group
            values instead of a list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["A", "A", "B", "B", "C"],
        ...         "N": [1, 2, 2, 4, 2],
        ...         "bar": ["k", "l", "m", "m", "l"],
        ...     }
        ... )
        >>> df.partition_by(groups="foo", maintain_order=True)
        [shape: (2, 3)
         ┌─────┬─────┬─────┐
         │ foo ┆ N   ┆ bar │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ str │
         ╞═════╪═════╪═════╡
         │ A   ┆ 1   ┆ k   │
         │ A   ┆ 2   ┆ l   │
         └─────┴─────┴─────┘,
         shape: (2, 3)
         ┌─────┬─────┬─────┐
         │ foo ┆ N   ┆ bar │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ str │
         ╞═════╪═════╪═════╡
         │ B   ┆ 2   ┆ m   │
         │ B   ┆ 4   ┆ m   │
         └─────┴─────┴─────┘,
         shape: (1, 3)
         ┌─────┬─────┬─────┐
         │ foo ┆ N   ┆ bar │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ str │
         ╞═════╪═════╪═════╡
         │ C   ┆ 2   ┆ l   │
         └─────┴─────┴─────┘]
        >>> df.partition_by(groups="foo", maintain_order=True, as_dict=True)
        {'A': shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ N   ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ A   ┆ 1   ┆ k   │
        │ A   ┆ 2   ┆ l   │
        └─────┴─────┴─────┘, 'B': shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ N   ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ B   ┆ 2   ┆ m   │
        │ B   ┆ 4   ┆ m   │
        └─────┴─────┴─────┘, 'C': shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ N   ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ C   ┆ 2   ┆ l   │
        └─────┴─────┴─────┘}

        """
        if isinstance(groups, str):
            groups = [groups]
        elif not isinstance(groups, list):
            groups = list(groups)

        if as_dict:
            out: dict[Any, Self] = {}
            if len(groups) == 1:
                for _df in self._df.partition_by(groups, maintain_order):
                    df = self._from_pydf(_df)
                    out[df[groups][0, 0]] = df
            else:
                for _df in self._df.partition_by(groups, maintain_order):
                    df = self._from_pydf(_df)
                    out[df[groups].row(0)] = df

            return out

        else:
            return [
                self._from_pydf(_df)
                for _df in self._df.partition_by(groups, maintain_order)
            ]

    def shift(self, periods: int) -> Self:
        """
        Shift values by the given period.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        See Also
        --------
        shift_and_fill

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.shift(periods=1)
        shape: (3, 3)
        ┌──────┬──────┬──────┐
        │ foo  ┆ bar  ┆ ham  │
        │ ---  ┆ ---  ┆ ---  │
        │ i64  ┆ i64  ┆ str  │
        ╞══════╪══════╪══════╡
        │ null ┆ null ┆ null │
        │ 1    ┆ 6    ┆ a    │
        │ 2    ┆ 7    ┆ b    │
        └──────┴──────┴──────┘
        >>> df.shift(periods=-1)
        shape: (3, 3)
        ┌──────┬──────┬──────┐
        │ foo  ┆ bar  ┆ ham  │
        │ ---  ┆ ---  ┆ ---  │
        │ i64  ┆ i64  ┆ str  │
        ╞══════╪══════╪══════╡
        │ 2    ┆ 7    ┆ b    │
        │ 3    ┆ 8    ┆ c    │
        │ null ┆ null ┆ null │
        └──────┴──────┴──────┘

        """
        return self._from_pydf(self._df.shift(periods))

    def shift_and_fill(self, periods: int, fill_value: int | str | float) -> DataFrame:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with this value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.shift_and_fill(periods=1, fill_value=0)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 0   ┆ 0   ┆ 0   │
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        return (
            self.lazy()
            .shift_and_fill(periods, fill_value)
            .collect(no_optimization=True)
        )

    def is_duplicated(self) -> pli.Series:
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
        return pli.wrap_s(self._df.is_duplicated())

    def is_unique(self) -> pli.Series:
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
        return pli.wrap_s(self._df.is_unique())

    def lazy(self) -> pli.LazyFrame:
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
        <polars.LazyFrame object at ...>

        """
        return pli.wrap_ldf(self._df.lazy())

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
        by enabling the experimental setting ``Config.set_auto_structify(True)``:

        >>> from polars.datatypes import INTEGER_DTYPES
        >>> with pl.Config() as cfg:
        ...     cfg.set_auto_structify(True)  # doctest: +IGNORE_RESULT
        ...     df.select(
        ...         is_odd=(pl.col(INTEGER_DTYPES) % 2).suffix("_is_odd"),
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
        return self._from_pydf(
            self.lazy()
            .select(exprs, *more_exprs, **named_exprs)
            .collect(no_optimization=True)
            ._df
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
    ) -> DataFrame:
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
        ...         (pl.col("c").is_not()).alias("not c"),
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
        ...     (pl.col("c").is_not()).alias("not c"),
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
        ...     not_c=pl.col("c").is_not(),
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
        by enabling the experimental setting ``Config.set_auto_structify(True)``:

        >>> with pl.Config() as cfg:
        ...     cfg.set_auto_structify(True)  # doctest: +IGNORE_RESULT
        ...     df.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().suffix("_diff"),
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
        return (
            self.lazy()
            .with_columns(exprs, *more_exprs, **named_exprs)
            .collect(no_optimization=True)
        )

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
                f"Strategy: '{strategy}' not understood. "
                f"Choose one of {{'first',  'all'}}"
            )

    @overload
    def max(self, axis: Literal[0] = ...) -> Self:
        ...

    @overload
    def max(self, axis: Literal[1]) -> pli.Series:
        ...

    @overload
    def max(self, axis: int = 0) -> Self | pli.Series:
        ...

    def max(self, axis: int = 0) -> Self | pli.Series:
        """
        Aggregate the columns of this DataFrame to their maximum value.

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
        if axis == 0:
            return self._from_pydf(self._df.max())
        if axis == 1:
            return pli.wrap_s(self._df.hmax())
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    @overload
    def min(self, axis: Literal[0] = ...) -> Self:
        ...

    @overload
    def min(self, axis: Literal[1]) -> pli.Series:
        ...

    @overload
    def min(self, axis: int = 0) -> Self | pli.Series:
        ...

    def min(self, axis: int = 0) -> Self | pli.Series:
        """
        Aggregate the columns of this DataFrame to their minimum value.

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
        if axis == 0:
            return self._from_pydf(self._df.min())
        if axis == 1:
            return pli.wrap_s(self._df.hmin())
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

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
    ) -> pli.Series:
        ...

    @overload
    def sum(
        self,
        *,
        axis: int = 0,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | pli.Series:
        ...

    def sum(
        self,
        *,
        axis: int = 0,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | pli.Series:
        """
        Aggregate the columns of this DataFrame to their sum value.

        Parameters
        ----------
        axis
            Either 0 or 1.
        null_strategy : {'ignore', 'propagate'}
            This argument is only used if axis == 1.

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
        >>> df.sum(axis=1)
        shape: (3,)
        Series: 'foo' [str]
        [
                "16a"
                "27b"
                "38c"
        ]

        """
        if axis == 0:
            return self._from_pydf(self._df.sum())
        if axis == 1:
            return pli.wrap_s(self._df.hsum(null_strategy))
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

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
    ) -> pli.Series:
        ...

    @overload
    def mean(
        self,
        *,
        axis: int = 0,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | pli.Series:
        ...

    def mean(
        self,
        axis: int = 0,
        null_strategy: NullStrategy = "ignore",
    ) -> Self | pli.Series:
        """
        Aggregate the columns of this DataFrame to their mean value.

        Parameters
        ----------
        axis
            Either 0 or 1.
        null_strategy : {'ignore', 'propagate'}
            This argument is only used if axis == 1.

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
        >>> df.mean(axis=1)
        shape: (3,)
        Series: 'foo' [f64]
        [
            2.666667
            3.0
            5.5
        ]

        """
        if axis == 0:
            return self._from_pydf(self._df.mean())
        if axis == 1:
            return pli.wrap_s(self._df.hmean(null_strategy))
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    def std(self, ddof: int = 1) -> Self:
        """
        Aggregate the columns of this DataFrame to their standard deviation value.

        Parameters
        ----------
        ddof
            Degrees of freedom

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
            Degrees of freedom

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

    def product(self) -> Self:
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
        return self.select(pli.all().product())

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
        self, columns: Sequence[str] | None = None, *, separator: str = "_"
    ) -> Self:
        """
        Get one hot encoded dummy variables.

        Parameters
        ----------
        columns:
            A subset of columns to convert to dummy variables. ``None`` means
            "all columns".
        separator
            Separator/delimiter used when generating column names.

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

        """
        if isinstance(columns, str):
            columns = [columns]
        return self._from_pydf(self._df.to_dummies(columns, separator))

    @deprecate_nonkeyword_arguments(
        message=(
            "All arguments of DataFrame.unique except for 'subset' will be keyword-only in the next breaking release."
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
        ... )
        >>> df.unique()
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
        >>> df.unique(subset=["bar", "ham"])
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> df.unique(keep="last")
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

        return self._from_pydf(self._df.unique(maintain_order, subset, keep))

    def n_unique(
        self, subset: str | pli.Expr | Sequence[str | pli.Expr] | None = None
    ) -> int:
        """
        Return the number of unique rows, or the number of unique row-subsets.

        Parameters
        ----------
        subset
            One or more columns/expressions that define what to count;
            omit to return the count of unique rows.

        Notes
        -----
        This method operates at the ``DataFrame`` level; to operate on subsets at the
        expression level you can make use of struct-packing instead, for example:

        >>> expr_unique_subset = pl.struct(["a", "b"]).n_unique()

        If instead you want to count the number of unique values per-column, you can
        also use expression-level syntax to return a new frame containing that result:

        >>> df = pl.DataFrame([[1, 2, 3], [1, 2, 4]], schema=["a", "b", "c"])
        >>> df_nunique = df.select(pl.all().n_unique())

        In aggregate context there is also an equivalent method for returning the
        unique values per-group:

        >>> df_agg_nunique = df.groupby(by=["a"]).n_unique()

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
        >>> # simple columns subset
        >>> df.n_unique(subset=["b", "c"])
        4
        >>> # expression subset
        >>> df.n_unique(
        ...     subset=[
        ...         (pl.col("a") // 2),
        ...         (pl.col("c") | (pl.col("b") >= 2)),
        ...     ],
        ... )
        3

        """
        if isinstance(subset, str):
            subset = [pli.col(subset)]
        elif isinstance(subset, pli.Expr):
            subset = [subset]

        if isinstance(subset, Sequence) and len(subset) == 1:
            expr = pli.expr_to_lit_or_expr(subset[0], str_to_lit=False)
        else:
            struct_fields = pli.all() if (subset is None) else subset
            expr = pli.struct(struct_fields)  # type: ignore[call-overload]

        df = self.lazy().select(expr.n_unique()).collect()
        return 0 if df.is_empty() else df.row(0)[0]

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
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Self:
        """
        Sample from this DataFrame.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `frac`. Defaults to 1 if
            `frac` is None.
        frac
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

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
        if n is not None and frac is not None:
            raise ValueError("cannot specify both `n` and `frac`")

        if seed is None:
            seed = random.randint(0, 10000)

        if n is None and frac is not None:
            return self._from_pydf(
                self._df.sample_frac(frac, with_replacement, shuffle, seed)
            )

        if n is None:
            n = 1
        return self._from_pydf(self._df.sample_n(n, with_replacement, shuffle, seed))

    def fold(
        self, operation: Callable[[pli.Series, pli.Series], pli.Series]
    ) -> pli.Series:
        """
        Apply a horizontal reduction on a DataFrame.

        This can be used to effectively determine aggregations on a row level, and can
        be applied to any DataType that can be supercasted (casted to a similar parent
        type).

        An example of the supercast rules when applying an arithmetic operation on two
        DataTypes are for instance:

        Int8 + Utf8 = Utf8
        Float32 + Int64 = Float32
        Float32 + Float64 = Float64

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
        by_predicate: pli.Expr | None = ...,
        named: Literal[False] = ...,
    ) -> tuple[Any, ...]:
        ...

    @overload
    def row(
        self,
        index: int | None = ...,
        *,
        by_predicate: pli.Expr | None = ...,
        named: Literal[True],
    ) -> dict[str, Any]:
        ...

    def row(
        self,
        index: int | None = None,
        *,
        by_predicate: pli.Expr | None = None,
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
        Tuple (default) or dictionary of row values.

        Notes
        -----
        The ``index`` and ``by_predicate`` params are mutually exclusive. Additionally,
        to ensure clarity, the `by_predicate` parameter must be supplied by keyword.

        When using ``by_predicate`` it is an error condition if anything other than
        one row is returned; more than one row raises ``TooManyRowsReturned``, and
        zero rows will raise ``NoRowsReturned`` (both inherit from ``RowsException``).

        Warning
        -------
        You should NEVER use this method to iterate over a DataFrame; if you absolutely
        require row-iteration you should strongly prefer ``iter_rows()`` instead.

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

        Specify ``named=True`` to get a dictionary instead with a mapping of column
        names to row values.

        >>> df.row(2, named=True)
        {'foo': 3, 'bar': 8, 'ham': 'c'}

        Use ``by_predicate`` to return the row that matches the given predicate.

        >>> df.row(by_predicate=(pl.col("ham") == "b"))
        (2, 7, 'b')

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        rows : Materialises all frame data as a list of rows.

        """
        if index is not None and by_predicate is not None:
            raise ValueError(
                "Cannot set both 'index' and 'by_predicate'; mutually exclusive"
            )
        elif isinstance(index, pli.Expr):
            raise TypeError("Expressions should be passed to the 'by_predicate' param")

        if index is not None:
            row = self._df.row_tuple(index)
            if named:
                return dict(zip(self.columns, row))
            else:
                return row

        elif by_predicate is not None:
            if not isinstance(by_predicate, pli.Expr):
                raise TypeError(
                    f"Expected 'by_predicate to be an expression; "
                    f"found {type(by_predicate)}"
                )
            rows = self.filter(by_predicate).rows()
            n_rows = len(rows)
            if n_rows > 1:
                raise TooManyRowsReturned(
                    f"Predicate <{by_predicate!s}> returned {n_rows} rows"
                )
            elif n_rows == 0:
                raise NoRowsReturned(f"Predicate <{by_predicate!s}> returned no rows")

            row = rows[0]
            if named:
                return dict(zip(self.columns, row))
            else:
                return row
        else:
            raise ValueError("One of 'index' or 'by_predicate' must be set")

    @overload
    def rows(self, named: Literal[False] = ...) -> list[tuple[Any, ...]]:
        ...

    @overload
    def rows(self, named: Literal[True]) -> list[dict[str, Any]]:
        ...

    def rows(self, named: bool = False) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
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
        If you have ``ns``-precision temporal values you should be aware that python
        natively only supports up to ``us``-precision; if this matters you should export
        to a different format.

        Warnings
        --------
        Row-iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods.

        Returns
        -------
        A list of tuples (default) or dictionaries of row values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.rows()
        [(1, 2), (3, 4), (5, 6)]
        >>> df.rows(named=True)
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).

        """
        if named:
            # Load these into the local namespace for a minor performance boost
            dict_, zip_, columns = dict, zip, self.columns
            return [dict_(zip_(columns, row)) for row in self._df.row_tuples()]
        else:
            return self._df.row_tuples()

    @overload
    def iter_rows(
        self, named: Literal[False] = ..., buffer_size: int = ...
    ) -> Iterator[tuple[Any, ...]]:
        ...

    @overload
    def iter_rows(
        self, named: Literal[True], buffer_size: int = ...
    ) -> Iterator[dict[str, Any]]:
        ...

    def iter_rows(
        self, named: bool = False, buffer_size: int = 500
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
            value to zero disables row buffering.

        Notes
        -----
        If you have ``ns``-precision temporal values you should be aware that python
        natively only supports up to ``us``-precision; if this matters you should export
        to a different format.

        Warnings
        --------
        Row iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods.

        Returns
        -------
        An iterator of tuples (default) or dictionaries of python row values.

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

        See Also
        --------
        rows : Materialises all frame data as a list of rows.

        """
        # load into the local namespace for a modest performance boost in the hot loops
        columns, get_row, dict_, zip_ = self.columns, self.row, dict, zip

        # note: buffering rows results in a 2-4x speedup over individual calls
        # to ".row(i)", so it should only be disabled in extremely specific cases.
        if buffer_size:
            load_pyarrow_dicts = (
                named
                and _PYARROW_AVAILABLE
                # note: 'ns' precision instantiates values as pandas types - avoid
                and not any((getattr(tp, "tu", None) == "ns") for tp in self.dtypes)
            )
            for offset in range(0, self.height, buffer_size):
                zerocopy_slice = self.slice(offset, buffer_size)
                if load_pyarrow_dicts:
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

        Using ``iter_slices`` is an efficient way to chunk-iterate over DataFrames and
        any supported frame export/conversion types; for example, as RecordBatches:

        >>> for frame in df.iter_slices(n_rows=15_000):
        ...     record_batch = frame.to_arrow().to_batches()[0]
        ...     print(record_batch, "\n<< ", len(record_batch))
        ...
        pyarrow.RecordBatch
        a: int32
        b: date32[day]
        c: large_string
        << 15000
        pyarrow.RecordBatch
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

    def shrink_to_fit(self, in_place: bool = False) -> Self:
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

    def take_every(self, n: int) -> Self:
        """
        Take every nth row in the DataFrame and return as a new DataFrame.

        Examples
        --------
        >>> s = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        >>> s.take_every(2)
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

    def hash_rows(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> pli.Series:
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
        return pli.wrap_s(self._df.hash_rows(k0, k1, k2, k3))

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
        ... )
        >>> df.interpolate()
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

    def to_struct(self, name: str) -> pli.Series:
        """
        Convert a ``DataFrame`` to a ``Series`` of type ``Struct``.

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
        return pli.wrap_s(self._df.to_struct(name))

    def unnest(self, names: str | Sequence[str]) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "before": ["foo", "bar"],
        ...         "t_a": [1, 2],
        ...         "t_b": ["a", "b"],
        ...         "t_c": [True, None],
        ...         "t_d": [[1, 2], [3]],
        ...         "after": ["baz", "womp"],
        ...     }
        ... ).select(["before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after"])
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
        if isinstance(names, str):
            names = [names]
        return self._from_pydf(self._df.unnest(names))

    @typing.no_type_check
    def pearson_corr(self, **kwargs: dict[str, Any]) -> DataFrame:
        """
        Return Pearson product-moment correlation coefficients.

        See numpy corrcoef for more information.

        Notes
        -----
        This functionality requires numpy to be installed.

        Parameters
        ----------
        kwargs
            keyword arguments are passed to numpy corrcoef

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [3, 2, 1], "ham": [7, 8, 9]})
        >>> df.pearson_corr()
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
        return DataFrame(
            np.corrcoef(self, **kwargs),
            schema=self.columns,
        )

    def merge_sorted(self, other: DataFrame, key: str) -> Self:
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

        """
        return self._from_pydf(
            self.lazy()
            .merge_sorted(other.lazy(), key)
            .collect(no_optimization=True)
            ._df
        )

    def update(
        self, other: DataFrame, on: None | str | Sequence[str] = None, how: str = "left"
    ) -> DataFrame:
        """
        Update the values in this `DataFrame` with the non-null values in `other`.

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
            DataFrame that will be used to update the values
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
        return self.lazy().update(other.lazy(), on, how).collect(no_optimization=True)


def _prepare_other_arg(other: Any, length: int | None = None) -> pli.Series:
    # if not a series create singleton series such that it will broadcast
    value = other
    if not isinstance(other, pli.Series):
        if isinstance(other, str):
            pass
        elif isinstance(other, Sequence):
            raise ValueError("Operation not supported.")
        other = pli.Series("", [other])

    if length and length > 1:
        other = other.extend_constant(value=value, n=length - 1)

    return other

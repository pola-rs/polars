"""Module containing logic related to eager DataFrames."""
from __future__ import annotations

import os
import sys
import warnings
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TextIO,
    TypeVar,
    overload,
)

from polars import internals as pli
from polars._html import NotebookFormatter
from polars.datatypes import (
    Boolean,
    ColumnsType,
    DataType,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    get_idx_type,
    py_type_to_dtype,
)
from polars.internals.construction import (
    arrow_to_pydf,
    dict_to_pydf,
    numpy_to_pydf,
    pandas_to_pydf,
    sequence_to_pydf,
    series_to_pydf,
)
from polars.internals.slice import PolarsSlice
from polars.utils import (
    _prepare_row_count_args,
    _process_null_values,
    deprecated_alias,
    format_path,
    handle_projection_columns,
    is_bool_sequence,
    is_int_sequence,
    is_str_sequence,
    range_to_slice,
)

from .lazy_frame import LazyFrame  # noqa: F401

try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pyarrow as pa

    # do not remove these
    import pyarrow.compute
    import pyarrow.parquet

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# A type variable used to refer to a polars.DataFrame or any subclass of it.
# Used to annotate DataFrame methods which returns the same type as self.
DF = TypeVar("DF", bound="DataFrame")

if TYPE_CHECKING:
    # these aliases are used to annotate DataFrame.__getitem__()
    # MultiRowSelector indexes into the vertical axis and
    # MultiColSelector indexes into the horizontal axis
    # NOTE: wrapping these as strings is necessary for Python <3.10
    MultiRowSelector: TypeAlias = "slice | range | list[int] | list[bool] | pli.Series"
    MultiColSelector: TypeAlias = (
        "slice | range | list[int] | list[bool] | list[str] | pli.Series"
    )


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame._from_pydf(df)


def _prepare_other_arg(other: Any) -> pli.Series:
    # if not a series create singleton series such that it will broadcast
    if not isinstance(other, pli.Series):
        if isinstance(other, str):
            pass
        elif isinstance(other, Sequence):
            raise ValueError("Operation not supported.")

        other = pli.Series("", [other])
    return other


class DataFrameMetaClass(type):
    """
    Custom metaclass for DataFrame class.

    This metaclass is responsible for constructing the relationship between the
    DataFrame class and the LazyFrame class. Originally, without inheritance, the
    relationship is as follows:

    DataFrame <-> LazyFrame

    This two-way relationship is represented by the following pointers:
        - cls._lazyframe_class: A pointer on the DataFrame (sub)class to a LazyFrame
            (sub)class. This class property can be used in DataFrame methods in order
            to construct new lazy dataframes.
        - cls._lazyframe_class._dataframe_class: A pointer on the LazyFrame (sub)class
            back to the original DataFrame (sub)class. This allows LazyFrame methods to
            construct new non-lazy dataframes with the correct type. This pointer should
            always be set to cls such that the following is always `True`:
                `type(cls) is type(cls.lazy().collect())`.

    If an end user subclasses DataFrame like so:

    >>> class MyDataFrame(pl.DataFrame):
    ...     pass
    ...

    Then the following class is dynamically created by the metaclass and saved on the
    class variable `MyDataFrame._lazyframe_class`.

    >>> class LazyMyDataFrame(pl.DataFrame):
    ...     _dataframe_class = MyDataFrame
    ...

    If an end user needs to extend both `DataFrame` and `LazyFrame`, it can be done like
    so:

    >>> class MyLazyFrame(pl.LazyFrame):
    ...     @classmethod
    ...     @property
    ...     def _dataframe_class(cls):
    ...         return MyDataFrame
    ...

    >>> class MyDataFrame(pl.DataFrame):
    ...     _lazyframe_class = MyLazyFrame
    ...

    """

    def __init__(cls, name: str, bases: tuple, clsdict: dict) -> None:
        """Construct new DataFrame class."""
        if not bases:
            # This is not a subclass of DataFrame and we can simply hard-link to
            # LazyFrame instead of dynamically defining a new subclass of LazyFrame.
            cls._lazyframe_class = LazyFrame
        elif cls._lazyframe_class is LazyFrame:
            # This is a subclass of DataFrame which has *not* specified a custom
            # LazyFrame subclass by setting `cls._lazyframe_class`. We must therefore
            # dynamically create a subclass of LazyFrame with `_dataframe_class` set
            # to `cls` in order to preserve types after `.lazy().collect()` roundtrips.
            cls._lazyframe_class = type(  # type: ignore[assignment]
                f"Lazy{name}",
                (LazyFrame,),
                {"_dataframe_class": cls},
            )
        super().__init__(name, bases, clsdict)


class DataFrame(metaclass=DataFrameMetaClass):
    """
    A DataFrame is a two-dimensional data structure that represents data as a table
    with rows and columns.

    Parameters
    ----------
    data : dict, Sequence, ndarray, Series, or pandas.DataFrame
        Two-dimensional data in various forms. dict must contain Sequences.
        Sequence may contain Series or other Sequences.
    columns : Sequence of str or (str,DataType) pairs, default None
        Column labels to use for resulting DataFrame. If specified, overrides any
        labels already present in the data. Must match data dimensions.
    orient : {'col', 'row'}, default None
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is inferred by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.

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
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   │
    └─────┴─────┘

    Notice that the dtype is automatically inferred as a polars Int64:

    >>> df.dtypes
    [<class 'polars.datatypes.Int64'>, <class 'polars.datatypes.Int64'>]

    In order to specify dtypes for your columns, initialize the DataFrame with a list
    of typed Series:

    >>> data = [
    ...     pl.Series("col1", [1, 2], dtype=pl.Float32),
    ...     pl.Series("col2", [3, 4], dtype=pl.Int64),
    ... ]
    >>> df2 = pl.DataFrame(data)
    >>> df2
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    Or set the `columns` parameter with a list of (name,dtype) pairs (compatible with
    all of the other valid data parameter types):

    >>> data = {"col1": [1, 2], "col2": [3, 4]}
    >>> df3 = pl.DataFrame(data, columns=[("col1", pl.Float32), ("col2", pl.Int64)])
    >>> df3
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    Constructing a DataFrame from a numpy ndarray, specifying column names:

    >>> import numpy as np
    >>> data = np.array([(1, 2), (3, 4)], dtype=np.int64)
    >>> df4 = pl.DataFrame(data, columns=["a", "b"], orient="col")
    >>> df4
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   │
    └─────┴─────┘

    Constructing a DataFrame from a list of lists, row orientation inferred:

    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> df4 = pl.DataFrame(data, columns=["a", "b", "c"])
    >>> df4
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┘

    """

    def __init__(
        self,
        data: (
            dict[str, Sequence[Any]]
            | Sequence[Any]
            | np.ndarray
            | pa.Table
            | pd.DataFrame
            | pli.Series
            | None
        ) = None,
        columns: ColumnsType | None = None,
        orient: Literal["col", "row"] | None = None,
    ):
        if data is None:
            self._df = dict_to_pydf({}, columns=columns)

        elif isinstance(data, dict):
            self._df = dict_to_pydf(data, columns=columns)

        elif _NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            self._df = numpy_to_pydf(data, columns=columns, orient=orient)

        elif _PYARROW_AVAILABLE and isinstance(data, pa.Table):
            self._df = arrow_to_pydf(data, columns=columns)

        elif isinstance(data, Sequence) and not isinstance(data, str):
            self._df = sequence_to_pydf(data, columns=columns, orient=orient)

        elif isinstance(data, pli.Series):
            self._df = series_to_pydf(data, columns=columns)

        elif _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            if not _PYARROW_AVAILABLE:  # pragma: no cover
                raise ImportError(
                    "'pyarrow' is required for converting a pandas DataFrame to a"
                    " polars DataFrame."
                )
            self._df = pandas_to_pydf(data, columns=columns)

        else:
            raise ValueError("DataFrame constructor not called properly.")

    def estimated_size(self) -> int:
        """
        Return an estimation of the total (heap) allocated size of the `DataFrame` in
        bytes.

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, [`StructArray`]'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        unchanged. However, this function will yield a smaller number. This is because
        this function returns the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.
        """
        return self._df.estimated_size()

    @classmethod
    def _from_pydf(cls: type[DF], py_df: PyDataFrame) -> DF:
        """Construct Polars DataFrame from FFI PyDataFrame object."""
        df = cls.__new__(cls)
        df._df = py_df
        return df

    @classmethod
    def _from_dicts(
        cls: type[DF],
        data: Sequence[dict[str, Any]],
        infer_schema_length: int | None = 100,
    ) -> DF:
        pydf = PyDataFrame.read_dicts(data, infer_schema_length)
        return cls._from_pydf(pydf)

    @classmethod
    def _from_dict(
        cls: type[DF],
        data: dict[str, Sequence[Any]],
        columns: Sequence[str] | None = None,
    ) -> DF:
        """
        Construct a DataFrame from a dictionary of sequences.

        Parameters
        ----------
        data : dict of sequences
            Two-dimensional data represented as a dictionary. dict must contain
            Sequences.
        columns : Sequence of str, default None
            Column labels to use for resulting DataFrame. If specified, overrides any
            labels already present in the data. Must match data dimensions.

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(dict_to_pydf(data, columns=columns))

    @classmethod
    def _from_records(
        cls: type[DF],
        data: Sequence[Sequence[Any]],
        columns: Sequence[str] | None = None,
        orient: Literal["col", "row"] | None = None,
    ) -> DF:
        """
        Construct a DataFrame from a sequence of sequences.

        Parameters
        ----------
        data : Sequence of sequences
            Two-dimensional data represented as a sequence of sequences.
        columns : Sequence of str, default None
            Column labels to use for resulting DataFrame. Must match data dimensions.
            If not specified, columns will be named `column_0`, `column_1`, etc.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows. If None,
            the orientation is inferred by matching the columns and data dimensions. If
            this does not yield conclusive results, column orientation is used.

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(sequence_to_pydf(data, columns=columns, orient=orient))

    @classmethod
    def _from_numpy(
        cls: type[DF],
        data: np.ndarray,
        columns: Sequence[str] | None = None,
        orient: Literal["col", "row"] | None = None,
    ) -> DF:
        """
        Construct a DataFrame from a numpy ndarray.

        Parameters
        ----------
        data : numpy ndarray
            Two-dimensional data represented as a numpy ndarray.
        columns : Sequence of str, default None
            Column labels to use for resulting DataFrame. Must match data dimensions.
            If not specified, columns will be named `column_0`, `column_1`, etc.
        orient : {'col', 'row'}, default None
            Whether to interpret two-dimensional data as columns or as rows. If None,
            the orientation is inferred by matching the columns and data dimensions. If
            this does not yield conclusive results, column orientation is used.

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(numpy_to_pydf(data, columns=columns, orient=orient))

    @classmethod
    def _from_arrow(
        cls: type[DF],
        data: pa.Table,
        columns: Sequence[str] | None = None,
        rechunk: bool = True,
    ) -> DF:
        """
        Construct a DataFrame from an Arrow table.

        This operation will be zero copy for the most part. Types that are not
        supported by Polars may be cast to the closest supported type.

        Parameters
        ----------
        data : numpy ndarray or Sequence of sequences
            Two-dimensional data represented as Arrow table.
        columns : Sequence of str, default None
            Column labels to use for resulting DataFrame. Must match data dimensions.
            If not specified, existing Array table columns are used, with missing names
            named as `column_0`, `column_1`, etc.
        rechunk : bool, default True
            Make sure that all data is contiguous.

        Returns
        -------
        DataFrame

        """
        return cls._from_pydf(arrow_to_pydf(data, columns=columns, rechunk=rechunk))

    @classmethod
    def _from_pandas(
        cls: type[DF],
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        rechunk: bool = True,
        nan_to_none: bool = True,
    ) -> DF:
        """
        Construct a Polars DataFrame from a pandas DataFrame.

        Parameters
        ----------
        data : pandas DataFrame
            Two-dimensional data represented as a pandas DataFrame.
        columns : Sequence of str, default None
            Column labels to use for resulting DataFrame. If specified, overrides any
            labels already present in the data. Must match data dimensions.
        rechunk : bool, default True
            Make sure that all data is contiguous.
        nan_to_none : bool, default True
            If data contains NaN values PyArrow will convert the NaN to None

        Returns
        -------
        DataFrame

        """
        # path for table without rows that keeps datatype
        if data.shape[0] == 0:
            series = []
            for name in data.columns:
                pd_series = data[name]
                if pd_series.dtype == np.dtype("O"):
                    series.append(pli.Series(name, [], dtype=Utf8))
                else:
                    col = pli.Series(name, pd_series)
                    series.append(pli.Series(name, col))
            return cls(series)

        return cls._from_pydf(
            pandas_to_pydf(
                data, columns=columns, rechunk=rechunk, nan_to_none=nan_to_none
            )
        )

    @classmethod
    def _read_csv(
        cls: type[DF],
        file: str | Path | BinaryIO | bytes,
        has_header: bool = True,
        columns: list[int] | list[str] | None = None,
        sep: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = r'"',
        skip_rows: int = 0,
        dtypes: None | (Mapping[str, type[DataType]] | list[type[DataType]]) = None,
        null_values: str | list[str] | dict[str, str] | None = None,
        ignore_errors: bool = False,
        parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema_length: int | None = 100,
        batch_size: int = 8192,
        n_rows: int | None = None,
        encoding: Literal["utf8", "utf8-lossy"] = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
    ) -> DF:
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
            path = format_path(file)
        else:
            path = None
            if isinstance(file, BytesIO):
                file = file.getvalue()
            if isinstance(file, StringIO):
                file = file.getvalue().encode()

        dtype_list: list[tuple[str, type[DataType]]] | None = None
        dtype_slice: list[type[DataType]] | None = None
        if dtypes is not None:
            if isinstance(dtypes, dict):
                dtype_list = []
                for k, v in dtypes.items():
                    dtype_list.append((k, py_type_to_dtype(v)))
            elif isinstance(dtypes, list):
                dtype_slice = dtypes
            else:
                raise ValueError("dtype arg should be list or dict")

        processed_null_values = _process_null_values(null_values)

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
            elif is_str_sequence(columns, False):
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
            parse_dates,
            skip_rows_after_header,
            _prepare_row_count_args(row_count_name, row_count_offset),
            sample_size=sample_size,
            eol_char=eol_char,
        )
        return self

    @classmethod
    def _read_parquet(
        cls: type[DF],
        file: str | Path | BinaryIO,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
        parallel: str = "auto",
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        low_memory: bool = False,
    ) -> DF:
        """
        Read into a DataFrame from a parquet file.

        See Also
        --------
        read_parquet

        """
        if isinstance(file, (str, Path)):
            file = format_path(file)

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
            elif is_str_sequence(columns, False):
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
        cls: type[DF],
        file: str | Path | BinaryIO,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
    ) -> DF:
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
            file = format_path(file)
        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_avro(file, columns, projection, n_rows)
        return self

    @classmethod
    def _read_ipc(
        cls: type[DF],
        file: str | Path | BinaryIO,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        rechunk: bool = True,
        memory_map: bool = True,
    ) -> DF:
        """
        Read into a DataFrame from Arrow IPC stream format. This is also called the
        Feather (v2) format.

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
            file = format_path(file)

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
            elif is_str_sequence(columns, False):
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
    def _read_json(
        cls: type[DF],
        file: str | Path | IOBase,
        json_lines: bool = False,
    ) -> DF:
        """
        See Also
        --------
        read_json
        """
        if isinstance(file, StringIO):
            file = BytesIO(file.getvalue().encode())
        elif isinstance(file, (str, Path)):
            file = format_path(file)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_json(file, json_lines)
        return self

    def to_arrow(self) -> pa.Table:
        """
        Collect the underlying arrow arrays in an Arrow Table.
        This operation is mostly zero copy.

        Data types that do copy:
            - CategoricalType
        """
        if not _PYARROW_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "'pyarrow' is required for converting a polars DataFrame to an Arrow"
                " Table."
            )
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
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 300      │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ null     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2        │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
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

    @overload
    def to_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: Literal[True],
    ) -> str:
        ...

    @overload
    def to_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: Literal[False] = ...,
    ) -> None:
        ...

    @overload
    def to_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: bool = ...,
    ) -> str | None:
        ...

    def to_json(
        self,
        file: IOBase | str | Path | None = None,
        pretty: bool = False,
        row_oriented: bool = False,
        json_lines: bool = False,
        *,
        to_string: bool = False,
    ) -> str | None:  # pragma: no cover
        """
        .. deprecated:: 0.13.12
            Use :func:`write_json` instead.
        """
        warnings.warn(
            "'to_json' is deprecated. please use 'write_json'", DeprecationWarning
        )
        return self.write_json(
            file, pretty, row_oriented, json_lines, to_string=to_string
        )

    @overload
    def write_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: Literal[True],
    ) -> str:
        ...

    @overload
    def write_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: Literal[False] = ...,
    ) -> None:
        ...

    @overload
    def write_json(
        self,
        file: IOBase | str | Path | None = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: bool = ...,
    ) -> str | None:
        ...

    def write_json(
        self,
        file: IOBase | str | Path | None = None,
        pretty: bool = False,
        row_oriented: bool = False,
        json_lines: bool = False,
        *,
        to_string: bool = False,
    ) -> str | None:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            Write to this file instead of returning a string.
        pretty
            Pretty serialize json.
        row_oriented
            Write to row oriented json. This is slower, but more common.
        json_lines
            Write to Json Lines format
        to_string
            Ignore file argument and return a string.

        """
        if isinstance(file, (str, Path)):
            file = format_path(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if to_string or file is None or to_string_io:
            with BytesIO() as buf:
                self._df.to_json(buf, pretty, row_oriented, json_lines)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._df.to_json(file, pretty, row_oriented, json_lines)
        return None

    def to_pandas(
        self, *args: Any, date_as_object: bool = False, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Cast to a pandas DataFrame.
        This requires that pandas and pyarrow are installed.
        This operation clones data.

        Parameters
        ----------
        args
            Arguments will be sent to pyarrow.Table.to_pandas.
        date_as_object
            Cast dates to objects. If False, convert to datetime64[ns] dtype.
        kwargs
            Arguments will be sent to pyarrow.Table.to_pandas.

        Examples
        --------
        >>> import pandas
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> pandas_df = df.to_pandas()
        >>> type(pandas_df)
        <class 'pandas.core.frame.DataFrame'>

        """
        if not _PYARROW_AVAILABLE:  # pragma: no cover
            raise ImportError("'pyarrow' is required when using to_pandas().")
        record_batches = self._df.to_pandas()
        tbl = pa.Table.from_batches(record_batches)
        return tbl.to_pandas(*args, date_as_object=date_as_object, **kwargs)

    def write_csv(
        self,
        file: TextIO | BytesIO | str | Path | None = None,
        has_header: bool = True,
        sep: str = ",",
        quote: str = '"',
        batch_size: int = 1024,
    ) -> str | None:
        """
        Write Dataframe to comma-separated values file (csv).

        Parameters
        ----------
        file
            File path to which the file should be written.
        has_header
            Whether to include header in the CSV output.
        sep
            Separate CSV fields with this symbol.
        quote
            byte to use as quoting character
        batch_size
            rows that will be processed per thread

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
        if len(quote) > 1:
            raise ValueError("only single byte quote char is allowed")
        if file is None:
            buffer = BytesIO()
            self._df.to_csv(buffer, has_header, ord(sep), ord(quote), batch_size)
            return str(buffer.getvalue(), encoding="utf-8")

        if isinstance(file, (str, Path)):
            file = format_path(file)

        self._df.to_csv(file, has_header, ord(sep), ord(quote), batch_size)
        return None

    def to_csv(
        self,
        file: TextIO | BytesIO | str | Path | None = None,
        has_header: bool = True,
        sep: str = ",",
    ) -> str | None:  # pragma: no cover
        """
        .. deprecated:: 0.13.12
            Use :func:`write_csv` instead.
        """
        warnings.warn(
            "'to_csv' is deprecated. please use 'write_csv'", DeprecationWarning
        )
        return self.write_csv(file, has_header, sep)

    def write_avro(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: Literal["uncompressed", "snappy", "deflate"] = "uncompressed",
    ) -> None:
        """
        Write to Apache Avro file.

        Parameters
        ----------
        file
            File path to which the file should be written.
        compression
            Compression method. Choose one of:
                - "uncompressed"
                - "snappy"
                - "deflate"

        """
        if isinstance(file, (str, Path)):
            file = format_path(file)

        self._df.to_avro(file, compression)

    def to_avro(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: Literal["uncompressed", "snappy", "deflate"] = "uncompressed",
    ) -> None:  # pragma: no cover
        """
        .. deprecated:: 0.13.12
            Use :func:`write_avro` instead.
        """
        warnings.warn(
            "'to_avro' is deprecated. please use 'write_avro'", DeprecationWarning
        )
        return self.write_avro(file, compression)

    def write_ipc(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: Literal["uncompressed", "lz4", "zstd"] | None = "uncompressed",
    ) -> None:
        """
        Write to Arrow IPC binary stream, or a feather file.

        Parameters
        ----------
        file
            File path to which the file should be written.
        compression
            Compression method. Choose one of:
                - "uncompressed"
                - "lz4"
                - "zstd"

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = format_path(file)

        self._df.to_ipc(file, compression)

    def to_ipc(
        self,
        file: BinaryIO | BytesIO | str | Path,
        compression: Literal["uncompressed", "lz4", "zstd"] | None = "uncompressed",
    ) -> None:  # pragma: no cover
        """
        .. deprecated:: 0.13.12
            Use :func:`write_ipc` instead.
        """
        warnings.warn(
            "'to_ipc' is deprecated. please use 'write_ipc'", DeprecationWarning
        )
        return self.write_ipc(file, compression)

    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert every row to a dictionary.

        Note that this is slow.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.to_dicts()
        [{'foo': 1, 'bar': 4}, {'foo': 2, 'bar': 5}, {'foo': 3, 'bar': 6}]

        """
        pydf = self._df
        names = self.columns

        return [
            {k: v for k, v in zip(names, pydf.row_tuple(i))}
            for i in range(0, self.height)
        ]

    def transpose(
        self: DF,
        include_header: bool = False,
        header_name: str = "column",
        column_names: Iterator[str] | Sequence[str] | None = None,
    ) -> DF:
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
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
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

    def write_parquet(
        self,
        file: str | Path | BytesIO,
        *,
        compression: (
            Literal["uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"]
            | str
            | None
        ) = "lz4",
        compression_level: int | None = None,
        statistics: bool = False,
        row_group_size: int | None = None,
        use_pyarrow: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Write the DataFrame to disk in parquet format.

        Parameters
        ----------
        file
            File path to which the file should be written.
        compression
            Compression method. Choose one of:

            - "uncompressed" (not supported by pyarrow)
            - "snappy"
            - "gzip"
            - "lzo"
            - "brotli"
            - "lz4"
            - "zstd"

            The default compression "lz4" (actually lz4raw) has very good performance,
            but may not yet been supported by older readers. If you want more
            compatability guarantees, consider using "snappy".
        compression_level
            The level of compression to use. Higher compression means smaller files on
            disk.

            - "gzip" : min-level: 0, max-level: 10.
            - "brotli" : min-level: 0, max-level: 11.
            - "zstd" : min-level: 1, max-level: 22.
        statistics
            Write statistics to the parquet headers. This requires extra compute.
        row_group_size
            Size of the row groups. If None (default), the chunks of the `DataFrame` are
            used. Writing in smaller chunks may reduce memory pressure and improve
            writing speeds. This argument has no effect if 'pyarrow' is used.
        use_pyarrow
            Use C++ parquet implementation vs rust parquet implementation.
            At the moment C++ supports more features.
        kwargs
            Arguments are passed to ``pyarrow.parquet.write_table``.

        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = format_path(file)

        if use_pyarrow:
            if not _PYARROW_AVAILABLE:  # pragma: no cover
                raise ImportError(
                    "'pyarrow' is required when using"
                    " 'write_parquet(..., use_pyarrow=True)'."
                )

            tbl = self.to_arrow()

            data = {}

            for i, column in enumerate(tbl):
                # extract the name before casting
                if column._name is None:
                    name = f"column_{i}"
                else:
                    name = column._name

                data[name] = column
            tbl = pa.table(data)

            pa.parquet.write_table(
                table=tbl,
                where=file,
                compression=compression,
                write_statistics=statistics,
                **kwargs,
            )
        else:
            self._df.to_parquet(
                file, compression, compression_level, statistics, row_group_size
            )

    def to_parquet(
        self,
        file: str | Path | BytesIO,
        compression: (
            Literal["uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"]
            | str
            | None
        ) = "snappy",
        statistics: bool = False,
        use_pyarrow: bool = False,
        **kwargs: Any,
    ) -> None:  # pragma: no cover
        """
        .. deprecated:: 0.13.12
            Use :func:`write_parquet` instead.
        """
        warnings.warn(
            "'to_parquet' is deprecated. please use 'write_parquet'", DeprecationWarning
        )
        return self.write_parquet(
            file,
            compression=compression,
            statistics=statistics,
            use_pyarrow=use_pyarrow,
            **kwargs,
        )

    def to_numpy(self) -> np.ndarray:
        """
        Convert DataFrame to a 2d numpy array.
        This operation clones data.

        Notes
        -----
        If you're attempting to convert Utf8 to an array you'll need to install
        `pyarrow`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> numpy_array = df.to_numpy()
        >>> type(numpy_array)
        <class 'numpy.ndarray'>

        """
        if not _NUMPY_AVAILABLE:
            raise ImportError("'numpy' is required for this functionality.")
        out = self._df.to_numpy()
        if out is None:
            return np.vstack(
                [self.to_series(i).to_numpy() for i in range(self.width)]
            ).T
        else:
            return out

    def _comp(
        self: DF, other: Any, op: Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]
    ) -> DF:
        """Compare a DataFrame with another object."""
        if isinstance(other, DataFrame):
            return self._compare_to_other_df(other, op)
        else:
            return self._compare_to_non_df(other, op)

    def _compare_to_other_df(
        self: DF,
        other: DataFrame,
        op: Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"],
    ) -> DF:
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

        return combined.select(expr)  # type: ignore[return-value]

    def _compare_to_non_df(
        self: DF,
        other: Any,
        op: Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"],
    ) -> DF:
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

    def __eq__(self: DF, other: Any) -> DF:  # type: ignore[override]
        return self._comp(other, "eq")

    def __ne__(self: DF, other: Any) -> DF:  # type: ignore[override]
        return self._comp(other, "neq")

    def __gt__(self: DF, other: Any) -> DF:
        return self._comp(other, "gt")

    def __lt__(self: DF, other: Any) -> DF:
        return self._comp(other, "lt")

    def __ge__(self: DF, other: Any) -> DF:
        return self._comp(other, "gt_eq")

    def __le__(self: DF, other: Any) -> DF:
        return self._comp(other, "lt_eq")

    def __getstate__(self) -> list[pli.Series]:
        return self.get_columns()

    def __setstate__(self, state) -> None:  # type: ignore[no-untyped-def]
        self._df = DataFrame(state)._df

    def __mul__(self: DF, other: DataFrame | pli.Series | int | float | bool) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.mul_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.mul(other._s))

    def __truediv__(self: DF, other: DataFrame | pli.Series | int | float | bool) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.div_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.div(other._s))

    def __add__(
        self: DF,
        other: DataFrame | pli.Series | int | float | bool | str,
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.add_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.add(other._s))

    def __sub__(self: DF, other: DataFrame | pli.Series | int | float | bool) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.sub_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.sub(other._s))

    def __mod__(self: DF, other: DataFrame | pli.Series | int | float | bool) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.rem_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.rem(other._s))

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item: Any) -> PySeries:
        """Access columns as attribute."""
        # it is important that we return an AttributeError here
        # this is used by ipython to check some private
        # `_ipython_canary_method_should_not_exist_`
        # if we return any other error than AttributeError pretty printing
        # will not work in notebooks.
        # See: https://github.com/jupyter/notebook/issues/2014
        if item.startswith("_"):
            raise AttributeError(item)
        try:  # pragma: no cover
            warnings.warn(
                "accessing series as Attribute of a DataFrame is deprecated",
                DeprecationWarning,
            )
            return pli.wrap_s(self._df.column(item))
        except Exception as exc:
            raise AttributeError(item) from exc

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __iter__(self) -> Iterator[Any]:
        return self.get_columns().__iter__()

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

    def _pos_idx(self, idx: int, dim: int) -> int:
        if idx >= 0:
            return idx
        else:
            return self.shape[dim] + idx

    def _pos_idxs(self, idxs: np.ndarray | pli.Series, dim: int) -> pli.Series:
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

        if _NUMPY_AVAILABLE and isinstance(idxs, np.ndarray):
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
    def __getitem__(self: DF, item: str) -> pli.Series:
        ...

    @overload
    def __getitem__(
        self: DF,
        item: int
        | np.ndarray
        | pli.Expr
        | list[pli.Expr]
        | MultiColSelector
        | tuple[int, MultiColSelector]
        | tuple[MultiRowSelector, MultiColSelector],
    ) -> DF:
        ...

    @overload
    def __getitem__(self: DF, item: tuple[MultiRowSelector, int]) -> pli.Series:
        ...

    @overload
    def __getitem__(self: DF, item: tuple[MultiRowSelector, str]) -> pli.Series:
        ...

    @overload
    def __getitem__(self: DF, item: tuple[int, int]) -> Any:
        ...

    @overload
    def __getitem__(self: DF, item: tuple[int, str]) -> Any:
        ...

    def __getitem__(
        self: DF,
        item: (
            str
            | int
            | np.ndarray
            | pli.Expr
            | list[pli.Expr]
            | MultiColSelector
            | tuple[int, MultiColSelector]
            | tuple[MultiRowSelector, MultiColSelector]
            | tuple[MultiRowSelector, int]
            | tuple[MultiRowSelector, str]
            | tuple[int, int]
            | tuple[int, str]
        ),
    ) -> DF | pli.Series:
        """Get item. Does quite a lot. Read the comments."""
        if isinstance(item, pli.Expr):  # pragma: no cover
            warnings.warn(
                "'using expressions in []' is deprecated. please use 'select'",
                DeprecationWarning,
            )
            return self.select(item)
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

                # slice and boolean mask
                # df[:2, [True, False, True]]
                if isinstance(col_selection, (Sequence, pli.Series)):
                    if (
                        isinstance(col_selection[0], bool)
                        or isinstance(col_selection, pli.Series)
                        and col_selection.dtype() == Boolean
                    ):
                        df = self.__getitem__(row_selection)
                        select = []
                        for col, valid in zip(df.columns, col_selection):
                            if valid:
                                select.append(col)
                        return df.select(select)

                # single slice
                # df[:, unknown]
                series = self.__getitem__(col_selection)
                # s[:]
                pli.wrap_s(series[row_selection])

            # df[2, :] (select row as df)
            if isinstance(row_selection, int):
                if isinstance(col_selection, (slice, list)) or (
                    _NUMPY_AVAILABLE and isinstance(col_selection, np.ndarray)
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
            return PolarsSlice(self).apply(item)  # type: ignore[return-value]

        # select rows by numpy mask or index
        # df[np.array([1, 2, 3])]
        # df[np.array([True, False, True])]
        if _NUMPY_AVAILABLE and isinstance(item, np.ndarray):
            if item.ndim != 1:
                raise ValueError("Only a 1D-Numpy array is supported as index.")
            if item.dtype.kind in ("i", "u"):
                # Numpy array with signed or unsigned integers.
                return self._from_pydf(
                    self._df.take_with_series(self._pos_idxs(item, dim=0).inner())
                )
            if isinstance(item[0], str):
                return self._from_pydf(self._df.select(item))
            if item.dtype == bool:
                warnings.warn(
                    "index notation '[]' is deprecated for boolean masks. Consider"
                    " using 'filter'.",
                    DeprecationWarning,
                )
                return self._from_pydf(self._df.filter(pli.Series("", item).inner()))

        if isinstance(item, Sequence):
            if isinstance(item[0], str):
                # select multiple columns
                # df[["foo", "bar"]]
                return self._from_pydf(self._df.select(item))
            elif isinstance(item[0], pli.Expr):
                return self.select(item)
            elif is_bool_sequence(item) or is_int_sequence(item):
                item = pli.Series("", item)  # fall through to next if isinstance

        if isinstance(item, pli.Series):
            dtype = item.dtype
            if dtype == Utf8:
                return self._from_pydf(self._df.select(item))
            if dtype == Boolean:
                return self._from_pydf(self._df.filter(item.inner()))
            if dtype == UInt32:
                return self._from_pydf(self._df.take_with_series(item.inner()))
            if dtype in {UInt8, UInt16, UInt64, Int8, Int16, Int32, Int64}:
                return self._from_pydf(
                    self._df.take_with_series(self._pos_idxs(item, dim=0).inner())
                )

        # if no data has been returned, the operation is not supported
        raise ValueError(
            f"Cannot __getitem__ on DataFrame with item: '{item}'"
            f" of type: '{type(item)}'."
        )

    def __setitem__(
        self, key: str | list | tuple[Any, str | int], value: Any
    ) -> None:  # pragma: no cover
        warnings.warn(
            "setting a DataFrame by indexing is deprecated; Consider using"
            " DataFrame.with_column",
            DeprecationWarning,
        )
        # df["foo"] = series
        if isinstance(key, str):
            try:
                self.replace(key, pli.Series(key, value))
            except Exception:
                self.hstack([pli.Series(key, value)], in_place=True)
        # df[["C", "D"]]
        elif isinstance(key, list):
            # TODO: Use python sequence constructors
            if not _NUMPY_AVAILABLE:
                raise ImportError("'numpy' is required for this functionality.")
            value = np.array(value)
            if value.ndim != 2:
                raise ValueError("can only set multiple columns with 2D matrix")
            if value.shape[1] != len(key):
                raise ValueError(
                    "matrix columns should be equal to list use to determine column"
                    " names"
                )
            for (i, name) in enumerate(key):
                self[name] = value[:, i]

        # df[a, b]
        elif isinstance(key, tuple):
            row_selection, col_selection = key

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

    def _repr_html_(self) -> str:
        """
        Format output data in HTML for display in Jupyter Notebooks.

        Output rows and columns can be modified by setting the following ENVIRONMENT
        variables:

        * POLARS_FMT_MAX_COLS: set the number of columns
        * POLARS_FMT_MAX_ROWS: set the number of rows

        """
        max_cols = int(os.environ.get("POLARS_FMT_MAX_COLS", default=75))
        max_rows = int(os.environ.get("POLARS_FMT_MAX_ROWS", default=25))
        return "\n".join(NotebookFormatter(self, max_cols, max_rows).render())

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

    def reverse(self: DF) -> DF:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ a   ┆ 1   │
        └─────┴─────┘

        """
        return self.select(pli.col("*").reverse())

    def rename(self: DF, mapping: dict[str, str]) -> DF:
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
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2     ┆ 7   ┆ b   │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3     ┆ 8   ┆ c   │
        └───────┴─────┴─────┘

        """
        return self.lazy().rename(mapping).collect(no_optimization=True)

    def insert_at_idx(self, index: int, series: pli.Series) -> None:
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
        >>> df.insert_at_idx(1, s)  # returns None
        >>> df
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ baz ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 97  ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 98  ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
        >>> df
        shape: (4, 4)
        ┌─────┬──────┬───────┬──────┐
        │ a   ┆ b    ┆ c     ┆ d    │
        │ --- ┆ ---  ┆ ---   ┆ ---  │
        │ i64 ┆ f64  ┆ bool  ┆ f64  │
        ╞═════╪══════╪═══════╪══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ -2.5 │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  ┆ 15.0 │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false ┆ 20.5 │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  ┆ 0.0  │
        └─────┴──────┴───────┴──────┘

        """
        if index < 0:
            index = len(self.columns) + index
        self._df.insert_at_idx(index, series._s)

    def filter(self: DF, predicate: pli.Expr) -> DF:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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

        """
        return (
            self.lazy()
            .filter(predicate)
            .collect(no_optimization=True, string_cache=False)
        )

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
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2     ┆ 7      ┆ b      │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
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
    def dtypes(self) -> list[type[DataType]]:
        """
        Get dtypes of columns in DataFrame. Dtypes can also be found in column headers when printing the DataFrame.

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
        [<class 'polars.datatypes.Int64'>, <class 'polars.datatypes.Float64'>, <class 'polars.datatypes.Utf8'>]
        >>> df
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7.0 ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        See Also
        --------
        schema : Returns a {colname:dtype} mapping.

        """  # noqa: E501
        return self._df.dtypes()

    @property
    def schema(self) -> dict[str, type[DataType]]:
        """
        Get a dict[column name, DataType]

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
        {'foo': <class 'polars.datatypes.Int64'>, 'bar': <class 'polars.datatypes.Float64'>, 'ham': <class 'polars.datatypes.Utf8'>}

        """  # noqa: E501
        return dict(zip(self.columns, self.dtypes))

    def describe(self: DF) -> DF:
        """
        Summary statistics for a DataFrame.

        Only summarizes numeric datatypes at the moment and returns nulls for
        non-numeric datatypes.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, 2.8, 3.0],
        ...         "b": [4, 5, 6],
        ...         "c": [True, False, True],
        ...     }
        ... )
        >>> df.describe()
        shape: (5, 4)
        ┌──────────┬──────────┬─────┬──────┐
        │ describe ┆ a        ┆ b   ┆ c    │
        │ ---      ┆ ---      ┆ --- ┆ ---  │
        │ str      ┆ f64      ┆ f64 ┆ f64  │
        ╞══════════╪══════════╪═════╪══════╡
        │ mean     ┆ 2.266667 ┆ 5.0 ┆ null │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ std      ┆ 1.101514 ┆ 1.0 ┆ null │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ min      ┆ 1.0      ┆ 4.0 ┆ 0.0  │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ max      ┆ 3.0      ┆ 6.0 ┆ 1.0  │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ median   ┆ 2.8      ┆ 5.0 ┆ null │
        └──────────┴──────────┴─────┴──────┘

        """

        def describe_cast(self: DF) -> DF:
            columns = []
            for s in self:
                if s.is_numeric() or s.is_boolean():
                    columns.append(s.cast(float))
                else:
                    columns.append(s)
            return self.__class__(columns)

        summary = self._from_pydf(
            pli.concat(
                [
                    describe_cast(self.mean()),
                    describe_cast(self.std()),
                    describe_cast(self.min()),
                    describe_cast(self.max()),
                    describe_cast(self.median()),
                ]
            )._df
        )
        summary.insert_at_idx(
            0, pli.Series("describe", ["mean", "std", "min", "max", "median"])
        )
        return summary

    def replace_at_idx(self, index: int, series: pli.Series) -> None:
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
        >>> x = pl.Series("apple", [10, 20, 30])
        >>> df.replace_at_idx(0, x)
        >>> df
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ apple ┆ bar ┆ ham │
        │ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64 ┆ str │
        ╞═══════╪═════╪═════╡
        │ 10    ┆ 6   ┆ a   │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 20    ┆ 7   ┆ b   │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 30    ┆ 8   ┆ c   │
        └───────┴─────┴─────┘

        """
        if index < 0:
            index = len(self.columns) + index
        self._df.replace_at_idx(index, series._s)

    @overload
    def sort(
        self: DF,
        by: str | pli.Expr | list[str] | list[pli.Expr],
        reverse: bool | list[bool] = ...,
        nulls_last: bool = ...,
        *,
        in_place: Literal[False] = ...,
    ) -> DF:
        ...

    @overload
    def sort(
        self,
        by: str | pli.Expr | list[str] | list[pli.Expr],
        reverse: bool | list[bool] = ...,
        nulls_last: bool = ...,
        *,
        in_place: Literal[True],
    ) -> None:
        ...

    @overload
    def sort(
        self: DF,
        by: str | pli.Expr | list[str] | list[pli.Expr],
        reverse: bool | list[bool] = ...,
        nulls_last: bool = ...,
        *,
        in_place: bool,
    ) -> DF | None:
        ...

    def sort(
        self: DF,
        by: str | pli.Expr | list[str] | list[pli.Expr],
        reverse: bool | list[bool] = False,
        nulls_last: bool = False,
        *,
        in_place: bool = False,
    ) -> DF | None:
        """
        Sort the DataFrame by column.

        Parameters
        ----------
        by
            By which column to sort. Only accepts string.
        reverse
            Reverse/descending sort.
        in_place
            Perform operation in-place.
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7.0 ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7.0 ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 6.0 ┆ a   │
        └─────┴─────┴─────┘

        """
        if type(by) is list or isinstance(by, pli.Expr):
            df = (
                self.lazy()
                .sort(by, reverse, nulls_last)
                .collect(no_optimization=True, string_cache=False)
            )
            if in_place:  # pragma: no cover
                warnings.warn(
                    "in-place sorting is deprecated; please use default sorting",
                    DeprecationWarning,
                )
                self._df = df._df
                return self
            return df
        if in_place:  # pragma: no cover
            warnings.warn(
                "in-place sorting is deprecated; please use default sorting",
                DeprecationWarning,
            )
            self._df.sort_in_place(by, reverse)
            return None
        else:
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

    def replace(self, column: str, new_col: pli.Series) -> None:
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
        >>> df
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 10  ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 20  ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 30  ┆ 6   │
        └─────┴─────┘

        """
        self._df.replace(column, new_col.inner())

    def slice(self: DF, offset: int, length: int | None = None) -> DF:
        """
        Slice this DataFrame over the rows direction.

        Parameters
        ----------
        offset
            Offset index.
        length
            Length of the slice.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        """
        if (length is not None) and length < 0:
            length = self.height - offset + length
        return self._from_pydf(self._df.slice(offset, length))

    def limit(self: DF, length: int = 5) -> DF:
        """
        Get first N rows as DataFrame.

        See Also
        --------
        head, tail, slice

        Parameters
        ----------
        length
            Amount of rows to take.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.limit(2)
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

        """
        return self.head(length)

    def head(self: DF, length: int = 5) -> DF:
        """
        Get first N rows as DataFrame.

        Parameters
        ----------
        length
            Length of the head.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        """
        return self._from_pydf(self._df.head(length))

    def tail(self: DF, length: int = 5) -> DF:
        """
        Get last N rows as DataFrame.

        Parameters
        ----------
        length
            Length of the tail.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 4   ┆ 9   ┆ d   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 5   ┆ 10  ┆ e   │
        └─────┴─────┴─────┘

        """
        return self._from_pydf(self._df.tail(length))

    def drop_nulls(self: DF, subset: str | list[str] | None = None) -> DF:
        """
        Return a new DataFrame where the null values are dropped.

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
        >>> df.drop_nulls()
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

        Drop a column if all values are null:

        >>> df[:, [not (s.null_count() == df.height) for s in df]]
        shape: (4, 2)
        ┌──────┬──────┐
        │ b    ┆ c    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 1    ┆ 1    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2    ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ null │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 1    ┆ 1    │
        └──────┴──────┘

        """
        if isinstance(subset, str):
            subset = [subset]
        return self._from_pydf(self._df.drop_nulls(subset))

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
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]})
        >>> df.pipe(cast_str_to_int, col_name="b")
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

    def with_row_count(self: DF, name: str = "row_nr", offset: int = 0) -> DF:
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
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1      ┆ 3   ┆ 4   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2      ┆ 5   ┆ 6   │
        └────────┴─────┴─────┘

        """
        return self._from_pydf(self._df.with_row_count(name, offset))

    def groupby(
        self: DF,
        by: str | pli.Expr | Sequence[str | pli.Expr],
        maintain_order: bool = False,
    ) -> GroupBy[DF]:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 11  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ b   ┆ 4   ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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
        if not isinstance(maintain_order, bool):
            raise TypeError(
                f"invalid input for groupby arg `maintain_order`: {maintain_order}."
            )
        if isinstance(by, str):
            by = [by]
        return GroupBy(
            self._df,
            by,  # type: ignore[arg-type]
            dataframe_class=self.__class__,
            maintain_order=maintain_order,
        )

    def groupby_rolling(
        self: DF,
        index_column: str,
        period: str,
        offset: str | None = None,
        closed: str = "right",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ) -> RollingGroupBy[DF]:
        """
        Create rolling groups based on a time column (or index value of type Int32,
        Int64).

        Different from a rolling groupby the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals use
        *groupby_dynamic*

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
        closed
            Defines if the window interval is closed or not.
            Any of {"left", "right", "both" "none"}
        by
            Also group by this column/these columns

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
        return RollingGroupBy(self, index_column, period, offset, closed, by)

    def groupby_dynamic(
        self,
        index_column: str,
        every: str,
        period: str | None = None,
        offset: str | None = None,
        truncate: bool = True,
        include_boundaries: bool = False,
        closed: str = "right",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ) -> DynamicGroupBy:
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
        closed
            Defines if the window interval is closed or not.
            Any of {"left", "right", "both" "none"}
        by
            Also group by this column/these columns

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
        │ datetime[ns]        ┆ i64 │
        ╞═════════════════════╪═════╡
        │ 2021-12-16 00:00:00 ┆ 0   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 00:30:00 ┆ 1   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 01:30:00 ┆ 3   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 4   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 02:30:00 ┆ 5   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2021-12-16 03:00:00 ┆ 6   │
        └─────────────────────┴─────┘

        Group by windows of 1 hour starting at 2021-12-16 00:00:00.

        >>> (
        ...     df.groupby_dynamic("time", every="1h").agg(
        ...         [
        ...             pl.col("time").min().alias("time_min"),
        ...             pl.col("time").max().alias("time_max"),
        ...         ]
        ...     )
        ... )
        shape: (4, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ time                ┆ time_min            ┆ time_max            │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╡
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 00:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 00:30:00 ┆ 2021-12-16 01:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 01:30:00 ┆ 2021-12-16 02:00:00 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 02:30:00 ┆ 2021-12-16 03:00:00 │
        └─────────────────────┴─────────────────────┴─────────────────────┘

        The window boundaries can also be added to the aggregation result

        >>> (
        ...     df.groupby_dynamic("time", every="1h", include_boundaries=True).agg(
        ...         [pl.col("time").count().alias("time_count")]
        ...     )
        ... )
        shape: (4, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        ┆ u32        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 00:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        └─────────────────────┴─────────────────────┴─────────────────────┴────────────┘

        When closed="left", should not include right end of interval
        [lower_bound, upper_bound)

        >>> (
        ...     df.groupby_dynamic("time", every="1h", closed="left").agg(
        ...         [
        ...             pl.col("time").count().alias("time_count"),
        ...             pl.col("time").list().alias("time_agg_list"),
        ...         ]
        ...     )
        ... )
        shape: (4, 3)
        ┌─────────────────────┬────────────┬─────────────────────────────────────┐
        │ time                ┆ time_count ┆ time_agg_list                       │
        │ ---                 ┆ ---        ┆ ---                                 │
        │ datetime[ns]        ┆ u32        ┆ list[datetime[ns]]                  │
        ╞═════════════════════╪════════════╪═════════════════════════════════════╡
        │ 2021-12-16 00:00:00 ┆ 2          ┆ [2021-12-16 00:00:00, 2021-12-16... │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2          ┆ [2021-12-16 01:00:00, 2021-12-16... │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 2          ┆ [2021-12-16 02:00:00, 2021-12-16... │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 03:00:00 ┆ 1          ┆ [2021-12-16 03:00:00]               │
        └─────────────────────┴────────────┴─────────────────────────────────────┘

        When closed="both" the time values at the window boundaries belong to 2 groups.

        >>> (
        ...     df.groupby_dynamic("time", every="1h", closed="both").agg(
        ...         [pl.col("time").count().alias("time_count")]
        ...     )
        ... )
        shape: (5, 2)
        ┌─────────────────────┬────────────┐
        │ time                ┆ time_count │
        │ ---                 ┆ ---        │
        │ datetime[ns]        ┆ u32        │
        ╞═════════════════════╪════════════╡
        │ 2021-12-16 00:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 00:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
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
        │ datetime[ns]        ┆ str    │
        ╞═════════════════════╪════════╡
        │ 2021-12-16 00:00:00 ┆ a      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 00:30:00 ┆ a      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ a      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:30:00 ┆ b      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ b      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:30:00 ┆ a      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 03:00:00 ┆ a      │
        └─────────────────────┴────────┘
        >>> (
        ...     df.groupby_dynamic(
        ...         "time",
        ...         every="1h",
        ...         closed="both",
        ...         by="groups",
        ...         include_boundaries=True,
        ...     ).agg([pl.col("time").count().alias("time_count")])
        ... )
        shape: (7, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ str    ┆ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        ┆ u32        │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ a      ┆ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 00:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 04:00:00 ┆ 2021-12-16 03:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ b      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 1          │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴────────────┘


        Dynamic groupby on an index column

        >>> df = pl.DataFrame(
        ...     {
        ...         "idx": np.arange(6),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> (
        ...     df.groupby_dynamic(
        ...         "idx",
        ...         every="2i",
        ...         period="3i",
        ...         include_boundaries=True,
        ...     ).agg(pl.col("A").list().alias("A_agg_list"))
        ... )
        shape: (3, 4)
        ┌─────────────────┬─────────────────┬─────┬─────────────────┐
        │ _lower_boundary ┆ _upper_boundary ┆ idx ┆ A_agg_list      │
        │ ---             ┆ ---             ┆ --- ┆ ---             │
        │ i64             ┆ i64             ┆ i64 ┆ list[str]       │
        ╞═════════════════╪═════════════════╪═════╪═════════════════╡
        │ 0               ┆ 3               ┆ 0   ┆ ["A", "B", "B"] │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2               ┆ 5               ┆ 2   ┆ ["B", "B", "C"] │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4               ┆ 7               ┆ 4   ┆ ["C"]           │
        └─────────────────┴─────────────────┴─────┴─────────────────┘

        """  # noqa: E501
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
        )

    def upsample(
        self: DF,
        time_column: str,
        every: str,
        offset: str | None = None,
        by: str | Sequence[str] | None = None,
        maintain_order: bool = False,
    ) -> DF:
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
        >>> (
        ...     df.upsample(
        ...         time_column="time", every="1mo", by="groups", maintain_order=True
        ...     ).select(pl.all().forward_fill())
        ... )
        shape: (7, 3)
        ┌─────────────────────┬────────┬────────┐
        │ time                ┆ groups ┆ values │
        │ ---                 ┆ ---    ┆ ---    │
        │ datetime[μs]        ┆ str    ┆ i64    │
        ╞═════════════════════╪════════╪════════╡
        │ 2021-02-01 00:00:00 ┆ A      ┆ 0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-03-01 00:00:00 ┆ A      ┆ 0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-04-01 00:00:00 ┆ A      ┆ 0      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-05-01 00:00:00 ┆ A      ┆ 2      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-04-01 00:00:00 ┆ B      ┆ 1      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-05-01 00:00:00 ┆ B      ┆ 1      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2021-06-01 00:00:00 ┆ B      ┆ 3      │
        └─────────────────────┴────────┴────────┘

        """
        if by is None:
            by = []
        if isinstance(by, str):
            by = [by]
        if offset is None:
            offset = "0ns"

        return self._from_pydf(
            self._df.upsample(by, time_column, every, offset, maintain_order)
        )

    @deprecated_alias(df="other")
    def join_asof(
        self: DF,
        other: DataFrame,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | list[str] | None = None,
        by_right: str | list[str] | None = None,
        by: str | list[str] | None = None,
        strategy: str = "backward",
        suffix: str = "_right",
        tolerance: str | int | float | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> DF:
        """
        Perform an asof join. This is similar to a left-join except that we
        match on nearest key rather than equal keys.

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
        strategy
            One of {'forward', 'backward'}
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2017-05-12 00:00:00 ┆ 82.66      ┆ 4411 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2018-05-12 00:00:00 ┆ 83.12      ┆ 4566 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2019-05-12 00:00:00 ┆ 83.52      ┆ 4696 │
        └─────────────────────┴────────────┴──────┘

        """
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

    @deprecated_alias(df="other")
    def join(
        self: DF,
        other: DataFrame,
        left_on: str | pli.Expr | list[str | pli.Expr] | None = None,
        right_on: str | pli.Expr | list[str | pli.Expr] | None = None,
        on: str | pli.Expr | list[str | pli.Expr] | None = None,
        how: str = "inner",
        suffix: str = "_right",
        asof_by: str | list[str] | None = None,
        asof_by_left: str | list[str] | None = None,
        asof_by_right: str | list[str] | None = None,
    ) -> DF:
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
        how
            Join strategy
                - "inner"
                - "left"
                - "outer"
                - "asof"
                - "cross"
                - "semi"
                - "anti"
        suffix
            Suffix to append to columns with a duplicate name.
        asof_by
            join on these columns before doing asof join
        asof_by_left
            join on these columns before doing asof join
        asof_by_right
            join on these columns before doing asof join

        Returns
        -------
            Joined DataFrame

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2    ┆ 7.0  ┆ b   ┆ y     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ null ┆ null ┆ d   ┆ z     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3    ┆ 8.0  ┆ c   ┆ null  │
        └──────┴──────┴─────┴───────┘

        **Asof join**
        This is similar to a left-join except that we match on near keys rather than
        equal keys.
        The direction is backward
        The keys must be sorted to perform an asof join

        **Joining on columns with categorical data**
        See pl.StringCache().

        """
        if how == "asof":  # pragma: no cover
            warnings.warn(
                "using asof join via DataFrame.join is deprecated, please use"
                " DataFrame.join_asof",
                DeprecationWarning,
            )
        if how == "cross":
            return self._from_pydf(self._df.join(other._df, [], [], how, suffix))

        left_on_: list[str | pli.Expr] | None
        if isinstance(left_on, (str, pli.Expr)):
            left_on_ = [left_on]
        else:
            left_on_ = left_on

        right_on_: list[str | pli.Expr] | None
        if isinstance(right_on, (str, pli.Expr)):
            right_on_ = [right_on]
        else:
            right_on_ = right_on

        if isinstance(on, (str, pli.Expr)):
            left_on_ = [on]
            right_on_ = [on]
        elif isinstance(on, list):
            left_on_ = on
            right_on_ = on

        if left_on_ is None or right_on_ is None:
            raise ValueError("You should pass the column to join on as an argument.")

        if (
            isinstance(left_on_[0], pli.Expr)
            or isinstance(right_on_[0], pli.Expr)
            or asof_by_left is not None
            or asof_by_right is not None
            or asof_by is not None
        ):
            return (
                self.lazy()
                .join(
                    other.lazy(),
                    left_on,
                    right_on,
                    on=on,
                    how=how,
                    suffix=suffix,
                    asof_by_right=asof_by_right,
                    asof_by_left=asof_by_left,
                    asof_by=asof_by,
                )
                .collect(no_optimization=True)
            )
        else:
            return self._from_pydf(
                self._df.join(other._df, left_on_, right_on_, how, suffix)
            )

    def apply(
        self: DF,
        f: Callable[[tuple[Any, ...]], Any],
        return_dtype: type[DataType] | None = None,
        inference_size: int = 256,
    ) -> DF:
        """
        Apply a custom function over the rows of the DataFrame. The rows are passed as
        tuple.

        Implementing logic using this .apply method is generally slower and more memory
        intensive than implementing the same logic using the expression API because:

        - with .apply the logic is implemented in Python but with an expression the
          logic is implemented in Rust
        - with .apply the DataFrame is materialized in memory
        - expressions can be parallelised
        - expressions can be optimised

        If possible, use the expression API for best performance.

        Parameters
        ----------
        f
            Custom function/ lambda function.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema

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
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4        ┆ 15       │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 6        ┆ 24       │
        └──────────┴──────────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.select([pl.col("foo") * 2, pl.col("bar") * 3])
        ... )  # doctest: +IGNORE_RESULT

        Return a Series by mapping each row to a scalar:

        >>> df.apply(lambda t: (t[0] * 2 + t[1]))
        shape: (3, 1)
        ┌───────┐
        │ apply │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 1     │
        ├╌╌╌╌╌╌╌┤
        │ 9     │
        ├╌╌╌╌╌╌╌┤
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

    def with_column(self: DF, column: pli.Series | pli.Expr) -> DF:
        """
        Return a new DataFrame with the column added or replaced.

        Parameters
        ----------
        column
            Series, where the name of the Series refers to the column in the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.with_column((pl.col("b") ** 2).alias("b_squared"))  # added
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
        >>> df.with_column(pl.col("a") ** 2)  # replaced
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
        if isinstance(column, list):
            raise ValueError(
                "`with_column` expects a single expression, not a list. Consider using"
                " `with_columns`"
            )
        if isinstance(column, pli.Expr):
            return self.with_columns([column])
        else:
            return self._from_pydf(self._df.with_column(column._s))

    @overload
    def hstack(
        self: DF,
        columns: list[pli.Series] | DataFrame,
        in_place: Literal[False] = False,
    ) -> DF:
        ...

    @overload
    def hstack(
        self: DF,
        columns: list[pli.Series] | DataFrame,
        in_place: Literal[True],
    ) -> None:
        ...

    def hstack(
        self: DF,
        columns: list[pli.Series] | DataFrame,
        in_place: bool = False,
    ) -> DF | None:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   ┆ 20    │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   ┆ 30    │
        └─────┴─────┴─────┴───────┘

        """
        if not isinstance(columns, list):
            columns = columns.get_columns()
        if in_place:
            self._df.hstack_mut([s.inner() for s in columns])
            return None
        else:
            return self._from_pydf(self._df.hstack([s.inner() for s in columns]))

    @overload
    def vstack(self, df: DataFrame, in_place: Literal[True]) -> None:
        ...

    @overload
    def vstack(self: DF, df: DataFrame, in_place: Literal[False] = ...) -> DF:
        ...

    @overload
    def vstack(self: DF, df: DataFrame, in_place: bool) -> DF | None:
        ...

    def vstack(self: DF, df: DataFrame, in_place: bool = False) -> DF | None:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 4   ┆ 9   ┆ d   │
        └─────┴─────┴─────┘

        """
        if in_place:
            self._df.vstack_mut(df._df)
            return None
        else:
            return self._from_pydf(self._df.vstack(df._df))

    def extend(self, other: DataFrame) -> None:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from `vstack` which adds the chunks from `other` to the chunks of this
        `DataFrame` `extent` appends the data from `other` to the underlying memory
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
        >>> df1.extend(df2)  # returns None
        >>> df1
        shape: (6, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 10  ┆ 40  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 20  ┆ 50  │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 30  ┆ 60  │
        └─────┴─────┘

        """
        self._df.extend(other._df)

    def drop(self: DF, name: str | list[str]) -> DF:
        """
        Remove column from DataFrame and return as new.

        Parameters
        ----------
        name
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
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7.0 │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8.0 │
        └─────┴─────┘

        """
        if isinstance(name, list):
            df = self.clone()

            for n in name:
                df._df.drop_in_place(n)
            return df

        return self._from_pydf(self._df.drop(name))

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

    def select_at_idx(self, idx: int) -> pli.Series:
        """
        Select column at index location.

        Parameters
        ----------
        idx
            Location of selection.

        .. deprecated:: 0.10.20

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.select_at_idx(1)
        shape: (3,)
        Series: 'bar' [i64]
        [
                6
                7
                8
        ]

        """
        if idx < 0:
            idx = len(self.columns) + idx
        return pli.wrap_s(self._df.select_at_idx(idx))

    def cleared(self: DF) -> DF:
        """
        Create an empty copy of the current DataFrame, with identical schema but no
        data.

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

    def clone(self: DF) -> DF:
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
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  │
        └─────┴──────┴───────┘

        """
        return self._from_pydf(self._df.clone())

    def __copy__(self: DF) -> DF:
        return self.clone()

    def __deepcopy__(self: DF, memo: None = None) -> DF:
        return self.clone()

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
        self: DF, strategy: str | pli.Expr | Any, limit: int | None = None
    ) -> DF:
        """
        Fill null values using a filling strategy, literal, or Expr.

        .. seealso::

            :func:`fill_nan`

        Parameters
        ----------
        strategy
            One of {"backward", "forward", "min", "max", "mean", "one", "zero"}
            or an expression.
        limit
            The number of consecutive null values to forward/backward fill.
            Only valid if ``strategy`` is 'forward' or 'backward'.

        Returns
        -------
            DataFrame with None values replaced by the filling strategy.

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
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 99  ┆ 99.0 │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        """
        if isinstance(strategy, pli.Expr):
            return self.lazy().fill_null(strategy).collect(no_optimization=True)
        if not isinstance(strategy, str):
            return self.fill_null(pli.lit(strategy))
        return self._from_pydf(self._df.fill_null(strategy, limit))

    def fill_nan(self: DF, fill_value: pli.Expr | int | float) -> DF:
        """
        Fill floating point NaN values by an Expression evaluation.

        .. seealso::

            :func:`fill_null`

        Warnings
        --------
        NOTE that floating point NaNs (Not a Number) are not missing values!
        to replace missing values, use :func:`fill_null`.

        Parameters
        ----------
        fill_value
            Value to fill NaN with.

        Returns
        -------
            DataFrame with NaN replaced with fill_value

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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 2.0  ┆ 4.0  │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 99.0 ┆ 99.0 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 4.0  ┆ 13.0 │
        └──────┴──────┘

        """
        return self.lazy().fill_nan(fill_value).collect(no_optimization=True)

    def explode(
        self: DF,
        columns: str | list[str] | pli.Expr | list[pli.Expr],
    ) -> DF:
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
        ...         "letters": ["c", "c", "a", "c", "a", "b"],
        ...         "nrs": [[1, 2], [1, 3], [4, 3], [5, 5, 5], [6], [2, 1, 2]],
        ...     }
        ... )
        >>> df
        shape: (6, 2)
        ┌─────────┬───────────┐
        │ letters ┆ nrs       │
        │ ---     ┆ ---       │
        │ str     ┆ list[i64] │
        ╞═════════╪═══════════╡
        │ c       ┆ [1, 2]    │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ [1, 3]    │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ [4, 3]    │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ c       ┆ [5, 5, 5] │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ a       ┆ [6]       │
        ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ b       ┆ [2, 1, 2] │
        └─────────┴───────────┘
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
        return self.lazy().explode(columns).collect(no_optimization=True)

    def pivot(
        self: DF,
        values: list[str] | str,
        index: list[str] | str,
        columns: list[str] | str,
        aggregate_fn: str = "first",
        maintain_order: bool = True,
        sort_columns: bool = False,
    ) -> DF:
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
        aggregate_fn
            Any of:

            - "sum"
            - "max"
            - "min"
            - "mean"
            - "median"
            - "first"
            - "last"
            - "count"

        maintain_order
            Sort the grouped keys so that the output order is predictable.
        sort_columns
            Sort the transposed columns by name. Default is by order of discovery.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ two ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        """
        if isinstance(values, str):
            values = [values]
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]
        return self._from_pydf(
            self._df.pivot2(
                values, index, columns, aggregate_fn, maintain_order, sort_columns
            )
        )

    def melt(
        self: DF,
        id_vars: list[str] | str | None = None,
        value_vars: list[str] | str | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> DF:
        """
        Unpivot a DataFrame from wide to long format, optionally leaving identifiers
        set.

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
        ... )
        >>> df.melt(id_vars="a", value_vars=["b", "c"])
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
        return self._from_pydf(
            self._df.melt(id_vars, value_vars, value_name, variable_name)
        )

    @overload
    def partition_by(
        self: DF,
        groups: str | list[str],
        maintain_order: bool,
        *,
        as_dict: Literal[False] = ...,
    ) -> list[DF]:
        ...

    @overload
    def partition_by(
        self: DF,
        groups: str | list[str],
        maintain_order: bool,
        *,
        as_dict: Literal[True],
    ) -> dict[Any, DF]:
        ...

    @overload
    def partition_by(
        self: DF,
        groups: str | list[str],
        maintain_order: bool,
        *,
        as_dict: bool,
    ) -> list[DF] | dict[Any, DF]:
        ...

    def partition_by(
        self: DF,
        groups: str | list[str],
        maintain_order: bool = True,
        *,
        as_dict: bool = False,
    ) -> list[DF] | dict[Any, DF]:
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
         ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
         │ A   ┆ 2   ┆ l   │
         └─────┴─────┴─────┘,
         shape: (2, 3)
         ┌─────┬─────┬─────┐
         │ foo ┆ N   ┆ bar │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ str │
         ╞═════╪═════╪═════╡
         │ B   ┆ 2   ┆ m   │
         ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
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

        """
        if isinstance(groups, str):
            groups = [groups]

        if as_dict:
            out: dict[Any, DF] = {}
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

    def shift(self: DF, periods: int) -> DF:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 1    ┆ 6    ┆ a    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 3    ┆ 8    ┆ c    │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ null ┆ null ┆ null │
        └──────┴──────┴──────┘

        """
        return self._from_pydf(self._df.shift(periods))

    def shift_and_fill(self: DF, periods: int, fill_value: int | str | float) -> DF:
        """
        Shift the values by a given period and fill the parts that will be empty due to
        this operation with the result of the `fill_value` expression.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 6   ┆ a   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        return (
            self.lazy()
            .shift_and_fill(periods, fill_value)
            .collect(no_optimization=True, string_cache=False)
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

        """
        return pli.wrap_s(self._df.is_unique())

    def lazy(self: DF) -> pli.LazyFrame[DF]:
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

        """
        return self._lazyframe_class._from_pyldf(self._df.lazy())

    def select(
        self: DF,
        exprs: (
            str
            | pli.Expr
            | Sequence[str | pli.Expr | bool | int | float | pli.Series]
            | pli.Series
        ),
    ) -> DF:
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
        ... )
        >>> df.select("foo")
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
        return (
            self.lazy()
            .select(exprs)  # type: ignore[arg-type]
            .collect(no_optimization=True, string_cache=False)
        )

    def with_columns(
        self: DF,
        exprs: pli.Expr | pli.Series | Sequence[pli.Expr | pli.Series] | None = None,
        **named_exprs: pli.Expr | pli.Series,
    ) -> DF:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
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
        >>> df.with_columns(
        ...     d=pl.col("a") * pl.col("b"),
        ...     e=pl.col("c").is_not(),
        ... )
        shape: (4, 5)
        ┌─────┬──────┬───────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ d    ┆ e     │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 0.5  ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  ┆ 8.0  ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false ┆ 30.0 ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  ┆ 52.0 ┆ false │
        └─────┴──────┴───────┴──────┴───────┘

        """
        if exprs is not None and not isinstance(exprs, Sequence):
            exprs = [exprs]
        return (
            self.lazy()
            .with_columns(exprs, **named_exprs)
            .collect(no_optimization=True, string_cache=False)
        )

    def n_chunks(self) -> int:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame.

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

        """
        return self._df.n_chunks()

    @overload
    def max(self: DF, axis: Literal[0] = ...) -> DF:
        ...

    @overload
    def max(self, axis: Literal[1]) -> pli.Series:
        ...

    @overload
    def max(self: DF, axis: int = 0) -> DF | pli.Series:
        ...

    def max(self: DF, axis: int = 0) -> DF | pli.Series:
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
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 3   ┆ 8   ┆ null │
        └─────┴─────┴──────┘

        """
        if axis == 0:
            return self._from_pydf(self._df.max())
        if axis == 1:
            return pli.wrap_s(self._df.hmax())
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    @overload
    def min(self: DF, axis: Literal[0] = ...) -> DF:
        ...

    @overload
    def min(self, axis: Literal[1]) -> pli.Series:
        ...

    @overload
    def min(self: DF, axis: int = 0) -> DF | pli.Series:
        ...

    def min(self: DF, axis: int = 0) -> DF | pli.Series:
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
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ null │
        └─────┴─────┴──────┘

        """
        if axis == 0:
            return self._from_pydf(self._df.min())
        if axis == 1:
            return pli.wrap_s(self._df.hmin())
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    @overload
    def sum(self: DF, *, axis: Literal[0] = ..., null_strategy: str = "ignore") -> DF:
        ...

    @overload
    def sum(self, *, axis: Literal[1], null_strategy: str = "ignore") -> pli.Series:
        ...

    @overload
    def sum(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> DF | pli.Series:
        ...

    def sum(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> DF | pli.Series:
        """
        Aggregate the columns of this DataFrame to their sum value.

        Parameters
        ----------
        axis
            either 0 or 1
        null_strategy
            {'ignore', 'propagate'}
            this argument is only used if axis == 1

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
        if axis == 0:
            return self._from_pydf(self._df.sum())
        if axis == 1:
            return pli.wrap_s(self._df.hsum(null_strategy))
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    @overload
    def mean(self: DF, *, axis: Literal[0] = ..., null_strategy: str = "ignore") -> DF:
        ...

    @overload
    def mean(self, *, axis: Literal[1], null_strategy: str = "ignore") -> pli.Series:
        ...

    @overload
    def mean(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> DF | pli.Series:
        ...

    def mean(self: DF, axis: int = 0, null_strategy: str = "ignore") -> DF | pli.Series:
        """
        Aggregate the columns of this DataFrame to their mean value.

        Parameters
        ----------
        axis
            either 0 or 1
        null_strategy
            {'ignore', 'propagate'}
            this argument is only used if axis == 1

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.mean()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null │
        └─────┴─────┴──────┘

        Note: the mean of booleans evaluates to null.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [True, True, False],
        ...         "b": [True, True, True],
        ...     }
        ... )
        >>> df.mean()
        shape: (1, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ bool ┆ bool │
        ╞══════╪══════╡
        │ null ┆ null │
        └──────┴──────┘

        Instead, cast to numeric type:

        >>> df.select(pl.all().cast(pl.UInt8)).mean()
        shape: (1, 2)
        ┌──────────┬─────┐
        │ a        ┆ b   │
        │ ---      ┆ --- │
        │ f64      ┆ f64 │
        ╞══════════╪═════╡
        │ 0.666667 ┆ 1.0 │
        └──────────┴─────┘

        """
        if axis == 0:
            return self._from_pydf(self._df.mean())
        if axis == 1:
            return pli.wrap_s(self._df.hmean(null_strategy))
        raise ValueError("Axis should be 0 or 1.")  # pragma: no cover

    def std(self: DF) -> DF:
        """
        Aggregate the columns of this DataFrame to their standard deviation value.

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

        """
        return self._from_pydf(self._df.std())

    def var(self: DF) -> DF:
        """
        Aggregate the columns of this DataFrame to their variance value.

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

        """
        return self._from_pydf(self._df.var())

    def median(self: DF) -> DF:
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

    def product(self: DF) -> DF:
        """
        Aggregate the columns of this DataFrame to their product values

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

    def quantile(self: DF, quantile: float, interpolation: str = "nearest") -> DF:
        """
        Aggregate the columns of this DataFrame to their quantile value.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options:
            ['nearest', 'higher', 'lower', 'midpoint', 'linear']

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

    def to_dummies(self: DF) -> DF:
        """
        Get one hot encoded dummy variables.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.to_dummies()
        shape: (3, 9)
        ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
        │ foo_1 ┆ foo_2 ┆ foo_3 ┆ bar_6 ┆ bar_7 ┆ bar_8 ┆ ham_a ┆ ham_b ┆ ham_c │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
        │ 1     ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0     │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 1     ┆ 0     │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 1     ┆ 0     ┆ 0     ┆ 1     │
        └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

        """
        return self._from_pydf(self._df.to_dummies())

    def distinct(
        self: DF,
        maintain_order: bool = True,
        subset: str | list[str] | None = None,
        keep: str = "first",
    ) -> DF:
        """
        .. deprecated:: 0.13.13
            Use :func:`unique` instead.
        """
        return self.unique(maintain_order, subset, keep)

    def unique(
        self: DF,
        maintain_order: bool = True,
        subset: str | list[str] | None = None,
        keep: str = "first",
    ) -> DF:
        """
        Drop duplicate rows from this DataFrame.

        Warnings
        --------
        Note that this fails if there is a column of type `List` in the DataFrame or
        subset.

        Parameters
        ----------
        maintain_order
            Keep the same order as the original DataFrame. This requires more work to
            compute.
        subset
            Subset to use to compare rows
        keep
            any of {"first", "last"}

        Returns
        -------
        DataFrame with unique rows

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, True, True],
        ...     }
        ... )
        >>> df.unique()
        shape: (5, 3)
        ┌─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     │
        │ --- ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 4.0  ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3   ┆ 10.0 ┆ false │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 4   ┆ 13.0 ┆ true  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 5   ┆ 14.0 ┆ true  │
        └─────┴──────┴───────┘

        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return self._from_pydf(self._df.unique(maintain_order, subset, keep))

    def rechunk(self: DF) -> DF:
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable
        performance.
        """
        return self._from_pydf(self._df.rechunk())

    def null_count(self: DF) -> DF:
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
        self: DF,
        n: int | None = None,
        frac: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> DF:
        """
        Sample from this DataFrame by setting either `n` or `frac`.

        Parameters
        ----------
        n
            Number of samples < self.len() .
        frac
            Fraction between 0.0 and 1.0 .
        with_replacement
            Sample with replacement.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Initialization seed. If None is given a random seed is used.

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
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘

        """
        if n is not None and frac is not None:
            raise ValueError("n and frac were both supplied")

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
        Apply a horizontal reduction on a DataFrame. This can be used to effectively
        determine aggregations on a row level, and can be applied to any DataType that
        can be supercasted (casted to a similar parent type).

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

    def row(self, index: int) -> tuple[Any]:
        """
        Get a row as tuple.

        Parameters
        ----------
        index
            Row index.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.row(2)
        (3, 8, 'c')

        """
        return self._df.row_tuple(index)

    def rows(self) -> list[tuple]:
        """
        Convert columnar data to rows as python tuples.

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

        """
        return self._df.row_tuples()

    @overload
    def shrink_to_fit(self: DF, in_place: Literal[False] = ...) -> DF:
        ...

    @overload
    def shrink_to_fit(self, in_place: Literal[True]) -> None:
        ...

    @overload
    def shrink_to_fit(self: DF, in_place: bool) -> DF | None:
        ...

    def shrink_to_fit(self: DF, in_place: bool = False) -> DF | None:
        """
        Shrink memory usage of this DataFrame to fit the exact capacity needed to hold
        the data.
        """
        if in_place:
            self._df.shrink_to_fit()
            return None
        else:
            df = self.clone()
            df._df.shrink_to_fit()
            return df

    def take_every(self: DF, n: int) -> DF:
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
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 7   │
        └─────┴─────┘

        """
        return self.select(pli.col("*").take_every(n))

    @deprecated_alias(k0="seed", k1="seed_1", k2="seed_2", k3="seed_3")
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
        >>> df.hash_rows(seed=42)
        shape: (4,)
        Series: '' [u64]
        [
            1381515935931787907
            14326417405130769253
            12561864296213327929
            11391467306893437193
        ]

        """
        k0 = seed
        k1 = seed_1 if seed_1 is not None else seed
        k2 = seed_2 if seed_2 is not None else seed
        k3 = seed_3 if seed_3 is not None else seed
        return pli.wrap_s(self._df.hash_rows(k0, k1, k2, k3))

    def interpolate(self: DF) -> DF:
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
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 5   ┆ 7    ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 9   ┆ 9    ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ 10  ┆ null ┆ 9   │
        └─────┴──────┴─────┘

        """
        return self.select(pli.col("*").interpolate())

    def is_empty(self) -> bool:
        """
        Check if the dataframe is empty

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
        Convert a ``DataFrame`` to a ``Series`` of type ``Struct``

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

    def unnest(self: DF, names: str | list[str]) -> DF:
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
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ bar    ┆ 2   ┆ b   ┆ null ┆ [3]       ┆ womp  │
        └────────┴─────┴─────┴──────┴───────────┴───────┘

        """
        if isinstance(names, str):
            names = [names]
        return self._from_pydf(self._df.unnest(names))


class RollingGroupBy(Generic[DF]):
    """
    A rolling grouper. This has an `.agg` method which will allow you to run all polars
    expressions in a groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        period: str,
        offset: str | None,
        closed: str = "none",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ):
        self.df = df
        self.time_column = index_column
        self.period = period
        self.offset = offset
        self.closed = closed
        self.by = by

    def agg(
        self,
        column_to_agg: (
            list[tuple[str, list[str]]]
            | dict[str, str | list[str]]
            | list[pli.Expr]
            | pli.Expr
        ),
    ) -> DF:
        return (
            self.df.lazy()
            .groupby_rolling(
                self.time_column, self.period, self.offset, self.closed, self.by
            )
            .agg(column_to_agg)  # type: ignore[arg-type]
            .collect(no_optimization=True, string_cache=False)
        )


class DynamicGroupBy(Generic[DF]):
    """
    A dynamic grouper. This has an `.agg` method which will allow you to run all polars
    expressions in a groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        every: str,
        period: str | None,
        offset: str | None,
        truncate: bool = True,
        include_boundaries: bool = True,
        closed: str = "none",
        by: str | list[str] | pli.Expr | list[pli.Expr] | None = None,
    ):
        self.df = df
        self.time_column = index_column
        self.every = every
        self.period = period
        self.offset = offset
        self.truncate = truncate
        self.include_boundaries = include_boundaries
        self.closed = closed
        self.by = by

    def agg(
        self,
        column_to_agg: (
            list[tuple[str, list[str]]]
            | dict[str, str | list[str]]
            | list[pli.Expr]
            | pli.Expr
        ),
    ) -> DF:
        return (
            self.df.lazy()
            .groupby_dynamic(
                self.time_column,
                self.every,
                self.period,
                self.offset,
                self.truncate,
                self.include_boundaries,
                self.closed,
                self.by,
            )
            .agg(column_to_agg)  # type: ignore[arg-type]
            .collect(no_optimization=True, string_cache=False)
        )


class GroupBy(Generic[DF]):
    """
    Starts a new GroupBy operation.

    You can also loop over this Object to loop over `DataFrames` with unique groups.

    Examples
    --------
    >>> df = pl.DataFrame({"foo": ["a", "a", "b"], "bar": [1, 2, 3]})
    >>> for group in df.groupby("foo"):
    ...     print(group)
    ... # doctest: +IGNORE_RESULT
    ...
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ a   ┆ 1   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ a   ┆ 2   │
    └─────┴─────┘
    shape: (1, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ b   ┆ 3   │
    └─────┴─────┘

    """

    def __init__(
        self,
        df: PyDataFrame,
        by: str | list[str],
        dataframe_class: type[DF],
        maintain_order: bool = False,
    ):
        """
        Construct class representing a group by operation over the given dataframe.

        Parameters
        ----------
        df
            PyDataFrame to perform operation over.
        by
            Column(s) to group by.
        dataframe_class
            The class used to wrap around the given dataframe. Used to construct new
            dataframes returned from the group by operation.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more
            expensive than a default groupby. Note that this only works in expression
            aggregations.

        """
        self._df = df
        self._dataframe_class = dataframe_class
        self.by = by
        self.maintain_order = maintain_order

    def __getitem__(self, item: Any) -> GBSelection[DF]:
        print(
            "accessing GroupBy by index is deprecated, consider using the `.agg` method"
        )
        return self._select(item)

    def _select(self, columns: str | list[str]) -> GBSelection[DF]:  # pragma: no cover
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns.

        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg`"
            " method",
            DeprecationWarning,
        )
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(
            self._df,
            self.by,
            columns,
            dataframe_class=self._dataframe_class,
        )

    def __iter__(self) -> Iterable[Any]:
        groups_df = self.groups()
        groups = groups_df["groups"]
        df = self._dataframe_class._from_pydf(self._df)
        for i in range(groups_df.height):
            yield df[groups[i]]

    def get_group(self, group_value: Any | tuple[Any]) -> DF:
        """
        Select a single group as a new DataFrame.

        .. deprecated:: 0.13.32
            Use :func:`partition_by` instead.

        Parameters
        ----------
        group_value
            Group to select.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["one", "one", "one", "two", "two", "two"],
        ...         "bar": ["A", "B", "C", "A", "B", "C"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.groupby("foo").get_group("one")
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ baz │
        │ --- ┆ --- ┆ --- │
        │ str ┆ str ┆ i64 │
        ╞═════╪═════╪═════╡
        │ one ┆ A   ┆ 1   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ one ┆ B   ┆ 2   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ one ┆ C   ┆ 3   │
        └─────┴─────┴─────┘

        """
        groups_df = self.groups()
        groups = groups_df["groups"]

        if not isinstance(group_value, list):
            group_value = [group_value]

        by = self.by
        if not isinstance(by, list):
            by = [by]

        mask = None
        for column, group_val in zip(by, group_value):
            local_mask = groups_df[column] == group_val
            if mask is None:
                mask = local_mask
            else:
                mask = mask & local_mask

        # should be only one match
        try:
            groups_idx = groups[mask][0]  # type: ignore[index]
        except IndexError:
            raise ValueError(f"no group: {group_value} found") from None

        df = self._dataframe_class._from_pydf(self._df)
        return df[groups_idx]

    def groups(self) -> DF:  # pragma: no cover
        """
        Return a `DataFrame` with:

        * the groupby keys
        * the group indexes aggregated as lists

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, True, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d").groups().sort(by="d")
        shape: (3, 2)
        ┌────────┬───────────┐
        │ d      ┆ groups    │
        │ ---    ┆ ---       │
        │ str    ┆ list[u32] │
        ╞════════╪═══════════╡
        │ Apple  ┆ [0, 2, 3] │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ Banana ┆ [4, 5]    │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ Orange ┆ [1]       │
        └────────┴───────────┘

        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg`"
            " method",
            DeprecationWarning,
        )
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, None, "groups")
        )

    def apply(self, f: Callable[[DataFrame], DataFrame]) -> DF:
        """
        Apply a function over the groups as a sub-DataFrame.

        Implementing logic using this .apply method is generally slower and more memory
        intensive than implementing the same logic using the expression API because:

        - with .apply the logic is implemented in Python but with an expression the
          logic is implemented in Rust
        - with .apply the DataFrame is materialized in memory
        - expressions can be parallelised
        - expressions can be optimised

        If possible use the expression API for best performance.

        Parameters
        ----------
        f
            Custom function.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": [0, 1, 2, 3, 4],
        ...         "color": ["red", "green", "green", "red", "red"],
        ...         "shape": ["square", "triangle", "square", "triangle", "square"],
        ...     }
        ... )
        >>> df
        shape: (5, 3)
        ┌─────┬───────┬──────────┐
        │ id  ┆ color ┆ shape    │
        │ --- ┆ ---   ┆ ---      │
        │ i64 ┆ str   ┆ str      │
        ╞═════╪═══════╪══════════╡
        │ 0   ┆ red   ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 1   ┆ green ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ green ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ red   ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ red   ┆ square   │
        └─────┴───────┴──────────┘

        For each color group sample two rows:

        >>> (
        ...     df.groupby("color").apply(lambda group_df: group_df.sample(2))
        ... )  # doctest: +IGNORE_RESULT
        shape: (4, 3)
        ┌─────┬───────┬──────────┐
        │ id  ┆ color ┆ shape    │
        │ --- ┆ ---   ┆ ---      │
        │ i64 ┆ str   ┆ str      │
        ╞═════╪═══════╪══════════╡
        │ 1   ┆ green ┆ triangle │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ green ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 4   ┆ red   ┆ square   │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ 3   ┆ red   ┆ triangle │
        └─────┴───────┴──────────┘

        It is better to implement this with an expression:

        >>> (
        ...     df.filter(pl.arange(0, pl.count()).shuffle().over("color") < 2)
        ... )  # doctest: +IGNORE_RESULT

        """
        return self._dataframe_class._from_pydf(self._df.groupby_apply(self.by, f))

    def agg(
        self,
        column_to_agg: (
            list[tuple[str, list[str]]]
            | dict[str, str | list[str]]
            | list[pli.Expr]
            | pli.Expr
        ),
    ) -> DF:
        """
        Use multiple aggregations on columns. This can be combined with complete lazy
        API and is considered idiomatic polars.

        Parameters
        ----------
        column_to_agg
            map column to aggregation functions.

        Returns
        -------
        Result of groupby split apply operations.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": ["one", "two", "two", "one", "two"], "bar": [5, 3, 2, 4, 1]}
        ... )
        >>> df.groupby("foo", maintain_order=True).agg(
        ...     [
        ...         pl.sum("bar").suffix("_sum"),
        ...         pl.col("bar").sort().tail(2).sum().suffix("_tail_sum"),
        ...     ]
        ... )
        shape: (2, 3)
        ┌─────┬─────────┬──────────────┐
        │ foo ┆ bar_sum ┆ bar_tail_sum │
        │ --- ┆ ---     ┆ ---          │
        │ str ┆ i64     ┆ i64          │
        ╞═════╪═════════╪══════════════╡
        │ one ┆ 9       ┆ 9            │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ two ┆ 6       ┆ 5            │
        └─────┴─────────┴──────────────┘

        """
        # a single list comprehension would be cleaner, but mypy complains on different
        # lines for py3.7 vs py3.10 about typing errors, so this is the same logic,
        # but broken down into two small functions
        def _str_to_list(y: Any) -> Any:
            return [y] if isinstance(y, str) else y

        def _wrangle(x: Any) -> list:
            return [(xi[0], _str_to_list(xi[1])) for xi in x]

        if isinstance(column_to_agg, pli.Expr):
            column_to_agg = [column_to_agg]
        if isinstance(column_to_agg, dict):
            column_to_agg = _wrangle(column_to_agg.items())
        elif isinstance(column_to_agg, list):

            if isinstance(column_to_agg[0], tuple):
                column_to_agg = _wrangle(column_to_agg)

            elif isinstance(column_to_agg[0], pli.Expr):
                return (
                    self._dataframe_class._from_pydf(self._df)
                    .lazy()
                    .groupby(self.by, maintain_order=self.maintain_order)
                    .agg(column_to_agg)  # type: ignore[arg-type]
                    .collect(no_optimization=True, string_cache=False)
                )
            else:
                raise ValueError(
                    f"argument: {column_to_agg} not understood, have you passed a list"
                    " of expressions?"
                )
        else:
            raise ValueError(
                f"argument: {column_to_agg} not understood, have you passed a list of"
                " expressions?"
            )

        return self._dataframe_class._from_pydf(
            self._df.groupby_agg(self.by, column_to_agg)
        )

    def head(self, n: int = 5) -> DF:
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
        return (
            self._dataframe_class._from_pydf(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .head(n)
            .collect(no_optimization=True, string_cache=False)
        )

    def tail(self, n: int = 5) -> DF:
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
        >>> (df.groupby("letters").tail(2).sort("letters"))
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
        return (
            self._dataframe_class._from_pydf(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .tail(n)
            .collect(no_optimization=True, string_cache=False)
        )

    def _select_all(self) -> GBSelection[DF]:
        """Select all columns for aggregation."""
        return GBSelection(
            self._df,
            self.by,
            None,
            dataframe_class=self._dataframe_class,
        )

    def pivot(
        self, pivot_column: str | list[str], values_column: str | list[str]
    ) -> PivotOps[DF]:
        """
        Do a pivot operation based on the group key, a pivot column and an aggregation
        function on the values column.

        .. note::
            Polars'/arrow memory is not ideal for transposing operations like pivots.
            If you have a relatively large table, consider using a groupby over a pivot.

        Parameters
        ----------
        pivot_column
            Column to pivot.
        values_column
            Column that will be aggregated.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["one", "one", "one", "two", "two", "two"],
        ...         "bar": ["A", "B", "C", "A", "B", "C"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.groupby("foo", maintain_order=True).pivot(  # doctest: +SKIP
        ...     pivot_column="bar", values_column="baz"
        ... ).first()
        shape: (2, 4)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ A   ┆ B   ┆ C   │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╡
        │ one ┆ 1   ┆ 2   ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ two ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        """
        if isinstance(pivot_column, str):
            pivot_column = [pivot_column]
        if isinstance(values_column, str):
            values_column = [values_column]
        return PivotOps(
            self._df,
            self.by,
            pivot_column,
            values_column,
            dataframe_class=self._dataframe_class,
        )

    def first(self) -> DF:
        """
        Aggregate the first values in the group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).first()
        shape: (3, 4)
        ┌────────┬─────┬──────┬───────┐
        │ d      ┆ a   ┆ b    ┆ c     │
        │ ---    ┆ --- ┆ ---  ┆ ---   │
        │ str    ┆ i64 ┆ f64  ┆ bool  │
        ╞════════╪═════╪══════╪═══════╡
        │ Apple  ┆ 1   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 4   ┆ 13.0 ┆ false │
        └────────┴─────┴──────┴───────┘

        """
        return self.agg(pli.all().first())

    def last(self) -> DF:
        """
        Aggregate the last values in the group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).last()
        shape: (3, 4)
        ┌────────┬─────┬──────┬───────┐
        │ d      ┆ a   ┆ b    ┆ c     │
        │ ---    ┆ --- ┆ ---  ┆ ---   │
        │ str    ┆ i64 ┆ f64  ┆ bool  │
        ╞════════╪═════╪══════╪═══════╡
        │ Apple  ┆ 3   ┆ 10.0 ┆ false │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ true  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 5   ┆ 14.0 ┆ true  │
        └────────┴─────┴──────┴───────┘

        """
        return self.agg(pli.all().last())

    def sum(self) -> DF:
        """
        Reduce the groups to the sum.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).sum()
        shape: (3, 4)
        ┌────────┬─────┬──────┬─────┐
        │ d      ┆ a   ┆ b    ┆ c   │
        │ ---    ┆ --- ┆ ---  ┆ --- │
        │ str    ┆ i64 ┆ f64  ┆ u32 │
        ╞════════╪═════╪══════╪═════╡
        │ Apple  ┆ 6   ┆ 14.5 ┆ 2   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ 1   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 9   ┆ 27.0 ┆ 1   │
        └────────┴─────┴──────┴─────┘

        """
        return self.agg(pli.all().sum())

    def min(self) -> DF:
        """
        Reduce the groups to the minimal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).min()
        shape: (3, 4)
        ┌────────┬─────┬──────┬─────┐
        │ d      ┆ a   ┆ b    ┆ c   │
        │ ---    ┆ --- ┆ ---  ┆ --- │
        │ str    ┆ i64 ┆ f64  ┆ u32 │
        ╞════════╪═════╪══════╪═════╡
        │ Apple  ┆ 1   ┆ 0.5  ┆ 0   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ 1   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 4   ┆ 13.0 ┆ 0   │
        └────────┴─────┴──────┴─────┘

        """
        return self.agg(pli.all().min())

    def max(self) -> DF:
        """
        Reduce the groups to the maximal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).max()
        shape: (3, 4)
        ┌────────┬─────┬──────┬─────┐
        │ d      ┆ a   ┆ b    ┆ c   │
        │ ---    ┆ --- ┆ ---  ┆ --- │
        │ str    ┆ i64 ┆ f64  ┆ u32 │
        ╞════════╪═════╪══════╪═════╡
        │ Apple  ┆ 3   ┆ 10.0 ┆ 1   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Orange ┆ 2   ┆ 0.5  ┆ 1   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 5   ┆ 14.0 ┆ 1   │
        └────────┴─────┴──────┴─────┘

        """
        return self.agg(pli.all().max())

    def count(self) -> DF:
        """
        Count the number of values in each group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).count()
        shape: (3, 2)
        ┌────────┬───────┐
        │ d      ┆ count │
        │ ---    ┆ ---   │
        │ str    ┆ u32   │
        ╞════════╪═══════╡
        │ Apple  ┆ 3     │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Orange ┆ 1     │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Banana ┆ 2     │
        └────────┴───────┘

        """
        return self.agg(pli.lazy_functions.count())

    def mean(self) -> DF:
        """
        Reduce the groups to the mean values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "c": [True, True, True, False, False, True],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d", maintain_order=True).mean()
        shape: (3, 4)
        ┌────────┬─────┬──────────┬──────┐
        │ d      ┆ a   ┆ b        ┆ c    │
        │ ---    ┆ --- ┆ ---      ┆ ---  │
        │ str    ┆ f64 ┆ f64      ┆ bool │
        ╞════════╪═════╪══════════╪══════╡
        │ Apple  ┆ 2.0 ┆ 4.833333 ┆ null │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Orange ┆ 2.0 ┆ 0.5      ┆ null │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 4.5 ┆ 13.5     ┆ null │
        └────────┴─────┴──────────┴──────┘

        """
        return self.agg(pli.all().mean())

    def n_unique(self) -> DF:
        """
        Count the unique values per group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 1, 3, 4, 5],
        ...         "b": [0.5, 0.5, 0.5, 10, 13, 14],
        ...         "d": ["Apple", "Banana", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )

        >>> df.groupby("d", maintain_order=True).n_unique()
        shape: (2, 3)
        ┌────────┬─────┬─────┐
        │ d      ┆ a   ┆ b   │
        │ ---    ┆ --- ┆ --- │
        │ str    ┆ u32 ┆ u32 │
        ╞════════╪═════╪═════╡
        │ Apple  ┆ 2   ┆ 2   │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ Banana ┆ 3   ┆ 3   │
        └────────┴─────┴─────┘

        """
        return self.agg(pli.all().n_unique())

    def quantile(self, quantile: float, interpolation: str = "nearest") -> DF:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options:
            ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "d": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).quantile(1)
        shape: (3, 3)
        ┌────────┬─────┬──────┐
        │ d      ┆ a   ┆ b    │
        │ ---    ┆ --- ┆ ---  │
        │ str    ┆ f64 ┆ f64  │
        ╞════════╪═════╪══════╡
        │ Apple  ┆ 3.0 ┆ 10.0 │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Orange ┆ 2.0 ┆ 0.5  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 5.0 ┆ 14.0 │
        └────────┴─────┴──────┘

        """
        return self.agg(pli.all().quantile(quantile, interpolation))

    def median(self) -> DF:
        """
        Return the median per group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 4, 10, 13, 14],
        ...         "d": ["Apple", "Banana", "Apple", "Apple", "Banana", "Banana"],
        ...     }
        ... )
        >>> df.groupby("d", maintain_order=True).median()
        shape: (2, 3)
        ┌────────┬─────┬──────┐
        │ d      ┆ a   ┆ b    │
        │ ---    ┆ --- ┆ ---  │
        │ str    ┆ f64 ┆ f64  │
        ╞════════╪═════╪══════╡
        │ Apple  ┆ 2.0 ┆ 4.0  │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ Banana ┆ 4.0 ┆ 13.0 │
        └────────┴─────┴──────┘

        """
        return self.agg(pli.all().median())

    def agg_list(self) -> DF:
        """
        Aggregate the groups into Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
        >>> df.groupby("a", maintain_order=True).agg_list()
        shape: (2, 2)
        ┌─────┬───────────┐
        │ a   ┆ b         │
        │ --- ┆ ---       │
        │ str ┆ list[i64] │
        ╞═════╪═══════════╡
        │ one ┆ [1, 3]    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
        │ two ┆ [2, 4]    │
        └─────┴───────────┘

        """
        return self.agg(pli.all().list())


class PivotOps(Generic[DF]):
    """Utility class returned in a pivot operation."""

    def __init__(
        self,
        df: DataFrame,
        by: str | list[str],
        pivot_column: str | list[str],
        values_column: str | list[str],
        dataframe_class: type[DF],
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column
        self._dataframe_class = dataframe_class

    def first(self) -> DF:
        """Get the first value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self) -> DF:
        """Get the sum per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self) -> DF:
        """Get the minimal value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self) -> DF:
        """Get the maximal value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self) -> DF:
        """Get the mean value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def count(self) -> DF:
        """Count the values per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "count")
        )

    def median(self) -> DF:
        """Get the median value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )

    def last(self) -> DF:
        """Get the last value per group."""
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "last")
        )


class GBSelection(Generic[DF]):
    """Utility class returned in a groupby operation."""

    def __init__(
        self,
        df: PyDataFrame,
        by: str | list[str],
        selection: list[str] | None,
        dataframe_class: type[DF],
    ):
        self._df = df
        self.by = by
        self.selection = selection
        self._dataframe_class = dataframe_class

    def first(self) -> DF:
        """Aggregate the first values in the group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "first")
        )

    def last(self) -> DF:
        """Aggregate the last values in the group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "last")
        )

    def sum(self) -> DF:
        """Reduce the groups to the sum."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "sum")
        )

    def min(self) -> DF:
        """Reduce the groups to the minimal value."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "min")
        )

    def max(self) -> DF:
        """Reduce the groups to the maximal value."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "max")
        )

    def count(self) -> DF:
        """
        Count the number of values in each group.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3, 4],
        ...         "bar": ["a", "b", "c", "a"],
        ...     }
        ... )
        >>> df.groupby("bar", maintain_order=True).count()  # counts nulls
        shape: (3, 2)
        ┌─────┬───────┐
        │ bar ┆ count │
        │ --- ┆ ---   │
        │ str ┆ u32   │
        ╞═════╪═══════╡
        │ a   ┆ 2     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b   ┆ 1     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c   ┆ 1     │
        └─────┴───────┘

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "count")
        )

    def mean(self) -> DF:
        """Reduce the groups to the mean values."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "mean")
        )

    def n_unique(self) -> DF:
        """Count the unique values per group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "n_unique")
        )

    def quantile(self, quantile: float, interpolation: str = "nearest") -> DF:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options:
            ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby_quantile(self.by, self.selection, quantile, interpolation)
        )

    def median(self) -> DF:
        """Return the median per group."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "median")
        )

    def agg_list(self) -> DF:
        """Aggregate the groups into Series."""
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "agg_list")
        )

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: type[DataType] | None = None,
    ) -> DF:
        """Apply a function over the groups."""
        df = self.agg_list()
        if self.selection is None:
            raise TypeError(
                "apply not available for Groupby.select_all(). Use select() instead."
            )
        for name in self.selection:
            s = df.drop_in_place(name + "_agg_list").apply(func, return_dtype)
            s.rename(name, in_place=True)
            df[name] = s

        return df

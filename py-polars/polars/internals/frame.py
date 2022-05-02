"""
Module containing logic related to eager DataFrames
"""
import os
import sys
import warnings
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal  # pragma: no cover

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute
    import pyarrow.parquet

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from polars import internals as pli
from polars.internals.construction import (
    ColumnsType,
    arrow_to_pydf,
    dict_to_pydf,
    numpy_to_pydf,
    pandas_to_pydf,
    sequence_to_pydf,
    series_to_pydf,
)

from .lazy_frame import LazyFrame, wrap_ldf  # noqa: F401

try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars._html import NotebookFormatter
from polars.datatypes import Boolean, DataType, UInt32, Utf8, py_type_to_dtype
from polars.utils import (
    _prepare_row_count_args,
    _process_null_values,
    format_path,
    handle_projection_columns,
    is_int_sequence,
    is_str_sequence,
    range_to_slice,
)

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


# A type variable used to refer to a polars.DataFrame or any subclass of it.
# Used to annotate DataFrame methods which returns the same type as self.
DF = TypeVar("DF", bound="DataFrame")


def wrap_df(df: "PyDataFrame") -> "DataFrame":
    return DataFrame._from_pydf(df)


def _prepare_other_arg(other: Any) -> "pli.Series":
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
            cls._lazyframe_class = type(  # type: ignore
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
    of typed Series, or set the columns parameter with a list of (name,dtype) pairs:

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

    # or, equivalent... (and also compatible with all of the other valid data parameter types):
    >>> df3 = pl.DataFrame(data, columns=[("col1", pl.Float32), ("col2", pl.Int64)])
    >>> df3
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
        data: Optional[
            Union[
                Dict[str, Sequence[Any]],
                Sequence[Any],
                np.ndarray,
                "pa.Table",
                "pd.DataFrame",
                "pli.Series",
            ]
        ] = None,
        columns: Optional[ColumnsType] = None,
        orient: Optional[str] = None,
    ):
        if data is None:
            self._df = dict_to_pydf({}, columns=columns)

        elif isinstance(data, dict):
            self._df = dict_to_pydf(data, columns=columns)

        elif isinstance(data, np.ndarray):
            self._df = numpy_to_pydf(data, columns=columns, orient=orient)

        elif _PYARROW_AVAILABLE and isinstance(data, pa.Table):
            self._df = arrow_to_pydf(data, columns=columns)

        elif isinstance(data, Sequence) and not isinstance(data, str):
            self._df = sequence_to_pydf(data, columns=columns, orient=orient)

        elif isinstance(data, pli.Series):
            self._df = series_to_pydf(data, columns=columns)

        elif _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            if not _PYARROW_AVAILABLE:
                raise ImportError(  # pragma: no cover
                    "'pyarrow' is required for converting a pandas DataFrame to a polars DataFrame."
                )
            self._df = pandas_to_pydf(data, columns=columns)

        else:
            raise ValueError("DataFrame constructor not called properly.")

    def estimated_size(self) -> int:
        """
        Returns an estimation of the total (heap) allocated size of the `DataFrame` in bytes.

        This estimation is the sum of the size of its buffers, validity, including nested arrays.
        Multiple arrays may share buffers and bitmaps. Therefore, the size of 2 arrays is not the
        sum of the sizes computed from this function. In particular, [`StructArray`]'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer unchanged.
        However, this function will yield a smaller number. This is because this function returns
        the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.
        """
        return self._df.estimated_size()

    @classmethod
    def _from_pydf(cls: Type[DF], py_df: "PyDataFrame") -> DF:
        """
        Construct Polars DataFrame from FFI PyDataFrame object.
        """
        df = cls.__new__(cls)
        df._df = py_df
        return df

    @classmethod
    def _from_dicts(
        cls: Type[DF],
        data: Sequence[Dict[str, Any]],
    ) -> DF:
        pydf = PyDataFrame.read_dicts(data)
        return cls._from_pydf(pydf)

    @classmethod
    def _from_dict(
        cls: Type[DF],
        data: Dict[str, Sequence[Any]],
        columns: Optional[Sequence[str]] = None,
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
        cls: Type[DF],
        data: Union[np.ndarray, Sequence[Sequence[Any]]],
        columns: Optional[Sequence[str]] = None,
        orient: Optional[str] = None,
    ) -> DF:
        """
        Construct a DataFrame from a numpy ndarray or sequence of sequences.

        Parameters
        ----------
        data : numpy ndarray or Sequence of sequences
            Two-dimensional data represented as numpy ndarray or sequence of sequences.
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
        if isinstance(data, np.ndarray):
            pydf = numpy_to_pydf(data, columns=columns, orient=orient)
        else:
            pydf = sequence_to_pydf(data, columns=columns, orient=orient)
        return cls._from_pydf(pydf)

    @classmethod
    def _from_arrow(
        cls: Type[DF],
        data: "pa.Table",
        columns: Optional[Sequence[str]] = None,
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
        cls: Type[DF],
        data: "pd.DataFrame",
        columns: Optional[Sequence[str]] = None,
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
        cls: Type[DF],
        file: Union[str, Path, BinaryIO, bytes],
        has_header: bool = True,
        columns: Optional[Union[List[int], List[str]]] = None,
        sep: str = ",",
        comment_char: Optional[str] = None,
        quote_char: Optional[str] = r'"',
        skip_rows: int = 0,
        dtypes: Optional[
            Union[Mapping[str, Type[DataType]], List[Type[DataType]]]
        ] = None,
        null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
        ignore_errors: bool = False,
        parse_dates: bool = False,
        n_threads: Optional[int] = None,
        infer_schema_length: Optional[int] = 100,
        batch_size: int = 8192,
        n_rows: Optional[int] = None,
        encoding: str = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        skip_rows_after_header: int = 0,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
        sample_size: int = 1024,
    ) -> DF:
        """
        see pl.read_csv
        """
        self = cls.__new__(cls)

        path: Optional[str]
        if isinstance(file, (str, Path)):
            path = format_path(file)
        else:
            path = None
            if isinstance(file, BytesIO):
                file = file.getvalue()
            if isinstance(file, StringIO):
                file = file.getvalue().encode()

        dtype_list: Optional[List[Tuple[str, Type[DataType]]]] = None
        dtype_slice: Optional[List[Type[DataType]]] = None
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
                    "cannot use glob patterns and unamed dtypes as `dtypes` argument; Use dtypes: Mapping[str, Type[DataType]"
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
            )
            if columns is None:
                return self._from_pydf(scan.collect()._df)
            elif is_str_sequence(columns, False):
                return self._from_pydf(scan.select(columns).collect()._df)
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns` argument; Use columns: List[str]"
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
        )
        return self

    @classmethod
    def _read_parquet(
        cls: Type[DF],
        file: Union[str, Path, BinaryIO],
        columns: Optional[Union[List[int], List[str]]] = None,
        n_rows: Optional[int] = None,
        parallel: bool = True,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
    ) -> DF:
        """
        Read into a DataFrame from a parquet file.

        Parameters
        ----------
        file
            Path to a file or a file-like object. Any valid filepath can be used.
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
        n_rows
            Stop reading from parquet file after reading ``n_rows``.
        parallel
            Read the parquet file in parallel. The single threaded reader consumes less memory.
        """
        if isinstance(file, (str, Path)):
            file = format_path(file)
        if isinstance(file, str) and "*" in file:
            from polars import scan_parquet

            scan = scan_parquet(
                file,
                n_rows=n_rows,
                rechunk=True,
                parallel=parallel,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
            )

            if columns is None:
                return cls._from_pydf(scan.collect()._df)
            elif is_str_sequence(columns, False):
                return cls._from_pydf(scan.select(columns).collect()._df)
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns` argument; Use columns: List[str]"
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
        )
        return self

    @classmethod
    def _read_avro(
        cls: Type[DF],
        file: Union[str, Path, BinaryIO],
        columns: Optional[Union[List[int], List[str]]] = None,
        n_rows: Optional[int] = None,
    ) -> DF:
        """
        Read into a DataFrame from Apache Avro format.

        Parameters
        ----------
        file
            Path to a file or a file-like object.
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
        cls: Type[DF],
        file: Union[str, Path, BinaryIO],
        columns: Optional[Union[List[int], List[str]]] = None,
        n_rows: Optional[int] = None,
        row_count_name: Optional[str] = None,
        row_count_offset: int = 0,
        rechunk: bool = True,
    ) -> DF:
        """
        Read into a DataFrame from Arrow IPC stream format. This is also called the Feather (v2) format.

        Parameters
        ----------
        file
            Path to a file or a file-like object.
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
        n_rows
            Stop reading from IPC file after reading ``n_rows``.
        rechunk
            Make sure that all data is contiguous.

        Returns
        -------
        DataFrame
        """

        if isinstance(file, (str, Path)):
            file = format_path(file)
        if isinstance(file, str) and "*" in file:
            from polars import scan_ipc

            scan = scan_ipc(
                file,
                n_rows=n_rows,
                rechunk=rechunk,
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
            )
            if columns is None:
                scan.collect()
            elif is_str_sequence(columns, False):
                scan.select(columns).collect()
            else:
                raise ValueError(
                    "cannot use glob patterns and integer based projection as `columns` argument; Use columns: List[str]"
                )

        projection, columns = handle_projection_columns(columns)
        self = cls.__new__(cls)
        self._df = PyDataFrame.read_ipc(
            file,
            columns,
            projection,
            n_rows,
            _prepare_row_count_args(row_count_name, row_count_offset),
        )
        return self

    @classmethod
    def _read_json(
        cls: Type[DF],
        file: Union[str, Path, IOBase],
        json_lines: bool = False,
    ) -> DF:
        """
        See Also pl.read_json
        """
        if isinstance(file, StringIO):
            file = BytesIO(file.getvalue().encode())
        elif isinstance(file, (str, Path)):
            file = format_path(file)

        self = cls.__new__(cls)
        self._df = PyDataFrame.read_json(file, json_lines)
        return self

    def to_arrow(self) -> "pa.Table":
        """
        Collect the underlying arrow arrays in an Arrow Table.
        This operation is mostly zero copy.

        Data types that do copy:
            - CategoricalType
        """
        if not _PYARROW_AVAILABLE:
            raise ImportError(  # pragma: no cover
                "'pyarrow' is required for converting a polars DataFrame to an Arrow Table."
            )
        record_batches = self._df.to_arrow()
        return pa.Table.from_batches(record_batches)

    @overload
    def to_dict(self, as_series: Literal[True] = ...) -> Dict[str, "pli.Series"]:
        ...

    @overload
    def to_dict(self, as_series: Literal[False]) -> Dict[str, List[Any]]:
        ...

    @overload
    def to_dict(
        self, as_series: bool = True
    ) -> Union[Dict[str, "pli.Series"], Dict[str, List[Any]]]:
        ...

    def to_dict(
        self, as_series: bool = True
    ) -> Union[Dict[str, "pli.Series"], Dict[str, List[Any]]]:
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
        file: Optional[Union[IOBase, str, Path]] = ...,
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
        file: Optional[Union[IOBase, str, Path]] = ...,
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
        file: Optional[Union[IOBase, str, Path]] = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: bool = ...,
    ) -> Optional[str]:
        ...

    def to_json(
        self,
        file: Optional[Union[IOBase, str, Path]] = None,
        pretty: bool = False,
        row_oriented: bool = False,
        json_lines: bool = False,
        *,
        to_string: bool = False,
    ) -> Optional[str]:
        """
        .. deprecated:: 0.13.12
            Please use `write_json`
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
        file: Optional[Union[IOBase, str, Path]] = ...,
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
        file: Optional[Union[IOBase, str, Path]] = ...,
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
        file: Optional[Union[IOBase, str, Path]] = ...,
        pretty: bool = ...,
        row_oriented: bool = ...,
        json_lines: bool = ...,
        *,
        to_string: bool = ...,
    ) -> Optional[str]:
        ...

    def write_json(
        self,
        file: Optional[Union[IOBase, str, Path]] = None,
        pretty: bool = False,
        row_oriented: bool = False,
        json_lines: bool = False,
        *,
        to_string: bool = False,
    ) -> Optional[str]:
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
    ) -> "pd.DataFrame":  # noqa: F821
        """
        Cast to a Pandas DataFrame. This requires that Pandas is installed.
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
        record_batches = self._df.to_pandas()
        tbl = pa.Table.from_batches(record_batches)
        return tbl.to_pandas(*args, date_as_object=date_as_object, **kwargs)

    def write_csv(
        self,
        file: Optional[Union[TextIO, BytesIO, str, Path]] = None,
        has_header: bool = True,
        sep: str = ",",
        quote: str = '"',
    ) -> Optional[str]:
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

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.write_csv("new_file.csv", sep=",")

        """
        if len(sep) > 1:
            raise ValueError("only single byte separator is allowed")
        if len(quote) > 1:
            raise ValueError("only single byte quote char is allowed")
        if file is None:
            buffer = BytesIO()
            self._df.to_csv(buffer, has_header, ord(sep), ord(quote))
            return str(buffer.getvalue(), encoding="utf-8")

        if isinstance(file, (str, Path)):
            file = format_path(file)

        self._df.to_csv(file, has_header, ord(sep), ord(quote))
        return None

    def to_csv(
        self,
        file: Optional[Union[TextIO, BytesIO, str, Path]] = None,
        has_header: bool = True,
        sep: str = ",",
    ) -> Optional[str]:
        """
        .. deprecated:: 0.13.12
            Please use `write_csv`
        """
        warnings.warn(
            "'to_csv' is deprecated. please use 'write_csv'", DeprecationWarning
        )
        return self.write_csv(file, has_header, sep)

    def write_avro(
        self,
        file: Union[BinaryIO, BytesIO, str, Path],
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
        file: Union[BinaryIO, BytesIO, str, Path],
        compression: Literal["uncompressed", "snappy", "deflate"] = "uncompressed",
    ) -> None:
        """
        .. deprecated:: 0.13.12
            Please use `write_avro`
        """
        warnings.warn(
            "'to_avro' is deprecated. please use 'write_avro'", DeprecationWarning
        )
        return self.write_avro(file, compression)

    def write_ipc(
        self,
        file: Union[BinaryIO, BytesIO, str, Path],
        compression: Optional[Literal["uncompressed", "lz4", "zstd"]] = "uncompressed",
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
        file: Union[BinaryIO, BytesIO, str, Path],
        compression: Optional[Literal["uncompressed", "lz4", "zstd"]] = "uncompressed",
    ) -> None:
        """
        .. deprecated:: 0.13.12
            Please use `write_ipc`
        """
        warnings.warn(
            "'to_ipc' is deprecated. please use 'write_ipc'", DeprecationWarning
        )
        return self.write_ipc(file, compression)

    def to_dicts(self) -> List[Dict[str, Any]]:
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
        column_names: Optional[Union[Iterator[str], Sequence[str]]] = None,
    ) -> DF:
        """
        Transpose a DataFrame over the diagonal.

        Parameters
        ----------
        include_header:
            If set, the column names will be added as first column.
        header_name:
            If `include_header` is set, this determines the name of the column that will be inserted
        column_names:
            Optional generator/iterator that yields column names. Will be used to replace the columns in the DataFrame.

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

        # replace the auto generated column names with a list

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

        Replace the auto generated column with column names from a generator function

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
        file: Union[str, Path, BytesIO],
        compression: Optional[
            Union[
                Literal[
                    "uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"
                ],
                str,
            ]
        ] = "lz4",
        statistics: bool = False,
        use_pyarrow: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Write the DataFrame disk in parquet format.

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
        statistics
            Write statistics to the parquet headers. This requires extra compute.
        use_pyarrow
            Use C++ parquet implementation vs rust parquet implementation.
            At the moment C++ supports more features.

        **kwargs are passed to pyarrow.parquet.write_table
        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, (str, Path)):
            file = format_path(file)

        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(  # pragma: no cover
                    "'pyarrow' is required when using 'write_parquet(..., use_pyarrow=True)'."
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
            self._df.to_parquet(file, compression, statistics)

    def to_parquet(
        self,
        file: Union[str, Path, BytesIO],
        compression: Optional[
            Union[
                Literal[
                    "uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"
                ],
                str,
            ]
        ] = "snappy",
        statistics: bool = False,
        use_pyarrow: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        .. deprecated:: 0.13.12
            Please use `write_parquet`
        """
        warnings.warn(
            "'to_parquet' is deprecated. please use 'write_parquet'", DeprecationWarning
        )
        return self.write_parquet(file, compression, statistics, use_pyarrow, **kwargs)

    def to_numpy(self) -> np.ndarray:
        """
        Convert DataFrame to a 2d numpy array.
        This operation clones data.

        Notes
        -----
        If you're attempting to convert Utf8 to an array you'll need to install `pyarrow`.

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

    def __getstate__(self):  # type: ignore
        return self.get_columns()

    def __setstate__(self, state):  # type: ignore
        self._df = DataFrame(state)._df

    def __mul__(
        self: DF, other: Union["DataFrame", "pli.Series", int, float, bool]
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.mul_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.mul(other._s))

    def __truediv__(
        self: DF, other: Union["DataFrame", "pli.Series", int, float, bool]
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.div_df(other._df))

        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.div(other._s))

    def __add__(
        self: DF,
        other: Union["DataFrame", "pli.Series", int, float, bool, str],
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.add_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.add(other._s))

    def __sub__(
        self: DF, other: Union["DataFrame", "pli.Series", int, float, bool]
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.sub_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.sub(other._s))

    def __mod__(
        self: DF, other: Union["DataFrame", "pli.Series", int, float, bool]
    ) -> DF:
        if isinstance(other, DataFrame):
            return self._from_pydf(self._df.rem_df(other._df))
        other = _prepare_other_arg(other)
        return self._from_pydf(self._df.rem(other._s))

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item: Any) -> "PySeries":
        """
        Access columns as attribute.
        """
        # it is important that we return an AttributeError here
        # this is used by ipython to check some private
        # `_ipython_canary_method_should_not_exist_`
        # if we return any other error than AttributeError pretty printing
        # will not work in notebooks.
        # See: https://github.com/jupyter/notebook/issues/2014
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            warnings.warn(
                "accessing series as Attribute of a DataFrame is deprecated",
                DeprecationWarning,
            )
            return pli.wrap_s(self._df.column(item))
        except Exception:
            raise AttributeError(item)

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

    # __getitem__() mostly returns a dataframe. The major exception is when a string is passed in. Note that there are
    # more subtle cases possible where a non-string value leads to a Series.
    @overload
    def __getitem__(self, item: str) -> "pli.Series":
        ...

    @overload
    def __getitem__(
        self: DF,
        item: Union[
            int, range, slice, np.ndarray, "pli.Expr", "pli.Series", List, tuple
        ],
    ) -> DF:
        ...

    def __getitem__(
        self: DF,
        item: Union[
            str, int, range, slice, np.ndarray, "pli.Expr", "pli.Series", List, tuple
        ],
    ) -> Union[DF, "pli.Series"]:
        """
        Does quite a lot. Read the comments.
        """
        if isinstance(item, pli.Expr):
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
                if isinstance(col_selection, (slice, list, np.ndarray)):
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
                # select by column indexes
                if isinstance(col_selection[0], int):
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
            # special case df[::-1]
            if item.start is None and item.stop is None and item.step == -1:
                return self.select(pli.col("*").reverse())

            if getattr(item, "end", False):
                raise ValueError("A slice with steps larger than 1 is not supported.")
            if item.start is None:
                start = 0
            else:
                start = item.start
            if item.stop is None:
                stop = self.height
            else:
                stop = item.stop

            length = stop - start
            if item.step is None:
                # df[start:stop]
                return self.slice(start, length)
            else:
                # df[start:stop:step]
                return self.select(
                    pli.col("*").slice(start, length).take_every(item.step)
                )

        # select rows by numpy mask or index
        # df[[1, 2, 3]]
        # df[[true, false, true]]
        if isinstance(item, np.ndarray):
            if item.dtype == int:
                return self._from_pydf(self._df.take(item))
            if isinstance(item[0], str):
                return self._from_pydf(self._df.select(item))
            if item.dtype == bool:
                return self._from_pydf(self._df.filter(pli.Series("", item).inner()))

        if isinstance(item, Sequence):
            if isinstance(item[0], str):
                # select multiple columns
                # df[["foo", "bar"]]
                return self._from_pydf(self._df.select(item))
            elif isinstance(item[0], pli.Expr):
                return self.select(item)
            elif type(item[0]) == bool:
                item = pli.Series("", item)  # fall through to next if isinstance
            elif is_int_sequence(item):
                return self._from_pydf(
                    self._df.take([self._pos_idx(i, dim=0) for i in item])
                )

        if isinstance(item, pli.Series):
            dtype = item.dtype
            if dtype == Boolean:
                return self._from_pydf(self._df.filter(item.inner()))
            if dtype == UInt32:
                return self._from_pydf(self._df.take_with_series(item.inner()))

        # if no data has been returned, the operation is not supported
        raise NotImplementedError

    def __setitem__(
        self, key: Union[str, List, Tuple[Any, Union[str, int]]], value: Any
    ) -> None:
        warnings.warn(
            "setting a DataFrame by indexing is deprecated; Consider using DataFrame.with_column",
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
            value = np.array(value)
            if len(value.shape) != 2:
                raise ValueError("can only set multiple columns with 2D matrix")
            if value.shape[1] != len(key):
                raise ValueError(
                    "matrix columns should be equal to list use to determine column names"
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
                s = self[:, col_selection]  # type: ignore
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
            raise NotImplementedError

    def __len__(self) -> int:
        return self.height

    def _repr_html_(self) -> str:
        """
        Used by jupyter notebooks to get a html table.

        Output rows and columns can be modified by setting the following ENVIRONMENT variables:

        * POLARS_FMT_MAX_COLS: set the number of columns
        * POLARS_FMT_MAX_ROWS: set the number of rows
        """
        max_cols = int(os.environ.get("POLARS_FMT_MAX_COLS", default=75))
        max_rows = int(os.environ.get("POLARS_FMT_MAX_ROWS", default=25))
        return "\n".join(NotebookFormatter(self, max_cols, max_rows).render())

    def to_series(self, index: int = 0) -> "pli.Series":
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
        return pli.wrap_s(self._df.select_at_idx(index))

    def rename(self: DF, mapping: Dict[str, str]) -> DF:
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

    def insert_at_idx(self, index: int, series: "pli.Series") -> None:
        """
        Insert a Series at a certain column index. This operation is in place.

        Parameters
        ----------
        index
            Column to insert the new `Series` column.
        series
            `Series` to insert.
        """
        self._df.insert_at_idx(index, series._s)

    def filter(self: DF, predicate: "pli.Expr") -> DF:
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
    def shape(self) -> Tuple[int, int]:
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
    def columns(self) -> List[str]:
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
    def dtypes(self) -> List[Type[DataType]]:
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
        │ 1   ┆ 6   ┆ a   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        See Also
        --------
        schema : Return a dict of [column name, dtype]
        """
        return self._df.dtypes()

    @property
    def schema(self) -> Dict[str, Type[DataType]]:
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

        """
        return dict(zip(self.columns, self.dtypes))

    def describe(self: DF) -> DF:
        """
        Summary statistics for a DataFrame. Only summarizes numeric datatypes at the moment and returns nulls for non numeric datatypes.

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
        ┌──────────┬────────────────────┬─────┬──────┐
        │ describe ┆ a                  ┆ b   ┆ c    │
        │ ---      ┆ ---                ┆ --- ┆ ---  │
        │ str      ┆ f64                ┆ f64 ┆ f64  │
        ╞══════════╪════════════════════╪═════╪══════╡
        │ mean     ┆ 2.2666666666666666 ┆ 5   ┆ null │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ std      ┆ 1.1015141094572205 ┆ 1   ┆ null │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ min      ┆ 1                  ┆ 4   ┆ 0.0  │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ max      ┆ 3                  ┆ 6   ┆ 1    │
        ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ median   ┆ 2.8                ┆ 5   ┆ null │
        └──────────┴────────────────────┴─────┴──────┘

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

    def replace_at_idx(self, index: int, series: "pli.Series") -> None:
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
        self._df.replace_at_idx(index, series._s)

    @overload
    def sort(
        self: DF,
        by: Union[str, "pli.Expr", List[str], List["pli.Expr"]],
        reverse: Union[bool, List[bool]] = ...,
        nulls_last: bool = ...,
        *,
        in_place: Literal[False] = ...,
    ) -> DF:
        ...

    @overload
    def sort(
        self,
        by: Union[str, "pli.Expr", List[str], List["pli.Expr"]],
        reverse: Union[bool, List[bool]] = ...,
        nulls_last: bool = ...,
        *,
        in_place: Literal[True],
    ) -> None:
        ...

    @overload
    def sort(
        self: DF,
        by: Union[str, "pli.Expr", List[str], List["pli.Expr"]],
        reverse: Union[bool, List[bool]] = ...,
        nulls_last: bool = ...,
        *,
        in_place: bool,
    ) -> Optional[DF]:
        ...

    def sort(
        self: DF,
        by: Union[str, "pli.Expr", List[str], List["pli.Expr"]],
        reverse: Union[bool, List[bool]] = False,
        nulls_last: bool = False,
        *,
        in_place: bool = False,
    ) -> Optional[DF]:
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
        │ 3   ┆ 8   ┆ c   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 6   ┆ a   │
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
        │ 3   ┆ 64  ┆ c   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 49  ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 36  ┆ a   │
        └─────┴─────┴─────┘

        """
        if type(by) is list or isinstance(by, pli.Expr):
            df = (
                self.lazy()
                .sort(by, reverse, nulls_last)
                .collect(no_optimization=True, string_cache=False)
            )
            if in_place:
                warnings.warn(
                    "in-place sorting is deprecated; please use default sorting",
                    DeprecationWarning,
                )
                self._df = df._df
                return self
            return df
        if in_place:
            warnings.warn(
                "in-place sorting is deprecated; please use default sorting",
                DeprecationWarning,
            )
            self._df.sort_in_place(by, reverse)
            return None
        else:
            return self._from_pydf(self._df.sort(by, reverse, nulls_last))

    def frame_equal(self, other: "DataFrame", null_equal: bool = True) -> bool:
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

    def replace(self, column: str, new_col: "pli.Series") -> None:
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

    def slice(self: DF, offset: int, length: int) -> DF:
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
        │ 2   ┆ 7   ┆ b   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        """
        if length < 0:
            length = self.height - offset + length
        return self._from_pydf(self._df.slice(offset, length))

    def limit(self: DF, length: int = 5) -> DF:
        """
        Get first N rows as DataFrame.

        See Also `DataFrame.head`

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

    def drop_nulls(self: DF, subset: Optional[Union[str, List[str]]] = None) -> DF:
        """
        Return a new DataFrame where the null values are dropped.

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
        by: Union[str, "pli.Expr", Sequence[str], Sequence["pli.Expr"]],
        maintain_order: bool = False,
    ) -> "GroupBy[DF]":
        """
        Start a groupby operation.

        Parameters
        ----------
        by
            Column(s) to group by.
        maintain_order
            Make sure that the order of the groups remain consistent. This is more expensive than a default groupby.
            Note that this only works in expression aggregations.

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
        >>> df.groupby("a")["b"].sum().sort(by="a")
        shape: (3, 2)
        ┌─────┬───────┐
        │ a   ┆ b_sum │
        │ --- ┆ ---   │
        │ str ┆ i64   │
        ╞═════╪═══════╡
        │ a   ┆ 4     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b   ┆ 11    │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c   ┆ 6     │
        └─────┴───────┘

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
        if isinstance(by, str):
            by = [by]
        return GroupBy(
            self._df,
            by,  # type: ignore
            dataframe_class=self.__class__,
            maintain_order=maintain_order,
        )

    def groupby_rolling(
        self: DF,
        index_column: str,
        period: str,
        offset: Optional[str] = None,
        closed: str = "right",
        by: Optional[Union[str, List[str], "pli.Expr", List["pli.Expr"]]] = None,
    ) -> "RollingGroupBy[DF]":
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

        - **"1i"      # length 1**
        - **"10i"     # length 10**

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

        return RollingGroupBy(self, index_column, period, offset, closed, by)

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
    ) -> "DynamicGroupBy":
        """
        Groups based on a time value (or index value of type Int32, Int64). Time windows are calculated and rows are assigned to windows.
        Different from a normal groupby is that a row can be member of multiple groups. The time/index window could
        be seen as a rolling window, with a window size determined by dates/times/values instead of slots in the DataFrame.

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
            add the lower and upper bound of the window to the "_lower_bound" and "_upper_bound" columns.
            this will impact performance because it's harder to parallelize
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
        shape: (3, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ time                ┆ time_min            ┆ time_max            │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╡
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
        shape: (3, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        ┆ u32        │
        ╞═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        └─────────────────────┴─────────────────────┴─────────────────────┴────────────┘

        When closed="left", should not include right end of interval [lower_bound, upper_bound)

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
        │ datetime[ns]        ┆ u32        ┆ list [datetime[ns]]                 │
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
        shape: (4, 2)
        ┌─────────────────────┬────────────┐
        │ time                ┆ time_count │
        │ ---                 ┆ ---        │
        │ datetime[ns]        ┆ u32        │
        ╞═════════════════════╪════════════╡
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
        shape: (6, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ str    ┆ datetime[ns]        ┆ datetime[ns]        ┆ datetime[ns]        ┆ u32        │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
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
        │ i64             ┆ i64             ┆ i64 ┆ list [str]      │
        ╞═════════════════╪═════════════════╪═════╪═════════════════╡
        │ 0               ┆ 3               ┆ 0   ┆ ["A", "B", "B"] │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2               ┆ 5               ┆ 2   ┆ ["B", "B", "C"] │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 4               ┆ 7               ┆ 4   ┆ ["C"]           │
        └─────────────────┴─────────────────┴─────┴─────────────────┘

        """

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
        offset: Optional[str] = None,
        by: Optional[Union[str, Sequence[str]]] = None,
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
        │ datetime[ns]        ┆ str    ┆ i64    │
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

    def join_asof(
        self: DF,
        df: "DataFrame",
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
        return (
            self.lazy()
            .join_asof(
                df.lazy(),
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

    def join(
        self: DF,
        df: "DataFrame",
        left_on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        right_on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        on: Optional[Union[str, "pli.Expr", List[Union[str, "pli.Expr"]]]] = None,
        how: str = "inner",
        suffix: str = "_right",
        asof_by: Optional[Union[str, List[str]]] = None,
        asof_by_left: Optional[Union[str, List[str]]] = None,
        asof_by_right: Optional[Union[str, List[str]]] = None,
    ) -> DF:
        """
        SQL like joins.

        Parameters
        ----------
        df
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
        │ 1   ┆ 6   ┆ a   ┆ x     │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘

        >>> df.join(other_df, on="ham", how="outer")
        shape: (4, 4)
        ┌──────┬──────┬─────┬───────┐
        │ foo  ┆ bar  ┆ ham ┆ apple │
        │ ---  ┆ ---  ┆ --- ┆ ---   │
        │ i64  ┆ f64  ┆ str ┆ str   │
        ╞══════╪══════╪═════╪═══════╡
        │ 1    ┆ 6    ┆ a   ┆ x     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2    ┆ 7    ┆ b   ┆ y     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ null ┆ null ┆ d   ┆ z     │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3    ┆ 8    ┆ c   ┆ null  │
        └──────┴──────┴─────┴───────┘

        **Asof join**
        This is similar to a left-join except that we match on near keys rather than equal keys.
        The direction is backward
        The keys must be sorted to perform an asof join

        """
        if how == "asof":
            warnings.warn(
                "using asof join via DataFrame.join is deprecated, please use DataFrame.join_asof",
                DeprecationWarning,
            )
        if how == "cross":
            return self._from_pydf(self._df.join(df._df, [], [], how, suffix))

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
                    df.lazy(),
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
                self._df.join(df._df, left_on_, right_on_, how, suffix)
            )

    def apply(
        self: DF,
        f: Callable[[Tuple[Any, ...]], Any],
        return_dtype: Optional[Type[DataType]] = None,
        inference_size: int = 256,
    ) -> DF:
        """
        Apply a custom function over the rows of the DataFrame. The rows are passed as tuple.

        Beware, this is slow.

        Parameters
        ----------
        f
            Custom function/ lambda function.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema

        """
        out, is_df = self._df.apply(f, return_dtype, inference_size)
        if is_df:
            return self._from_pydf(out)
        else:
            return self._from_pydf(pli.wrap_s(out).to_frame()._df)

    def with_column(self: DF, column: Union["pli.Series", "pli.Expr"]) -> DF:
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
        if isinstance(column, pli.Expr):
            return self.with_columns([column])
        else:
            return self._from_pydf(self._df.with_column(column._s))

    def hstack(
        self: DF,
        columns: Union[List["pli.Series"], "DataFrame"],
        in_place: bool = False,
    ) -> Optional[DF]:
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
    def vstack(self, df: "DataFrame", in_place: Literal[True]) -> None:
        ...

    @overload
    def vstack(self: DF, df: "DataFrame", in_place: Literal[False] = ...) -> DF:
        ...

    @overload
    def vstack(self: DF, df: "DataFrame", in_place: bool) -> Optional[DF]:
        ...

    def vstack(self: DF, df: "DataFrame", in_place: bool = False) -> Optional[DF]:
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

    def extend(self, other: "DataFrame") -> None:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from `vstack` which adds the chunks from `other` to the chunks of this `DataFrame`
        `extent` appends the data from `other` to the underlying memory locations and thus may cause a reallocation.

        If this does not cause a reallocation, the resulting data structure will not have any extra chunks
        and thus will yield faster queries.

        Prefer `extend` over `vstack` when you want to do a query after a single append. For instance during
        online operations where you add `n` rows and rerun a query.

        Prefer `vstack` over `extend` when you want to append many times before doing a query. For instance
        when you read in multiple files and when to store them in a single `DataFrame`.
        In the latter case, finish the sequence of `vstack` operations with a `rechunk`.

        Parameters
        ----------
        other
            DataFrame to vertically add.
        """
        self._df.extend(other._df)

    def drop(self: DF, name: Union[str, List[str]]) -> DF:
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
        │ 1   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   │
        └─────┴─────┘

        """
        if isinstance(name, list):
            df = self.clone()

            for name in name:
                df._df.drop_in_place(name)
            return df

        return self._from_pydf(self._df.drop(name))

    def drop_in_place(self, name: str) -> "pli.Series":
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

    def select_at_idx(self, idx: int) -> "pli.Series":
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
        return pli.wrap_s(self._df.select_at_idx(idx))

    def clone(self: DF) -> DF:
        """
        Very cheap deep clone.
        """
        return self._from_pydf(self._df.clone())

    def __copy__(self: DF) -> DF:
        return self.clone()

    def __deepcopy__(self: DF, memodict={}) -> DF:  # type: ignore
        return self.clone()

    def get_columns(self) -> List["pli.Series"]:
        """
        Get the DataFrame as a List of Series.
        """
        return list(map(lambda s: pli.wrap_s(s), self._df.get_columns()))

    def get_column(self, name: str) -> "pli.Series":
        """
        Get a single column as Series by name.
        """
        return self[name]

    def fill_null(self: DF, strategy: Union[str, "pli.Expr", Any]) -> DF:
        """
        Fill null values using a filling strategy, literal, or Expr.

        Parameters
        ----------
        strategy
            One of:
            - "backward"
            - "forward"
            - "mean"
            - "min'
            - "max"
            - "zero"
            - "one"
            Or an expression.

        Returns
        -------
            DataFrame with None values replaced by the filling strategy.
        """
        if isinstance(strategy, pli.Expr):
            return self.lazy().fill_null(strategy).collect(no_optimization=True)
        if not isinstance(strategy, str):
            return self.fill_null(pli.lit(strategy))
        return self._from_pydf(self._df.fill_null(strategy))

    def fill_nan(self: DF, fill_value: Union["pli.Expr", int, float]) -> DF:
        """
        Fill floating point NaN values by an Expression evaluation.

        Warnings
        --------
        NOTE that floating point NaN (No a Number) are not missing values!
        to replace missing values, use `fill_null`.

        Parameters
        ----------
        fill_value
            value to fill NaN with

        Returns
        -------
            DataFrame with NaN replaced with fill_value
        """
        return self.lazy().fill_nan(fill_value).collect(no_optimization=True)

    def explode(
        self: DF,
        columns: Union[str, List[str], "pli.Expr", List["pli.Expr"]],
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
        return self.lazy().explode(columns).collect(no_optimization=True)

    def pivot(
        self: DF,
        values: Union[List[str], str],
        index: Union[List[str], str],
        columns: Union[List[str], str],
        aggregate_fn: str = "first",
        maintain_order: bool = True,
        sort_columns: bool = False,
    ) -> DF:
        """
        Create a spreadsheet-style pivot table as a DataFrame.

        Parameters
        ----------
        values
            Column values to aggregate. Can be multiple columns if the *columns* arguments contains multiple columns as well
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
        id_vars: Optional[Union[List[str], str]] = None,
        value_vars: Optional[Union[List[str], str]] = None,
        variable_name: Optional[str] = None,
        value_name: Optional[str] = None,
    ) -> DF:
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

    def partition_by(
        self, groups: Union[str, List[str]], maintain_order: bool = True
    ) -> List[DF]:
        """
        Split into multiple DataFrames partitioned by groups.

        Parameters
        ----------
        groups
            Groups to partition by
        maintain_order
            Keep predictable output order. This is slower as it requires and extra sort operation.

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

        return [
            self._from_pydf(_df)  # type: ignore
            for _df in self._df.partition_by(groups, maintain_order)
        ]

    def shift(self: DF, periods: int) -> DF:
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

    def shift_and_fill(
        self: DF, periods: int, fill_value: Union[int, str, float]
    ) -> DF:
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

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

    def is_duplicated(self) -> "pli.Series":
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

    def is_unique(self) -> "pli.Series":
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

    def lazy(self: DF) -> "pli.LazyFrame[DF]":
        """
        Start a lazy query from this point. This returns a `LazyFrame` object.

        Operations on a `LazyFrame` are not executed until this is requested by either calling:

        * `.fetch()` (run on a small number of rows)
        * `.collect()` (run on all data)
        * `.describe_plan()` (print unoptimized query plan)
        * `.describe_optimized_plan()` (print optimized query plan)
        * `.show_graph()` (show (un)optimized query plan) as graphiz graph)

        Lazy operations are advised because they allow for query optimization and more parallelization.
        """
        return self._lazyframe_class._from_pyldf(self._df.lazy())

    def select(
        self: DF,
        exprs: Union[
            str,
            "pli.Expr",
            Sequence[Union[str, "pli.Expr", bool, int, float, "pli.Series"]],
            "pli.Series",
        ],
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
            .select(exprs)  # type: ignore
            .collect(no_optimization=True, string_cache=False)
        )

    def with_columns(self: DF, exprs: Union["pli.Expr", List["pli.Expr"]]) -> DF:
        """
        Add or overwrite multiple columns in a DataFrame.

        Parameters
        ----------
        exprs
            List of Expressions that evaluate to columns.
        """
        if not isinstance(exprs, list):
            exprs = [exprs]
        return (
            self.lazy()
            .with_columns(exprs)
            .collect(no_optimization=True, string_cache=False)
        )

    def n_chunks(self) -> int:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame.
        """
        return self._df.n_chunks()

    @overload
    def max(self: DF, axis: Literal[0] = ...) -> DF:
        ...

    @overload
    def max(self, axis: Literal[1]) -> "pli.Series":
        ...

    @overload
    def max(self: DF, axis: int = 0) -> Union[DF, "pli.Series"]:
        ...

    def max(self: DF, axis: int = 0) -> Union[DF, "pli.Series"]:
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
    def min(self, axis: Literal[1]) -> "pli.Series":
        ...

    @overload
    def min(self: DF, axis: int = 0) -> Union[DF, "pli.Series"]:
        ...

    def min(self: DF, axis: int = 0) -> Union[DF, "pli.Series"]:
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
    def sum(self, *, axis: Literal[1], null_strategy: str = "ignore") -> "pli.Series":
        ...

    @overload
    def sum(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union[DF, "pli.Series"]:
        ...

    def sum(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union[DF, "pli.Series"]:
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
    def mean(self, *, axis: Literal[1], null_strategy: str = "ignore") -> "pli.Series":
        ...

    @overload
    def mean(
        self: DF, *, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union[DF, "pli.Series"]:
        ...

    def mean(
        self: DF, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union[DF, "pli.Series"]:
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
        │ 2   ┆ 7   ┆ null │
        └─────┴─────┴──────┘

        Note: the mean of booleans evaluates to null.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [True, True, False],
        ...         "b": [True, True, True],
        ...     }
        ... )
        # mean evaluates to null
        >>> df.mean()
        shape: (1, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ bool ┆ bool │
        ╞══════╪══════╡
        │ null ┆ null │
        └──────┴──────┘
        # instead, cast to numeric type
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
        │ 1   ┆ 1   ┆ null │
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
        │ 1   ┆ 1   ┆ null │
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
        │ 2   ┆ 7   ┆ null │
        └─────┴─────┴──────┘

        """
        return self._from_pydf(self._df.median())

    def product(self: DF) -> DF:
        """
        Aggregate the columns of this DataFrame to their product values
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
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

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
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2   ┆ 7   ┆ null │
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
        ┌───────┬───────┬───────┬───────┬─────┬───────┬───────┬───────┬───────┐
        │ foo_1 ┆ foo_2 ┆ foo_3 ┆ bar_6 ┆ ... ┆ bar_8 ┆ ham_a ┆ ham_b ┆ ham_c │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆     ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆     ┆ u8    ┆ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╪═══════╪═════╪═══════╪═══════╪═══════╪═══════╡
        │ 1     ┆ 0     ┆ 0     ┆ 1     ┆ ... ┆ 0     ┆ 1     ┆ 0     ┆ 0     │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 0     ┆ 1     ┆ 0     ┆ 0     ┆ ... ┆ 0     ┆ 0     ┆ 1     ┆ 0     │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 0     ┆ 0     ┆ 1     ┆ 0     ┆ ... ┆ 1     ┆ 0     ┆ 0     ┆ 1     │
        └───────┴───────┴───────┴───────┴─────┴───────┴───────┴───────┴───────┘

        """
        return self._from_pydf(self._df.to_dummies())

    def distinct(
        self: DF,
        maintain_order: bool = True,
        subset: Optional[Union[str, List[str]]] = None,
        keep: str = "first",
    ) -> DF:
        """
        .. deprecated:: 0.13.13
            Please use `unique`
        """
        return self.unique(maintain_order, subset, keep)

    def unique(
        self: DF,
        maintain_order: bool = True,
        subset: Optional[Union[str, List[str]]] = None,
        keep: str = "first",
    ) -> DF:
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
        return self._from_pydf(self._df.unique(maintain_order, subset, keep))

    def rechunk(self: DF) -> DF:
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable performance.
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
        n: Optional[int] = None,
        frac: Optional[float] = None,
        with_replacement: bool = False,
        seed: Optional[int] = None,
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
            return self._from_pydf(self._df.sample_frac(frac, with_replacement, seed))

        if n is None:
            n = 1

        return self._from_pydf(self._df.sample_n(n, with_replacement, seed))

    def fold(
        self, operation: Callable[["pli.Series", "pli.Series"], "pli.Series"]
    ) -> "pli.Series":
        """
        Apply a horizontal reduction on a DataFrame. This can be used to effectively
        determine aggregations on a row level, and can be applied to any DataType that
        can be supercasted (casted to a similar parent type).

        An example of the supercast rules when applying an arithmetic operation on two DataTypes are for instance:

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
            4
            5
            9
        ]

        A horizontal minimum operation:

        >>> df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
        >>> df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2))
        shape: (3,)
        Series: 'a' [f64]
        [
            1
            1
            3
        ]

        A horizontal string concattenation:

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

    def row(self, index: int) -> Tuple[Any]:
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

    def rows(self) -> List[Tuple]:
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
    def shrink_to_fit(self: DF, in_place: bool) -> Optional[DF]:
        ...

    def shrink_to_fit(self: DF, in_place: bool = False) -> Optional[DF]:
        """
        Shrink memory usage of this DataFrame to fit the exact capacity needed to hold the data.
        """
        if in_place:
            self._df.shrink_to_fit()
            return None
        else:
            df = self.clone()
            df._df.shrink_to_fit()
            return df

    def hash_rows(
        self, k0: int = 0, k1: int = 1, k2: int = 2, k3: int = 3
    ) -> "pli.Series":
        """
        Hash and combine the rows in this DataFrame.
        Hash value is UInt64.

        Parameters
        ----------
        k0
            seed parameter
        k1
            seed parameter
        k2
            seed parameter
        k3
            seed parameter

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.hash(k0=42)  # doctest: +SKIP
        shape: (3,)
        Series: '' [u64]
        [
            1208206736888326229
            8040480609798856146
            18282897888575762835
        ]
        """
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
        """
        return self.height == 0

    def to_struct(self, name: str) -> "pli.Series":
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
        Series: 'nums' [struct[2]{'a': i64, 'b': str}]
        [
                {1,"one"}
                {2,"two"}
                {3,"three"}
                {4,"four"}
                {5,"five"}
        ]

        """
        return pli.wrap_s(self._df.to_struct(name))

    def unnest(self: DF, names: Union[str, List[str]]) -> DF:
        """
        Decompose a struct into its fields. The fields will be inserted in to the `DataFrame` on the
        location of the `struct` type.

        Parameters
        ----------
        names
           Names of the struct columns that will be decomposed by its fields

        Examples
        --------

        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "int": [1, 2],
        ...             "str": ["a", "b"],
        ...             "bool": [True, None],
        ...             "list": [[1, 2], [3]],
        ...         }
        ...     )
        ...     .to_struct("my_struct")
        ...     .to_frame()
        ... )
        >>> df
        shape: (2, 1)
        ┌─────────────────────────────┐
        │ my_struct                   │
        │ ---                         │
        │ struct[4]{'int',...,'list'} │
        ╞═════════════════════════════╡
        │ {1,"a",true,[1, 2]}         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {2,"b",null,[3]}            │
        └─────────────────────────────┘
        >>> df.unnest("my_struct")
        shape: (2, 4)
        ┌─────┬─────┬──────┬────────────┐
        │ int ┆ str ┆ bool ┆ list       │
        │ --- ┆ --- ┆ ---  ┆ ---        │
        │ i64 ┆ str ┆ bool ┆ list [i64] │
        ╞═════╪═════╪══════╪════════════╡
        │ 1   ┆ a   ┆ true ┆ [1, 2]     │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ b   ┆ null ┆ [3]        │
        └─────┴─────┴──────┴────────────┘

        """
        if isinstance(names, str):
            names = [names]
        return self._from_pydf(self._df.unnest(names))


class RollingGroupBy(Generic[DF]):
    """
    A rolling grouper. This has an `.agg` method which will allow you to run all polars expressions
    in a groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        period: str,
        offset: Optional[str],
        closed: str = "none",
        by: Optional[Union[str, List[str], "pli.Expr", List["pli.Expr"]]] = None,
    ):
        self.df = df
        self.time_column = index_column
        self.period = period
        self.offset = offset
        self.closed = closed
        self.by = by

    def agg(
        self,
        column_to_agg: Union[
            List[Tuple[str, List[str]]],
            Dict[str, Union[str, List[str]]],
            List["pli.Expr"],
            "pli.Expr",
        ],
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
    A dynamic grouper. This has an `.agg` method which will allow you to run all polars expressions
    in a groupby context.
    """

    def __init__(
        self,
        df: DF,
        index_column: str,
        every: str,
        period: Optional[str],
        offset: Optional[str],
        truncate: bool = True,
        include_boundaries: bool = True,
        closed: str = "none",
        by: Optional[Union[str, List[str], "pli.Expr", List["pli.Expr"]]] = None,
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
        column_to_agg: Union[
            List[Tuple[str, List[str]]],
            Dict[str, Union[str, List[str]]],
            List["pli.Expr"],
            "pli.Expr",
        ],
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
        df: "PyDataFrame",
        by: Union[str, List[str]],
        dataframe_class: Type[DF],
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

    def __getitem__(self, item: Any) -> "GBSelection[DF]":
        print(
            "accessing GroupBy by index is deprecated, consider using the `.agg` method"
        )
        return self._select(item)

    def _select(self, columns: Union[str, List[str]]) -> "GBSelection[DF]":
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns.
        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg` method",
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

    def get_group(self, group_value: Union[Any, Tuple[Any]]) -> DF:
        """
        Select a single group as a new DataFrame.

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
            groups_idx = groups[mask][0]  # type: ignore
        except IndexError:
            raise ValueError(f"no group: {group_value} found")

        df = self._dataframe_class._from_pydf(self._df)
        return df[groups_idx]

    def groups(self) -> DF:
        """
        Return a `DataFrame` with:

        * the groupby keys
        * the group indexes aggregated as lists
        """
        warnings.warn(
            "accessing GroupBy by index is deprecated, consider using the `.agg` method",
            DeprecationWarning,
        )
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, None, "groups")
        )

    def apply(self, f: Callable[[DataFrame], DataFrame]) -> DF:
        """
        Apply a function over the groups as a sub-DataFrame.

        Beware, this is slow.

        Parameters
        ----------
        f
            Custom function.

        Returns
        -------
        DataFrame
        """
        return self._dataframe_class._from_pydf(self._df.groupby_apply(self.by, f))

    def agg(
        self,
        column_to_agg: Union[
            List[Tuple[str, List[str]]],
            Dict[str, Union[str, List[str]]],
            List["pli.Expr"],
            "pli.Expr",
        ],
    ) -> DF:
        """
        Use multiple aggregations on columns. This can be combined with complete lazy API
        and is considered idiomatic polars.

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
        >>> df.groupby("foo").agg(
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

                pass
            else:
                raise ValueError(
                    f"argument: {column_to_agg} not understood, have you passed a list of expressions?"
                )
        else:
            raise ValueError(
                f"argument: {column_to_agg} not understood, have you passed a list of expressions?"
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

    def _select_all(self) -> "GBSelection[DF]":
        """
        Select all columns for aggregation.
        """
        return GBSelection(
            self._df,
            self.by,
            None,
            dataframe_class=self._dataframe_class,
        )

    def pivot(
        self, pivot_column: Union[str, List[str]], values_column: Union[str, List[str]]
    ) -> "PivotOps[DF]":
        """
        Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.

        .. note::
            Polars'/arrow memory is not ideal for transposing operations like pivots. If you have a relatively large
            table, consider using a groupby over a pivot.

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
        >>> df.groupby("foo").pivot(pivot_column="bar", values_column="baz").first()
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
        """
        return self.agg(pli.all().first())

    def last(self) -> DF:
        """
        Aggregate the last values in the group.
        """
        return self.agg(pli.all().last())

    def sum(self) -> DF:
        """
        Reduce the groups to the sum.
        """
        return self.agg(pli.all().sum())

    def min(self) -> DF:
        """
        Reduce the groups to the minimal value.
        """
        return self.agg(pli.all().min())

    def max(self) -> DF:
        """
        Reduce the groups to the maximal value.
        """
        return self.agg(pli.all().max())

    def count(self) -> DF:
        """
        Count the number of values in each group.
        """
        return self.agg(pli.lazy_functions.count())

    def mean(self) -> DF:
        """
        Reduce the groups to the mean values.
        """
        return self.agg(pli.all().mean())

    def n_unique(self) -> DF:
        """
        Count the unique values per group.
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
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        """
        return self.agg(pli.all().quantile(quantile, interpolation))

    def median(self) -> DF:
        """
        Return the median per group.
        """
        return self.agg(pli.all().median())

    def agg_list(self) -> DF:
        """
        Aggregate the groups into Series.

        Examples
        --------

        >>> df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
        >>> df.groupby("a").agg_list()
        shape: (2, 2)
        ┌─────┬────────────┐
        │ a   ┆ b          │
        │ --- ┆ ---        │
        │ str ┆ list [i64] │
        ╞═════╪════════════╡
        │ one ┆ [1, 3]     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ two ┆ [2, 4]     │
        └─────┴────────────┘

        """
        return self.agg(pli.all().list())


class PivotOps(Generic[DF]):
    """
    Utility class returned in a pivot operation.
    """

    def __init__(
        self,
        df: DataFrame,
        by: Union[str, List[str]],
        pivot_column: Union[str, List[str]],
        values_column: Union[str, List[str]],
        dataframe_class: Type[DF],
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column
        self._dataframe_class = dataframe_class

    def first(self) -> DF:
        """
        Get the first value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self) -> DF:
        """
        Get the sum per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self) -> DF:
        """
        Get the minimal value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self) -> DF:
        """
        Get the maximal value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self) -> DF:
        """
        Get the mean value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def count(self) -> DF:
        """
        Count the values per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "count")
        )

    def median(self) -> DF:
        """
        Get the median value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )

    def last(self) -> DF:
        """
        Get the last value per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "last")
        )


class GBSelection(Generic[DF]):
    """
    Utility class returned in a groupby operation.
    """

    def __init__(
        self,
        df: "PyDataFrame",
        by: Union[str, List[str]],
        selection: Optional[List[str]],
        dataframe_class: Type[DF],
    ):
        self._df = df
        self.by = by
        self.selection = selection
        self._dataframe_class = dataframe_class

    def first(self) -> DF:
        """
        Aggregate the first values in the group.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "first")
        )

    def last(self) -> DF:
        """
        Aggregate the last values in the group.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "last")
        )

    def sum(self) -> DF:
        """
        Reduce the groups to the sum.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "sum")
        )

    def min(self) -> DF:
        """
        Reduce the groups to the minimal value.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "min")
        )

    def max(self) -> DF:
        """
        Reduce the groups to the maximal value.
        """
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
        >>> df.groupby("bar").count()  # counts nulls
        shape: (3, 2)
        ┌─────┬───────┐
        │ bar ┆ count │
        │ --- ┆ ---   │
        │ str ┆ u32   │
        ╞═════╪═══════╡
        │ c   ┆ 1     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ a   ┆ 2     │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b   ┆ 1     │
        └─────┴───────┘

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "count")
        )

    def mean(self) -> DF:
        """
        Reduce the groups to the mean values.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "mean")
        )

    def n_unique(self) -> DF:
        """
        Count the unique values per group.
        """
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
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        """
        return self._dataframe_class._from_pydf(
            self._df.groupby_quantile(self.by, self.selection, quantile, interpolation)
        )

    def median(self) -> DF:
        """
        Return the median per group.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "median")
        )

    def agg_list(self) -> DF:
        """
        Aggregate the groups into Series.
        """
        return self._dataframe_class._from_pydf(
            self._df.groupby(self.by, self.selection, "agg_list")
        )

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> DF:
        """
        Apply a function over the groups.
        """
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

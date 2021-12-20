"""
Module containing logic related to eager DataFrames
"""
import os
import sys
import typing as tp
from datetime import timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Type,
    Union,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

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
    arrow_to_pydf,
    dict_to_pydf,
    numpy_to_pydf,
    pandas_to_pydf,
    sequence_to_pydf,
    series_to_pydf,
)

try:
    from polars.polars import PyDataFrame, PySeries

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True

from polars._html import NotebookFormatter
from polars.datatypes import Boolean, DataType, Datetime, UInt32, py_type_to_dtype
from polars.utils import _process_null_values

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


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


class DataFrame:
    """
    A DataFrame is a two-dimensional data structure that represents data as a table
    with rows and columns.

    Parameters
    ----------
    data : dict, Sequence, ndarray, Series, or pandas.DataFrame
        Two-dimensional data in various forms. dict must contain Sequences.
        Sequence may contain Series or other Sequences.
    columns : Sequence of str, default None
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
    of Series instead:

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
    │ 1    ┆ 3    │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ 2    ┆ 4    │
    └──────┴──────┘

    Constructing a DataFrame from a numpy ndarray, specifying column names:

    >>> import numpy as np
    >>> data = np.array([(1, 2), (3, 4)], dtype=np.int64)
    >>> df3 = pl.DataFrame(data, columns=["a", "b"], orient="col")
    >>> df3  # doctest: +IGNORE_RESULT
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
        columns: Optional[Sequence[str]] = None,
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
                raise ImportError(
                    "'pyarrow' is required for converting a pandas DataFrame to a polars DataFrame."
                )
            self._df = pandas_to_pydf(data, columns=columns)

        else:
            raise ValueError("DataFrame constructor not called properly.")

    @classmethod
    def _from_pydf(cls, py_df: "PyDataFrame") -> "DataFrame":
        """
        Construct Polars DataFrame from FFI PyDataFrame object.
        """
        df = cls.__new__(cls)
        df._df = py_df
        return df

    @classmethod
    def _from_dicts(cls, data: Sequence[Dict[str, Any]]) -> "DataFrame":
        pydf = PyDataFrame.read_dicts(data)
        return DataFrame._from_pydf(pydf)

    @classmethod
    def _from_dict(
        cls,
        data: Dict[str, Sequence[Any]],
        columns: Optional[Sequence[str]] = None,
    ) -> "DataFrame":
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
        cls,
        data: Union[np.ndarray, Sequence[Sequence[Any]]],
        columns: Optional[Sequence[str]] = None,
        orient: Optional[str] = None,
    ) -> "DataFrame":
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
        cls,
        data: "pa.Table",
        columns: Optional[Sequence[str]] = None,
        rechunk: bool = True,
    ) -> "DataFrame":
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
        cls,
        data: "pd.DataFrame",
        columns: Optional[Sequence[str]] = None,
        rechunk: bool = True,
        nan_to_none: bool = True,
    ) -> "DataFrame":
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
        return cls._from_pydf(
            pandas_to_pydf(
                data, columns=columns, rechunk=rechunk, nan_to_none=nan_to_none
            )
        )

    @staticmethod
    def read_csv(
        file: Union[str, BinaryIO, bytes],
        has_header: bool = True,
        columns: Optional[Union[tp.List[int], tp.List[str]]] = None,
        sep: str = ",",
        comment_char: Optional[str] = None,
        quote_char: Optional[str] = r'"',
        skip_rows: int = 0,
        dtypes: Optional[
            Union[Dict[str, Type[DataType]], tp.List[Type[DataType]]]
        ] = None,
        null_values: Optional[Union[str, tp.List[str], Dict[str, str]]] = None,
        ignore_errors: bool = False,
        parse_dates: bool = False,
        n_threads: Optional[int] = None,
        infer_schema_length: Optional[int] = 100,
        batch_size: int = 8192,
        n_rows: Optional[int] = None,
        encoding: str = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
    ) -> "DataFrame":
        """
        Read a CSV file into a Dataframe.

        Parameters
        ----------
        file
            Path to a file or file like object.
        has_header
            Indicate if the first row of dataset is a header or not.
            If set to False, column names will be autogenrated in the
            following format: ``column_x``, with ``x`` being an
            enumeration over every column in the dataset starting at 1.
        columns
            Columns to select. Accepts a list of column indices (starting
            at zero) or a list of column names.
        sep
            Character to use as delimiter in the file.
        comment_char
            Character that indicates the start of a comment line, for
            instance ``#``.
        quote_char
            Single byte character used for csv quoting, default = ''.
            Set to None to turn off special handling and escaping of quotes.
        skip_rows
            Start reading after ``skip_rows`` lines.
        dtypes
            Overwrite dtypes during inference.
        null_values
            Values to interpret as null values. You can provide a:
              - ``str``: All values equal to this string will be null.
              - ``List[str]``: A null value per column.
              - ``Dict[str, str]``: A dictionary that maps column name to a
                                    null value string.
        ignore_errors
            Try to keep reading lines if some lines yield errors.
            First try ``infer_schema_length=0`` to read all columns as
            ``pl.Utf8`` to check which values might cause an issue.
        parse_dates
            Try to automatically parse dates. If this does not succeed,
            the column remains of data type ``pl.Utf8``.
        n_threads
            Number of threads to use in csv parsing.
            Defaults to the number of physical cpu's of your system.
        infer_schema_length
            Maximum number of lines to read to infer schema.
            If set to 0, all columns will be read as ``pl.Utf8``.
            If set to ``None``, a full table scan will be done (slow).
        batch_size
            Number of lines to read into the buffer at once.
            Modify this to change performance.
        n_rows
            Stop reading from CSV file after reading ``n_rows``.
            During multi-threaded parsing, an upper bound of ``n_rows``
            rows cannot be guaranteed.
        encoding
            Allowed encodings: ``utf8`` or ``utf8-lossy``.
            Lossy means that invalid utf8 values are replaced with ``�``
            characters.
        low_memory
            Reduce memory usage at expense of performance.
        rechunk
            Make sure that all columns are contiguous in memory by
            aggregating the chunks into a single array.

        Returns
        -------
        DataFrame

        Examples
        --------

        >>> df = pl.read_csv("file.csv", sep=";", n_rows=25)  # doctest: +SKIP

        """
        self = DataFrame.__new__(DataFrame)

        path: Optional[str]
        if isinstance(file, str):
            path = file
        else:
            path = None
            if isinstance(file, BytesIO):
                file = file.getvalue()
            if isinstance(file, StringIO):
                file = file.getvalue().encode()

        projection: Optional[tp.List[int]] = None
        if columns:
            if isinstance(columns, list):
                if all(isinstance(i, int) for i in columns):
                    projection = columns  # type: ignore
                    columns = None
                elif not all(isinstance(i, str) for i in columns):
                    raise ValueError(
                        "columns arg should contain a list of all integers or all strings values."
                    )

        dtype_list: Optional[tp.List[Tuple[str, Type[DataType]]]] = None
        dtype_slice: Optional[tp.List[Type[DataType]]] = None
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
        )
        return self

    @staticmethod
    def read_parquet(
        file: Union[str, BinaryIO],
        columns: Optional[Union[tp.List[int], tp.List[str]]] = None,
        n_rows: Optional[int] = None,
    ) -> "DataFrame":
        """
        Read into a DataFrame from a parquet file.

        Parameters
        ----------
        file
            Path to a file or a file like object. Any valid filepath can be used.
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
        n_rows
            Stop reading from parquet file after reading ``n_rows``.
        """
        projection: Optional[tp.List[int]] = None
        if columns:
            if isinstance(columns, list):
                if all(isinstance(i, int) for i in columns):
                    projection = columns  # type: ignore
                    columns = None
                elif not all(isinstance(i, str) for i in columns):
                    raise ValueError(
                        "columns arg should contain a list of all integers or all strings values."
                    )

        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_parquet(file, columns, projection, n_rows)
        return self

    @staticmethod
    def read_ipc(
        file: Union[str, BinaryIO],
        columns: Optional[Union[tp.List[int], tp.List[str]]] = None,
        n_rows: Optional[int] = None,
    ) -> "DataFrame":
        """
        Read into a DataFrame from Arrow IPC stream format. This is also called the Feather (v2) format.

        Parameters
        ----------
        file
            Path to a file or a file like object.
        columns
            Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
        n_rows
            Stop reading from IPC file after reading ``n_rows``.

        Returns
        -------
        DataFrame
        """
        projection: Optional[tp.List[int]] = None
        if columns:
            if isinstance(columns, list):
                if all(isinstance(i, int) for i in columns):
                    projection = columns  # type: ignore
                    columns = None
                elif not all(isinstance(i, str) for i in columns):
                    raise ValueError(
                        "columns arg should contain a list of all integers or all strings values."
                    )

        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_ipc(file, columns, projection, n_rows)
        return self

    @staticmethod
    def read_json(file: Union[str, BytesIO]) -> "DataFrame":
        """
        Read into a DataFrame from JSON format.

        Parameters
        ----------
        file
            Path to a file or a file like object.
        """
        if not isinstance(file, str):
            file = file.read().decode("utf8")
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_json(file)
        return self

    def to_arrow(self) -> "pa.Table":
        """
        Collect the underlying arrow arrays in an Arrow Table.
        This operation is mostly zero copy.

        Data types that do copy:
            - CategoricalType
        """
        if not _PYARROW_AVAILABLE:
            raise ImportError(
                "'pyarrow' is required for converting a polars DataFrame to an Arrow Table."
            )
        record_batches = self._df.to_arrow()
        return pa.Table.from_batches(record_batches)

    def to_dict(
        self, as_series: bool = True
    ) -> Union[Dict[str, "pli.Series"], Dict[str, tp.List[Any]]]:
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

    @tp.overload
    def to_json(
        self,
        file: Optional[Union[BytesIO, str, Path]] = ...,
        pretty: bool = ...,
        *,
        to_string: Literal[True],
    ) -> str:
        ...

    @tp.overload
    def to_json(
        self,
        file: Optional[Union[BytesIO, str, Path]] = ...,
        pretty: bool = ...,
        *,
        to_string: Literal[False] = ...,
    ) -> None:
        ...

    @tp.overload
    def to_json(
        self,
        file: Optional[Union[BytesIO, str, Path]] = ...,
        pretty: bool = ...,
        *,
        to_string: bool = ...,
    ) -> Optional[str]:
        ...

    def to_json(
        self,
        file: Optional[Union[BytesIO, str, Path]] = None,
        pretty: bool = False,
        *,
        to_string: bool = False,
    ) -> Optional[str]:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            Write to this file instead of returning an string.
        pretty
            Pretty serialize json.
        to_string
            Ignore file argument and return a string.
        """
        if to_string or file is None:
            file = BytesIO()
            self._df.to_json(file, pretty)
            file.seek(0)
            return file.read().decode("utf8")
        else:
            self._df.to_json(file, pretty)
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
        return self.to_arrow().to_pandas(*args, date_as_object=date_as_object, **kwargs)

    def to_csv(
        self,
        file: Optional[Union[TextIO, BytesIO, str, Path]] = None,
        has_header: bool = True,
        sep: str = ",",
    ) -> Optional[str]:
        """
        Write Dataframe to comma-separated values file (csv).

        Parameters
        ----------
        file
            File path to which the file should be written.
        has_header
            Whether or not to include header in the CSV output.
        sep
            Separate CSV fields with this symbol.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.to_csv("new_file.csv", sep=",")

        """
        if file is None:
            buffer = BytesIO()
            self._df.to_csv(buffer, has_header, ord(sep))
            return str(buffer.getvalue(), encoding="utf-8")

        if isinstance(file, Path):
            file = str(file)

        self._df.to_csv(file, has_header, ord(sep))
        return None

    def to_ipc(
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
        if isinstance(file, Path):
            file = str(file)

        self._df.to_ipc(file, compression)

    def to_dicts(self) -> tp.List[Dict[str, Any]]:
        pydf = self._df
        names = self.columns

        return [
            {k: v for k, v in zip(names, pydf.row_tuple(i))}
            for i in range(0, self.height)
        ]

    def transpose(
        self,
        include_header: bool = False,
        header_name: str = "column",
        column_names: Optional[Union[tp.Iterator[str], tp.Sequence[str]]] = None,
    ) -> "pli.DataFrame":
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
        df = wrap_df(self._df.transpose(include_header, header_name))
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
        use_pyarrow
            Use C++ parquet implementation vs rust parquet implementation.
            At the moment C++ supports more features.

        **kwargs are passed to pyarrow.parquet.write_table
        """
        if compression is None:
            compression = "uncompressed"
        if isinstance(file, Path):
            file = str(file)

        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using 'to_parquet(..., use_pyarrow=True)'."
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
                table=tbl, where=file, compression=compression, **kwargs
            )
        else:
            self._df.to_parquet(file, compression)

    def to_numpy(self) -> np.ndarray:
        """
        Convert DataFrame to a 2d numpy array.
        This operation clones data.

        Examples
        --------

        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> numpy_array = df.to_numpy()
        >>> type(numpy_array)
        <class 'numpy.ndarray'>

        """
        return np.vstack([self.to_series(i).to_numpy() for i in range(self.width)]).T

    def __getstate__(self):  # type: ignore
        return self.get_columns()

    def __setstate__(self, state):  # type: ignore
        self._df = DataFrame(state)._df

    def __mul__(self, other: Any) -> "DataFrame":
        other = _prepare_other_arg(other)
        return wrap_df(self._df.mul(other._s))

    def __truediv__(self, other: Any) -> "DataFrame":
        other = _prepare_other_arg(other)
        return wrap_df(self._df.div(other._s))

    def __add__(self, other: Any) -> "DataFrame":
        other = _prepare_other_arg(other)
        return wrap_df(self._df.add(other._s))

    def __sub__(self, other: Any) -> "DataFrame":
        other = _prepare_other_arg(other)
        return wrap_df(self._df.sub(other._s))

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item: Any) -> "PySeries":
        """
        Access columns as attribute.
        """
        try:
            return pli.wrap_s(self._df.column(item))
        except RuntimeError:
            raise AttributeError(f"{item} not found")

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

    def __getitem__(self, item: Any) -> Any:
        """
        Does quite a lot. Read the comments.
        """
        if hasattr(item, "_pyexpr"):
            return self.select(item)
        if isinstance(item, np.ndarray):
            item = pli.Series("", item)
        # select rows and columns at once
        # every 2d selection, i.e. tuple is row column order, just like numpy
        if isinstance(item, tuple):
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
                    series = [self.to_series(i) for i in col_selection]
                    df = DataFrame(series)
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
            step: Optional[int]
            # maybe we can slice instead of take by indices
            if item.step != 1:
                step = item.step
            else:
                step = None
            slc = slice(item.start, item.stop, step)
            return self[slc]

        # df[:]
        if isinstance(item, slice):
            # special case df[::-1]
            if item.start is None and item.stop is None and item.step == -1:
                return self.select(pli.col("*").reverse())  # type: ignore

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
                    pli.col("*").slice(start, length).take_every(item.step)  # type: ignore
                )

        # select multiple columns
        # df["foo", "bar"]
        if isinstance(item, Sequence):
            if isinstance(item[0], str):
                return wrap_df(self._df.select(item))
            elif isinstance(item[0], pli.Expr):
                return self.select(item)

        # select rows by mask or index
        # df[[1, 2, 3]]
        # df[true, false, true]
        if isinstance(item, np.ndarray):
            if item.dtype == int:
                return wrap_df(self._df.take(item))
            if isinstance(item[0], str):
                return wrap_df(self._df.select(item))
        if isinstance(item, (pli.Series, Sequence)):
            if isinstance(item, Sequence):
                # only bool or integers allowed
                if type(item[0]) == bool:
                    item = pli.Series("", item)
                else:
                    return wrap_df(
                        self._df.take([self._pos_idx(i, dim=0) for i in item])
                    )
            dtype = item.dtype
            if dtype == Boolean:
                return wrap_df(self._df.filter(item.inner()))
            if dtype == UInt32:
                return wrap_df(self._df.take_with_series(item.inner()))

    def __setitem__(self, key: Union[str, int, Tuple[Any, Any]], value: Any) -> None:
        # df["foo"] = series
        if isinstance(key, str):
            try:
                self.replace(key, pli.Series(key, value))
            except Exception:
                self.hstack([pli.Series(key, value)], in_place=True)
        # df[idx] = series
        elif isinstance(key, int):
            assert isinstance(value, pli.Series)
            self.replace_at_idx(key, value)
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

    def rename(self, mapping: Dict[str, str]) -> "DataFrame":
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
        df = self.clone()
        for k, v in mapping.items():
            df._df.rename(k, v)
        return df

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

    def filter(self, predicate: "pli.Expr") -> "DataFrame":
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
    def columns(self) -> tp.List[str]:
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
    def dtypes(self) -> tp.List[Type[DataType]]:
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
        return {c: self[c].dtype for c in self.columns}

    def describe(self) -> "DataFrame":
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

        def describe_cast(self: "DataFrame") -> "DataFrame":
            columns = []
            for s in self:
                if s.is_numeric() or s.is_boolean():
                    columns.append(s.cast(float))
                else:
                    columns.append(s)
            return DataFrame(columns)

        summary = pli.concat(
            [
                describe_cast(self.mean()),  # type: ignore
                describe_cast(self.std()),
                describe_cast(self.min()),  # type: ignore
                describe_cast(self.max()),  # type: ignore
                describe_cast(self.median()),
            ]
        )
        summary.insert_at_idx(  # type: ignore
            0, pli.Series("describe", ["mean", "std", "min", "max", "median"])
        )
        return summary  # type: ignore

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

    @tp.overload
    def sort(
        self,
        by: Union[str, "pli.Expr", tp.List[str], tp.List["pli.Expr"]],
        reverse: Union[bool, tp.List[bool]] = ...,
        *,
        in_place: Literal[False] = ...,
    ) -> "DataFrame":
        ...

    @tp.overload
    def sort(
        self,
        by: Union[str, "pli.Expr", tp.List[str], tp.List["pli.Expr"]],
        reverse: Union[bool, tp.List[bool]] = ...,
        *,
        in_place: Literal[True],
    ) -> None:
        ...

    @tp.overload
    def sort(
        self,
        by: Union[str, "pli.Expr", tp.List[str], tp.List["pli.Expr"]],
        reverse: Union[bool, tp.List[bool]] = ...,
        *,
        in_place: bool,
    ) -> Optional["DataFrame"]:
        ...

    def sort(
        self,
        by: Union[str, "pli.Expr", tp.List[str], tp.List["pli.Expr"]],
        reverse: Union[bool, tp.List[bool]] = False,
        *,
        in_place: bool = False,
    ) -> Optional["DataFrame"]:
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
                .sort(by, reverse)
                .collect(no_optimization=True, string_cache=False)
            )
            if in_place:
                self._df = df._df
                return self
            return df
        if in_place:
            self._df.sort_in_place(by, reverse)
            return None
        else:
            return wrap_df(self._df.sort(by, reverse))

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
        """
        self._df.replace(column, new_col.inner())

    def slice(self, offset: int, length: int) -> "DataFrame":
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
        return wrap_df(self._df.slice(offset, length))

    def limit(self, length: int = 5) -> "DataFrame":
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

    def head(self, length: int = 5) -> "DataFrame":
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
        return wrap_df(self._df.head(length))

    def tail(self, length: int = 5) -> "DataFrame":
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
        return wrap_df(self._df.tail(length))

    def drop_nulls(self, subset: Optional[tp.List[str]] = None) -> "DataFrame":
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
        if subset is not None and isinstance(subset, str):
            subset = [subset]
        return wrap_df(self._df.drop_nulls(subset))

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

    def with_row_count(self, name: str = "row_nr") -> "DataFrame":
        """
        Add a column at index 0 that counts the rows.

        Parameters
        ----------
        name
            Name of the column to add.
        """
        return wrap_df(self._df.with_row_count(name))

    def groupby(
        self,
        by: Union[str, "pli.Expr", Sequence[str], Sequence["pli.Expr"]],
        maintain_order: bool = False,
    ) -> "GroupBy":
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
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ a   ┆ 3   ┆ 4   │
        └─────┴─────┴─────┘

        """
        if isinstance(by, str):
            by = [by]
        return GroupBy(self._df, by, maintain_order=maintain_order)  # type: ignore

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
    ) -> "DynamicGroupBy":
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
        │ datetime            ┆ i64 │
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
        ...         [pl.col("time").min(), pl.col("time").max()]
        ...     )
        ... )
        shape: (3, 3)
        ┌─────────────────────┬─────────────────────┬─────────────────────┐
        │ time                ┆ time_min            ┆ time_max            │
        │ ---                 ┆ ---                 ┆ ---                 │
        │ datetime            ┆ datetime            ┆ datetime            │
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
        ...         [pl.col("time").count()]
        ...     )
        ... )
        shape: (3, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ datetime            ┆ datetime            ┆ datetime            ┆ u32        │
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
        ...         [pl.col("time").count(), pl.col("time").list()]
        ...     )
        ... )
        shape: (3, 3)
        ┌─────────────────────┬────────────┬─────────────────────────────────────┐
        │ time                ┆ time_count ┆ time_agg_list                       │
        │ ---                 ┆ ---        ┆ ---                                 │
        │ datetime            ┆ u32        ┆ list [datetime]                     │
        ╞═════════════════════╪════════════╪═════════════════════════════════════╡
        │ 2021-12-16 00:00:00 ┆ 2          ┆ [2021-12-16 00:00:00, 2021-12-16... │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 2          ┆ [2021-12-16 01:00:00, 2021-12-16... │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 2          ┆ [2021-12-16 02:00:00, 2021-12-16... │
        └─────────────────────┴────────────┴─────────────────────────────────────┘

        When closed="both" the time values at the window boundaries belong to 2 groups.

        >>> (
        ...     df.groupby_dynamic("time", every="1h", closed="both").agg(
        ...         [pl.col("time").count()]
        ...     )
        ... )
        shape: (3, 2)
        ┌─────────────────────┬────────────┐
        │ time                ┆ time_count │
        │ ---                 ┆ ---        │
        │ datetime            ┆ u32        │
        ╞═════════════════════╪════════════╡
        │ 2021-12-16 00:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 01:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2021-12-16 02:00:00 ┆ 3          │
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
        │ datetime            ┆ str    │
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
        ...     ).agg([pl.col("time").count()])
        ... )
        shape: (4, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬────────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ time_count │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---        │
        │ str    ┆ datetime            ┆ datetime            ┆ datetime            ┆ u32        │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪════════════╡
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 3          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 1          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 2          │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 2          │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴────────────┘

        """

        return DynamicGroupBy(
            self,
            time_column,
            every,
            period,
            offset,
            truncate,
            include_boundaries,
            closed,
            by,
        )

    def upsample(self, by: str, interval: Union[str, timedelta]) -> "DataFrame":
        """
        Upsample a DataFrame at a regular frequency.

        .. warning::
            This API is experimental and may change without it being considered a breaking change.

        Parameters
        ----------
        by
            Column that will be used as key in the upsampling operation.
            This should be a datetime column.
        interval
            Interval periods.
        """
        if self[by].dtype != Datetime:
            raise ValueError(
                f"Column {by} should be of type datetime. Got {self[by].dtype}"
            )
        bounds = self.select(
            [pli.col(by).min().alias("low"), pli.col(by).max().alias("high")]
        )
        low = bounds["low"].dt[0]
        high = bounds["high"].dt[0]
        upsampled = pli.date_range(low, high, interval, name=by)
        return DataFrame(upsampled).join(self, on=by, how="left")

    def join(
        self,
        df: "DataFrame",
        left_on: Optional[
            Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]
        ] = None,
        right_on: Optional[
            Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]
        ] = None,
        on: Optional[Union[str, "pli.Expr", tp.List[Union[str, "pli.Expr"]]]] = None,
        how: str = "inner",
        suffix: str = "_right",
        asof_by: Optional[Union[str, tp.List[str]]] = None,
        asof_by_left: Optional[Union[str, tp.List[str]]] = None,
        asof_by_right: Optional[Union[str, tp.List[str]]] = None,
    ) -> "DataFrame":
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
        This is similar to a left-join except that we match on nearest key rather than equal keys.
        The keys must be sorted to perform an asof join

        """
        if how == "cross":
            return wrap_df(self._df.join(df._df, [], [], how, suffix))

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
            return wrap_df(self._df.join(df._df, left_on_, right_on_, how, suffix))

    def apply(
        self,
        f: Callable[[Tuple[Any]], Any],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> "pli.Series":
        """
        Apply a custom function over the rows of the DataFrame. The rows are passed as tuple.

        Beware, this is slow.

        Parameters
        ----------
        f
            Custom function/ lambda function.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        """
        return pli.wrap_s(self._df.apply(f, return_dtype))

    def with_column(self, column: Union["pli.Series", "pli.Expr"]) -> "DataFrame":
        """
        Return a new DataFrame with the column added or replaced.

        Parameters
        ----------
        column
            Series, where the name of the Series refers to the column in the DataFrame.
        """
        if isinstance(column, pli.Expr):
            return self.with_columns([column])
        else:
            return wrap_df(self._df.with_column(column._s))

    def with_column_renamed(self, existing_name: str, new_name: str) -> "DataFrame":
        """
        Return a new DataFrame with the column renamed.

        Parameters
        ----------
        existing_name
        new_name
        """
        return (
            self.lazy()
            .with_column_renamed(existing_name, new_name)
            .collect(no_optimization=True, string_cache=False)
        )

    def hstack(
        self, columns: Union[tp.List["pli.Series"], "DataFrame"], in_place: bool = False
    ) -> Optional["DataFrame"]:
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
            return wrap_df(self._df.hstack([s.inner() for s in columns]))

    @tp.overload
    def vstack(self, df: "DataFrame", in_place: Literal[True]) -> None:
        ...

    @tp.overload
    def vstack(self, df: "DataFrame", in_place: Literal[False] = ...) -> "DataFrame":
        ...

    @tp.overload
    def vstack(self, df: "DataFrame", in_place: bool) -> Optional["DataFrame"]:
        ...

    def vstack(self, df: "DataFrame", in_place: bool = False) -> Optional["DataFrame"]:
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
            return wrap_df(self._df.vstack(df._df))

    def drop(self, name: Union[str, tp.List[str]]) -> "DataFrame":
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

        return wrap_df(self._df.drop(name))

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

    def clone(self) -> "DataFrame":
        """
        Very cheap deep clone.
        """
        return wrap_df(self._df.clone())

    def __copy__(self) -> "DataFrame":
        return self.clone()

    def __deepcopy__(self, memodict={}) -> "DataFrame":  # type: ignore
        return self.clone()

    def get_columns(self) -> tp.List["pli.Series"]:
        """
        Get the DataFrame as a List of Series.
        """
        return list(map(lambda s: pli.wrap_s(s), self._df.get_columns()))

    def get_column(self, name: str) -> "pli.Series":
        """
        Get a single column as Series by name.
        """
        return self[name]

    def fill_null(self, strategy: Union[str, "pli.Expr"]) -> "DataFrame":
        """
        Fill None/missing values by a filling strategy or an Expression evaluation.

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
            DataFrame with None replaced with the filling strategy.
        """
        if isinstance(strategy, pli.Expr):
            return self.lazy().fill_null(strategy).collect(no_optimization=True)
        if not isinstance(strategy, str):
            return self.fill_null(pli.lit(strategy))
        return wrap_df(self._df.fill_null(strategy))

    def fill_nan(self, fill_value: Union["pli.Expr", int, float]) -> "DataFrame":
        """
        Fill None/missing values by a an Expression evaluation.

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
        self, columns: Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]]
    ) -> "DataFrame":
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
        return self.lazy().explode(columns).collect(no_optimization=True)

    def melt(
        self, id_vars: Union[tp.List[str], str], value_vars: Union[tp.List[str], str]
    ) -> "DataFrame":
        """
        Unpivot DataFrame to long format.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables.

        value_vars
            Values to use as identifier variables.

        Returns
        -------

        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        return wrap_df(self._df.melt(id_vars, value_vars))

    def shift(self, periods: int) -> "DataFrame":
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
        return wrap_df(self._df.shift(periods))

    def shift_and_fill(
        self, periods: int, fill_value: Union[int, str, float]
    ) -> "DataFrame":
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
        """
        return pli.wrap_s(self._df.is_duplicated())

    def is_unique(self) -> "pli.Series":
        """
        Get a mask of all unique rows in this DataFrame.
        """
        return pli.wrap_s(self._df.is_unique())

    def lazy(self) -> "pli.LazyFrame":
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
        return pli.wrap_ldf(self._df.lazy())

    def select(
        self,
        exprs: Union[
            str,
            "pli.Expr",
            Sequence[Union[str, "pli.Expr"]],
            Sequence[bool],
            Sequence[int],
            Sequence[float],
            "pli.Series",
        ],
    ) -> "DataFrame":
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
            self.lazy().select(exprs).collect(no_optimization=True, string_cache=False)  # type: ignore
        )

    def with_columns(
        self, exprs: Union["pli.Expr", tp.List["pli.Expr"]]
    ) -> "DataFrame":
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

    def max(self, axis: int = 0) -> Union["DataFrame", "pli.Series"]:
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
            return wrap_df(self._df.max())
        if axis == 1:
            return pli.wrap_s(self._df.hmax())
        raise ValueError("Axis should be 0 or 1.")

    def min(self, axis: int = 0) -> Union["DataFrame", "pli.Series"]:
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
            return wrap_df(self._df.min())
        if axis == 1:
            return pli.wrap_s(self._df.hmin())
        raise ValueError("Axis should be 0 or 1.")

    def sum(
        self, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union["DataFrame", "pli.Series"]:
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
            return wrap_df(self._df.sum())
        if axis == 1:
            return pli.wrap_s(self._df.hsum(null_strategy))
        raise ValueError("Axis should be 0 or 1.")

    def mean(
        self, axis: int = 0, null_strategy: str = "ignore"
    ) -> Union["DataFrame", "pli.Series"]:
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

        """
        if axis == 0:
            return wrap_df(self._df.mean())
        if axis == 1:
            return pli.wrap_s(self._df.hmean(null_strategy))
        raise ValueError("Axis should be 0 or 1.")

    def std(self) -> "DataFrame":
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
        return wrap_df(self._df.std())

    def var(self) -> "DataFrame":
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
        return wrap_df(self._df.var())

    def median(self) -> "DataFrame":
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
        return wrap_df(self._df.median())

    def quantile(self, quantile: float, interpolation: str = "nearest") -> "DataFrame":
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
        return wrap_df(self._df.quantile(quantile, interpolation))

    def to_dummies(self) -> "DataFrame":
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
        return wrap_df(self._df.to_dummies())

    def drop_duplicates(
        self,
        maintain_order: bool = True,
        subset: Optional[Union[str, tp.List[str]]] = None,
    ) -> "DataFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_df(self._df.drop_duplicates(maintain_order, subset))

    def rechunk(self) -> "DataFrame":
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable performance.
        """
        return wrap_df(self._df.rechunk())

    def null_count(self) -> "DataFrame":
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
        return wrap_df(self._df.null_count())

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        with_replacement: bool = False,
        seed: int = 0,
    ) -> "DataFrame":
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
            Initialization seed

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sample(n=2)  # doctest: +IGNORE_RESULT
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
        if n is not None:
            return wrap_df(self._df.sample_n(n, with_replacement, seed))
        return wrap_df(self._df.sample_frac(frac, with_replacement, seed))

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
        Series: '' [str]
        [
            "foo11.0"
            "bar22.0"
            null
        ]

        Parameters
        ----------
        operation
            function that takes two `Series` and returns a `Series`.

        """
        if self.width == 1:
            return self.to_series(0)
        df = self
        acc = operation(df.to_series(0), df.to_series(1))

        for i in range(2, df.width):
            acc = operation(acc, df.to_series(i))
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

    def rows(self) -> tp.List[Tuple]:
        """
        Convert columnar data to rows as python tuples.
        """
        return self._df.row_tuples()

    def shrink_to_fit(self, in_place: bool = False) -> Optional["DataFrame"]:
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

        Hash value is UInt64

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

    def interpolate(self) -> "DataFrame":
        """
        Interpolate intermediate values. The interpolation method is linear.
        """
        return self.select(pli.col("*").interpolate())

    def is_empty(self) -> bool:
        """
        Check if the dataframe is empty
        """
        return self.height == 0


class DynamicGroupBy:
    """
    A dynamic grouper. This has an `.agg` method which will allow you to run all polars expressions
    in a groupby context.
    """

    def __init__(
        self,
        df: "DataFrame",
        time_column: str,
        every: str,
        period: Optional[str],
        offset: Optional[str],
        truncate: bool = True,
        include_boundaries: bool = True,
        closed: str = "none",
        by: Optional[Union[str, tp.List[str], "pli.Expr", tp.List["pli.Expr"]]] = None,
    ):
        self.df = df
        self.time_column = time_column
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
            tp.List[Tuple[str, tp.List[str]]],
            Dict[str, Union[str, tp.List[str]]],
            tp.List["pli.Expr"],
            "pli.Expr",
        ],
    ) -> DataFrame:
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


class GroupBy:
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
        by: Union[str, tp.List[str]],
        maintain_order: bool = False,
    ):
        self._df = df
        self.by = by
        self.maintain_order = maintain_order

    def __getitem__(self, item: Any) -> "GBSelection":
        return self._select(item)

    def _select(self, columns: Union[str, tp.List[str]]) -> "GBSelection":
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(self._df, self.by, columns)

    def __iter__(self) -> Iterable[Any]:
        groups_df = self.groups()
        groups = groups_df["groups"]
        df = wrap_df(self._df)
        for i in range(groups_df.height):
            yield df[groups[i]]

    def get_group(self, group_value: Union[Any, Tuple[Any]]) -> DataFrame:
        """
        Select a single group as a new DataFrame.

        Parameters
        ----------
        group_value
            Group to select.
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
            groups_idx = groups[mask][0]
        except IndexError:
            raise ValueError(f"no group: {group_value} found")

        df = wrap_df(self._df)
        return df[groups_idx]

    def groups(self) -> DataFrame:
        """
        Return a `DataFrame` with:

        * the groupby keys
        * the group indexes aggregated as lists
        """
        return wrap_df(self._df.groupby(self.by, None, "groups"))

    def apply(self, f: Callable[[DataFrame], DataFrame]) -> DataFrame:
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
        return wrap_df(self._df.groupby_apply(self.by, f))

    def agg(
        self,
        column_to_agg: Union[
            tp.List[Tuple[str, tp.List[str]]],
            Dict[str, Union[str, tp.List[str]]],
            tp.List["pli.Expr"],
            "pli.Expr",
        ],
    ) -> DataFrame:
        """
                Use multiple aggregations on columns. This can be combined with complete lazy API
                and is considered idiomatic polars.

                Parameters
                ----------
                column_to_agg
                    map column to aggregation functions.

                Use lazy API syntax (recommended)

                >>> [pl.col("foo").sum(), pl.col("bar").min()]  # doctest: +SKIP

                Column name to aggregation with tuples        time_column: str,
                every: str,
                period: str,
                offset: str,
                truncate: bool = True
        :

                >>> [
                ...     ("foo", ["sum", "n_unique", "min"]),
                ...     ("bar", ["max"]),
                ... ]  # doctest: +SKIP

                Column name to aggregation with dict:
                >>> {"foo": ["sum", "n_unique", "min"], "bar": "max"}  # doctest: +SKIP

                Returns
                -------
                Result of groupby split apply operations.


                Examples
                --------

                Use lazy API:

                >>> df.groupby(["foo", "bar"]).agg(
                ...     [
                ...         pl.sum("ham"),
                ...         pl.col("spam").tail(4).sum(),
                ...     ]
                ... )  # doctest: +SKIP

                Use a dict:

                >>> df.groupby(["foo", "bar"]).agg(
                ...     {
                ...         "spam": ["sum", "min"],
                ...     }
                ... )  # doctest: +SKIP
                shape: (3, 2)
                ┌─────┬─────┐
                │ foo ┆ bar │
                │ --- ┆ --- │
                │ str ┆ i64 │
                ╞═════╪═════╡
                │ a   ┆ 1   │
                ├╌╌╌╌╌┼╌╌╌╌╌┤
                │ a   ┆ 2   │
                ├╌╌╌╌╌┼╌╌╌╌╌┤
                │ b   ┆ 3   │
                └─────┴─────┘

        """
        if isinstance(column_to_agg, pli.Expr):
            column_to_agg = [column_to_agg]
        if isinstance(column_to_agg, dict):
            column_to_agg = [
                (column, [agg] if isinstance(agg, str) else agg)
                for (column, agg) in column_to_agg.items()
            ]
        elif isinstance(column_to_agg, list):

            if isinstance(column_to_agg[0], tuple):
                column_to_agg = [  # type: ignore[misc]
                    (column, [agg] if isinstance(agg, str) else agg)  # type: ignore[misc]
                    for (column, agg) in column_to_agg
                ]

            elif isinstance(column_to_agg[0], pli.Expr):
                return (
                    wrap_df(self._df)
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

        return wrap_df(self._df.groupby_agg(self.by, column_to_agg))

    def head(self, n: int = 5) -> DataFrame:
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
            wrap_df(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .head(n)
            .collect(no_optimization=True, string_cache=False)
        )

    def tail(self, n: int = 5) -> DataFrame:
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
            wrap_df(self._df)
            .lazy()
            .groupby(self.by, self.maintain_order)
            .tail(n)
            .collect(no_optimization=True, string_cache=False)
        )

    def _select_all(self) -> "GBSelection":
        """
        Select all columns for aggregation.
        """
        return GBSelection(self._df, self.by, None)

    def pivot(self, pivot_column: str, values_column: str) -> "PivotOps":
        """
        Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.

        Parameters
        ----------
        pivot_column
            Column to pivot.
        values_column
            Column that will be aggregated.
        """
        return PivotOps(self._df, self.by, pivot_column, values_column)

    def first(self) -> DataFrame:
        """
        Aggregate the first values in the group.
        """
        return self._select_all().first()

    def last(self) -> DataFrame:
        """
        Aggregate the last values in the group.
        """
        return self._select_all().last()

    def sum(self) -> DataFrame:
        """
        Reduce the groups to the sum.
        """
        return self._select_all().sum()

    def min(self) -> DataFrame:
        """
        Reduce the groups to the minimal value.
        """
        return self._select_all().min()

    def max(self) -> DataFrame:
        """
        Reduce the groups to the maximal value.
        """
        return self._select_all().max()

    def count(self) -> DataFrame:
        """
        Count the number of values in each group.
        """
        return self._select_all().count()

    def mean(self) -> DataFrame:
        """
        Reduce the groups to the mean values.
        """
        return self._select_all().mean()

    def n_unique(self) -> DataFrame:
        """
        Count the unique values per group.
        """
        return self._select_all().n_unique()

    def quantile(self, quantile: float, interpolation: str = "nearest") -> DataFrame:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        """
        return self._select_all().quantile(quantile, interpolation)

    def median(self) -> DataFrame:
        """
        Return the median per group.
        """
        return self._select_all().median()

    def agg_list(self) -> DataFrame:
        """
        Aggregate the groups into Series.
        """
        return self._select_all().agg_list()


class PivotOps:
    """
    Utility class returned in a pivot operation.
    """

    def __init__(
        self,
        df: DataFrame,
        by: Union[str, tp.List[str]],
        pivot_column: str,
        values_column: str,
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column

    def first(self) -> DataFrame:
        """
        Get the first value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self) -> DataFrame:
        """
        Get the sum per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self) -> DataFrame:
        """
        Get the minimal value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self) -> DataFrame:
        """
        Get the maximal value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self) -> DataFrame:
        """
        Get the mean value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def count(self) -> DataFrame:
        """
        Count the values per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "count")
        )

    def median(self) -> DataFrame:
        """
        Get the median value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )


class GBSelection:
    """
    Utility class returned in a groupby operation.
    """

    def __init__(
        self,
        df: "PyDataFrame",
        by: Union[str, tp.List[str]],
        selection: Optional[tp.List[str]],
    ):
        self._df = df
        self.by = by
        self.selection = selection

    def first(self) -> DataFrame:
        """
        Aggregate the first values in the group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "first"))

    def last(self) -> DataFrame:
        """
        Aggregate the last values in the group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "last"))

    def sum(self) -> DataFrame:
        """
        Reduce the groups to the sum.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "sum"))

    def min(self) -> DataFrame:
        """
        Reduce the groups to the minimal value.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "min"))

    def max(self) -> DataFrame:
        """
        Reduce the groups to the maximal value.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "max"))

    def count(self) -> DataFrame:
        """
        Count the number of values in each group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "count"))

    def mean(self) -> DataFrame:
        """
        Reduce the groups to the mean values.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "mean"))

    def n_unique(self) -> DataFrame:
        """
        Count the unique values per group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "n_unique"))

    def quantile(self, quantile: float, interpolation: str = "nearest") -> DataFrame:
        """
        Compute the quantile per group.

        Parameters
        ----------
        quantile
            quantile between 0.0 and 1.0

        interpolation
            interpolation type, options: ['nearest', 'higher', 'lower', 'midpoint', 'linear']

        """
        return wrap_df(
            self._df.groupby_quantile(self.by, self.selection, quantile, interpolation)
        )

    def median(self) -> DataFrame:
        """
        Return the median per group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "median"))

    def agg_list(self) -> DataFrame:
        """
        Aggregate the groups into Series.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "agg_list"))

    def apply(
        self,
        func: Callable[[Any], Any],
        return_dtype: Optional[Type[DataType]] = None,
    ) -> DataFrame:
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

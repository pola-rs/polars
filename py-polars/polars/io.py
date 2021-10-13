from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Type,
    Union,
    overload,
)
from urllib.request import urlopen

try:
    import pyarrow as pa
    import pyarrow.csv
    import pyarrow.feather
    import pyarrow.parquet

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

import polars as pl

from .convert import from_arrow

try:
    from polars.polars import ipc_schema as _ipc_schema
except ImportError:
    pass

try:
    import connectorx as cx

    _WITH_CX = True
except ImportError:
    _WITH_CX = False

try:
    import fsspec
    from fsspec.utils import infer_storage_options

    _WITH_FSSPEC = True
except ImportError:
    _WITH_FSSPEC = False

__all__ = [
    "read_csv",
    "read_parquet",
    "read_json",
    "read_sql",
    "read_ipc",
    "scan_csv",
    "scan_parquet",
    "read_ipc_schema",
]


def _process_http_file(path: str) -> BytesIO:
    with urlopen(path) as f:
        return BytesIO(f.read())


@overload
def _prepare_file_arg(
    file: Union[str, List[str], Path, BinaryIO, bytes], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO]]:
    ...


@overload
def _prepare_file_arg(
    file: Union[str, TextIO, Path, BinaryIO, bytes], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO]]:
    ...


@overload
def _prepare_file_arg(
    file: Union[str, List[str], TextIO, Path, BinaryIO, bytes], **kwargs: Any
) -> ContextManager[Union[str, List[str], BinaryIO, List[BinaryIO]]]:
    ...


def _prepare_file_arg(
    file: Union[str, List[str], TextIO, Path, BinaryIO, bytes], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO, List[str], List[BinaryIO]]]:
    """
    Utility for read_[csv, parquet]. (not to be used by scan_[csv, parquet]).
    Returned value is always usable as a context.

    A `StringIO`, `BytesIO` file is returned as a `BytesIO`
    A local path is returned as a string
    An http url is read into a buffer and returned as a `BytesIO`

    When fsspec is installed, remote file(s) is (are) opened with
    `fsspec.open(file, **kwargs)` or `fsspec.open_files(file, **kwargs)`.
    """

    # Small helper to use a variable as context
    @contextmanager
    def managed_file(file: Any) -> Iterator[Any]:
        try:
            yield file
        finally:
            pass

    if isinstance(file, StringIO):
        return BytesIO(file.read().encode("utf8"))
    if isinstance(file, BytesIO):
        return file
    if isinstance(file, Path):
        return managed_file(str(file))
    if isinstance(file, str):
        if _WITH_FSSPEC:
            if infer_storage_options(file)["protocol"] == "file":
                return managed_file(file)
            return fsspec.open(file, **kwargs)
        if file.startswith("http"):
            return _process_http_file(file)
    if isinstance(file, list) and bool(file) and all(isinstance(f, str) for f in file):
        if _WITH_FSSPEC:
            if all(infer_storage_options(f)["protocol"] == "file" for f in file):
                return managed_file(file)
            return fsspec.open_files(file, **kwargs)
    return managed_file(file)


def update_columns(df: "pl.DataFrame", new_columns: List[str]) -> "pl.DataFrame":
    if df.width > len(new_columns):
        cols = df.columns
        for i, name in enumerate(new_columns):
            cols[i] = name
        new_columns = cols
    df.columns = new_columns
    return df


def read_csv(
    file: Union[str, TextIO, Path, BinaryIO, bytes],
    infer_schema_length: int = 100,
    batch_size: int = 8192,
    has_headers: bool = True,
    ignore_errors: bool = False,
    stop_after_n_rows: Optional[int] = None,
    skip_rows: int = 0,
    projection: Optional[List[int]] = None,
    sep: str = ",",
    columns: Optional[List[str]] = None,
    rechunk: bool = True,
    encoding: str = "utf8",
    n_threads: Optional[int] = None,
    dtype: Optional[Dict[str, Type["pl.DataType"]]] = None,
    new_columns: Optional[List[str]] = None,
    use_pyarrow: bool = False,
    low_memory: bool = False,
    comment_char: Optional[str] = None,
    quote_char: Optional[str] = r'"',
    storage_options: Optional[Dict] = None,
    null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
    parse_dates: bool = True,
) -> "pl.DataFrame":
    """
    Read into a DataFrame from a csv file.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO`` or ``BytesIO``.
        If ``fsspec`` is installed, it will be used to open remote files
    infer_schema_length
        Maximum number of lines to read to infer schema.
    batch_size
        Number of lines to read into the buffer at once. Modify this to change performance.
    has_headers
        Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`,
        `x` being an enumeration over every column in the dataset starting at 1.
    ignore_errors
        Try to keep reading lines if some lines yield errors.
    stop_after_n_rows
        After n rows are read from the CSV, it stops reading.
        During multi-threaded parsing, an upper bound of `n` rows
        cannot be guaranteed.
    skip_rows
        Start reading after `skip_rows`.
    projection
        Indexes of columns to select. Note that column indexes count from zero.
    sep
        Delimiter/ value separator.
    columns
        Columns to project/ select.
    rechunk
        Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
    encoding
        - "utf8"
        - "utf8-lossy"
    n_threads
        Number of threads to use in csv parsing. Defaults to the number of physical cpu's of your system.
    dtype
        Overwrite the dtypes during inference.
    new_columns
        Rename columns to these right after parsing. If the given list is shorted than the width of the DataFrame the
        remaining columns will have their original name.
    use_pyarrow
        Try to use pyarrow's native CSV parser. This is not always possible. The set of arguments given to this function
        determine if it is possible to use pyarrows native parser. Note that pyarrow and polars may have a different
        strategy regarding type inference.
    low_memory
        Reduce memory usage in expense of performance.
    comment_char
        character that indicates the start of a comment line, for instance '#'.
    quote_char
        single byte character that is used for csv quoting, default = ''. Set to None to turn special handling and escaping
        of quotes off.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.
    null_values
        Values to interpret as null values. You can provide a:

        - str -> all values encountered equal to this string will be null
        - List[str] -> A null value per column.
        - Dict[str, str] -> A dictionary that maps column name to a null value string.
    parse_dates
        Try to automatically parse dates. If this not succeeds, the column remains
        of data type Utf8.

    Returns
    -------
    DataFrame
    """
    if isinstance(file, bytes) and len(file) == 0:
        raise ValueError("no date in bytes")

    storage_options = storage_options or {}

    if columns and not has_headers:
        for column in columns:
            if not column.startswith("column_"):
                raise ValueError(
                    'Specified column names do not start with "column_", '
                    "but autogenerated header names were requested."
                )

    if use_pyarrow and not _PYARROW_AVAILABLE:
        raise ImportError(
            "'pyarrow' is required when using 'read_csv(..., use_pyarrow=True)'."
        )

    if (
        use_pyarrow
        and dtype is None
        and stop_after_n_rows is None
        and n_threads is None
        and encoding == "utf8"
        and not low_memory
        and null_values is None
        and parse_dates
    ):
        include_columns = None

        if columns:
            if not has_headers:
                # Convert 'column_1', 'column_2', ... column names to 'f0', 'f1', ... column names for pyarrow,
                # if CSV file does not contain a header.
                include_columns = [f"f{int(column[7:]) - 1}" for column in columns]
            else:
                include_columns = columns

        if not columns and projection:
            # Convert column indices from projection to 'f0', 'f1', ... column names for pyarrow.
            include_columns = [f"f{column_idx}" for column_idx in projection]

        with _prepare_file_arg(file, **storage_options) as data:
            tbl = pa.csv.read_csv(
                data,
                pa.csv.ReadOptions(
                    skip_rows=skip_rows, autogenerate_column_names=not has_headers
                ),
                pa.csv.ParseOptions(delimiter=sep),
                pa.csv.ConvertOptions(
                    column_types=None,
                    include_columns=include_columns,
                    include_missing_columns=ignore_errors,
                ),
            )

        if not has_headers:
            # Rename 'f0', 'f1', ... columns names autogenated by pyarrow to 'column_1', 'column_2', ...
            tbl = tbl.rename_columns(
                [f"column_{int(column[1:]) + 1}" for column in tbl.column_names]
            )

        df = from_arrow(tbl, rechunk)
        if new_columns:
            return update_columns(df, new_columns)  # type: ignore
        return df  # type: ignore

    with _prepare_file_arg(file, **storage_options) as data:
        df = pl.DataFrame.read_csv(
            file=data,
            infer_schema_length=infer_schema_length,
            batch_size=batch_size,
            has_headers=has_headers,
            ignore_errors=ignore_errors,
            stop_after_n_rows=stop_after_n_rows,
            skip_rows=skip_rows,
            projection=projection,
            sep=sep,
            columns=columns,
            rechunk=rechunk,
            encoding=encoding,
            n_threads=n_threads,
            dtype=dtype,
            low_memory=low_memory,
            comment_char=comment_char,
            quote_char=quote_char,
            null_values=null_values,
            parse_dates=parse_dates,
        )

    if new_columns:
        return update_columns(df, new_columns)
    return df


def scan_csv(
    file: Union[str, Path],
    has_headers: bool = True,
    ignore_errors: bool = False,
    sep: str = ",",
    skip_rows: int = 0,
    stop_after_n_rows: Optional[int] = None,
    cache: bool = True,
    dtype: Optional[Dict[str, Type["pl.DataType"]]] = None,
    low_memory: bool = False,
    comment_char: Optional[str] = None,
    quote_char: Optional[str] = r'"',
    null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
) -> "pl.LazyFrame":
    """
    Lazily read from a csv file.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a file.
    has_headers
        If the CSV file has headers or not.
    ignore_errors
        Try to keep reading lines if some lines yield errors.
    sep
        Delimiter/ value separator.
    skip_rows
        Start reading after `skip_rows`.
    stop_after_n_rows
        After n rows are read from the CSV, it stops reading.
        During multi-threaded parsing, an upper bound of `n` rows cannot be guaranteed.
    cache
        Cache the result after reading.
    dtype
        Overwrite the dtypes during inference.
    low_memory
        Reduce memory usage in expense of performance.
    comment_char
        character that indicates the start of a comment line, for instance '#'.
    quote_char
        single byte character that is used for csv quoting, default = ''. Set to None to turn special handling and escaping
        of quotes off.
    null_values
        Values to interpret as null values. You can provide a:

        - str -> all values encountered equal to this string will be null
        - List[str] -> A null value per column.
        - Dict[str, str] -> A dictionary that maps column name to a null value string.
    """
    if isinstance(file, Path):
        file = str(file)
    return pl.LazyFrame.scan_csv(
        file=file,
        has_headers=has_headers,
        sep=sep,
        ignore_errors=ignore_errors,
        skip_rows=skip_rows,
        stop_after_n_rows=stop_after_n_rows,
        cache=cache,
        dtype=dtype,
        low_memory=low_memory,
        comment_char=comment_char,
        quote_char=quote_char,
        null_values=null_values,
    )


def scan_parquet(
    file: Union[str, Path],
    stop_after_n_rows: Optional[int] = None,
    cache: bool = True,
) -> "pl.LazyFrame":
    """
    Lazily read from a parquet file.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a file.
    stop_after_n_rows
        After n rows are read from the parquet, it stops reading.
    cache
        Cache the result after reading.
    """
    if isinstance(file, Path):
        file = str(file)
    return pl.LazyFrame.scan_parquet(
        file=file, stop_after_n_rows=stop_after_n_rows, cache=cache
    )


def read_ipc_schema(
    file: Union[str, BinaryIO, Path, bytes]
) -> Dict[str, Type["pl.DataType"]]:
    """
    Get a schema of the IPC file without reading data.

    Parameters
    ----------
    file
        Path to a file or a file like object.


    Returns
    -------
    Dictionary mapping column names to datatypes
    """
    return _ipc_schema(file)


def read_ipc(
    file: Union[str, BinaryIO, Path, bytes],
    use_pyarrow: bool = _PYARROW_AVAILABLE,
    memory_map: bool = True,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
) -> "pl.DataFrame":
    """
    Read into a DataFrame from Arrow IPC stream format. This is also called the feather format.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        If ``fsspec`` is installed, it will be used to open remote files
    use_pyarrow
        Use pyarrow or the native rust reader.
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when 'use_pyarrow=True'
    columns
        Columns to project/ select.
        Only valid when 'use_pyarrow=True'
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.

    Returns
    -------
    DataFrame
    """
    storage_options = storage_options or {}
    with _prepare_file_arg(file, **storage_options) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using 'read_ipc(..., use_pyarrow=True)'."
                )
            tbl = pa.feather.read_table(data, memory_map=memory_map, columns=columns)
            return pl.DataFrame._from_arrow(tbl)
        return pl.DataFrame.read_ipc(data)


def read_parquet(
    source: Union[str, List[str], Path, BinaryIO, bytes],
    use_pyarrow: bool = _PYARROW_AVAILABLE,
    stop_after_n_rows: Optional[int] = None,
    memory_map: bool = True,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
    **kwargs: Any,
) -> "pl.DataFrame":
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    source
        Path to a file, list of files, or a file like object. If the path is a directory, that directory will be used
        as partition aware scan.
        If ``fsspec`` is installed, it will be used to open remote files
    use_pyarrow
        Use pyarrow instead of the rust native parquet reader. The pyarrow reader is more stable.
    stop_after_n_rows
        After n rows are read from the parquet, it stops reading.
        Only valid when 'use_pyarrow=False'
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when 'use_pyarrow=True'
    columns
        Columns to project/ select.
        Only valid when 'use_pyarrow=True'
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.
    **kwargs
        kwargs for [pyarrow.parquet.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html)

    Returns
    -------
    DataFrame
    """
    if use_pyarrow:
        if stop_after_n_rows:
            raise ValueError(
                "'stop_after_n_rows' cannot be used with 'use_pyarrow=True'."
            )
    else:
        if columns:
            raise ValueError("'columns' cannot be used with 'use_pyarrow=False'.")
    storage_options = storage_options or {}
    with _prepare_file_arg(source, **storage_options) as source_prep:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using 'read_parquet(..., use_pyarrow=True)'."
                )
            return from_arrow(  # type: ignore[return-value]
                pa.parquet.read_table(
                    source_prep, memory_map=memory_map, columns=columns, **kwargs
                )
            )
        return pl.DataFrame.read_parquet(
            source_prep, stop_after_n_rows=stop_after_n_rows
        )


def read_json(source: Union[str, BytesIO]) -> "pl.DataFrame":
    """
    Read into a DataFrame from JSON format.

    Parameters
    ----------
    source
        Path to a file or a file like object.
    """
    return pl.DataFrame.read_json(source)


def read_sql(
    sql: Union[List[str], str],
    connection_uri: str,
    partition_on: Optional[str] = None,
    partition_range: Optional[Tuple[int, int]] = None,
    partition_num: Optional[int] = None,
) -> "pl.DataFrame":
    """
    Read a SQL query into a DataFrame
    Make sure to install connextorx>=0.2

    # Sources
    Supports reading a sql query from the following data sources:

    * Postgres
    * Mysql
    * Sqlite
    * Redshift (through postgres protocol)
    * Clickhouse (through mysql protocol)

    ## Source not supported?
    If a database source is not supported, pandas can be used to load the query:

    >>>> df = pl.from_pandas(pd.read_sql(sql, engine))

    Parameters
    ----------
    sql
        raw sql query
    connection_uri
        connectorx connection uri:
            - "postgresql://username:password@server:port/database"
    partition_on
      the column to partition the result.
    partition_range
      the value range of the partition column.
    partition_num
      how many partition to generate.


    Examples
    --------

    ## Single threaded
    Read a DataFrame from a SQL using a single thread:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_sql(query, uri)

    ## Using 10 threads
    Read a DataFrame parallelly using 10 threads by automatically partitioning the provided SQL on the partition column:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> read_sql(query, uri, partition_on="partition_col", partition_num=10)

    ## Using
    Read a DataFrame parallel using 2 threads by manually providing two partition SQLs:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = ["SELECT * FROM lineitem WHERE partition_col <= 10", "SELECT * FROM lineitem WHERE partition_col > 10"]
    >>> read_sql(uri, queries)

    """
    if _WITH_CX:
        tbl = cx.read_sql(
            conn=connection_uri,
            query=sql,
            return_type="arrow",
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
        )
        return pl.from_arrow(tbl)  # type: ignore[return-value]
    else:
        raise ImportError(
            "connectorx is not installed." "Please run pip install connectorx>=0.2.0a3"
        )

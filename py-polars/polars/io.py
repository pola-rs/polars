from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    TextIO,
    Tuple,
    Type,
    Union,
    overload,
)
from urllib.request import urlopen

from polars.utils import handle_projection_columns

try:
    import pyarrow as pa
    import pyarrow.csv
    import pyarrow.feather
    import pyarrow.parquet

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

from polars.convert import from_arrow
from polars.datatypes import DataType
from polars.internals import DataFrame, LazyFrame

try:
    from polars.polars import ipc_schema as _ipc_schema
except ImportError:  # pragma: no cover
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


def update_columns(df: DataFrame, new_columns: List[str]) -> DataFrame:
    if df.width > len(new_columns):
        cols = df.columns
        for i, name in enumerate(new_columns):
            cols[i] = name
        new_columns = cols
    df.columns = new_columns
    return df


def read_csv(
    file: Union[str, TextIO, BytesIO, Path, BinaryIO, bytes],
    has_header: bool = True,
    columns: Optional[Union[List[int], List[str]]] = None,
    new_columns: Optional[List[str]] = None,
    sep: str = ",",
    comment_char: Optional[str] = None,
    quote_char: Optional[str] = r'"',
    skip_rows: int = 0,
    dtypes: Optional[Union[Mapping[str, Type[DataType]], List[Type[DataType]]]] = None,
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
    use_pyarrow: bool = False,
    storage_options: Optional[Dict] = None,
    offset_schema_inference: int = 0,
    **kwargs: Any,
) -> DataFrame:
    """
    Read a CSV file into a Dataframe.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        By file-like object, we refer to objects with a ``read()``
        method, such as a file handler (e.g. via builtin ``open``
        function) or ``StringIO`` or ``BytesIO``.
        If ``fsspec`` is installed, it will be used to open remote
        files.
    has_header
        Indicate if the first row of dataset is a header or not.
        If set to False, column names will be autogenrated in the
        following format: ``column_x``, with ``x`` being an
        enumeration over every column in the dataset starting at 1.
    columns
        Columns to select. Accepts a list of column indices (starting
        at zero) or a list of column names.
    new_columns
        Rename columns right after parsing the CSV file. If the given
        list is shorter than the width of the DataFrame the remaining
        columns will have their original name.
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
    use_pyarrow
        Try to use pyarrow's native CSV parser.
        This is not always possible. The set of arguments given to
        this function determines if it is possible to use pyarrow's
        native parser. Note that pyarrow and polars may have a
        different strategy regarding type inference.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a
        particular storage connection.
        e.g. host, port, username, password, etc.
    offset_schema_inference
        Start schema parsing of the header at this offset

    Returns
    -------
    DataFrame
    """

    # Map legacy arguments to current ones and remove them from kwargs.
    has_header = kwargs.pop("has_headers", has_header)
    dtypes = kwargs.pop("dtype", dtypes)
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if columns is None:
        columns = kwargs.pop("projection", None)

    projection, columns = handle_projection_columns(columns)

    if isinstance(file, bytes) and len(file) == 0:
        raise ValueError("no date in bytes")

    storage_options = storage_options or {}

    if columns and not has_header:
        for column in columns:
            if isinstance(column, str) and not column.startswith("column_"):
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
        and dtypes is None
        and n_rows is None
        and n_threads is None
        and encoding == "utf8"
        and not low_memory
        and null_values is None
        and parse_dates
    ):
        include_columns = None

        if columns:
            if not has_header:
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
                    skip_rows=skip_rows, autogenerate_column_names=not has_header
                ),
                pa.csv.ParseOptions(delimiter=sep),
                pa.csv.ConvertOptions(
                    column_types=None,
                    include_columns=include_columns,
                    include_missing_columns=ignore_errors,
                ),
            )

        if not has_header:
            # Rename 'f0', 'f1', ... columns names autogenated by pyarrow to 'column_1', 'column_2', ...
            tbl = tbl.rename_columns(
                [f"column_{int(column[1:]) + 1}" for column in tbl.column_names]
            )

        df = from_arrow(tbl, rechunk)
        if new_columns:
            return update_columns(df, new_columns)  # type: ignore
        return df  # type: ignore

    if new_columns and dtypes and isinstance(dtypes, dict):
        current_columns = None

        # As new column names are not available yet while parsing the CSV file, rename column names in
        # dtypes to old names (if possible) so they can be used during CSV parsing.
        if columns:
            if len(columns) < len(new_columns):
                raise ValueError(
                    "More new colum names are specified than there are selected columns."
                )

            # Get column names of requested columns.
            current_columns = columns[0 : len(new_columns)]
        elif not has_header:
            # When there are no header, column names are autogenerated (and known).

            if projection:
                if columns and len(columns) < len(new_columns):
                    raise ValueError(
                        "More new colum names are specified than there are selected columns."
                    )
                # Convert column indices from projection to 'column_1', 'column_2', ... column names.
                current_columns = [
                    f"column_{column_idx + 1}" for column_idx in projection
                ]
            else:
                # Generate autogenerated 'column_1', 'column_2', ... column names for new column names.
                current_columns = [
                    f"column_{column_idx}"
                    for column_idx in range(1, len(new_columns) + 1)
                ]
        else:
            # When a header is present, column names are not known yet.

            if len(dtypes) <= len(new_columns):
                # If dtypes dictionary contains less or same amount of values than new column names
                # a list of dtypes can be created if all listed column names in dtypes dictionary
                # appear in the first consecutive new column names.
                dtype_list = [
                    dtypes[new_column_name]
                    for new_column_name in new_columns[0 : len(dtypes)]
                    if new_column_name in dtypes
                ]

                if len(dtype_list) == len(dtypes):
                    dtypes = dtype_list

        if current_columns and isinstance(dtypes, dict):
            new_to_current = {
                new_column: current_column
                for new_column, current_column in zip(new_columns, current_columns)
            }
            # Change new column names to current column names in dtype.
            dtypes = {
                new_to_current.get(column_name, column_name): column_dtype
                for column_name, column_dtype in dtypes.items()
            }

    with _prepare_file_arg(file, **storage_options) as data:
        df = DataFrame._read_csv(
            file=data,
            has_header=has_header,
            columns=columns if columns else projection,
            sep=sep,
            comment_char=comment_char,
            quote_char=quote_char,
            skip_rows=skip_rows,
            dtypes=dtypes,
            null_values=null_values,
            ignore_errors=ignore_errors,
            parse_dates=parse_dates,
            n_threads=n_threads,
            infer_schema_length=infer_schema_length,
            batch_size=batch_size,
            n_rows=n_rows,
            encoding=encoding,
            low_memory=low_memory,
            rechunk=rechunk,
            offset_schema_inference=offset_schema_inference,
        )

    if new_columns:
        return update_columns(df, new_columns)
    return df


def scan_csv(
    file: Union[str, Path],
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
    low_memory: bool = False,
    rechunk: bool = True,
    offset_schema_inference: int = 0,
    **kwargs: Any,
) -> LazyFrame:
    """
    Lazily read from a CSV file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and
    projections to the scan level, thereby potentially reducing
    memory overhead.

    Parameters
    ----------
    file
        Path to a file.
    has_header
        Indicate if the first row of dataset is a header or not.
        If set to False, column names will be autogenrated in the
        following format: ``column_x``, with ``x`` being an
        enumeration over every column in the dataset starting at 1.
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
    cache
        Cache the result after reading.
    with_column_names
        Apply a function over the column names.
        This can be used to update a schema just in time, thus before
        scanning.
    infer_schema_length
        Maximum number of lines to read to infer schema.
        If set to 0, all columns will be read as ``pl.Utf8``.
        If set to ``None``, a full table scan will be done (slow).
    n_rows
        Stop reading from CSV file after reading ``n_rows``.
    low_memory
        Reduce memory usage in expense of performance.
    rechunk
        Reallocate to contiguous memory when all chunks/ files are parsed.
    offset_schema_inference
        Start schema parsing of the header at this offset

    Examples
    --------
    >>> (
    ...     pl.scan_csv("my_long_file.csv")  # lazy, doesn't do a thing
    ...     .select(
    ...         ["a", "c"]
    ...     )  # select only 2 columns (other columns will not be read)
    ...     .filter(
    ...         pl.col("a") > 10
    ...     )  # the filter is pushed down the the scan, so less data read in memory
    ...     .fetch(100)  # pushed a limit of 100 rows to the scan level
    ... )  # doctest: +SKIP

    We can use `with_column_names` to modify the header before scanning:

    >>> df = pl.DataFrame(
    ...     {"BrEeZaH": [1, 2, 3, 4], "LaNgUaGe": ["is", "terrible", "to", "read"]}
    ... )
    >>> df.to_csv("mydf.csv")
    >>> pl.scan_csv(
    ...     "mydf.csv", with_column_names=lambda cols: [col.lower() for col in cols]
    ... ).fetch()
    shape: (4, 2)
    ┌─────────┬──────────┐
    │ breezah ┆ language │
    │ ---     ┆ ---      │
    │ i64     ┆ str      │
    ╞═════════╪══════════╡
    │ 1       ┆ is       │
    ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
    │ 2       ┆ terrible │
    ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
    │ 3       ┆ to       │
    ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
    │ 4       ┆ read     │
    └─────────┴──────────┘


    """

    # Map legacy arguments to current ones and remove them from kwargs.
    has_header = kwargs.pop("has_headers", has_header)
    dtypes = kwargs.pop("dtype", dtypes)
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if isinstance(file, Path):
        file = str(file)

    return LazyFrame.scan_csv(
        file=file,
        has_header=has_header,
        sep=sep,
        comment_char=comment_char,
        quote_char=quote_char,
        skip_rows=skip_rows,
        dtypes=dtypes,
        null_values=null_values,
        ignore_errors=ignore_errors,
        cache=cache,
        with_column_names=with_column_names,
        infer_schema_length=infer_schema_length,
        n_rows=n_rows,
        low_memory=low_memory,
        rechunk=rechunk,
        offset_schema_inference=offset_schema_inference,
    )


def scan_ipc(
    file: Union[str, Path],
    n_rows: Optional[int] = None,
    cache: bool = True,
    rechunk: bool = True,
    **kwargs: Any,
) -> LazyFrame:
    """
    Lazily read from an Arrow IPC (Feather v2) file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a IPC file.
    n_rows
        Stop reading from IPC file after reading ``n_rows``.
    cache
        Cache the result after reading.
    rechunk
        Reallocate to contiguous memory when all chunks/ files are parsed.
    """

    # Map legacy arguments to current ones and remove them from kwargs.
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if isinstance(file, Path):
        file = str(file)

    return LazyFrame.scan_ipc(file=file, n_rows=n_rows, cache=cache, rechunk=rechunk)


def scan_parquet(
    file: Union[str, Path],
    n_rows: Optional[int] = None,
    cache: bool = True,
    parallel: bool = True,
    rechunk: bool = True,
    **kwargs: Any,
) -> LazyFrame:
    """
    Lazily read from a parquet file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a file.
    n_rows
        Stop reading from parquet file after reading ``n_rows``.
    cache
        Cache the result after reading.
    parallel
        Read the parquet file in parallel. The single threaded reader consumes less memory.
    rechunk
        In case of reading multiple files via a glob pattern rechunk the final DataFrame into contiguous memory chunks.
    """

    # Map legacy arguments to current ones and remove them from kwargs.
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if isinstance(file, Path):
        file = str(file)

    return LazyFrame.scan_parquet(
        file=file, n_rows=n_rows, cache=cache, parallel=parallel, rechunk=rechunk
    )


def read_ipc_schema(
    file: Union[str, BinaryIO, Path, bytes]
) -> Dict[str, Type[DataType]]:
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
    file: Union[str, BinaryIO, BytesIO, Path, bytes],
    columns: Optional[Union[List[int], List[str]]] = None,
    n_rows: Optional[int] = None,
    use_pyarrow: bool = _PYARROW_AVAILABLE,
    memory_map: bool = True,
    storage_options: Optional[Dict] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read into a DataFrame from Arrow IPC (Feather v2) file.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        If ``fsspec`` is installed, it will be used to open remote files.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
    n_rows
        Stop reading from IPC file after reading ``n_rows``.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Use pyarrow or the native rust reader.
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when ``use_pyarrow=True``.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.

    Returns
    -------
    DataFrame
    """

    # Map legacy arguments to current ones and remove them from kwargs.
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if columns is None:
        columns = kwargs.pop("projection", None)

    if use_pyarrow:
        if n_rows:
            raise ValueError("``n_rows`` cannot be used with ``use_pyarrow=True``.")

    storage_options = storage_options or {}
    with _prepare_file_arg(file, **storage_options) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using 'read_ipc(..., use_pyarrow=True)'."
                )

            tbl = pa.feather.read_table(data, memory_map=memory_map, columns=columns)
            return DataFrame._from_arrow(tbl)

        return DataFrame._read_ipc(
            data,
            columns=columns,
            n_rows=n_rows,
        )


def read_parquet(
    source: Union[str, List[str], Path, BinaryIO, BytesIO, bytes],
    columns: Optional[Union[List[int], List[str]]] = None,
    n_rows: Optional[int] = None,
    use_pyarrow: bool = False,
    memory_map: bool = True,
    storage_options: Optional[Dict] = None,
    parallel: bool = True,
    **kwargs: Any,
) -> DataFrame:
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    source
        Path to a file, list of files, or a file like object. If the path is a directory, that directory will be used
        as partition aware scan.
        If ``fsspec`` is installed, it will be used to open remote files.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
    n_rows
        Stop reading from parquet file after reading ``n_rows``.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Use pyarrow instead of the rust native parquet reader. The pyarrow reader is more stable.
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when ``use_pyarrow=True``.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.
    parallel
        Read the parquet file in parallel. The single threaded reader consumes less memory.
    **kwargs
        kwargs for [pyarrow.parquet.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html)

    Returns
    -------
    DataFrame
    """

    # Map legacy arguments to current ones and remove them from kwargs.
    n_rows = kwargs.pop("stop_after_n_rows", n_rows)

    if columns is None:
        columns = kwargs.pop("projection", None)

    if use_pyarrow:
        if n_rows:
            raise ValueError("``n_rows`` cannot be used with ``use_pyarrow=True``.")

    storage_options = storage_options or {}
    with _prepare_file_arg(source, **storage_options) as source_prep:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using 'read_parquet(..., use_pyarrow=True)'."
                )

            return from_arrow(  # type: ignore[return-value]
                pa.parquet.read_table(
                    source_prep,
                    memory_map=memory_map,
                    columns=columns,
                    **kwargs,
                )
            )

        return DataFrame._read_parquet(
            source_prep, columns=columns, n_rows=n_rows, parallel=parallel
        )


def read_json(source: Union[str, BytesIO]) -> DataFrame:
    """
    Read into a DataFrame from JSON format.

    Parameters
    ----------
    source
        Path to a file or a file like object.
    """
    return DataFrame._read_json(source)


def read_sql(
    sql: Union[List[str], str],
    connection_uri: str,
    partition_on: Optional[str] = None,
    partition_range: Optional[Tuple[int, int]] = None,
    partition_num: Optional[int] = None,
    protocol: Optional[str] = None,
) -> DataFrame:
    """
    Read a SQL query into a DataFrame.
    Make sure to install connectorx>=0.2

    # Sources
    Supports reading a sql query from the following data sources:

    * Postgres
    * Mysql
    * Sqlite
    * Redshift (through postgres protocol)
    * Clickhouse (through mysql protocol)

    ## Source not supported?
    If a database source is not supported, pandas can be used to load the query:

    >>> import pandas as pd
    >>> df = pl.from_pandas(pd.read_sql(sql, engine))  # doctest: +SKIP

    Parameters
    ----------
    sql
        raw sql query.
    connection_uri
        connectorx connection uri:
            - "postgresql://username:password@server:port/database"
    partition_on
      the column on which to partition the result.
    partition_range
      the value range of the partition column.
    partition_num
      how many partitions to generate.
    protocol
      backend-specific transfer protocol directive; see connectorx documentation for details.

    Examples
    --------

    ## Single threaded
    Read a DataFrame from a SQL query using a single thread:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_sql(query, uri)  # doctest: +SKIP

    ## Using 10 threads
    Read a DataFrame in parallel using 10 threads by automatically partitioning the provided SQL on the partition column:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> pl.read_sql(
    ...     query, uri, partition_on="partition_col", partition_num=10
    ... )  # doctest: +SKIP

    ## Using
    Read a DataFrame in parallel using 2 threads by explicitly providing two SQL queries:

    >>> uri = "postgresql://username:password@server:port/database"
    >>> queries = [
    ...     "SELECT * FROM lineitem WHERE partition_col <= 10",
    ...     "SELECT * FROM lineitem WHERE partition_col > 10",
    ... ]
    >>> pl.read_sql(uri, queries)  # doctest: +SKIP

    """
    if _WITH_CX:
        tbl = cx.read_sql(
            conn=connection_uri,
            query=sql,
            return_type="arrow",
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
        )
        return from_arrow(tbl)  # type: ignore[return-value]
    else:
        raise ImportError(
            "connectorx is not installed." "Please run pip install connectorx>=0.2.2"
        )

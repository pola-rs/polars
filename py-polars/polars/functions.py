from typing import (
    Union,
    TextIO,
    Optional,
    List,
    BinaryIO,
    Sequence,
    Dict,
    Any,
    Type,
    Iterator,
    ContextManager,
    overload,
)
import builtins
import urllib.request
import io
from io import StringIO, BytesIO
from contextlib import contextmanager
from pathlib import Path


import numpy as np

try:
    import pandas as pd
except ImportError:
    pass

try:
    import fsspec
    from fsspec.implementations.local import make_path_posix
    from fsspec.utils import infer_compression, infer_storage_options

    WITH_FSSPEC = True
except ImportError:
    WITH_FSSPEC = False

import pyarrow as pa
import pyarrow.parquet
import pyarrow.csv
import pyarrow.compute

from .frame import DataFrame
from .series import Series
from .lazy import LazyFrame
from .datatypes import DataType


def _process_http_file(path: str) -> io.BytesIO:
    with urllib.request.urlopen(path) as f:
        return io.BytesIO(f.read())


@overload
def _prepare_file_arg(
    file: Union[str, TextIO, Path, BinaryIO], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO]]:
    ...


@overload
def _prepare_file_arg(
    file: Union[str, List[str], TextIO, Path, BinaryIO], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO, List[str], List[BinaryIO]]]:
    ...


def _prepare_file_arg(
    file: Union[str, List[str], TextIO, Path, BinaryIO], **kwargs: Any
) -> ContextManager[Union[str, BinaryIO, List[str], List[BinaryIO]]]:
    """
    Utility for read_[csv, parquet]. (not to be used by scan_[csv, parquet]).
    Returned value is always usable as a context.

    A `StringIO`, `BytesIO` file is returned as a `BytesIO`
    A local path is returned as a string
    An http url is read into a buffer and returned as a `BytesIO`

    When fsspec is installed, except for `StringIO`, `BytesIO` and local
    uncompressed files, the file is opened with `fsspec.open(file, **kwargs)`,
    in which case, the compression is inferred.
    """

    compression = kwargs.pop("compression", "infer")

    # Small helper to use a variable as context
    @contextmanager
    def managed_file(file: Any) -> Iterator[Any]:
        try:
            yield file
        finally:
            pass

    if isinstance(file, StringIO):
        return io.BytesIO(file.read().encode("utf8"))
    if isinstance(file, BytesIO):
        return file
    if isinstance(file, Path):
        return managed_file(str(file))
    if isinstance(file, str):
        if WITH_FSSPEC:
            compressed = infer_compression(file) is not None
            local = infer_storage_options(file)["protocol"] == "file"
            if local and not compressed:
                return managed_file(make_path_posix(file))
            return fsspec.open(file, compression=compression, **kwargs)
        if file.startswith("http"):
            return _process_http_file(file)
    if isinstance(file, list) and bool(file) and all(isinstance(f, str) for f in file):
        if WITH_FSSPEC:
            compressed = any(infer_compression(f) is not None for f in file)
            local = all(infer_storage_options(f)["protocol"] == "file" for f in file)
            if local and not compressed:
                return managed_file(list(map(make_path_posix, file)))
            return fsspec.open_files(file, compression=compression, **kwargs)
    return managed_file(file)


def get_dummies(df: DataFrame) -> DataFrame:
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


def read_csv(
    file: Union[str, TextIO, Path, BinaryIO],
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
    dtype: Optional[Dict[str, Type[DataType]]] = None,
    new_columns: Optional[List[str]] = None,
    use_pyarrow: bool = True,
    low_memory: bool = False,
    comment_char: Optional[str] = None,
    storage_options: Optional[Dict] = None,
    null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
) -> DataFrame:
    """
    Read into a DataFrame from a csv file.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO`` or ``BytesIO``.
        If ``fsspec`` is installed, it will be used to open non-local or compressed files
    infer_schema_length
        Maximum number of lines to read to infer schema.
    batch_size
        Number of lines to read into the buffer at once. Modify this to change performance.
    has_headers
        Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`,
        `x` being an enumeration over every column in the dataset.
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
        Rename columns to these right after parsing. Note that the length of this list must equal the width of the DataFrame
        that's parsed.
    use_pyarrow
        Try to use pyarrow's native CSV parser. This is not always possible. The set of arguments given to this function
        determine if it is possible to use pyarrows native parser. Note that pyarrow and polars may have a different
        strategy regarding type inference.
    low_memory
        Reduce memory usage in expense of performance.
    comment_char
        character that indicates the start of a comment line, for instance '#'.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.
    null_values
        Values to interpret as null values. You can provide a:

        - str -> all values encountered equal to this string will be null
        - List[str] -> A null value per column.
        - Dict[str, str] -> A dictionary that maps column name to a null value string.

    Returns
    -------
    DataFrame
    """

    storage_options = storage_options or {}

    if columns and not has_headers:
        for column in columns:
            if not column.startswith("column_"):
                raise ValueError(
                    'Specified column names do not start with "column_", '
                    "but autogenerated header names were requested."
                )

    if (
        use_pyarrow
        and dtype is None
        and stop_after_n_rows is None
        and n_threads is None
        and encoding == "utf8"
        and not low_memory
        and null_values is None
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

        if new_columns:
            tbl = tbl.rename_columns(new_columns)
        elif not has_headers:
            # Rename 'f0', 'f1', ... columns names autogenated by pyarrow to 'column_1', 'column_2', ...
            tbl = tbl.rename_columns(
                [f"column_{int(column[1:]) + 1}" for column in tbl.column_names]
            )

        return from_arrow(tbl, rechunk)  # type: ignore

    with _prepare_file_arg(file, **storage_options) as data:
        df = DataFrame.read_csv(
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
            null_values=null_values,
        )

    if new_columns:
        df.columns = new_columns
    return df


def scan_csv(
    file: Union[str, Path],
    has_headers: bool = True,
    ignore_errors: bool = False,
    sep: str = ",",
    skip_rows: int = 0,
    stop_after_n_rows: Optional[int] = None,
    cache: bool = True,
    dtype: Optional[Dict[str, Type[DataType]]] = None,
    low_memory: bool = False,
    comment_char: Optional[str] = None,
    null_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
) -> LazyFrame:
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
        During multi-threaded parsing, an upper bound of `n` rows
        cannot be guaranteed.
    cache
        Cache the result after reading.
    dtype
        Overwrite the dtypes during inference.
    low_memory
        Reduce memory usage in expense of performance.
    comment_char
        character that indicates the start of a comment line, for instance '#'.
    null_values
        Values to interpret as null values. You can provide a:

        - str -> all values encountered equal to this string will be null
        - List[str] -> A null value per column.
        - Dict[str, str] -> A dictionary that maps column name to a null value string.
    """
    if isinstance(file, Path):
        file = str(file)
    return LazyFrame.scan_csv(
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
        null_values=null_values,
    )


def scan_parquet(
    file: Union[str, Path],
    stop_after_n_rows: Optional[int] = None,
    cache: bool = True,
) -> LazyFrame:
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
    return LazyFrame.scan_parquet(
        file=file, stop_after_n_rows=stop_after_n_rows, cache=cache
    )


def read_ipc(
    file: Union[str, BinaryIO, Path],
    use_pyarrow: bool = True,
    storage_options: Optional[Dict] = None,
) -> DataFrame:
    """
    Read into a DataFrame from Arrow IPC stream format. This is also called the feather format.

    Parameters
    ----------
    file
        Path to a file or a file like object.
        If ``fsspec`` is installed, it will be used to open non-local or compressed files
    use_pyarrow
        Use pyarrow or rust arrow backend.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.

    Returns
    -------
    DataFrame
    """
    storage_options = storage_options or {}
    with _prepare_file_arg(file, **storage_options) as data:
        return DataFrame.read_ipc(data, use_pyarrow)  # type: ignore


def read_parquet(
    source: Union[str, BinaryIO, Path, List[str]],
    stop_after_n_rows: Optional[int] = None,
    memory_map: bool = True,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    source
        Path to a file | list of files, or a file like object. If the path is a directory, that directory will be used
        as partition aware scan.
        If ``fsspec`` is installed, it will be used to open non-local or compressed files
    stop_after_n_rows
        After n rows are read from the parquet, it stops reading. Note: this cannot be used in partition aware parquet
        reads.
    memory_map
        Memory map underlying file. This will likely increase performance.
    columns
        Columns to project/ select.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage connection, e.g. host, port, username, password, etc.
    **kwargs
        kwargs for [pyarrow.parquet.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html)

    Returns
    -------
    DataFrame
    """
    storage_options = storage_options or {}
    with _prepare_file_arg(source, **storage_options) as source_prep:
        if stop_after_n_rows is not None:
            return DataFrame.read_parquet(source_prep, stop_after_n_rows=stop_after_n_rows)  # type: ignore
        return from_arrow(  # type: ignore
            pa.parquet.read_table(
                source_prep, memory_map=memory_map, columns=columns, **kwargs
            )
        )


def arg_where(mask: Series) -> Series:
    """
    Get index values where Boolean mask evaluate True.

    Parameters
    ----------
    mask
        Boolean Series.

    Returns
    -------
    UInt32 Series
    """
    return mask.arg_true()


def from_arrow_table(table: pa.Table, rechunk: bool = True) -> DataFrame:
    """
    .. deprecated:: 7.3
        use `from_arrow`

    Create a DataFrame from an arrow Table.

    Parameters
    ----------
    a
        Arrow Table.
    rechunk
        Make sure that all data is contiguous.
    """
    return DataFrame.from_arrow(table, rechunk)


def from_arrow(
    a: Union[pa.Table, pa.Array], rechunk: bool = True
) -> Union[DataFrame, Series]:
    """
    Create a DataFrame from an arrow Table.

    Parameters
    ----------
    a
        Arrow Table.
    rechunk
        Make sure that all data is contiguous.
    """
    if isinstance(a, pa.Table):
        return DataFrame.from_arrow(a, rechunk)
    elif isinstance(a, pa.Array):
        return Series.from_arrow("", a)
    else:
        raise ValueError(f"expected arrow table / array, got {a}")


def _from_pandas_helper(a: pd.Series) -> pa.Array:  # noqa: F821
    dtype = a.dtype
    if dtype == "datetime64[ns]":
        # We first cast to ms because that's the unit of Date64,
        # Then we cast to via int64 to date64. Casting directly to Date64 lead to
        # loss of time information https://github.com/ritchie46/polars/issues/476
        arr = pa.array(np.array(a.values, dtype="datetime64[ms]"))
        arr = pa.compute.cast(arr, pa.int64())
        return pa.compute.cast(arr, pa.date64())
    elif dtype == "object" and isinstance(a.iloc[0], str):
        return pa.array(a, pa.large_utf8())
    else:
        return pa.array(a)


def from_pandas(
    df: Union[pd.DataFrame, pd.Series, pd.DatetimeIndex],
    rechunk: bool = True,  # noqa: F821
) -> DataFrame:
    """
    Convert from a pandas DataFrame to a polars DataFrame.

    Parameters
    ----------
    df
        DataFrame to convert.
    rechunk
        Make sure that all data is contiguous.

    Returns
    -------
    A Polars DataFrame
    """
    if isinstance(df, pd.Series) or isinstance(df, pd.DatetimeIndex):
        return from_arrow(_from_pandas_helper(df))  # type: ignore

    # Note: we first tried to infer the schema via pyarrow and then modify the schema if needed.
    # However arrow 3.0 determines the type of a string like this:
    #       pa.array(array).type
    # needlessly allocating and failing when the string is too large for the string dtype.
    data = {}

    for name in df.columns:
        s = df[name]
        data[name] = _from_pandas_helper(s)

    table = pa.table(data)
    return from_arrow(table, rechunk)  # type: ignore


def concat(dfs: List[DataFrame], rechunk: bool = True) -> DataFrame:
    """
    Aggregate all the Dataframes in a List of DataFrames to a single DataFrame.

    Parameters
    ----------
    dfs
        DataFrames to concatenate.
    rechunk
        rechunk the final DataFrame.
    """
    assert len(dfs) > 0
    df = dfs[0].clone()
    for i in builtins.range(1, len(dfs)):
        try:
            df = df.vstack(dfs[i], in_place=False)  # type: ignore
        # could have a double borrow (one mutable one ref)
        except RuntimeError:
            df.vstack(dfs[i].clone(), in_place=True)

    if rechunk:
        return df.rechunk()
    return df


def repeat(val: Union[int, float, str], n: int, name: Optional[str] = None) -> Series:
    """
    Repeat a single value n times and collect into a Series.

    Parameters
    ----------
    val
        Value to repeat.
    n
        Number of repeats.
    name
        Optional name of the Series.
    """
    if name is None:
        name = ""
    if isinstance(val, str):
        s = Series._repeat(name, val, n)
        s.rename(name)
        return s
    else:
        return Series.from_arrow(name, pa.repeat(val, n))


def read_json(
    source: Union[str, StringIO, Path],
) -> DataFrame:
    """
    Read into a DataFrame from JSON format.

    Parameters
    ----------
    source
        Path to a file or a file like object.
    """
    return DataFrame.read_json(source)  # type: ignore


def from_rows(
    rows: Sequence[Sequence[Any]],
    column_names: Optional[List[str]] = None,
    column_name_mapping: Optional[Dict[int, str]] = None,
) -> DataFrame:
    """
    Create a DataFrame from rows. This should only be used as a last resort, as this is more expensive than
    creating from columnar data.

    Parameters
    ----------
    rows
        rows.
    column_names
        column names to use for the DataFrame.
    column_name_mapping
        map column index to a new name:
        Example:
        ```python
            column_mapping: {0: "first_column, 3: "fourth column"}
        ```
    """
    return DataFrame.from_rows(rows, column_names, column_name_mapping)


def read_sql(sql: str, engine: Any) -> DataFrame:
    """
    # Preface
    Deprecated by design. Will not have a long future support and no guarantees given whatsoever.
    Want backwards compatibility?

    Use:

    ```python
    df = pl.from_pandas(pd.read_sql(sql, engine))
    ```

    The support is limited because I want something better.

    # Docstring
    Load a DataFrame from a database by sending a raw sql query.
    Make sure to install sqlalchemy ^1.4

    Parameters
    ----------
    sql
        raw sql query
    engine : sqlalchemy engine
        make sure to install sqlalchemy ^1.4
    """
    try:
        # pandas sql loading is faster.
        # conversion from pandas to arrow is very cheap compared to db driver
        import pandas as pd

        return from_pandas(pd.read_sql(sql, engine))
    except ImportError:
        from sqlalchemy import text

        with engine.connect() as con:
            result = con.execute(text(sql))

        rows = result.fetchall()
        return from_rows(rows, list(result.keys()))

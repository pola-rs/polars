from typing import Union, TextIO, Optional, List, BinaryIO
import numpy as np
from pathlib import Path
from .frame import DataFrame
from .series import Series
from .lazy import LazyFrame
import pyarrow as pa
import pyarrow.parquet
import pyarrow.csv
import pyarrow.compute
import builtins
import urllib.request
import io

from typing import Dict
from .datatypes import DataType


def _process_http_file(path: str) -> io.BytesIO:
    with urllib.request.urlopen(path) as f:
        return io.BytesIO(f.read())


def _prepare_file_arg(
    file: Union[str, TextIO, Path, BinaryIO]
) -> Union[str, TextIO, Path, BinaryIO]:
    """
    Utility for read_[csv, parquet]. (not to be used by scan_[csv, parquet]).

    Does one of:
        - A path.Path object is converted to a string
        - a raw file on the web is downloaded into a buffer.
    """
    if isinstance(file, Path):
        file = str(file)

    if isinstance(file, str) and file.startswith("http"):
        file = _process_http_file(file)

    return file


def get_dummies(df: DataFrame) -> DataFrame:
    return df.to_dummies()


def read_csv(
    file: Union[str, TextIO, Path],
    infer_schema_length: int = 100,
    batch_size: int = 64,
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
    dtype: "Optional[Dict[str, DataType]]" = None,
    new_columns: "Optional[List[str]]" = None,
    use_pyarrow: bool = True,
) -> "DataFrame":
    """
    Read into a DataFrame from a csv file.

    Parameters
    ----------
    file
        Path to a file or a file like object.
    infer_schema_length
        Maximum number of lines to read to infer schema.
    batch_size
        Number of lines to read into the buffer at once. Modify this to change performance.
    has_headers
        If the CSV file has headers or not.
    ignore_errors
        Try to keep reading lines if some lines yield errors.
    stop_after_n_rows
        After n rows are read from the CSV stop reading. During multi-threaded parsing, an upper bound of `n` rows
        cannot be guaranteed.
    skip_rows
        Start reading after `skip_rows`.
    projection
        Indexes of columns to select
    sep
        Delimiter/ value separator
    columns
        Columns to project/ select
    rechunk
        Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
    encoding
        - "utf8"
        _ "utf8-lossy"
    n_threads
        Number of threads to use in csv parsing. Defaults to the number of physical cpu's of you system.
    dtype
        Overwrite the dtypes during inference
    use_pyarrow
        Use pyarrow's native CSV parser.

    Returns
    -------
    DataFrame
    """
    file = _prepare_file_arg(file)

    if (
        use_pyarrow
        and dtype is None
        and has_headers
        and projection is None
        and sep == ","
        and columns is None
        and stop_after_n_rows is None
        and not ignore_errors
        and n_threads is None
        and encoding == "utf8"
    ):
        tbl = pa.csv.read_csv(file, pa.csv.ReadOptions(skip_rows=skip_rows))
        return from_arrow(tbl, rechunk)

    df = DataFrame.read_csv(
        file=file,
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
    stop_after_n_rows: "Optional[int]" = None,
    cache: bool = True,
    dtype: "Optional[Dict[str, DataType]]" = None,
) -> "LazyFrame":
    """
    Lazily read from a csv file.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a file
    has_headers
        If the CSV file has headers or not.
    ignore_errors
        Try to keep reading lines if some lines yield errors.
    sep
        Delimiter/ value separator
    skip_rows
        Start reading after `skip_rows`.
    stop_after_n_rows
        After n rows are read from the CSV stop reading.
        During multi-threaded parsing, an upper bound of `n` rows
        cannot be guaranteed.
    cache
        Cache the result after reading
    dtype
        Overwrite the dtypes during inference
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
    )


def scan_parquet(
    file: Union[str, Path],
    stop_after_n_rows: "Optional[int]" = None,
    cache: bool = True,
) -> "LazyFrame":
    """
    Lazily read from a parquet file.

    This allows the query optimizer to push down predicates and projections to the scan level,
    thereby potentially reducing memory overhead.

    Parameters
    ----------
    file
        Path to a file
    stop_after_n_rows
        After n rows are read from the parquet stops reading.
    cache
        Cache the result after reading
    """
    if isinstance(file, Path):
        file = str(file)
    return LazyFrame.scan_parquet(
        file=file, stop_after_n_rows=stop_after_n_rows, cache=cache
    )


def read_ipc(file: Union[str, BinaryIO, Path]) -> "DataFrame":
    """
    Read into a DataFrame from Arrow IPC stream format. This is also called the feather format.

    Parameters
    ----------
    file
        Path to a file or a file like object.

    Returns
    -------
    DataFrame
    """
    file = _prepare_file_arg(file)
    return DataFrame.read_ipc(file)


def read_parquet(
    source: "Union[str, BinaryIO, Path, List[str]]",
    stop_after_n_rows: "Optional[int]" = None,
    memory_map=True,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> "DataFrame":
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    source
        Path to a file | list of files, or a file like object. If the path is a directory, that directory will be used
        as partition aware scan.
    stop_after_n_rows
        After n rows are read from the parquet stops reading. Note: this cannot be used in partition aware parquet reads.
    memory_map
        Memory map underlying file. This will likely increase performance.
    columns
        Columns to project / select
    **kwargs
        kwargs for [pyarrow.parquet.read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html)

    Returns
    -------
    DataFrame
    """
    source = _prepare_file_arg(source)
    if stop_after_n_rows is not None:
        return DataFrame.read_parquet(source, stop_after_n_rows=stop_after_n_rows)
    else:
        return from_arrow(
            pa.parquet.read_table(
                source, memory_map=memory_map, columns=columns, **kwargs
            )
        )


def arg_where(mask: "Series"):
    """
    Get index values where Boolean mask evaluate True.

    Parameters
    ----------
    mask
        Boolean Series

    Returns
    -------
    UInt32 Series
    """
    return mask.arg_true()


def from_arrow_table(table: pa.Table, rechunk: bool = True) -> "DataFrame":
    """
    .. deprecated:: 7.3
        use `from_arrow`

    Create a DataFrame from an arrow Table

    Parameters
    ----------
    a
        Arrow Table
    rechunk
        Make sure that all data is contiguous.
    """
    return DataFrame.from_arrow(table, rechunk)


def from_arrow(a: "Union[pa.Table, pa.Array]", rechunk: bool = True) -> "DataFrame":
    """
    Create a DataFrame from an arrow Table

    Parameters
    ----------
    a
        Arrow Table
    rechunk
        Make sure that all data is contiguous.
    """
    if isinstance(a, pa.Table):
        return DataFrame.from_arrow(a, rechunk)
    if isinstance(a, pa.Array):
        return Series.from_arrow("", a)
    raise ValueError(f"expected arrow table / array, got {a}")


def from_pandas(
    df: "pandas.DataFrame", rechunk: bool = True  # noqa: F821
) -> "DataFrame":
    """
    Convert from a pandas DataFrame to a polars DataFrame

    Parameters
    ----------
    df
        DataFrame to convert
    rechunk
        Make sure that all data is contiguous.

    Returns
    -------
    A Polars DataFrame
    """

    # Note: we first tried to infer the schema via pyarrow and then modify the schema if needed.
    # However arrow 3.0 determines the type of a string like this:
    #       pa.array(array).type
    # needlessly allocating and failing when the string is too large for the string dtype.
    data = {}

    for (name, dtype) in zip(df.columns, df.dtypes):
        if dtype == "object" and isinstance(df[name][0], str):
            data[name] = pa.array(df[name], pa.large_utf8())
        elif dtype == "datetime64[ns]":
            # We first cast to ms because that's the unit of Date64
            # Then we cast to via int64 to date64. Casting directly to Date64 lead to
            # loss of time information https://github.com/ritchie46/polars/issues/476
            arr = pa.array(np.array(df[name].values, dtype="datetime64[ms]"))
            arr = pa.compute.cast(arr, pa.int64())
            data[name] = pa.compute.cast(arr, pa.date64())
        else:
            data[name] = pa.array(df[name])

    table = pa.table(data)
    return from_arrow(table, rechunk)


def concat(dfs: "List[DataFrame]", rechunk=True) -> "DataFrame":
    """
    Aggregate all the Dataframe in a List of DataFrames to a single DataFrame

    Parameters
    ----------
    dfs
        DataFrames to concatenate
    rechunk
        rechunk the final DataFrame
    """
    assert len(dfs) > 0
    df = dfs[0]
    for i in builtins.range(1, len(dfs)):
        try:
            df = df.vstack(dfs[i], in_place=False)
        # could have a double borrow (one mutable one ref)
        except RuntimeError:
            df.vstack(dfs[i].clone(), in_place=True)

    if rechunk:
        return df.rechunk()
    return df


def arange(
    lower: int, upper: int, step: Optional[int] = None, name: Optional[str] = None
) -> Series:
    """
    Create a Series that ranges from lower bound to upper bound.
    Parameters
    ----------
    lower
        Lower bound value.
    upper
        Upper bound value.
    step
        Optional step size. If none given, the step size will be 1.
    name
        Name of the Series
    """
    if name is None:
        name = ""
    return Series(name, np.arange(lower, upper, step), nullable=False)


def repeat(
    val: "Union[int, float, str]", n: int, name: Optional[str] = None
) -> "Series":
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

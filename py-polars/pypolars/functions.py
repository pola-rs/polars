from typing import Union, TextIO, Optional, List, BinaryIO

from .frame import DataFrame
from .lazy import LazyFrame


def get_dummies(df: DataFrame) -> DataFrame:
    return df.to_dummies()


def read_csv(
    file: Union[str, TextIO],
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
        Delimiter/ value seperator
    columns
        Columns to project/ select
    rechunk
        Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
    encoding
        - "utf8"
        _ "utf8-lossy"
    n_threads
        Number of threads to use in csv parsing. Defaults to the number of physical cpu's of you system.

    Returns
    -------
    DataFrame
    """

    return DataFrame.read_csv(
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
    )


def scan_csv(
    file: str,
    has_headers: bool = True,
    ignore_errors: bool = False,
    sep: str = ",",
    skip_rows: int = 0,
    stop_after_n_rows: "Optional[int]" = None,
    cache: bool = True,
) -> "LazyFrame":
    """
    Read into a DataFrame from a csv file.

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
    """
    return LazyFrame.scan_csv(
        file=file,
        has_headers=has_headers,
        sep=sep,
        ignore_errors=ignore_errors,
        skip_rows=skip_rows,
        stop_after_n_rows=stop_after_n_rows,
        cache=cache,
    )


def scan_parquet(
    file: str, stop_after_n_rows: "Optional[int]" = None, cache: bool = True
) -> "LazyFrame":
    """
    Read into a DataFrame from a csv file.

    Parameters
    ----------
    file
        Path to a file
    stop_after_n_rows
        After n rows are read from the parquet stops reading.
    cache
        Cache the result after reading
    """
    return LazyFrame.scan_parquet(
        file=file, stop_after_n_rows=stop_after_n_rows, cache=cache
    )


def read_ipc(file: Union[str, BinaryIO]) -> "DataFrame":
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
    return DataFrame.read_ipc(file)


def read_parquet(
    file: Union[str, BinaryIO],
) -> "DataFrame":
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    file
        Path to a file or a file like object.

    Returns
    -------
    DataFrame
    """
    return DataFrame.read_parquet(file)


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

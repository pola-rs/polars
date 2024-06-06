from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Sequence

import polars._reexport as pl
from polars._utils.deprecation import deprecate_renamed_parameter
from polars._utils.various import (
    is_str_sequence,
    normalize_filepath,
)
from polars._utils.wrap import wrap_df, wrap_ldf
from polars.dependencies import import_optional
from polars.io._utils import (
    is_glob_pattern,
    is_local_file,
    parse_columns_arg,
    parse_row_index_args,
    prepare_file_arg,
)
from polars.io.ipc.anonymous_scan import _scan_ipc_fsspec

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame, PyLazyFrame
    from polars.polars import read_ipc_schema as _read_ipc_schema

if TYPE_CHECKING:
    from polars import DataFrame, DataType, LazyFrame


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def read_ipc(
    source: str | Path | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    use_pyarrow: bool = False,
    memory_map: bool = True,
    storage_options: dict[str, Any] | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = True,
) -> DataFrame:
    """
    Read into a DataFrame from Arrow IPC (Feather v2) file.

    See "File or Random Access format" on https://arrow.apache.org/docs/python/ipc.html.
    Arrow IPC files are also known as Feather (v2) files.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance). If `fsspec` is installed, it will be used
        to open remote files.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from IPC file after reading `n_rows`.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Use pyarrow or the native Rust reader.
    memory_map
        Try to memory map the file. This can greatly improve performance on repeated
        queries as the OS may cache pages.
        Only uncompressed IPC files can be memory mapped.
    storage_options
        Extra options that make sense for `fsspec.open()` or a particular storage
        connection, e.g. host, port, username, password, etc.
    row_index_name
        Insert a row index column with the given name into the DataFrame as the first
        column. If set to `None` (default), no row index column is created.
    row_index_offset
        Start the row index at this offset. Cannot be negative.
        Only used if `row_index_name` is set.
    rechunk
        Make sure that all data is contiguous.

    Returns
    -------
    DataFrame

    Warnings
    --------
    If `memory_map` is set, the bytes on disk are mapped 1:1 to memory.
    That means that you cannot write to the same filename.
    E.g. `pl.read_ipc("my_file.arrow").write_ipc("my_file.arrow")` will fail.
    """
    if use_pyarrow and n_rows and not memory_map:
        msg = "`n_rows` cannot be used with `use_pyarrow=True` and `memory_map=False`"
        raise ValueError(msg)

    with prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            pyarrow_feather = import_optional(
                "pyarrow.feather",
                err_prefix="",
                err_suffix="is required when using 'read_ipc(..., use_pyarrow=True)'",
            )
            tbl = pyarrow_feather.read_table(
                data,
                memory_map=memory_map,
                columns=columns,
            )
            df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
            if row_index_name is not None:
                df = df.with_row_index(row_index_name, row_index_offset)
            if n_rows is not None:
                df = df.slice(0, n_rows)
            return df

        return _read_ipc_impl(
            data,
            columns=columns,
            n_rows=n_rows,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            rechunk=rechunk,
            memory_map=memory_map,
        )


def _read_ipc_impl(
    source: str | Path | IO[bytes] | bytes,
    *,
    columns: Sequence[int] | Sequence[str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = True,
    memory_map: bool = True,
) -> DataFrame:
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    if isinstance(columns, str):
        columns = [columns]

    if isinstance(source, str) and is_glob_pattern(source) and is_local_file(source):
        scan = scan_ipc(
            source,
            n_rows=n_rows,
            rechunk=rechunk,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            memory_map=memory_map,
        )
        if columns is None:
            df = scan.collect()
        elif is_str_sequence(columns, allow_str=False):
            df = scan.select(columns).collect()
        else:
            msg = (
                "cannot use glob patterns and integer based projection as `columns` argument"
                "\n\nUse columns: List[str]"
            )
            raise TypeError(msg)
        return df

    projection, columns = parse_columns_arg(columns)
    pydf = PyDataFrame.read_ipc(
        source,
        columns,
        projection,
        n_rows,
        parse_row_index_args(row_index_name, row_index_offset),
        memory_map=memory_map,
    )
    return wrap_df(pydf)


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def read_ipc_stream(
    source: str | Path | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    use_pyarrow: bool = False,
    storage_options: dict[str, Any] | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = True,
) -> DataFrame:
    """
    Read into a DataFrame from Arrow IPC record batch stream.

    See "Streaming format" on https://arrow.apache.org/docs/python/ipc.html.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance). If `fsspec` is installed, it will be used
        to open remote files.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from IPC stream after reading `n_rows`.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Use pyarrow or the native Rust reader.
    storage_options
        Extra options that make sense for `fsspec.open()` or a particular storage
        connection, e.g. host, port, username, password, etc.
    row_index_name
        Insert a row index column with the given name into the DataFrame as the first
        column. If set to `None` (default), no row index column is created.
    row_index_offset
        Start the row index at this offset. Cannot be negative.
        Only used if `row_index_name` is set.
    rechunk
        Make sure that all data is contiguous.

    Returns
    -------
    DataFrame
    """
    with prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            pyarrow_ipc = import_optional(
                "pyarrow.ipc",
                err_prefix="",
                err_suffix="is required when using 'read_ipc_stream(..., use_pyarrow=True)'",
            )
            with pyarrow_ipc.RecordBatchStreamReader(data) as reader:
                tbl = reader.read_all()
                df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
                if row_index_name is not None:
                    df = df.with_row_index(row_index_name, row_index_offset)
                if n_rows is not None:
                    df = df.slice(0, n_rows)
                return df

        return _read_ipc_stream_impl(
            data,
            columns=columns,
            n_rows=n_rows,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            rechunk=rechunk,
        )


def _read_ipc_stream_impl(
    source: str | Path | IO[bytes] | bytes,
    *,
    columns: Sequence[int] | Sequence[str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = True,
) -> DataFrame:
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    if isinstance(columns, str):
        columns = [columns]

    projection, columns = parse_columns_arg(columns)
    pydf = PyDataFrame.read_ipc_stream(
        source,
        columns,
        projection,
        n_rows,
        parse_row_index_args(row_index_name, row_index_offset),
        rechunk,
    )
    return wrap_df(pydf)


def read_ipc_schema(source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]:
    """
    Get the schema of an IPC file without reading data.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance).

    Returns
    -------
    dict
        Dictionary mapping column names to datatypes
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)

    return _read_ipc_schema(source)


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def scan_ipc(
    source: str | Path | list[str] | list[Path],
    *,
    n_rows: int | None = None,
    cache: bool = True,
    rechunk: bool = False,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    storage_options: dict[str, Any] | None = None,
    memory_map: bool = True,
    retries: int = 0,
) -> LazyFrame:
    """
    Lazily read from an Arrow IPC (Feather v2) file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, thereby potentially reducing memory overhead.

    Parameters
    ----------
    source
        Path to a IPC file.
    n_rows
        Stop reading from IPC file after reading `n_rows`.
    cache
        Cache the result after reading.
    rechunk
        Reallocate to contiguous memory when all chunks/ files are parsed.
    row_index_name
        If not None, this will insert a row index column with give name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only use if the name is set)
    storage_options
        Extra options that make sense for `fsspec.open()` or a
        particular storage connection.
        e.g. host, port, username, password, etc.
    memory_map
        Try to memory map the file. This can greatly improve performance on repeated
        queries as the OS may cache pages.
        Only uncompressed IPC files can be memory mapped.
    retries
        Number of retries if accessing a cloud instance fails.

    """
    if isinstance(source, (str, Path)):
        can_use_fsspec = True
        source = normalize_filepath(source)
        sources = []
    else:
        can_use_fsspec = False
        sources = [normalize_filepath(source) for source in source]
        source = None  # type: ignore[assignment]

    # try fsspec scanner
    if can_use_fsspec and not is_local_file(source):  # type: ignore[arg-type]
        scan = _scan_ipc_fsspec(source, storage_options)  # type: ignore[arg-type]
        if n_rows:
            scan = scan.head(n_rows)
        if row_index_name is not None:
            scan = scan.with_row_index(row_index_name, row_index_offset)
        return scan

    pylf = PyLazyFrame.new_from_ipc(
        source,
        sources,
        n_rows,
        cache,
        rechunk,
        parse_row_index_args(row_index_name, row_index_offset),
        memory_map=memory_map,
        cloud_options=storage_options,
        retries=retries,
    )
    return wrap_ldf(pylf)

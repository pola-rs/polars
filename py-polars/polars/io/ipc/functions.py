from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

import polars._reexport as pl
from polars.dependencies import _PYARROW_AVAILABLE
from polars.io._utils import _prepare_file_arg
from polars.utils.deprecation import deprecate_renamed_parameter
from polars.utils.various import normalize_filepath

with contextlib.suppress(ImportError):
    from polars.polars import read_ipc_schema as _read_ipc_schema

if TYPE_CHECKING:
    from io import BytesIO

    from polars import DataFrame, DataType, LazyFrame


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def read_ipc(
    source: str | BinaryIO | BytesIO | Path | bytes,
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

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`). If `fsspec` is installed, it will be used to open
        remote files.
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

    with _prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                msg = "pyarrow is required when using `read_ipc(..., use_pyarrow=True)`"
                raise ModuleNotFoundError(msg)

            import pyarrow as pa
            import pyarrow.feather

            tbl = pa.feather.read_table(data, memory_map=memory_map, columns=columns)
            df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
            if row_index_name is not None:
                df = df.with_row_index(row_index_name, row_index_offset)
            if n_rows is not None:
                df = df.slice(0, n_rows)
            return df

        return pl.DataFrame._read_ipc(
            data,
            columns=columns,
            n_rows=n_rows,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            rechunk=rechunk,
            memory_map=memory_map,
        )


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def read_ipc_stream(
    source: str | BinaryIO | BytesIO | Path | bytes,
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

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`). If `fsspec` is installed, it will be used to open
        remote files.
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
    with _prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                msg = (
                    "'pyarrow' is required when using"
                    " 'read_ipc_stream(..., use_pyarrow=True)'"
                )
                raise ModuleNotFoundError(msg)

            import pyarrow as pa

            with pa.ipc.RecordBatchStreamReader(data) as reader:
                tbl = reader.read_all()
                df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
                if row_index_name is not None:
                    df = df.with_row_index(row_index_name, row_index_offset)
                if n_rows is not None:
                    df = df.slice(0, n_rows)
                return df

        return pl.DataFrame._read_ipc_stream(
            data,
            columns=columns,
            n_rows=n_rows,
            row_index_name=row_index_name,
            row_index_offset=row_index_offset,
            rechunk=rechunk,
        )


def read_ipc_schema(source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]:
    """
    Get the schema of an IPC file without reading data.

    Parameters
    ----------
    source
        Path to a file or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`).

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
    """
    return pl.LazyFrame._scan_ipc(
        source,
        n_rows=n_rows,
        cache=cache,
        rechunk=rechunk,
        row_index_name=row_index_name,
        row_index_offset=row_index_offset,
        storage_options=storage_options,
        memory_map=memory_map,
    )

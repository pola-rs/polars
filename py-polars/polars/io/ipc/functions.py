from __future__ import annotations

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

import polars._reexport as pl
from polars.dependencies import _PYARROW_AVAILABLE
from polars.io._utils import _prepare_file_arg
from polars.utils.various import normalize_filepath

with contextlib.suppress(ImportError):
    from polars.polars import read_ipc_schema as _read_ipc_schema

if TYPE_CHECKING:
    from io import BytesIO

    from polars import DataFrame, DataType, LazyFrame


def read_ipc(
    source: str | BinaryIO | BytesIO | Path | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    use_pyarrow: bool = False,
    memory_map: bool = True,
    storage_options: dict[str, Any] | None = None,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    rechunk: bool = True,
) -> DataFrame:
    """
    Read into a `DataFrame` from an Arrow IPC (Feather v2) file.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_. If `fsspec
        <https://filesystem-spec.readthedocs.io>`_ is installed, it will be used to open
        remote files.
    columns
        A list of column indices (starting at zero) or column names to read.
    n_rows
        The number of rows to read from the IPC file.
        Only used when `use_pyarrow=False`.
    use_pyarrow
        Whether to use the IPC reader from :mod:`pyarrow` instead of polars's.
    memory_map
        Whether to try to memory-map the file. This can greatly improve performance on
        repeated queries as the OS may cache pages. Only uncompressed IPC files can be
        memory-mapped.
    storage_options
        Extra options that make sense for `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_ for a
        particular storage connection, e.g. host, port, username, password, etc.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.

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
        raise ValueError(
            "`n_rows` cannot be used with `use_pyarrow=True` and `memory_map=False`"
        )

    with _prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ModuleNotFoundError(
                    "pyarrow is required when using `read_ipc(..., use_pyarrow=True)`"
                )

            import pyarrow as pa
            import pyarrow.feather

            tbl = pa.feather.read_table(data, memory_map=memory_map, columns=columns)
            df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
            if row_count_name is not None:
                df = df.with_row_count(row_count_name, row_count_offset)
            if n_rows is not None:
                df = df.slice(0, n_rows)
            return df

        return pl.DataFrame._read_ipc(
            data,
            columns=columns,
            n_rows=n_rows,
            row_count_name=row_count_name,
            row_count_offset=row_count_offset,
            rechunk=rechunk,
            memory_map=memory_map,
        )


def read_ipc_stream(
    source: str | BinaryIO | BytesIO | Path | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    use_pyarrow: bool = False,
    storage_options: dict[str, Any] | None = None,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    rechunk: bool = True,
) -> DataFrame:
    """
    Read into a `DataFrame` from an Arrow IPC record batch stream.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_. If `fsspec
        <https://filesystem-spec.readthedocs.io>`_ is installed, it will be used to open
        remote files.
    columns
        A list of column indices (starting at zero) or column names to read.
    n_rows
        The number of rows to read from the IPC stream.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Whether to use the IPC reader from :mod:`pyarrow` instead of polars's.
    storage_options
        Extra options that make sense for `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_ for a
        particular storage connection.
        e.g. host, port, username, password, etc.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.

    Returns
    -------
    DataFrame

    """
    with _prepare_file_arg(
        source, use_pyarrow=use_pyarrow, storage_options=storage_options
    ) as data:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ModuleNotFoundError(
                    "'pyarrow' is required when using"
                    " 'read_ipc_stream(..., use_pyarrow=True)'"
                )

            import pyarrow as pa

            with pa.ipc.RecordBatchStreamReader(data) as reader:
                tbl = reader.read_all()
                df = pl.DataFrame._from_arrow(tbl, rechunk=rechunk)
                if row_count_name is not None:
                    df = df.with_row_count(row_count_name, row_count_offset)
                if n_rows is not None:
                    df = df.slice(0, n_rows)
                return df

        return pl.DataFrame._read_ipc_stream(
            data,
            columns=columns,
            n_rows=n_rows,
            row_count_name=row_count_name,
            row_count_offset=row_count_offset,
            rechunk=rechunk,
        )


def read_ipc_schema(source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]:
    """
    Get the schema of an IPC file without reading data.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. from the builtin `open
        <https://docs.python.org/3/library/functions.html#open>`_ function) or `BytesIO
        <https://docs.python.org/3/library/io.html#io.BytesIO>`_.

    Returns
    -------
    dict
        A dictionary mapping column names to datatypes.

    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)

    return _read_ipc_schema(source)


def scan_ipc(
    source: str | Path | list[str] | list[Path],
    *,
    n_rows: int | None = None,
    cache: bool = True,
    rechunk: bool = True,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
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
        A path to an IPC file, or a glob pattern matching multiple files.
    n_rows
        The number of rows to read from the IPC file.
    cache
        Whether to cache the result after reading.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    storage_options
        Extra options that make sense for `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_ for a
        particular storage connection.
        e.g. host, port, username, password, etc.
    memory_map
        Whether to memory-map the underlying file. This can greatly improve performance
        on repeated queries as the operating system may cache pages.
        Only uncompressed IPC files can be memory-mapped.

    """
    return pl.LazyFrame._scan_ipc(
        source,
        n_rows=n_rows,
        cache=cache,
        rechunk=rechunk,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
        storage_options=storage_options,
        memory_map=memory_map,
    )

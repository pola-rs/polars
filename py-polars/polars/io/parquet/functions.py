from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import polars._reexport as pl
from polars.convert import from_arrow
from polars.dependencies import _PYARROW_AVAILABLE
from polars.io._utils import _prepare_file_arg
from polars.utils.various import normalise_filepath

with contextlib.suppress(ImportError):
    from polars.polars import read_parquet_schema as _read_parquet_schema

if TYPE_CHECKING:
    from io import BytesIO

    from polars import DataFrame, LazyFrame
    from polars.type_aliases import ParallelStrategy, PolarsDataType


def read_parquet(
    source: str | Path | BinaryIO | BytesIO | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    use_pyarrow: bool = False,
    memory_map: bool = True,
    storage_options: dict[str, Any] | None = None,
    parallel: ParallelStrategy = "auto",
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    low_memory: bool = False,
    pyarrow_options: dict[str, Any] | None = None,
    use_statistics: bool = True,
    rechunk: bool = True,
) -> DataFrame:
    """
    Read into a DataFrame from a parquet file.

    Notes
    -----
    This operation defaults to a `rechunk` operation at the end, meaning that
    all data will be stored continuously in memory.
    Set `rechunk=False` if you are benchmarking the parquet-reader. A `rechunk` is
    an expensive operation.

    Parameters
    ----------
    source
        Path to a file, or a file-like object. If the path is a directory, that
        directory will be used as partition aware scan.
        If ``fsspec`` is installed, it will be used to open remote files.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from parquet file after reading ``n_rows``.
        Only valid when `use_pyarrow=False`.
    use_pyarrow
        Use pyarrow instead of the Rust native parquet reader. The pyarrow reader is
        more stable.
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when ``use_pyarrow=True``.
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a particular storage
        connection, e.g. host, port, username, password, etc.
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        This determines the direction of parallelism. 'auto' will try to determine the
        optimal direction.
    row_count_name
        If not None, this will insert a row count column with give name into the
        DataFrame.
    row_count_offset
        Offset to start the row_count column (only use if the name is set).
    low_memory
        Reduce memory pressure at the expense of performance.
    pyarrow_options
        Keyword arguments for `pyarrow.parquet.read_table
        <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html>`_.
    use_statistics
        Use statistics in the parquet to determine if pages
        can be skipped from reading.
    rechunk
        Make sure that all columns are contiguous in memory by
        aggregating the chunks into a single array.

    Returns
    -------
    DataFrame

    """
    if use_pyarrow and n_rows:
        raise ValueError("``n_rows`` cannot be used with ``use_pyarrow=True``.")

    storage_options = storage_options or {}
    pyarrow_options = pyarrow_options or {}

    with _prepare_file_arg(
        source, use_pyarrow=use_pyarrow, **storage_options
    ) as source_prep:
        if use_pyarrow:
            if not _PYARROW_AVAILABLE:
                raise ImportError(
                    "'pyarrow' is required when using"
                    " 'read_parquet(..., use_pyarrow=True)'."
                )

            import pyarrow as pa
            import pyarrow.parquet

            return from_arrow(  # type: ignore[return-value]
                pa.parquet.read_table(
                    source_prep,
                    memory_map=memory_map,
                    columns=columns,
                    **pyarrow_options,
                )
            )

        return pl.DataFrame._read_parquet(
            source_prep,
            columns=columns,
            n_rows=n_rows,
            parallel=parallel,
            row_count_name=row_count_name,
            row_count_offset=row_count_offset,
            low_memory=low_memory,
            use_statistics=use_statistics,
            rechunk=rechunk,
        )


def read_parquet_schema(
    source: str | BinaryIO | Path | bytes,
) -> dict[str, PolarsDataType]:
    """
    Get the schema of a Parquet file without reading data.

    Parameters
    ----------
    source
        Path to a file or a file-like object.

    Returns
    -------
    Dictionary mapping column names to datatypes

    """
    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    return _read_parquet_schema(source)


def scan_parquet(
    source: str | Path,
    *,
    n_rows: int | None = None,
    cache: bool = True,
    parallel: ParallelStrategy = "auto",
    rechunk: bool = True,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    storage_options: dict[str, Any] | None = None,
    low_memory: bool = False,
    use_statistics: bool = True,
) -> LazyFrame:
    """
    Lazily read from a parquet file or multiple files via glob patterns.

    This allows the query optimizer to push down predicates and projections to the scan
    level, thereby potentially reducing memory overhead.

    Parameters
    ----------
    source
        Path to a file.
    n_rows
        Stop reading from parquet file after reading ``n_rows``.
    cache
        Cache the result after reading.
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        This determines the direction of parallelism. 'auto' will try to determine the
        optimal direction.
    rechunk
        In case of reading multiple files via a glob pattern rechunk the final DataFrame
        into contiguous memory chunks.
    row_count_name
        If not None, this will insert a row count column with give name into the
        DataFrame
    row_count_offset
        Offset to start the row_count column (only use if the name is set)
    storage_options
        Extra options that make sense for ``fsspec.open()`` or a
        particular storage connection.
        e.g. host, port, username, password, etc.
    low_memory
        Reduce memory pressure at the expense of performance.
    use_statistics
        Use statistics in the parquet to determine if pages
        can be skipped from reading.

    """
    if isinstance(source, (str, Path)):
        source = normalise_filepath(source)

    return pl.LazyFrame._scan_parquet(
        source,
        n_rows=n_rows,
        cache=cache,
        parallel=parallel,
        rechunk=rechunk,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
        storage_options=storage_options,
        low_memory=low_memory,
        use_statistics=use_statistics,
    )

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import polars._reexport as pl
from polars.convert import from_arrow
from polars.dependencies import _PYARROW_AVAILABLE
from polars.io._utils import _prepare_file_arg
from polars.utils.deprecation import deprecate_renamed_parameter
from polars.utils.various import is_int_sequence, normalize_filepath

with contextlib.suppress(ImportError):
    from polars.polars import read_parquet_schema as _read_parquet_schema

if TYPE_CHECKING:
    from polars import DataFrame, DataType, LazyFrame
    from polars.type_aliases import ParallelStrategy


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def read_parquet(
    source: str | Path | list[str] | list[Path] | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    parallel: ParallelStrategy = "auto",
    use_statistics: bool = True,
    hive_partitioning: bool = True,
    rechunk: bool = True,
    low_memory: bool = False,
    storage_options: dict[str, Any] | None = None,
    retries: int = 0,
    use_pyarrow: bool = False,
    pyarrow_options: dict[str, Any] | None = None,
    memory_map: bool = True,
) -> DataFrame:
    """
    Read into a DataFrame from a parquet file.

    Parameters
    ----------
    source
        Path to a file, or a file-like object (by file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via builtin `open`
        function) or `BytesIO`). If the path is a directory, files in that
        directory will all be read.
    columns
        Columns to select. Accepts a list of column indices (starting at zero) or a list
        of column names.
    n_rows
        Stop reading from parquet file after reading `n_rows`.
        Only valid when `use_pyarrow=False`.
    row_index_name
        Insert a row index column with the given name into the DataFrame as the first
        column. If set to `None` (default), no row index column is created.
    row_index_offset
        Start the row index at this offset. Cannot be negative.
        Only used if `row_index_name` is set.
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        This determines the direction of parallelism. 'auto' will try to determine the
        optimal direction.
    use_statistics
        Use statistics in the parquet to determine if pages
        can be skipped from reading.
    hive_partitioning
        Infer statistics and schema from hive partitioned URL and use them
        to prune reads.
    rechunk
        Make sure that all columns are contiguous in memory by
        aggregating the chunks into a single array.
    low_memory
        Reduce memory pressure at the expense of performance.
    storage_options
        Options that indicate how to connect to a cloud provider.
        If the cloud provider is not supported by Polars, the storage options
        are passed to `fsspec.open()`.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    retries
        Number of retries if accessing a cloud instance fails.
    use_pyarrow
        Use pyarrow instead of the Rust native parquet reader. The pyarrow reader is
        more stable.
    pyarrow_options
        Keyword arguments for `pyarrow.parquet.read_table
        <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html>`_.
    memory_map
        Memory map underlying file. This will likely increase performance.
        Only used when `use_pyarrow=True`.

    Returns
    -------
    DataFrame

    See Also
    --------
    scan_parquet
    scan_pyarrow_dataset

    Notes
    -----
    * Partitioned files:
        If you have a directory-nested (hive-style) partitioned dataset, you should
        use the :func:`scan_pyarrow_dataset` method instead.
    * When benchmarking:
        This operation defaults to a `rechunk` operation at the end, meaning that all
        data will be stored continuously in memory. Set `rechunk=False` if you are
        benchmarking the parquet-reader as `rechunk` can be an expensive operation
        that should not contribute to the timings.
    """
    # Dispatch to pyarrow if requested
    if use_pyarrow:
        if not _PYARROW_AVAILABLE:
            msg = (
                "'pyarrow' is required when using `read_parquet(..., use_pyarrow=True)`"
            )
            raise ModuleNotFoundError(msg)
        if n_rows is not None:
            msg = "`n_rows` cannot be used with `use_pyarrow=True`"
            raise ValueError(msg)

        import pyarrow as pa
        import pyarrow.parquet

        pyarrow_options = pyarrow_options or {}

        with _prepare_file_arg(
            source,  # type: ignore[arg-type]
            use_pyarrow=True,
            storage_options=storage_options,
        ) as source_prep:
            return from_arrow(  # type: ignore[return-value]
                pa.parquet.read_table(
                    source_prep,
                    memory_map=memory_map,
                    columns=columns,
                    **pyarrow_options,
                )
            )

    # Read binary types using `read_parquet`
    elif isinstance(source, (io.BufferedIOBase, io.RawIOBase, bytes)):
        with _prepare_file_arg(source, use_pyarrow=False) as source_prep:
            return pl.DataFrame._read_parquet(
                source_prep,
                columns=columns,
                n_rows=n_rows,
                parallel=parallel,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                low_memory=low_memory,
                use_statistics=use_statistics,
                rechunk=rechunk,
            )

    # For other inputs, defer to `scan_parquet`
    lf = scan_parquet(
        source,  # type: ignore[arg-type]
        n_rows=n_rows,
        row_index_name=row_index_name,
        row_index_offset=row_index_offset,
        parallel=parallel,
        use_statistics=use_statistics,
        hive_partitioning=hive_partitioning,
        rechunk=rechunk,
        low_memory=low_memory,
        cache=False,
        storage_options=storage_options,
        retries=retries,
    )

    if columns is not None:
        if is_int_sequence(columns):
            columns = [lf.columns[i] for i in columns]
        lf = lf.select(columns)

    return lf.collect(no_optimization=True)


def read_parquet_schema(source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]:
    """
    Get the schema of a Parquet file without reading data.

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

    return _read_parquet_schema(source)


@deprecate_renamed_parameter("row_count_name", "row_index_name", version="0.20.4")
@deprecate_renamed_parameter("row_count_offset", "row_index_offset", version="0.20.4")
def scan_parquet(
    source: str | Path | list[str] | list[Path],
    *,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    parallel: ParallelStrategy = "auto",
    use_statistics: bool = True,
    hive_partitioning: bool = True,
    rechunk: bool = False,
    low_memory: bool = False,
    cache: bool = True,
    storage_options: dict[str, Any] | None = None,
    retries: int = 0,
) -> LazyFrame:
    """
    Lazily read from a local or cloud-hosted parquet file (or files).

    This function allows the query optimizer to push down predicates and projections to
    the scan level, typically increasing performance and reducing memory overhead.

    Parameters
    ----------
    source
        Path(s) to a file
        If a single path is given, it can be a globbing pattern.
    n_rows
        Stop reading from parquet file after reading `n_rows`.
    row_index_name
        If not None, this will insert a row index column with the given name into the
        DataFrame
    row_index_offset
        Offset to start the row index column (only used if the name is set)
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        This determines the direction of parallelism. 'auto' will try to determine the
        optimal direction.
    use_statistics
        Use statistics in the parquet to determine if pages
        can be skipped from reading.
    hive_partitioning
        Infer statistics and schema from hive partitioned URL and use them
        to prune reads.
    rechunk
        In case of reading multiple files via a glob pattern rechunk the final DataFrame
        into contiguous memory chunks.
    low_memory
        Reduce memory pressure at the expense of performance.
    cache
        Cache the result after reading.
    storage_options
        Options that indicate how to connect to a cloud provider.
        If the cloud provider is not supported by Polars, the storage options
        are passed to `fsspec.open()`.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    retries
        Number of retries if accessing a cloud instance fails.

    See Also
    --------
    read_parquet
    scan_pyarrow_dataset

    Examples
    --------
    Scan a local Parquet file.

    >>> pl.scan_parquet("path/to/file.parquet")  # doctest: +SKIP

    Scan a file on AWS S3.

    >>> source = "s3://bucket/*.parquet"
    >>> pl.scan_parquet(source)  # doctest: +SKIP
    >>> storage_options = {
    ...     "aws_access_key_id": "<secret>",
    ...     "aws_secret_access_key": "<secret>",
    ...     "aws_region": "us-east-1",
    ... }
    >>> pl.scan_parquet(source, storage_options=storage_options)  # doctest: +SKIP
    """
    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    else:
        source = [normalize_filepath(source) for source in source]

    return pl.LazyFrame._scan_parquet(
        source,
        n_rows=n_rows,
        cache=cache,
        parallel=parallel,
        rechunk=rechunk,
        row_index_name=row_index_name,
        row_index_offset=row_index_offset,
        storage_options=storage_options,
        low_memory=low_memory,
        use_statistics=use_statistics,
        hive_partitioning=hive_partitioning,
        retries=retries,
    )

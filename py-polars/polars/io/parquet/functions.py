from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import polars._reexport as pl
from polars.convert import from_arrow
from polars.dependencies import _PYARROW_AVAILABLE
from polars.io._utils import _prepare_file_arg
from polars.utils.various import is_int_sequence, normalize_filepath

with contextlib.suppress(ImportError):
    from polars.polars import read_parquet_schema as _read_parquet_schema

if TYPE_CHECKING:
    from polars import DataFrame, DataType, LazyFrame
    from polars.type_aliases import ParallelStrategy


def read_parquet(
    source: str | Path | list[str] | list[Path] | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    n_rows: int | None = None,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
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
    Read into a `DataFrame` from an Apache Parquet file.

    Parameters
    ----------
    source
        A path to a file or a file-like object. By file-like object, we refer to objects
        that have a `read()` method, such as a file handler (e.g. via the builtin `open`
        function) or `BytesIO <https://docs.python.org/3/library/io.html#io.BytesIO>`_.
        If the path is a directory, all files in that directory will be read.
    columns
        A list of column indices (starting at zero) or column names to read.
    n_rows
        The number of rows to read from the Parquet file.
        Only used when `use_pyarrow=False`.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        The direction of parallelism. `'auto'` will try to determine the optimal
        direction.
    use_statistics
        Whether to use statistics in the parquet to determine if pages can be skipped
        from reading.
    hive_partitioning
        Whether to infer statistics and schema from hive partitioned URL and use them
        to prune reads.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    storage_options
        Options that indicate how to connect to a cloud provider.
        If the cloud provider is not supported by Polars, the storage options
        are passed to `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    retries
        The number of times to retry if accessing a cloud instance fails.
    use_pyarrow
        Whether to use the parquet reader from :mod:`pyarrow` instead of polars's.
        The PyArrow reader is more stable.
    pyarrow_options
        Keyword arguments for `pyarrow.parquet.read_table
        <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html>`_.
    memory_map
        Whether to memory-map the underlying file. This can greatly improve performance
        on repeated queries as the operating system may cache pages.
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
        This operation defaults to a :func:`DataFrame.rechunk` operation at the end,
        meaning that each column will be stored continuously in memory. Set
        `rechunk=False` if you are benchmarking the parquet reader, as rechunking can be
        an expensive operation that should not contribute to the timings.
    """
    # Dispatch to pyarrow if requested
    if use_pyarrow:
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError(
                "'pyarrow' is required when using `read_parquet(..., use_pyarrow=True)`"
            )
        if n_rows is not None:
            raise ValueError("`n_rows` cannot be used with `use_pyarrow=True`")

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
                row_count_name=row_count_name,
                row_count_offset=row_count_offset,
                low_memory=low_memory,
                use_statistics=use_statistics,
                rechunk=rechunk,
            )

    # For other inputs, defer to `scan_parquet`
    lf = scan_parquet(
        source,  # type: ignore[arg-type]
        n_rows=n_rows,
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
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

    return _read_parquet_schema(source)


def scan_parquet(
    source: str | Path | list[str] | list[Path],
    *,
    n_rows: int | None = None,
    row_count_name: str | None = None,
    row_count_offset: int = 0,
    parallel: ParallelStrategy = "auto",
    use_statistics: bool = True,
    hive_partitioning: bool = True,
    rechunk: bool = True,
    low_memory: bool = False,
    cache: bool = True,
    storage_options: dict[str, Any] | None = None,
    retries: int = 0,
) -> LazyFrame:
    """
    Lazily read from an Apache Parquet file, or multiple files via glob patterns.

    This function allows the query optimizer to push down predicates and projections to
    the scan level, typically increasing performance and reducing memory overhead.

    Parameters
    ----------
    source
        A path to an Apache Parquet file, or a glob pattern matching multiple files.
    n_rows
        The number of rows to read from the Parquet file.
    row_count_name
        If not `None`, add a row count column with this name as the first column.
    row_count_offset
        An integer offset to start the row count at; only used when `row_count_name`
        is not `None`.
    parallel : {'auto', 'columns', 'row_groups', 'none'}
        The direction of parallelism. `'auto'` will try to determine the optimal
        direction.
    use_statistics
        Whether to use statistics in the parquet to determine if pages can be skipped
        from reading.
    hive_partitioning
        Whether to infer statistics and schema from hive partitioned URL and use them
        to prune reads.
    rechunk
        Whether to ensure each column of the result is stored contiguously in
        memory; see :func:`DataFrame.rechunk` for details.
    low_memory
        Whether to reduce memory usage at the expense of speed.
    cache
        Whether to cache the result after reading.
    storage_options
        Options that indicate how to connect to a cloud provider.
        If the cloud provider is not supported by Polars, the storage options
        are passed to `fsspec.open()
        <https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open>`_.

        The cloud providers currently supported are AWS, GCP, and Azure.
        See supported keys here:

        * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
        * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
        * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_

        If `storage_options` is not provided, Polars will try to infer the information
        from environment variables.
    retries
        The number of times to retry if accessing a cloud instance fails.

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
        row_count_name=row_count_name,
        row_count_offset=row_count_offset,
        storage_options=storage_options,
        low_memory=low_memory,
        use_statistics=use_statistics,
        hive_partitioning=hive_partitioning,
        retries=retries,
    )

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Sequence

from polars._utils.deprecation import deprecate_renamed_parameter
from polars._utils.unstable import issue_unstable_warning
from polars._utils.various import (
    is_int_sequence,
    normalize_filepath,
)
from polars._utils.wrap import wrap_df, wrap_ldf
from polars.convert import from_arrow
from polars.dependencies import import_optional
from polars.io._utils import (
    is_local_file,
    is_supported_cloud,
    parse_columns_arg,
    parse_row_index_args,
    prepare_file_arg,
)
from polars.io.parquet.anonymous_scan import _scan_parquet_fsspec

with contextlib.suppress(ImportError):
    from polars.polars import PyDataFrame, PyLazyFrame
    from polars.polars import read_parquet_schema as _read_parquet_schema

if TYPE_CHECKING:
    from polars import DataFrame, DataType, LazyFrame
    from polars.type_aliases import ParallelStrategy, SchemaDict


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
    hive_schema: SchemaDict | None = None,
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
        Path to a file or a file-like object (by "file-like object" we refer to objects
        that have a `read()` method, such as a file handler like the builtin `open`
        function, or a `BytesIO` instance). If the path is a directory, files in that
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
        Infer statistics and schema from Hive partitioned URL and use them
        to prune reads.
    hive_schema
        The column names and data types of the columns by which the data is partitioned.
        If set to `None` (default), the schema of the Hive partitions is inferred.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
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
        Use PyArrow instead of the Rust-native Parquet reader. The PyArrow reader is
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
    * When benchmarking:
        This operation defaults to a `rechunk` operation at the end, meaning that all
        data will be stored continuously in memory. Set `rechunk=False` if you are
        benchmarking the parquet-reader as `rechunk` can be an expensive operation
        that should not contribute to the timings.
    """
    if hive_schema is not None:
        msg = "The `hive_schema` parameter of `read_parquet` is considered unstable."
        issue_unstable_warning(msg)

    # Dispatch to pyarrow if requested
    if use_pyarrow:
        if n_rows is not None:
            msg = "`n_rows` cannot be used with `use_pyarrow=True`"
            raise ValueError(msg)
        if hive_schema is not None:
            msg = (
                "cannot use `hive_partitions` with `use_pyarrow=True`"
                "\n\nHint: Pass `pyarrow_options` instead with a 'partitioning' entry."
            )
            raise TypeError(msg)
        return _read_parquet_with_pyarrow(
            source,
            columns=columns,
            storage_options=storage_options,
            pyarrow_options=pyarrow_options,
            memory_map=memory_map,
        )

    # Read file and bytes inputs using `read_parquet`
    elif isinstance(source, (io.IOBase, bytes)):
        return _read_parquet_binary(
            source,
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
        hive_schema=hive_schema,
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

    return lf.collect()


def _read_parquet_with_pyarrow(
    source: str | Path | list[str] | list[Path] | IO[bytes] | bytes,
    *,
    columns: list[int] | list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    pyarrow_options: dict[str, Any] | None = None,
    memory_map: bool = True,
) -> DataFrame:
    pyarrow_parquet = import_optional(
        "pyarrow.parquet",
        err_prefix="",
        err_suffix="is required when using `read_parquet(..., use_pyarrow=True)`",
    )
    pyarrow_options = pyarrow_options or {}

    with prepare_file_arg(
        source,  # type: ignore[arg-type]
        use_pyarrow=True,
        storage_options=storage_options,
    ) as source_prep:
        pa_table = pyarrow_parquet.read_table(
            source_prep,
            memory_map=memory_map,
            columns=columns,
            **pyarrow_options,
        )
    return from_arrow(pa_table)  # type: ignore[return-value]


def _read_parquet_binary(
    source: IO[bytes] | bytes,
    *,
    columns: Sequence[int] | Sequence[str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    parallel: ParallelStrategy = "auto",
    use_statistics: bool = True,
    rechunk: bool = True,
    low_memory: bool = False,
) -> DataFrame:
    projection, columns = parse_columns_arg(columns)
    row_index = parse_row_index_args(row_index_name, row_index_offset)

    with prepare_file_arg(source) as source_prep:
        pydf = PyDataFrame.read_parquet(
            source_prep,
            columns=columns,
            projection=projection,
            n_rows=n_rows,
            row_index=row_index,
            parallel=parallel,
            use_statistics=use_statistics,
            rechunk=rechunk,
            low_memory=low_memory,
        )
    return wrap_df(pydf)


def read_parquet_schema(source: str | Path | IO[bytes] | bytes) -> dict[str, DataType]:
    """
    Get the schema of a Parquet file without reading data.

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
    hive_schema: SchemaDict | None = None,
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
    hive_schema
        The column names and data types of the columns by which the data is partitioned.
        If set to `None` (default), the schema of the Hive partitions is inferred.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
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
    if hive_schema is not None:
        msg = "The `hive_schema` parameter of `scan_parquet` is considered unstable."
        issue_unstable_warning(msg)

    if isinstance(source, (str, Path)):
        source = normalize_filepath(source)
    else:
        source = [normalize_filepath(source) for source in source]

    return _scan_parquet_impl(
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
        hive_schema=hive_schema,
        retries=retries,
    )


def _scan_parquet_impl(
    source: str | list[str] | list[Path],
    *,
    n_rows: int | None = None,
    cache: bool = True,
    parallel: ParallelStrategy = "auto",
    rechunk: bool = True,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    storage_options: dict[str, object] | None = None,
    low_memory: bool = False,
    use_statistics: bool = True,
    hive_partitioning: bool = True,
    hive_schema: SchemaDict | None = None,
    retries: int = 0,
) -> LazyFrame:
    if isinstance(source, list):
        sources = source
        source = None  # type: ignore[assignment]
        can_use_fsspec = False
    else:
        can_use_fsspec = True
        sources = []

    # try fsspec scanner
    if (
        can_use_fsspec
        and not is_local_file(source)  # type: ignore[arg-type]
        and not is_supported_cloud(source)  # type: ignore[arg-type]
    ):
        scan = _scan_parquet_fsspec(source, storage_options)  # type: ignore[arg-type]
        if n_rows:
            scan = scan.head(n_rows)
        if row_index_name is not None:
            scan = scan.with_row_index(row_index_name, row_index_offset)
        return scan

    if storage_options:
        storage_options = list(storage_options.items())  # type: ignore[assignment]
    else:
        # Handle empty dict input
        storage_options = None

    pylf = PyLazyFrame.new_from_parquet(
        source,
        sources,
        n_rows,
        cache,
        parallel,
        rechunk,
        parse_row_index_args(row_index_name, row_index_offset),
        low_memory,
        cloud_options=storage_options,
        use_statistics=use_statistics,
        hive_partitioning=hive_partitioning,
        hive_schema=hive_schema,
        retries=retries,
    )
    return wrap_ldf(pylf)

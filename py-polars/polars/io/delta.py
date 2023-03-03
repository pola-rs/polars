from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from polars.convert import from_arrow
from polars.dependencies import _DELTALAKE_AVAILABLE, deltalake
from polars.io.pyarrow_dataset import scan_pyarrow_dataset
from polars.utils import deprecate_nonkeyword_arguments

if TYPE_CHECKING:
    from polars.dependencies import pyarrow as pa
    from polars.internals import DataFrame, LazyFrame


@deprecate_nonkeyword_arguments()
def read_delta(
    table_uri: str,
    version: int | None = None,
    columns: list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    delta_table_options: dict[str, Any] | None = None,
    pyarrow_options: dict[str, Any] | None = None,
) -> DataFrame:
    """
    Reads into a DataFrame from a Delta lake table.

    Parameters
    ----------
    table_uri
        Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported. But
        for the supported object storages - GCS, Azure and S3, there is no relative
        path support, and thus full URI must be provided.
    version
        Version of the Delta lake table.

        Note: If ``version`` is not provided, latest version of delta lake
        table is read.
    columns
        Columns to select. Accepts a list of column names.
    storage_options
        Extra options for the storage backends supported by `deltalake`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here
        <https://delta-io.github.io/delta-rs/python/usage.html?highlight=backend#loading-a-delta-table>`__.
    delta_table_options
        Additional keyword arguments while reading a Delta lake Table.
    pyarrow_options
        Keyword arguments while converting a Delta lake Table to pyarrow table.

    Returns
    -------
    DataFrame

    Examples
    --------
    Reads a Delta table from local filesystem.
    Note: Since version is not provided, latest version of the delta table is read.

    >>> table_path = "/path/to/delta-table/"
    >>> pl.read_delta(table_path)  # doctest: +SKIP

    Use the `pyarrow_options` parameter to read only certain partitions.
    Note: This should be preferred over using an equivalent `.filter()` on the resulting
    dataframe, as this avoids reading the data at all.

    >>> pl.read_delta(  # doctest: +SKIP
    ...     table_path,
    ...     pyarrow_options={"partitions": [("year", "=", "2021")]},
    ... )

    Reads a specific version of the Delta table from local filesystem.
    Note: This will fail if the provided version of the delta table does not exist.

    >>> pl.read_delta(table_path, version=1)  # doctest: +SKIP

    Reads a Delta table from AWS S3.
    See a list of supported storage options for S3 `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L423-L491>`__.

    >>> table_path = "s3://bucket/path/to/delta-table/"
    >>> storage_options = {
    ...     "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
    ...     "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.read_delta(table_path, storage_options=storage_options)  # doctest: +SKIP

    Reads a Delta table from Google Cloud storage (GCS).
    See a list of supported storage options for GCS `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L570-L577>`__.

    >>> table_path = "gs://bucket/path/to/delta-table/"
    >>> storage_options = {"SERVICE_ACCOUNT": "SERVICE_ACCOUNT_JSON_ABSOLUTE_PATH"}
    >>> pl.read_delta(table_path, storage_options=storage_options)  # doctest: +SKIP

    Reads a Delta table from Azure.

    Following type of table paths are supported,

    * az://<container>/<path>
    * adl://<container>/<path>
    * abfs://<container>/<path>

    See a list of supported storage options for Azure `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L524-L539>`__.

    >>> table_path = "az://container/path/to/delta-table/"
    >>> storage_options = {
    ...     "AZURE_STORAGE_ACCOUNT_NAME": "AZURE_STORAGE_ACCOUNT_NAME",
    ...     "AZURE_STORAGE_ACCOUNT_KEY": "AZURE_STORAGE_ACCOUNT_KEY",
    ... }
    >>> pl.read_delta(table_path, storage_options=storage_options)  # doctest: +SKIP

    Reads a Delta table with additional delta specific options. In the below example,
    `without_files` option is used which loads the table without file tracking
    information.

    >>> table_path = "/path/to/delta-table/"
    >>> delta_table_options = {"without_files": True}
    >>> pl.read_delta(
    ...     table_path, delta_table_options=delta_table_options
    ... )  # doctest: +SKIP

    """
    if delta_table_options is None:
        delta_table_options = {}

    if pyarrow_options is None:
        pyarrow_options = {}

    _, resolved_uri, _ = _resolve_delta_lake_uri(table_uri)

    dl_tbl = _get_delta_lake_table(
        table_path=resolved_uri,
        version=version,
        storage_options=storage_options,
        delta_table_options=delta_table_options,
    )

    return from_arrow(dl_tbl.to_pyarrow_table(columns=columns, **pyarrow_options))  # type: ignore[return-value]


@deprecate_nonkeyword_arguments()
def scan_delta(
    table_uri: str,
    version: int | None = None,
    raw_filesystem: pa.fs.FileSystem | None = None,
    storage_options: dict[str, Any] | None = None,
    delta_table_options: dict[str, Any] | None = None,
    pyarrow_options: dict[str, Any] | None = None,
) -> LazyFrame:
    """
    Lazily read from a Delta lake table.

    Parameters
    ----------
    table_uri
        Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported. But
        for the supported object storages - GCS, Azure and S3, there is no relative
        path support, and thus full URI must be provided.
    version
        Version of the Delta lake table.

        Note: If ``version`` is not provided, latest version of delta lake
        table is read.
    raw_filesystem
        A `pyarrow.fs.FileSystem` to read files from.

        Note: The root of the filesystem has to be adjusted to point at the root of
        the Delta lake table. The provided ``raw_filesystem`` is wrapped into a
        `pyarrow.fs.SubTreeFileSystem`

        More info is available `here
        <https://delta-io.github.io/delta-rs/python/usage.html?highlight=backend#custom-storage-backends>`__.
    storage_options
        Extra options for the storage backends supported by `deltalake`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here
        <https://delta-io.github.io/delta-rs/python/usage.html?highlight=backend#loading-a-delta-table>`__.
    delta_table_options
        Additional keyword arguments while reading a Delta lake Table.
    pyarrow_options
        Keyword arguments while converting a Delta lake Table to pyarrow table.
        Use this parameter when filtering on partitioned columns.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for a Delta table from local filesystem.
    Note: Since version is not provided, latest version of the delta table is read.

    >>> table_path = "/path/to/delta-table/"
    >>> pl.scan_delta(table_path).collect()  # doctest: +SKIP

    Use the `pyarrow_options` parameter to read only certain partitions.

    >>> pl.scan_delta(  # doctest: +SKIP
    ...     table_path,
    ...     pyarrow_options={"partitions": [("year", "=", "2021")]},
    ... )

    Creates a scan for a specific version of the Delta table from local filesystem.
    Note: This will fail if the provided version of the delta table does not exist.

    >>> pl.scan_delta(table_path, version=1).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from AWS S3.
    See a list of supported storage options for S3 `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L423-L491>`__.

    >>> table_path = "s3://bucket/path/to/delta-table/"
    >>> storage_options = {
    ...     "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
    ...     "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from Google Cloud storage (GCS).

    Note: This implementation relies on `pyarrow.fs` and thus has to rely on fsspec
    compatible filesystems as mentioned `here
    <https://arrow.apache.org/docs/python/filesystems.html#using-fsspec-compatible-filesystems-with-arrow>`__.
    So please ensure that `pyarrow` ,`fsspec` and `gcsfs` are installed.

    See a list of supported storage options for GCS `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L570-L577>`__.

    >>> import gcsfs  # doctest: +SKIP
    >>> from pyarrow.fs import PyFileSystem, FSSpecHandler  # doctest: +SKIP
    >>> storage_options = {"SERVICE_ACCOUNT": "SERVICE_ACCOUNT_JSON_ABSOLUTE_PATH"}
    >>> fs = gcsfs.GCSFileSystem(
    ...     project="my-project-id",
    ...     token=storage_options["SERVICE_ACCOUNT"],
    ... )  # doctest: +SKIP
    >>> # this pyarrow fs must be created and passed to scan_delta for GCS
    >>> pa_fs = PyFileSystem(FSSpecHandler(fs))  # doctest: +SKIP
    >>> table_path = "gs://bucket/path/to/delta-table/"
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options, raw_filesystem=pa_fs
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from Azure.

    Note: This implementation relies on `pyarrow.fs` and thus has to rely on fsspec
    compatible filesystems as mentioned `here
    <https://arrow.apache.org/docs/python/filesystems.html#using-fsspec-compatible-filesystems-with-arrow>`__.
    So please ensure that `pyarrow` ,`fsspec` and `adlfs` are installed.

    Following type of table paths are supported,

    * az://<container>/<path>
    * adl://<container>/<path>
    * abfs://<container>/<path>

    See a list of supported storage options for Azure `here
    <https://github.com/delta-io/delta-rs/blob/17999d24a58fb4c98c6280b9e57842c346b4603a/rust/src/builder.rs#L524-L539>`__.

    >>> import adlfs  # doctest: +SKIP
    >>> from pyarrow.fs import PyFileSystem, FSSpecHandler  # doctest: +SKIP
    >>> storage_options = {
    ...     "AZURE_STORAGE_ACCOUNT_NAME": "AZURE_STORAGE_ACCOUNT_NAME",
    ...     "AZURE_STORAGE_ACCOUNT_KEY": "AZURE_STORAGE_ACCOUNT_KEY",
    ... }
    >>> fs = adlfs.AzureBlobFileSystem(
    ...     account_name=storage_options["AZURE_STORAGE_ACCOUNT_NAME"],
    ...     account_key=storage_options["AZURE_STORAGE_ACCOUNT_KEY"],
    ... )  # doctest: +SKIP
    >>> # this pyarrow fs must be created and passed to scan_delta for Azure
    >>> pa_fs = PyFileSystem(FSSpecHandler(fs))  # doctest: +SKIP
    >>> table_path = "az://container/path/to/delta-table/"
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options, raw_filesystem=pa_fs
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table with additional delta specific options.
    In the below example, `without_files` option is used which loads the table without
    file tracking information.

    >>> table_path = "/path/to/delta-table/"
    >>> delta_table_options = {"without_files": True}
    >>> pl.scan_delta(
    ...     table_path, delta_table_options=delta_table_options
    ... ).collect()  # doctest: +SKIP

    """
    if delta_table_options is None:
        delta_table_options = {}

    if pyarrow_options is None:
        pyarrow_options = {}

    import pyarrow.fs as pa_fs

    # Resolve relative paths if not an object storage
    scheme, resolved_uri, normalized_path = _resolve_delta_lake_uri(table_uri)

    # Storage Backend
    if raw_filesystem is None:
        raw_filesystem, normalized_path = pa_fs.FileSystem.from_uri(resolved_uri)

    # SubTreeFileSystem requires normalized path
    subtree_fs_path = resolved_uri if scheme == "" else normalized_path
    filesystem = pa_fs.SubTreeFileSystem(subtree_fs_path, raw_filesystem)

    # deltalake can work with resolved paths
    dl_tbl = _get_delta_lake_table(
        table_path=resolved_uri,
        version=version,
        storage_options=storage_options,
        delta_table_options=delta_table_options,
    )

    # Must provide filesystem as DeltaStorageHandler is not serializable.
    pa_ds = dl_tbl.to_pyarrow_dataset(filesystem=filesystem, **pyarrow_options)
    return scan_pyarrow_dataset(pa_ds)


def _resolve_delta_lake_uri(table_uri: str) -> tuple[str, str, str]:
    from urllib.parse import ParseResult, urlparse

    parsed_result = urlparse(table_uri)
    scheme = parsed_result.scheme

    resolved_uri = str(
        Path(table_uri).expanduser().resolve(True) if scheme == "" else table_uri
    )

    normalized_path = str(ParseResult("", *parsed_result[1:]).geturl())
    return (scheme, resolved_uri, normalized_path)


def _get_delta_lake_table(
    table_path: str,
    version: int | None = None,
    storage_options: dict[str, Any] | None = None,
    delta_table_options: dict[str, Any] | None = None,
) -> deltalake.DeltaTable:
    """
    Initialise a Delta lake table for use in read and scan operations.

    Notes
    -----
    Make sure to install deltalake>=0.6.0. Read the documentation
    `here <https://delta-io.github.io/delta-rs/python/installation.html>`_.

    Returns
    -------
    DeltaTable

    """
    if not _DELTALAKE_AVAILABLE:
        raise ImportError(
            "deltalake is not installed. Please run `pip install deltalake>=0.6.0`."
        )

    if delta_table_options is None:
        delta_table_options = {}

    dl_tbl = deltalake.DeltaTable(
        table_uri=table_path,
        version=version,
        storage_options=storage_options,
        **delta_table_options,
    )

    return dl_tbl

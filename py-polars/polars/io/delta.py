from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from polars.convert import from_arrow
from polars.datatypes import Categorical, Null, Time
from polars.datatypes.convert import unpack_dtypes
from polars.dependencies import _DELTALAKE_AVAILABLE, deltalake
from polars.dependencies import pyarrow as pa
from polars.io.pyarrow_dataset import scan_pyarrow_dataset

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame
    from polars.type_aliases import PolarsDataType


def read_delta(
    source: str,
    *,
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
    source
        Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported but
        for the supported object storages - GCS, Azure and S3 full URI must be provided.
    version
        Version of the Delta lake table.

        Note: If ``version`` is not provided, the latest version of delta lake
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
    Note: Since version is not provided, the latest version of the delta table is read.

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
    <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants>`__.

    >>> table_path = "s3://bucket/path/to/delta-table/"
    >>> storage_options = {
    ...     "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
    ...     "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.read_delta(table_path, storage_options=storage_options)  # doctest: +SKIP

    Reads a Delta table from Google Cloud storage (GCS).
    See a list of supported storage options for GCS `here
    <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants>`__.

    >>> table_path = "gs://bucket/path/to/delta-table/"
    >>> storage_options = {"SERVICE_ACCOUNT": "SERVICE_ACCOUNT_JSON_ABSOLUTE_PATH"}
    >>> pl.read_delta(table_path, storage_options=storage_options)  # doctest: +SKIP

    Reads a Delta table from Azure.

    Following type of table paths are supported,

    * az://<container>/<path>
    * adl://<container>/<path>
    * abfs://<container>/<path>

    See a list of supported storage options for Azure `here
    <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.

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
    if pyarrow_options is None:
        pyarrow_options = {}

    resolved_uri = _resolve_delta_lake_uri(source)

    dl_tbl = _get_delta_lake_table(
        table_path=resolved_uri,
        version=version,
        storage_options=storage_options,
        delta_table_options=delta_table_options,
    )

    return from_arrow(dl_tbl.to_pyarrow_table(columns=columns, **pyarrow_options))  # type: ignore[return-value]


def scan_delta(
    source: str,
    *,
    version: int | None = None,
    storage_options: dict[str, Any] | None = None,
    delta_table_options: dict[str, Any] | None = None,
    pyarrow_options: dict[str, Any] | None = None,
) -> LazyFrame:
    """
    Lazily read from a Delta lake table.

    Parameters
    ----------
    source
        Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported but
        for the supported object storages - GCS, Azure and S3 full URI must be provided.
    version
        Version of the Delta lake table.

        Note: If ``version`` is not provided, the latest version of delta lake
        table is read.
    storage_options
        Extra options for the storage backends supported by `deltalake`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here
        <https://delta-io.github.io/delta-rs/python/usage.html?highlight=backend#loading-a-delta-table>`__.
    delta_table_options
        Additional keyword arguments while reading a Delta lake Table.
    pyarrow_options
        Keyword arguments while converting a Delta lake Table to pyarrow table.
        Use this parameter when filtering on partitioned columns or to read
        from a 'fsspec' supported filesystem.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for a Delta table from local filesystem.
    Note: Since version is not provided, the latest version of the delta table is read.

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
    <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants>`__.

    >>> table_path = "s3://bucket/path/to/delta-table/"
    >>> storage_options = {
    ...     "AWS_REGION": "eu-central-1",
    ...     "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
    ...     "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from Google Cloud storage (GCS).
    See a list of supported storage options for GCS `here
    <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants>`__.

    >>> table_path = "gs://bucket/path/to/delta-table/"
    >>> storage_options = {"SERVICE_ACCOUNT": "SERVICE_ACCOUNT_JSON_ABSOLUTE_PATH"}
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for a Delta table from Azure.
    Supported options for Azure are available `here
    <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.

    Following type of table paths are supported,

    * az://<container>/<path>
    * adl://<container>/<path>
    * abfs[s]://<container>/<path>

    >>> table_path = "az://container/path/to/delta-table/"
    >>> storage_options = {
    ...     "AZURE_STORAGE_ACCOUNT_NAME": "AZURE_STORAGE_ACCOUNT_NAME",
    ...     "AZURE_STORAGE_ACCOUNT_KEY": "AZURE_STORAGE_ACCOUNT_KEY",
    ... }
    >>> pl.scan_delta(
    ...     table_path, storage_options=storage_options
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
    if pyarrow_options is None:
        pyarrow_options = {}

    resolved_uri = _resolve_delta_lake_uri(source)
    dl_tbl = _get_delta_lake_table(
        table_path=resolved_uri,
        version=version,
        storage_options=storage_options,
        delta_table_options=delta_table_options,
    )

    pa_ds = dl_tbl.to_pyarrow_dataset(**pyarrow_options)
    return scan_pyarrow_dataset(pa_ds)


def _resolve_delta_lake_uri(table_uri: str, strict: bool = True) -> str:
    parsed_result = urlparse(table_uri)

    resolved_uri = str(
        Path(table_uri).expanduser().resolve(strict)
        if parsed_result.scheme == ""
        else table_uri
    )

    return resolved_uri


def _get_delta_lake_table(
    table_path: str,
    version: int | None = None,
    storage_options: dict[str, Any] | None = None,
    delta_table_options: dict[str, Any] | None = None,
) -> deltalake.DeltaTable:
    """
    Initialize a Delta lake table for use in read and scan operations.

    Notes
    -----
    Make sure to install deltalake>=0.8.0. Read the documentation
    `here <https://delta-io.github.io/delta-rs/python/installation.html>`_.

    """
    _check_if_delta_available()

    if delta_table_options is None:
        delta_table_options = {}

    dl_tbl = deltalake.DeltaTable(
        table_path,
        version=version,
        storage_options=storage_options,
        **delta_table_options,
    )

    return dl_tbl


def _check_if_delta_available() -> None:
    if not _DELTALAKE_AVAILABLE:
        raise ImportError(
            "deltalake is not installed. Please run `pip install deltalake>=0.9.0`."
        )


def _check_for_unsupported_types(dtypes: list[PolarsDataType]) -> None:
    schema_dtypes = unpack_dtypes(*dtypes)
    unsupported_types = {Time, Categorical, Null}
    overlap = schema_dtypes & unsupported_types

    if overlap:
        raise TypeError(f"dataframe contains unsupported data types: {overlap}")


def _convert_pa_schema_to_delta(schema: pa.schema) -> pa.schema:
    """Convert a PyArrow schema to a schema compatible with Delta Lake."""
    # TODO: Add time zone support
    dtype_map = {
        pa.uint8(): pa.int8(),
        pa.uint16(): pa.int16(),
        pa.uint32(): pa.int32(),
        pa.uint64(): pa.int64(),
        pa.timestamp("ns"): pa.timestamp("us"),
        pa.timestamp("ms"): pa.timestamp("us"),
        pa.large_string(): pa.string(),
        pa.large_binary(): pa.binary(),
    }

    def dtype_to_delta_dtype(dtype: pa.DataType) -> pa.DataType:
        # Handle nested types
        if isinstance(dtype, pa.LargeListType):
            return list_to_delta_dtype(dtype)
        elif isinstance(dtype, pa.StructType):
            return struct_to_delta_dtype(dtype)

        try:
            return dtype_map[dtype]
        except KeyError:
            return dtype

    def list_to_delta_dtype(dtype: pa.LargeListType) -> pa.ListType:
        nested_dtype = dtype.value_type
        nested_dtype_cast = dtype_to_delta_dtype(nested_dtype)
        return pa.list_(nested_dtype_cast)

    def struct_to_delta_dtype(dtype: pa.StructType) -> pa.StructType:
        fields = [dtype.field(i) for i in range(dtype.num_fields)]
        fields_cast = [pa.field(f.name, dtype_to_delta_dtype(f.type)) for f in fields]
        return pa.struct(fields_cast)

    return pa.schema([pa.field(f.name, dtype_to_delta_dtype(f.type)) for f in schema])

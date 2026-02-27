from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from polars._utils.wrap import wrap_ldf
from polars.io.cloud._utils import NoPickleOption
from polars.io.delta._dataset import DeltaDataset

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from typing import Literal

    from deltalake import DeltaTable

    from polars import DataFrame, LazyFrame
    from polars._typing import StorageOptionsDict
    from polars.io.cloud import CredentialProviderFunction


def read_delta(
    source: str | Path | DeltaTable,
    *,
    version: int | str | datetime | None = None,
    columns: list[str] | None = None,
    rechunk: bool | None = None,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    delta_table_options: dict[str, Any] | None = None,
    use_pyarrow: bool = False,
    pyarrow_options: dict[str, Any] | None = None,
) -> DataFrame:
    """
    Reads into a DataFrame from a Delta lake table.

    Parameters
    ----------
    source
        DeltaTable or a Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported but
        for the supported object storages - GCS, Azure and S3 full URI must be provided.
    version
        Numerical version or timestamp version of the Delta lake table.

        Note: If `version` is not provided, the latest version of delta lake
        table is read.
    columns
        Columns to select. Accepts a list of column names.
    rechunk
        Make sure that all columns are contiguous in memory by
        aggregating the chunks into a single array.
    storage_options
        Extra options for the storage backends supported by `deltalake`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here
        <https://delta-io.github.io/delta-rs/usage/loading-table/>`__.
    credential_provider
        Provide a function that can be called to provide cloud storage
        credentials. The function is expected to return a dictionary of
        credential keys along with an optional credential expiry time.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    delta_table_options
        Additional keyword arguments while reading a Delta lake Table.
    use_pyarrow
        Flag to enable pyarrow dataset reads.
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

    Reads a specific version of the Delta table from local filesystem.
    Note: This will fail if the provided version of the delta table does not exist.

    >>> pl.read_delta(table_path, version=1)  # doctest: +SKIP

    Time travel a delta table from local filesystem using a timestamp version.

    >>> pl.read_delta(
    ...     table_path, version=datetime(2020, 1, 1, tzinfo=timezone.utc)
    ... )  # doctest: +SKIP

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
    df = scan_delta(
        source=source,
        version=version,
        storage_options=storage_options,
        credential_provider=credential_provider,
        delta_table_options=delta_table_options,
        use_pyarrow=use_pyarrow,
        pyarrow_options=pyarrow_options,
        rechunk=rechunk,
    )

    if columns is not None:
        df = df.select(columns)
    return df.collect()


def scan_delta(
    source: str | Path | DeltaTable,
    *,
    version: int | str | datetime | None = None,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    delta_table_options: dict[str, Any] | None = None,
    use_pyarrow: bool = False,
    pyarrow_options: dict[str, Any] | None = None,
    rechunk: bool | None = None,
) -> LazyFrame:
    """
    Lazily read from a Delta lake table.

    Parameters
    ----------
    source
        DeltaTable or a Path or URI to the root of the Delta lake table.

        Note: For Local filesystem, absolute and relative paths are supported but
        for the supported object storages - GCS, Azure and S3 full URI must be provided.
    version
        Numerical version or timestamp version of the Delta lake table.

        Note: If `version` is not provided, the latest version of delta lake
        table is read.
    storage_options
        Extra options for the storage backends supported by `deltalake`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here
        <https://delta-io.github.io/delta-rs/usage/loading-table/>`__.
    credential_provider
        Provide a function that can be called to provide cloud storage
        credentials. The function is expected to return a dictionary of
        credential keys along with an optional credential expiry time.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    delta_table_options
        Additional keyword arguments while reading a Delta lake Table.
    use_pyarrow
        Flag to enable pyarrow dataset reads.
    pyarrow_options
        Keyword arguments while converting a Delta lake Table to pyarrow table.
        Use this parameter when filtering on partitioned columns or to read
        from a 'fsspec' supported filesystem.
    rechunk
        Make sure that all columns are contiguous in memory by
        aggregating the chunks into a single array.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for a Delta table from local filesystem.
    Note: Since version is not provided, the latest version of the delta table is read.

    >>> table_path = "/path/to/delta-table/"
    >>> pl.scan_delta(table_path).collect()  # doctest: +SKIP

    Creates a scan for a specific version of the Delta table from local filesystem.
    Note: This will fail if the provided version of the delta table does not exist.

    >>> pl.scan_delta(table_path, version=1).collect()  # doctest: +SKIP

    Time travel a delta table from local filesystem using a timestamp version.

    >>> pl.scan_delta(
    ...     table_path, version=datetime(2020, 1, 1, tzinfo=timezone.utc)
    ... ).collect()  # doctest: +SKIP

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
    from polars._plr import PyLazyFrame
    from polars.io.cloud.credential_provider._builder import (
        _init_credential_provider_builder,
    )

    table: DeltaTable | None = None

    if importlib.util.find_spec("deltalake") is not None:
        from deltalake import DeltaTable

        if isinstance(source, DeltaTable):
            table = source

    if table is None:
        credential_provider_builder = _init_credential_provider_builder(
            credential_provider, source, storage_options, "scan_delta"
        )
    elif credential_provider is not None and credential_provider != "auto":
        msg = "cannot use credential_provider when passing a DeltaTable object"
        raise ValueError(msg)
    else:
        credential_provider_builder = None

    del credential_provider

    if table is not None and (
        table._storage_options is not None or storage_options is not None
    ):
        storage_options = {
            **(table._storage_options or {}),
            **(storage_options or {}),
        }

    dataset = DeltaDataset(
        table_=NoPickleOption(table),
        table_uri_=str(source) if table is None else None,
        version=version,
        storage_options=storage_options,
        credential_provider_builder=credential_provider_builder,
        delta_table_options=delta_table_options,
        use_pyarrow=use_pyarrow,
        pyarrow_options=pyarrow_options,
        rechunk=rechunk or False,
    )

    return wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset))

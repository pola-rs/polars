from __future__ import annotations

import contextlib
import importlib
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from polars._utils.unstable import issue_unstable_warning
from polars._utils.wrap import wrap_ldf
from polars.exceptions import DuplicateError
from polars.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    from polars._typing import SchemaDict
    from polars.datatypes.classes import DataType
    from polars.io.cloud import (
        CredentialProviderFunction,
        CredentialProviderFunctionReturn,
    )
    from polars.lazyframe import LazyFrame


class Catalog:
    """
    Unity catalog client.

    .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    """

    def __init__(
        self,
        workspace_url: str,
        *,
        bearer_token: str | None = "auto",
    ) -> None:
        """
        Initialize a catalog client.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        workspace_url
            URL of the workspace, or alternatively the URL of the Unity catalog
            API endpoint.
        bearer_token
            Bearer token to authenticate with. This can also be set to:
            * "auto": Automatically retrieve bearer tokens from the environment.
            * "databricks-sdk": Use the Databricks SDK to retrieve and use the
            bearer token from the environment.
        """
        issue_unstable_warning("`Catalog` functionality is considered unstable.")

        if bearer_token == "databricks-sdk" or (
            bearer_token == "auto"
            # For security, in "auto" mode, only retrieve/use the token if:
            # * We are running inside a Databricks environment
            # * The `workspace_url` is pointing to Databricks and uses HTTPS
            and "DATABRICKS_RUNTIME_VERSION" in os.environ
            and workspace_url.startswith("https://")
            and (
                workspace_url.removeprefix("https://")
                .split("/", 1)[0]
                .endswith(".cloud.databricks.com")
            )
        ):
            bearer_token = self._get_databricks_token()

        if bearer_token == "auto":
            bearer_token = None

        self._client = PyCatalogClient.new(workspace_url, bearer_token)

    def list_catalogs(self) -> list[CatalogInfo]:
        """
        List the available catalogs.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
        """
        return self._client.list_catalogs()

    def list_namespaces(self, catalog_name: str) -> list[NamespaceInfo]:
        """
        List the available namespaces (unity schema) under the specified catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        """
        return self._client.list_namespaces(catalog_name)

    def list_tables(self, catalog_name: str, namespace: str) -> list[TableInfo]:
        """
        List the available tables under the specified schema.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        """
        return self._client.list_tables(catalog_name, namespace)

    def get_table_info(
        self, catalog_name: str, namespace: str, table_name: str
    ) -> TableInfo:
        """
        Retrieve the metadata of the specified table.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        table_name
            Name of the table.
        """
        return self._client.get_table_info(catalog_name, namespace, table_name)

    def _get_table_credentials(
        self, table_id: str, *, write: bool
    ) -> tuple[dict[str, str] | None, dict[str, str], int]:
        return self._client.get_table_credentials(table_id=table_id, write=write)

    def scan_table(
        self,
        catalog_name: str,
        namespace: str,
        table_name: str,
        *,
        delta_table_version: int | str | datetime | None = None,
        delta_table_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
        credential_provider: (
            CredentialProviderFunction | Literal["auto"] | None
        ) = "auto",
        retries: int = 2,
    ) -> LazyFrame:
        """
        Retrieve the metadata of the specified table.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        table_name
            Name of the table.
        delta_table_version
            Version of the table to scan (Deltalake only).
        delta_table_options
            Additional keyword arguments while reading a Deltalake table.
        storage_options
            Options that indicate how to connect to a cloud provider.

            The cloud providers currently supported are AWS, GCP, and Azure.
            See supported keys here:

            * `aws <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html>`_
            * `gcp <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html>`_
            * `azure <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html>`_
            * Hugging Face (`hf://`): Accepts an API key under the `token` parameter: \
            `{'token': '...'}`, or by setting the `HF_TOKEN` environment variable.

            If `storage_options` is not provided, Polars will try to infer the
            information from environment variables.
        credential_provider
            Provide a function that can be called to provide cloud storage
            credentials. The function is expected to return a dictionary of
            credential keys along with an optional credential expiry time.

            .. warning::
                This functionality is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.
        retries
            Number of retries if accessing a cloud instance fails.

        """
        table_info = self.get_table_info(catalog_name, namespace, table_name)
        storage_location, data_source_format = _extract_location_and_data_format(
            table_info, "scan table"
        )

        credential_provider, storage_options = self._init_credentials(
            credential_provider,
            storage_options,
            table_info,
            write=False,
            caller_name="Catalog.scan_table",
        )

        if data_source_format in ["DELTA", "DELTASHARING"]:
            from polars.io.delta import scan_delta

            return scan_delta(
                storage_location,
                version=delta_table_version,
                delta_table_options=delta_table_options,
                storage_options=storage_options,
                credential_provider=credential_provider,
            )

        if delta_table_version is not None:
            msg = (
                "cannot apply delta_table_version for table of type "
                f"{data_source_format}"
            )
            raise ValueError(msg)

        if delta_table_options is not None:
            msg = (
                "cannot apply delta_table_options for table of type "
                f"{data_source_format}"
            )
            raise ValueError(msg)

        if storage_options:
            storage_options = list(storage_options.items())  # type: ignore[assignment]
        else:
            # Handle empty dict input
            storage_options = None

        return wrap_ldf(
            self._client.scan_table(
                catalog_name,
                namespace,
                table_name,
                credential_provider=credential_provider,
                cloud_options=storage_options,
                retries=retries,
            )
        )

    def create_catalog(
        self,
        catalog_name: str,
        *,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> CatalogInfo:
        """
        Create a catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        comment
            Leaves a comment about the catalog.
        storage_root
            Base location at which to store the catalog.
        """
        return self._client.create_catalog(
            catalog_name=catalog_name, comment=comment, storage_root=storage_root
        )

    def delete_catalog(
        self,
        catalog_name: str,
        *,
        force: bool = False,
    ) -> None:
        """
        Delete a catalog.

        Note that depending on the table type and catalog server, this may not
        delete the actual data files from storage. For more details, please
        consult the documentation of the catalog provider you are using.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        force
            Forcibly delete the catalog even if it is not empty.
        """
        self._client.delete_catalog(catalog_name=catalog_name, force=force)

    def create_namespace(
        self,
        catalog_name: str,
        namespace: str,
        *,
        comment: str | None = None,
        storage_root: str | None = None,
    ) -> NamespaceInfo:
        """
        Create a namespace (unity schema) in the catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        comment
            Leaves a comment about the table.
        storage_root
            Base location at which to store the namespace.
        """
        return self._client.create_namespace(
            catalog_name=catalog_name,
            namespace=namespace,
            comment=comment,
            storage_root=storage_root,
        )

    def delete_namespace(
        self,
        catalog_name: str,
        namespace: str,
        *,
        force: bool = False,
    ) -> None:
        """
        Delete a namespace (unity schema) in the catalog.

        Note that depending on the table type and catalog server, this may not
        delete the actual data files from storage. For more details, please
        consult the documentation of the catalog provider you are using.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        force
            Forcibly delete the namespace even if it is not empty.
        """
        self._client.delete_namespace(
            catalog_name=catalog_name, namespace=namespace, force=force
        )

    def create_table(
        self,
        catalog_name: str,
        namespace: str,
        table_name: str,
        *,
        schema: SchemaDict | None,
        table_type: TableType,
        data_source_format: DataSourceFormat | None = None,
        comment: str | None = None,
        storage_root: str | None = None,
        properties: dict[str, str] | None = None,
    ) -> TableInfo:
        """
        Create a table in the catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        table_name
            Name of the table.
        schema
            Schema of the table.
        table_type
            Type of the table
        data_source_format
            Storage format of the table.
        comment
            Leaves a comment about the table.
        storage_root
            Base location at which to store the table.
        properties
            Extra key-value metadata to store.
        """
        return self._client.create_table(
            catalog_name=catalog_name,
            namespace=namespace,
            table_name=table_name,
            schema=schema,
            table_type=table_type,
            data_source_format=data_source_format,
            comment=comment,
            storage_root=storage_root,
            properties=list((properties or {}).items()),
        )

    def delete_table(
        self,
        catalog_name: str,
        namespace: str,
        table_name: str,
    ) -> None:
        """
        Delete the table stored at this location.

        Note that depending on the table type and catalog server, this may not
        delete the actual data files from storage. For more details, please
        consult the documentation of the catalog provider you are using.

        If you would like to perform manual deletions, the storage location of
        the files can be found using `get_table_info`.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        namespace
            Name of the namespace (unity schema).
        table_name
            Name of the table.
        """
        self._client.delete_table(
            catalog_name=catalog_name,
            namespace=namespace,
            table_name=table_name,
        )

    def _init_credentials(
        self,
        credential_provider: (CredentialProviderFunction | Literal["auto"] | None),
        storage_options: dict[str, Any] | None,
        table_info: TableInfo,
        *,
        write: bool,
        caller_name: str,
    ) -> tuple[CredentialProviderFunction | None, dict[str, Any] | None]:
        if credential_provider != "auto":
            return credential_provider, storage_options

        verbose = os.getenv("POLARS_VERBOSE") == "1"

        catalog_credential_provider = CatalogCredentialProvider(
            self, table_info.table_id, write=write
        )

        try:
            v = catalog_credential_provider._credentials_iter()
            storage_update_options = next(v)

            if storage_update_options:
                storage_options = {**(storage_options or {}), **storage_update_options}

            for _ in v:
                pass

        except Exception as e:
            if verbose:
                table_name = table_info.name
                table_id = table_info.table_id
                msg = (
                    f"error auto-initializing CatalogCredentialProvider: {e!r} "
                    f"{table_name = } ({table_id = })"
                )
                print(msg, file=sys.stderr)
        else:
            if verbose:
                table_name = table_info.name
                table_id = table_info.table_id
                msg = (
                    "auto-selected CatalogCredentialProvider for "
                    f"{table_name = } ({table_id = })"
                )
                print(msg, file=sys.stderr)

            return catalog_credential_provider, storage_options

        # This should generally not happen, but if using the temporary
        # credentials API fails for whatever reason, we fallback to our built-in
        # credential provider resolution.

        from polars.io.cloud.credential_provider import _maybe_init_credential_provider

        return _maybe_init_credential_provider(
            "auto", table_info.storage_location, storage_options, caller_name
        ), storage_options

    @classmethod
    def _get_databricks_token(cls) -> str:
        if importlib.util.find_spec("databricks.sdk") is None:
            msg = "could not get Databricks token: databricks-sdk is not installed"
            raise ImportError(msg)

        # We code like this to bypass linting
        m = importlib.import_module("databricks.sdk.core").__dict__

        return m["DefaultCredentials"]()(m["Config"]())()["Authorization"][7:]


class CatalogCredentialProvider:
    """Retrieves credentials from the Unity catalog temporary credentials API."""

    def __init__(self, catalog: Catalog, table_id: str, *, write: bool) -> None:
        self.catalog = catalog
        self.table_id = table_id
        self.write = write

    def __call__(self) -> CredentialProviderFunctionReturn:  # noqa: D102
        _, (creds, expiry) = self._credentials_iter()
        return creds, expiry

    def _credentials_iter(
        self,
    ) -> Generator[Any]:
        creds, storage_update_options, expiry = self.catalog._get_table_credentials(
            self.table_id, write=self.write
        )

        yield storage_update_options

        if not creds:
            table_id = self.table_id
            msg = (
                "did not receive credentials from temporary credentials API for "
                f"{table_id = }"
            )
            raise Exception(msg)  # noqa: TRY002

        yield creds, expiry


def _extract_location_and_data_format(
    table_info: TableInfo, operation: str
) -> tuple[str, DataSourceFormat]:
    if table_info.storage_location is None:
        msg = f"cannot {operation}: no storage_location found"
        raise ValueError(msg)

    if table_info.data_source_format is None:
        msg = f"cannot {operation}: no data_source_format found"
        raise ValueError(msg)

    return table_info.storage_location, table_info.data_source_format


@dataclass
class CatalogInfo:
    """Information for a catalog within a metastore."""

    name: str
    comment: str | None
    properties: dict[str, str]
    options: dict[str, str]
    storage_location: str | None
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None


@dataclass
class NamespaceInfo:
    """
    Information for a namespace within a catalog.

    This is also known by the name "schema" in unity catalog terminology.
    """

    name: str
    comment: str | None
    properties: dict[str, str]
    storage_location: str | None
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None


@dataclass
class TableInfo:
    """Information for a catalog table."""

    name: str
    comment: str | None
    table_id: str
    table_type: TableType
    storage_location: str | None
    data_source_format: DataSourceFormat | None
    columns: list[ColumnInfo] | None
    properties: dict[str, str]
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None

    def get_polars_schema(self) -> Schema | None:
        """
        Get the native polars schema of this table.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
        """
        issue_unstable_warning(
            "`get_polars_schema` functionality is considered unstable."
        )
        if self.columns is None:
            return None

        schema = Schema()

        for column_info in self.columns:
            if column_info.name in schema:
                msg = f"duplicate column name: {column_info.name}"
                raise DuplicateError(msg)
            schema[column_info.name] = column_info.get_polars_dtype()

        return schema


@dataclass
class ColumnInfo:
    """Information for a column within a catalog table."""

    name: str
    type_name: str
    type_text: str
    type_json: str
    position: int | None
    comment: str | None
    partition_index: int | None

    def get_polars_dtype(self) -> DataType:
        """
        Get the native polars datatype of this column.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
        """
        issue_unstable_warning(
            "`get_polars_dtype` functionality is considered unstable."
        )
        return PyCatalogClient.type_json_to_polars_type(self.type_json)


# TODO: Expose these type aliases to reference guide
TableType = Literal[
    "MANAGED",
    "EXTERNAL",
    "VIEW",
    "MATERIALIZED_VIEW",
    "STREAMING_TABLE",
    "MANAGED_SHALLOW_CLONE",
    "FOREIGN",
    "EXTERNAL_SHALLOW_CLONE",
]

DataSourceFormat = Literal[
    "DELTA",
    "CSV",
    "JSON",
    "AVRO",
    "PARQUET",
    "ORC",
    "TEXT",
    "UNITY_CATALOG",
    "DELTASHARING",
    "DATABRICKS_FORMAT",
    "REDSHIFT_FORMAT",
    "SNOWFLAKE_FORMAT",
    "SQLDW_FORMAT",
    "SALESFORCE_FORMAT",
    "BIGQUERY_FORMAT",
    "NETSUITE_FORMAT",
    "WORKDAY_RAAS_FORMAT",
    "HIVE_SERDE",
    "HIVE_CUSTOM",
    "VECTOR_INDEX_FORMAT",
]

# TODO: Move this back up after moving the data models to a separate file
with contextlib.suppress(ImportError):
    from polars.polars import PyCatalogClient

    PyCatalogClient.init_classes(
        catalog_info_cls=CatalogInfo,
        namespace_info_cls=NamespaceInfo,
        table_info_cls=TableInfo,
        column_info_cls=ColumnInfo,
    )

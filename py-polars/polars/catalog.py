from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from polars._utils.wrap import wrap_ldf

if TYPE_CHECKING:
    from datetime import datetime

    from polars.io.cloud import CredentialProviderFunction
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
        from polars.polars import PyCatalogClient

        if bearer_token == "databricks-sdk" or (
            bearer_token == "auto"
            # For security, in "auto" mode, only retrieve/use the token if:
            # * We are running inside a Databricks environment
            # * The `workspace_url` is pointing to Databricks
            and "DATABRICKS_RUNTIME_VERSION" in os.environ
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

    def list_schemas(self, catalog_name: str) -> list[SchemaInfo]:
        """
        List the available schemas under the specified catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        """
        return self._client.list_schemas(catalog_name)

    def list_tables(self, catalog_name: str, schema_name: str) -> list[TableInfo]:
        """
        List the available tables under the specified schema.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        catalog_name
            Name of the catalog.
        schema_name
            Name of the schema.
        """
        return self._client.list_tables(catalog_name, schema_name)

    def get_table_info(
        self, catalog_name: str, schema_name: str, table_name: str
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
        schema_name
            Name of the schema.
        table_name
            Name of the table.
        """
        return self._client.get_table_info(catalog_name, schema_name, table_name)

    def scan_table(
        self,
        catalog_name: str,
        schema_name: str,
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
        schema_name
            Name of the schema.
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
        table_info = self.get_table_info(catalog_name, schema_name, table_name)

        if (source := table_info.get("storage_location")) is None:
            msg = "cannot scan catalog table: no storage_location found"
            raise ValueError(msg)

        if (data_source_format := table_info.get("data_source_format")) is None:
            msg = "cannot scan catalog table: no data_source_format found"
            raise ValueError(msg)

        if data_source_format in ["DELTA", "DELTA_SHARING"]:
            from polars.io.delta import scan_delta

            if credential_provider is not None and credential_provider != "auto":
                msg = "credential_provider when scanning DELTA"
                raise NotImplementedError(msg)

            return scan_delta(
                source,
                version=delta_table_version,
                delta_table_options=delta_table_options,
                storage_options=storage_options,
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

        from polars.io.cloud.credential_provider import _maybe_init_credential_provider

        credential_provider = _maybe_init_credential_provider(
            credential_provider, source, storage_options, "Catalog.scan_table"
        )

        if storage_options:
            storage_options = list(storage_options.items())  # type: ignore[assignment]
        else:
            # Handle empty dict input
            storage_options = None

        return wrap_ldf(
            self._client.scan_table(
                catalog_name,
                schema_name,
                table_name,
                credential_provider=credential_provider,
                cloud_options=storage_options,
                retries=retries,
            )
        )

    @classmethod
    def _get_databricks_token(cls) -> str:
        cls._ensure_databricks_sdk_available()

        # We code like this to bypass linting
        m = importlib.import_module("databricks.sdk.core").__dict__

        return m["DefaultCredentials"]()(m["Config"]())()["Authorization"][7:]

    @staticmethod
    def _ensure_databricks_sdk_available() -> None:
        if importlib.util.find_spec("databricks.sdk") is None:
            msg = "could not get Databricks token: databricks-sdk is not installed"
            raise ImportError(msg)


class CatalogInfo(TypedDict):
    """Information for a catalog within a metastore."""

    name: str
    comment: str | None


class SchemaInfo(TypedDict):
    """Information for a schema within a catalog."""

    name: str
    comment: str | None


class TableInfo(TypedDict):
    """Information for a catalog table."""

    name: str
    comment: str | None
    table_id: str
    table_type: TableType
    storage_location: str | None
    data_source_format: DataSourceFormat | None
    columns: list[ColumnInfo] | None


class ColumnInfo(TypedDict):
    """Information for a column within a catalog table."""

    name: str
    type_text: str
    type_interval_type: str | None
    position: int | None
    comment: str | None
    partition_index: int | None


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
    "DELTA_SHARING",
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

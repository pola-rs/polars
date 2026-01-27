from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any

from polars._utils.logging import eprint
from polars.io.cloud.credential_provider._providers import (
    _get_credentials_from_provider_expiry_aware,
)
from polars.io.delta._utils import _extract_pl_data_statistics, _fill_missing_columns
from polars.io.parquet.functions import scan_parquet
from polars.io.scan_options.cast_options import ScanCastOptions
from polars.schema import Schema

if TYPE_CHECKING:
    from datetime import datetime

    from deltalake import DeltaTable

    from polars._typing import StorageOptionsDict
    from polars.io.cloud._utils import NoPickleOption
    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder
    from polars.lazyframe.frame import LazyFrame


@dataclass(kw_only=True)
class DeltaDataset:
    """Dataset interface for Delta."""

    table_: NoPickleOption[DeltaTable]
    table_uri_: str | None
    version: int | str | datetime | None

    storage_options: StorageOptionsDict | None
    credential_provider_builder: CredentialProviderBuilder | None
    delta_table_options: dict[str, Any] | None

    use_pyarrow: bool
    pyarrow_options: dict[str, Any] | None

    rechunk: bool

    #
    # PythonDatasetProvider interface functions
    #

    def schema(self) -> Schema:
        """Fetch the schema of the table."""
        return Schema(self.table().schema())

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[LazyFrame, str] | None:
        """Construct a LazyFrame scan."""
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "DeltaDataset: to_dataset_scan(): "
                f"version: {self.version}, "
                f"limit: {limit}, "
                f"projection: {projection}, "
                f"filter_columns: {filter_columns}, "
                f"use_pyarrow: {self.use_pyarrow}"
            )

        table = self.table()
        version = self.version if self.version is not None else table.version()
        version_key = str(version)

        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == version_key
        ):
            if verbose:
                eprint(
                    f"DeltaDataset: to_dataset_scan(): early return ({version_key = })"
                )

            return None

        if self.use_pyarrow:
            import polars.io.pyarrow_dataset.anonymous_scan
            from polars.lazyframe.frame import LazyFrame

            dataset = table.to_pyarrow_dataset(**(self.pyarrow_options or {}))

            func = partial(
                polars.io.pyarrow_dataset.anonymous_scan._scan_pyarrow_dataset_impl,
                dataset,
                n_rows=limit,
                predicate=pyarrow_predicate,
                with_columns=projection,
            )

            return LazyFrame._scan_python_function(
                dataset.schema, func, pyarrow=True, is_pure=True
            ), version_key

        table_md = table.metadata()
        partition_columns = set(table_md.partition_columns)

        schema = self.schema()
        hive_schema = Schema(
            {k: v for k, v in schema.items() if k in partition_columns}
        )

        start_time = perf_counter()

        if verbose:
            eprint("DeltaDataset: to_dataset_scan(): begin path expansion")

        paths = table.file_uris()

        if self.table_uri().startswith("lakefs://"):
            paths = [path.replace("lakefs://", "s3://") for path in paths]

        if verbose:
            elapsed = perf_counter() - start_time
            eprint(
                "DeltaDataset: to_dataset_scan(): "
                f"native scan_parquet(): "
                f"num_files: {len(paths)}, "
                f"path expansion time: {elapsed:.3f}s"
            )

        pl_table_statistics = _extract_pl_data_statistics(table)

        # Predicate pushown expects all statistics to be present for every column
        # that is not a partition column.
        pl_table_statistics = _fill_missing_columns(
            pl_table_statistics,
            table.schema(),
            table_md.partition_columns,
        )

        return scan_parquet(
            paths,
            hive_schema=hive_schema if len(partition_columns) > 0 else None,
            hive_partitioning=len(partition_columns) > 0,
            cast_options=ScanCastOptions._default_iceberg(),
            missing_columns="insert",
            extra_columns="ignore",
            storage_options=self.storage_options,
            credential_provider=self.credential_provider_builder,  # type: ignore[arg-type]
            rechunk=self.rechunk,
            _table_statistics=pl_table_statistics,
        ), version_key

    #
    # Accessors
    #

    def table_uri(self) -> str:
        """Fetch the table URI."""
        if self.table_uri_ is None:
            assert self.table_.get() is not None
            self.table_uri_ = self.table().table_uri

        return self.table_uri_

    def table(self) -> DeltaTable:
        """Fetch the DeltaTable object."""
        if self.table_.get() is None:
            from deltalake.exceptions import DeltaProtocolError
            from deltalake.table import (
                MAX_SUPPORTED_READER_VERSION,
                NOT_SUPPORTED_READER_VERSION,
                SUPPORTED_READER_FEATURES,
            )

            from polars.io.delta._utils import _get_delta_lake_table

            assert self.table_uri_ is not None

            credential_provider_creds = {}

            if self.credential_provider_builder and (
                provider := self.credential_provider_builder.build_credential_provider()
            ):
                credential_provider_creds = (
                    _get_credentials_from_provider_expiry_aware(provider) or {}
                )

            table = _get_delta_lake_table(
                table_path=self.table_uri_,
                version=self.version,
                storage_options=(
                    {**(self.storage_options or {}), **credential_provider_creds}
                    if self.storage_options is not None
                    or self.credential_provider_builder is not None
                    else None
                ),
                delta_table_options=self.delta_table_options,
            )

            table_protocol = table.protocol()

            if (
                table_protocol.min_reader_version > MAX_SUPPORTED_READER_VERSION
                or table_protocol.min_reader_version == NOT_SUPPORTED_READER_VERSION
            ):
                msg = (
                    f"The table's minimum reader version is {table_protocol.min_reader_version} "
                    f"but polars delta scanner only supports version 1 or {MAX_SUPPORTED_READER_VERSION} with these reader features: {SUPPORTED_READER_FEATURES}"
                )
                raise DeltaProtocolError(msg)
            if (
                table_protocol.min_reader_version >= 3
                and table_protocol.reader_features is not None
            ):
                missing_features = {*table_protocol.reader_features}.difference(
                    SUPPORTED_READER_FEATURES
                )
                if len(missing_features) > 0:
                    msg = f"The table has set these reader features: {missing_features} but these are not yet supported by the polars delta scanner."
                    raise DeltaProtocolError(msg)

            self.table_.set(table)

        return self.table_.get()  # type: ignore[return-value]

    def __getstate__(self) -> dict[str, Any]:
        self.table_uri()
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

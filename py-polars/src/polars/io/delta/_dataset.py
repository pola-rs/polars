from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any

import polars as pl
from polars._utils.logging import eprint
from polars._utils.various import parse_version
from polars.io.cloud.credential_provider._providers import (
    _get_credentials_from_provider_expiry_aware,
)
from polars.io.delta._utils import _extract_table_statistics_from_delta_add_actions
from polars.io.parquet.functions import scan_parquet
from polars.io.scan_options.cast_options import ScanCastOptions
from polars.schema import Schema

if TYPE_CHECKING:
    from datetime import datetime

    from deltalake import DeltaTable

    from polars._typing import DeletionFiles, StorageOptionsDict
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
        import polars as pl
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

        table_statistics = (
            _extract_table_statistics_from_delta_add_actions(
                pl.DataFrame(table.get_add_actions()),
                filter_columns=filter_columns,
                schema=schema,
                verbose=verbose,
            )
            if filter_columns is not None
            else None
        )

        reader_features = table.protocol().reader_features
        has_deletion_vectors = (
            reader_features is not None and "deletionVectors" in reader_features
        )

        deletion_files: DeletionFiles | None = None
        if has_deletion_vectors:
            import deltalake

            dv_min_version = (1, 4, 2)
            installed = parse_version(deltalake.__version__)
            if installed < dv_min_version:
                msg = (
                    f"reading delta deletion vectors requires "
                    f"deltalake >= {'.'.join(str(v) for v in dv_min_version)}, "
                    f"found {installed}."
                )
                raise ImportError(msg)

            def _deletion_vector_callback(
                requested_paths: pl.DataFrame,
            ) -> pl.DataFrame:
                delta_deletion_vectors = _fetch_deletion_vectors(table)
                if delta_deletion_vectors is None:
                    return pl.DataFrame(
                        {"selection_vector": [None] * len(requested_paths)},
                        schema={"selection_vector": pl.List(pl.Boolean)},
                    )
                return _extract_delta_deletion_vectors(
                    requested_paths, delta_deletion_vectors
                )

            deletion_files = (
                "delta-deletion-vector",
                _deletion_vector_callback,
            )
        else:
            deletion_files = None

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
            _table_statistics=table_statistics,
            _deletion_files=deletion_files,
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

            # Some reader features require explicit support by the engine (polars)
            SUPPORTED_READER_FEATURES.add("deletionVectors")

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


def _extract_delta_deletion_vectors(
    requested_paths: pl.DataFrame,
    delta_deletion_vectors: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extract the deletion_vectors for the provided requested_paths.

    Input requested_paths schema is "path": String.
    Output series schema is "selection_vector": List(Boolean), maintaining order.

    The selection_vector from deltalake is a keep-mask (True = keep).
    """
    assert requested_paths.schema == {"path": pl.String}

    delta_dv_schema = {"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)}
    delta_deletion_vectors = delta_deletion_vectors.select(delta_dv_schema.keys())
    assert delta_deletion_vectors.schema == delta_dv_schema

    file_prefix = "file://" if sys.platform != "win32" else "file:///"
    joined_df = (
        requested_paths.lazy()
        .with_columns(
            pl.col("path")
            .str.replace("^lakefs://", "s3://")
            .str.strip_prefix(file_prefix)
        )
        .join(
            delta_deletion_vectors.lazy().with_columns(
                pl.col("filepath")
                .str.replace("^lakefs://", "s3://")
                .str.strip_prefix(file_prefix)
            ),
            left_on="path",
            right_on="filepath",
            how="left",
            maintain_order="left",
        )
        .select(["selection_vector"])
        .collect()
    )

    assert joined_df.height == len(requested_paths)

    return joined_df


def _fetch_deletion_vectors(table: DeltaTable) -> pl.DataFrame | None:
    """
    Fetch the deletion_vectors, mapping file_uri to "deletion_vector".

    Schema: {"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)}

    The selection_vector from deltalake is a keep-mask (True = keep), so
    the more accurate term would be "selection_vector".

    Returns None if the table has no deletion vectors.
    """
    import polars._utils.logging

    verbose = polars._utils.logging.verbose()

    dv_table = pl.DataFrame(table.deletion_vectors())

    if verbose and dv_table.height > 0:
        eprint(f"DeltaDataset: has deletion_vectors, file_count: {len(dv_table)}")

    if len(dv_table) == 0:
        return None

    return dv_table

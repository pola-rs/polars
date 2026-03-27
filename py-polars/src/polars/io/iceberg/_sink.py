from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from polars._utils.logging import eprint
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    IcebergCatalogTableDescriptor,
    IcebergTableSerializer,
    IcebergTableWrap,
    _convert_iceberg_to_object_store_storage_options,
)
from polars.io.iceberg._utils import _normalize_windows_iceberg_file_uri
from polars.io.partition import _InternalPlPathProviderConfig

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import gen_uuid_v7

if TYPE_CHECKING:
    import pyiceberg.catalog
    import pyiceberg.table

    import polars as pl
    from polars._typing import StorageOptionsDict
    from polars.io.iceberg._dataset import SerializedTableState


def _ensure_table_has_catalog(table: pyiceberg.table.Table) -> None:
    from pyiceberg.catalog.noop import NoopCatalog

    if isinstance(table.catalog, NoopCatalog):
        msg = (
            "cannot sink to static Iceberg table: "
            f"{type(table) = }, {type(table.catalog) = }"
        )
        raise TypeError(msg)


class IcebergSinkTableSerializer(IcebergTableSerializer):
    @staticmethod
    def serialize_table(table: pyiceberg.table.Table) -> SerializedTableState:
        _ensure_table_has_catalog(table)

        return IcebergCatalogTableDescriptor(
            table_identifier=".".join(table.name()),
            catalog_config=IcebergCatalogConfig.from_catalog(table.catalog),
        )


# Passed to `pipe_with_schema` to defer sink resolution logic until IR resolution.
@dataclass(kw_only=True)
class AttachSink:
    sink_state: IcebergSinkState

    def __call__(
        self,
        lf: pl.LazyFrame,
        schema: pl.Schema,  # noqa: ARG002
    ) -> pl.LazyFrame:
        return self.sink_state._attach_sink_impl(lf)


@dataclass(kw_only=True)
class IcebergSinkState:
    table: IcebergTableWrap
    mode: Literal["append", "overwrite"]
    sink_uuid_str: str
    output_base_path_: str | None = None
    commit_result_df: pl.DataFrame | None = None

    @staticmethod
    def new(
        target: str | pyiceberg.table.Table,
        *,
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig | None = None,
        mode: Literal["append", "overwrite"] = "append",
        storage_options: StorageOptionsDict | None = None,
    ) -> IcebergSinkState:
        table: pyiceberg.table.Table | None = None

        if importlib.util.find_spec("pyiceberg.table") is not None:
            from pyiceberg.table import Table

            if isinstance(target, Table):
                table = target

        table_descriptor: IcebergCatalogTableDescriptor | None = None

        if isinstance(target, str):
            catalog_config = (
                IcebergCatalogConfig._from_api_parameter_or_environment_default(
                    catalog,
                    fn_name="sink_iceberg",
                )
            )

            table_descriptor = IcebergCatalogTableDescriptor(
                table_identifier=target,
                catalog_config=catalog_config,
            )

        if table is not None:
            _ensure_table_has_catalog(table)

        return IcebergSinkState(
            table=IcebergTableWrap(
                table_=NoPickleOption(table),
                table_descriptor_=table_descriptor,
                serializer=IcebergSinkTableSerializer(),
                iceberg_storage_properties=storage_options,
            ),
            mode=mode,
            sink_uuid_str=gen_uuid_v7().hex(),
        )

    def _get_converted_storage_options(self) -> dict[str, str] | None:
        return (
            _convert_iceberg_to_object_store_storage_options(
                self.table.iceberg_storage_properties
            )
            if self.table.iceberg_storage_properties is not None
            else None
        )

    def attach_sink(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.pipe_with_schema(AttachSink(sink_state=self))

    def _attach_sink_impl(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        from pyiceberg.table import TableProperties
        from pyiceberg.utils.properties import property_as_bool, property_as_int

        import polars as pl

        table = self.table.get()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        if table.spec().fields:
            msg = "sink to partitioned Iceberg table"
            raise NotImplementedError(msg)

        if table.sort_order().fields:
            msg = "sink to Iceberg table with sort order"
            raise NotImplementedError(msg)

        if location_provider_impl := table_properties.get(
            TableProperties.WRITE_PY_LOCATION_PROVIDER_IMPL
        ):
            msg = (
                "sink to Iceberg table with custom location provider"
                f" '{location_provider_impl}'"
            )
            raise NotImplementedError(msg)

        if property_as_bool(
            table_properties, TableProperties.OBJECT_STORE_ENABLED, False
        ):
            msg = f"sink to Iceberg table with '{TableProperties.OBJECT_STORE_ENABLED}'"
            raise NotImplementedError(msg)

        arrow_schema = self.table.arrow_schema()

        approximate_bytes_per_file = 2 * 1024 * 1024 * 1024

        if v := property_as_int(
            properties=table_metadata.properties,
            property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        ):
            estimated_compression_ratio = 4
            approximate_bytes_per_file = min(
                estimated_compression_ratio * v, (1 << 64) - 1
            )

        self.table.__getstate__()

        return lf.sink_parquet(
            pl.PartitionBy(
                _normalize_windows_iceberg_file_uri(self.output_base_path()),
                file_path_provider=PlIcebergPathProviderConfig(),
                approximate_bytes_per_file=approximate_bytes_per_file,
            ),
            arrow_schema=arrow_schema,
            storage_options=self._get_converted_storage_options(),
            lazy=True,
            _sinked_paths_callback=("iceberg-commit", self),
        )

    def commit(
        self,
        *,
        data_file_paths: list[str],
    ) -> pl.DataFrame:
        import polars as pl
        import polars._utils.logging

        function_start_instant = perf_counter()
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(f"IcebergSinkState[commit]: mode: '{self.mode}'")

        table = self.table.get()
        original_metadata_location = table.metadata_location

        if sys.platform == "win32":
            data_file_paths = [
                (f"file://{p[8:]}" if p.startswith("file:///") else p)
                for p in data_file_paths
            ]

        with table.transaction() as tx:
            if self.mode == "overwrite":
                from pyiceberg.expressions import AlwaysTrue

                tx.delete(AlwaysTrue())

            if verbose:
                eprint("IcebergSinkState[commit]: begin add_files")

            start_instant = perf_counter()

            tx.add_files(
                data_file_paths,
                check_duplicate_files=False,
            )

            if verbose:
                elapsed = perf_counter() - start_instant
                eprint(f"IcebergSinkState[commit]: finish add_files ({elapsed:.3f}s)")
                eprint("IcebergSinkState[commit]: begin transaction commit")

            start_instant = perf_counter()

        if verbose:
            now = perf_counter()
            elapsed = now - start_instant
            eprint(
                f"IcebergSinkState[commit]: finish transaction commit ({elapsed:.3f}s)"
            )

        new_metadata_location = table.metadata_location

        assert new_metadata_location != original_metadata_location

        self.commit_result_df = pl.DataFrame(
            {"metadata_path": new_metadata_location},
            schema={"metadata_path": pl.String},
            height=1,
        )

        if verbose:
            total_elapsed = now - function_start_instant

            eprint(
                f"IcebergSinkState[commit]: finished, total elapsed time: {total_elapsed:.3f}s"
            )

        return self.commit_result_df

    def output_base_path(self) -> str:
        if self.output_base_path_ is None:
            from pyiceberg.table import TableProperties

            table = self.table.get()
            table_metadata = table.metadata
            table_properties = table_metadata.properties

            output_base_path = (
                path.rstrip("/")
                if (path := table_properties.get(TableProperties.WRITE_DATA_PATH))
                else f"{table_metadata.location.rstrip('/')}/data"
            )

            output_base_path = f"{output_base_path}/{self.sink_uuid_str}/"
            self.output_base_path_ = output_base_path

        return self.output_base_path_

    @staticmethod
    def _from_data_dict(data_dict: dict[str, Any]) -> IcebergSinkState:
        table_wrap_dict = data_dict.pop("table")
        table_descriptor_dict = table_wrap_dict.pop("table_descriptor_")
        catalog_config_dict = table_descriptor_dict.pop("catalog_config")

        return IcebergSinkState(
            table=IcebergTableWrap(
                table_=NoPickleOption(),
                table_descriptor_=IcebergCatalogTableDescriptor(
                    catalog_config=IcebergCatalogConfig(**catalog_config_dict),
                    **table_descriptor_dict,
                ),
                serializer=IcebergSinkTableSerializer(),
                **table_wrap_dict,
            ),
            **data_dict,
        )


class PlIcebergPathProviderConfig(_InternalPlPathProviderConfig):
    pl_path_provider_id: ClassVar[str] = "iceberg"
    extension: ClassVar[Literal["parquet"]] = "parquet"

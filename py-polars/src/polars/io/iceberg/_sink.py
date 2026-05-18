from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar, Literal

from polars._utils.logging import eprint
from polars._utils.wrap import wrap_ldf
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
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
    from polars._plr import PyLazyFrame
    from polars._typing import StorageOptionsDict


@dataclass(kw_only=True)
class IcebergSinkState:
    py_catalog_class_module: str
    py_catalog_class_qualname: str

    catalog_name: str
    catalog_properties: dict[str, str]

    table_name: str
    mode: Literal["append", "overwrite"]
    iceberg_storage_properties: StorageOptionsDict

    sink_uuid_str: str

    table_: NoPickleOption[pyiceberg.table.Table]
    commit_result_df: NoPickleOption[pl.DataFrame]

    @staticmethod
    def new(
        target: str | pyiceberg.table.Table,
        *,
        mode: Literal["append", "overwrite"] = "append",
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig | None = None,
        storage_options: StorageOptionsDict | None = None,
    ) -> IcebergSinkState:
        catalog_config = (
            (
                IcebergCatalogConfig._from_api_parameter_or_environment_default(
                    catalog,
                    fn_name="sink_iceberg",
                )
            )
            if isinstance(target, str)
            else (
                IcebergCatalogConfig(
                    class_=type(target.catalog),
                    name=target.catalog.name,
                    properties=target.catalog.properties,
                )
            )
        )

        from pyiceberg.catalog.noop import NoopCatalog

        if catalog_config.class_ is NoopCatalog:
            msg = (
                "cannot sink to static Iceberg table: "
                f"{type(target) = }, {getattr(target, 'catalog', None) = }"
            )
            raise TypeError(msg)

        return IcebergSinkState(
            py_catalog_class_module=catalog_config.class_.__module__,
            py_catalog_class_qualname=catalog_config.class_.__qualname__,
            catalog_name=catalog_config.name,
            catalog_properties=catalog_config.properties,
            table_name=target if isinstance(target, str) else ".".join(target.name()),
            mode=mode,
            iceberg_storage_properties=storage_options or {},
            sink_uuid_str=gen_uuid_v7().hex(),
            table_=NoPickleOption(target if not isinstance(target, str) else None),
            commit_result_df=NoPickleOption(),
        )

    def table(self) -> pyiceberg.table.Table:
        if self.table_.get() is None:
            module = importlib.import_module(self.py_catalog_class_module)
            qualname_split = self.py_catalog_class_qualname.split(".")

            catalog_class: type[pyiceberg.catalog.Catalog] = getattr(
                module, qualname_split[0]
            )

            for part in qualname_split[1:]:
                catalog_class = getattr(catalog_class, part)

            catalog = catalog_class(self.catalog_name, **self.catalog_properties)
            self.table_.set(catalog.load_table(self.table_name))

        return self.table_.get()  # type: ignore[return-value]

    def _get_converted_storage_options(self) -> dict[str, str]:
        return _convert_iceberg_to_object_store_storage_options(
            self.iceberg_storage_properties
        )

    def attach_sink(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return wrap_ldf(lf._ldf.sink_iceberg(self))

    def _attach_resolved_sink(self, plf: PyLazyFrame) -> PyLazyFrame:
        from pyiceberg.table import TableProperties
        from pyiceberg.utils.properties import property_as_bool, property_as_int

        import polars as pl

        table = self.table()
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

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        arrow_schema = schema_to_pyarrow(table.schema())

        approximate_bytes_per_file = 2 * 1024 * 1024 * 1024

        if v := property_as_int(
            properties=table_metadata.properties,
            property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        ):
            estimated_compression_ratio = 4
            approximate_bytes_per_file = min(
                estimated_compression_ratio * v, (1 << 64) - 1
            )

        return (
            wrap_ldf(plf)
            .sink_parquet(
                pl.PartitionBy(
                    _normalize_windows_iceberg_file_uri(self.output_base_path()),
                    file_path_provider=PlIcebergPathProviderConfig(),
                    approximate_bytes_per_file=approximate_bytes_per_file,
                ),
                arrow_schema=arrow_schema,
                storage_options=self._get_converted_storage_options(),
                lazy=True,
            )
            ._ldf
        )

    def commit(self, data_file_paths: list[str]) -> pl.DataFrame:
        import polars as pl
        import polars._utils.logging

        function_start_instant = perf_counter()
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(f"IcebergSinkState[commit]: mode: '{self.mode}'")

        table = self.table()

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

        self.commit_result_df.set(
            pl.DataFrame(
                {"metadata_path": new_metadata_location},
                schema={"metadata_path": pl.String},
                height=1,
            )
        )

        if verbose:
            total_elapsed = now - function_start_instant

            eprint(
                f"IcebergSinkState[commit]: finished, total elapsed time: {total_elapsed:.3f}s"
            )

        return self.commit_result_df.get()  # type: ignore[return-value]

    def output_base_path(self) -> str:
        from pyiceberg.table import TableProperties

        table = self.table()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        output_base_path = (
            path.rstrip("/")
            if (path := table_properties.get(TableProperties.WRITE_DATA_PATH))
            else f"{table_metadata.location.rstrip('/')}/data"
        )

        return f"{output_base_path}/{self.sink_uuid_str}/"


class PlIcebergPathProviderConfig(_InternalPlPathProviderConfig):
    pl_path_provider_id: ClassVar[str] = "iceberg"
    extension: ClassVar[Literal["parquet"]] = "parquet"

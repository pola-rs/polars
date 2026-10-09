from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias

import polars._reexport as pl
from polars._utils.logging import eprint, verbose, verbose_print_sensitive
from polars._utils.various import qualified_type_name
from polars.exceptions import ComputeError
from polars.io.iceberg._cache import CachingFileIO
from polars.io.iceberg._plugin import (
    plugin_planner_required,
    plugin_scan,
    use_plugin_planner,
)
from polars.io.iceberg._utils import (
    IcebergStatisticsLoader,
    IdentityTransformedPartitionValuesBuilder,
    _new_pyiceberg_scan,
    _normalize_windows_iceberg_file_uri,
    extract_field_initial_default,
    filter_for_pyiceberg_reader,
    filter_for_scan_schema,
    filter_for_truncate_overflow,
    filter_with_nan_ordering,
    try_convert_pyarrow_predicate,
)
from polars.io.scan_options.cast_options import ScanCastOptions

if TYPE_CHECKING:
    import pyarrow as pa
    import pyiceberg.catalog
    import pyiceberg.schema
    import pyiceberg.table
    import pyiceberg.typedef

    from polars._typing import StorageOptionsDict
    from polars.io.cloud._utils import NoPickleOption
    from polars.lazyframe.frame import LazyFrame


class IcebergTableSerializer(ABC):
    @staticmethod
    @abstractmethod
    def serialize_table(table: pyiceberg.table.Table) -> SerializedTableState: ...


class IcebergScanTableSerializer(IcebergTableSerializer):
    @staticmethod
    def serialize_table(table: pyiceberg.table.Table) -> SerializedTableState:
        return table.metadata_location


@dataclass(kw_only=True)
class IcebergCatalogTableDescriptor:
    table_identifier: str | pyiceberg.typedef.Identifier
    catalog_config: IcebergCatalogConfig
    # Used for table loads when set; `catalog_config` is the fallback after
    # unpickling, and for descriptors pickled without this field.
    catalog_: NoPickleOption[pyiceberg.catalog.Catalog] | None = None


# Catalog classes whose instances, and those of their subclasses, can load tables
# from concurrent threads.
_REUSABLE_CATALOG_CLASSES: Final = frozenset(
    (
        "pyiceberg.catalog.glue.GlueCatalog",
        "pyiceberg.catalog.rest.RestCatalog",
        "pyiceberg.catalog.sql.SqlCatalog",
    )
)


def _reusable_catalog(
    catalog: pyiceberg.catalog.Catalog | None,
) -> pyiceberg.catalog.Catalog | None:
    if catalog is None or not any(
        qualified_type_name(cls) in _REUSABLE_CATALOG_CLASSES
        for cls in type(catalog).__mro__
    ):
        return None
    return catalog


SerializedTableState: TypeAlias = str | IcebergCatalogTableDescriptor


@dataclass(kw_only=True)
class IcebergTableWrap:
    table_: NoPickleOption[pyiceberg.table.Table]
    table_descriptor_: SerializedTableState | None
    serializer: IcebergTableSerializer
    iceberg_storage_properties: StorageOptionsDict | None

    def get(self) -> pyiceberg.table.Table:
        """Fetch the PyIceberg Table object."""
        if self.table_.get() is None:
            if verbose():
                from_ = (
                    "catalog table descriptor: "
                    f"{self.table_descriptor_.table_identifier = }, "
                    f"{self.table_descriptor_.catalog_config.class_ = }"
                    if isinstance(self.table_descriptor_, IcebergCatalogTableDescriptor)
                    else f"metadata path: {self.table_descriptor_}"
                )

                eprint(f"IcebergTableWrap: construct table from {from_}")

            assert self.table_descriptor_ is not None

            if isinstance(self.table_descriptor_, IcebergCatalogTableDescriptor):
                catalog_ = self.table_descriptor_.catalog_
                catalog = catalog_.get() if catalog_ is not None else None

                if catalog is None:
                    catalog = self.table_descriptor_.catalog_config.class_(
                        self.table_descriptor_.catalog_config.name,
                        **self.table_descriptor_.catalog_config.properties,
                    )
                elif verbose():
                    eprint("IcebergTableWrap: reuse catalog instance")

                table = catalog.load_table(self.table_descriptor_.table_identifier)
            else:
                from pyiceberg.table import StaticTable

                table = StaticTable.from_metadata(
                    metadata_location=self.table_descriptor_,
                    properties=self.iceberg_storage_properties or {},
                )

            self.table_.set(table)

        return self.table_.get()  # type: ignore[return-value]

    def arrow_schema(self) -> pa.schema:
        """Fetch the arrow schema of the table."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        return schema_to_pyarrow(self.get().schema())

    def __getstate__(self) -> dict[str, Any]:
        if (table := self.table_.get()) is not None:
            self.table_descriptor_ = self.serializer.serialize_table(table)

        assert self.table_descriptor_ is not None

        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state


@dataclass(kw_only=True)
class IcebergCatalogConfig:
    """
    Configuration for constructing a PyIceberg catalog.

    This is useful for constructing queries from a client that may not have
    access to a catalog server.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    class_: type[pyiceberg.catalog.Catalog]
    name: str
    properties: dict[str, str]

    @staticmethod
    def from_catalog(catalog: pyiceberg.catalog.Catalog) -> IcebergCatalogConfig:
        """
        Constructs an IcebergCatalogConfig from an instantiated PyIceberg catalog.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
        """
        return IcebergCatalogConfig(
            class_=type(catalog),
            name=catalog.name,
            properties=catalog.properties,
        )

    @staticmethod
    def _from_api_parameter_or_environment_default(
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig | None,
        *,
        fn_name: Literal["scan_iceberg", "sink_iceberg"],
    ) -> tuple[IcebergCatalogConfig, pyiceberg.catalog.Catalog | None]:
        """Return the catalog config, and the catalog instance when one is known."""
        import pyiceberg.catalog
        from pyiceberg.catalog.noop import NoopCatalog

        import polars._utils.logging
        from polars._utils.logging import eprint

        instance: pyiceberg.catalog.Catalog | None = None

        if isinstance(catalog, IcebergCatalogConfig):
            catalog_config = catalog
        elif isinstance(catalog, pyiceberg.catalog.Catalog):
            catalog_config = IcebergCatalogConfig.from_catalog(catalog)
            instance = catalog
        elif catalog is not None:
            msg = f"unknown type for `catalog` parameter: {type(catalog)}"
            raise TypeError(msg)
        else:
            if polars._utils.logging.verbose():
                eprint(f"{fn_name}(): calling pyiceberg.catalog.load_catalog()")

            try:
                default_catalog = pyiceberg.catalog.load_catalog()

            except Exception as error:
                static_metadata_hint = (
                    (
                        " "
                        "If you intended to pass a static metadata path, "
                        "ensure it is an absolute path."
                    )
                    if fn_name == "scan_iceberg"
                    else ""
                )

                msg = (
                    f"failed to load catalog for {fn_name}() ({error = }). "
                    "Configure the default PyIceberg catalog, or pass "
                    "a catalog via the 'catalog' parameter, or pass a PyIceberg "
                    "table object instead of the name."
                    f"{static_metadata_hint}"
                )
                raise ComputeError(msg) from error

            catalog_config = IcebergCatalogConfig.from_catalog(default_catalog)
            instance = default_catalog

        if catalog_config.class_ == NoopCatalog:
            msg = f"cannot use NoopCatalog with {fn_name}()"
            raise TypeError(msg)

        return catalog_config, instance


@dataclass(kw_only=True)
class IcebergScanResolver:
    """
    Iceberg scan resolver.

    Defers scan resolution to run during IR resolution.
    """

    table: IcebergTableWrap
    snapshot_id: int | None
    from_snapshot_id_exclusive: int | None
    to_snapshot_id_inclusive: int | None
    reader_override: Literal["native", "pyiceberg"] | None
    use_metadata_statistics: bool
    fast_deletion_count: bool
    use_pyiceberg_filter: bool
    # The table schema of the scan's output schema (without `snapshot_id`). Scans are
    # resolved by column name when collected, so they must not see another schema.
    schema_at_creation: pyiceberg.schema.Schema | None = field(
        default=None, init=False, repr=False
    )

    #
    # PythonDatasetProvider interface functions
    #

    def schema(self) -> pa.schema:
        """Fetch the schema of the table."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        if self.snapshot_id is None:
            if self.schema_at_creation is None:
                self.schema_at_creation = self.table.get().schema()

            return schema_to_pyarrow(self.schema_at_creation)

        snapshot = self.table.get().snapshot_by_id(self.snapshot_id)

        if snapshot is None:
            msg = f"iceberg snapshot ID not found: {self.snapshot_id}"
            raise ValueError(msg)

        schema_id = snapshot.schema_id

        if schema_id is None:
            msg = (
                f"IcebergScanResolver: requested snapshot {self.snapshot_id} "
                "did not contain a schema ID"
            )
            raise ValueError(msg)
        return schema_to_pyarrow(self.table.get().schemas()[schema_id])

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        statistics_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[LazyFrame, str] | None:
        """Construct a LazyFrame scan."""
        if (
            scan_data := self._to_dataset_scan_impl(
                existing_resolved_version_key=existing_resolved_version_key,
                limit=limit,
                projection=projection,
                filter_columns=filter_columns,
                statistics_columns=statistics_columns,
                pyarrow_predicate=pyarrow_predicate,
            )
        ) is None:
            return None

        return scan_data.to_lazyframe(), scan_data.snapshot_id_key

    def _to_dataset_scan_impl(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        statistics_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> _NativeIcebergScanData | _PyIcebergScanData | _PluginIcebergScanData | None:
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        iceberg_table_filter = None

        if (
            pyarrow_predicate is not None
            and self.use_metadata_statistics
            and self.use_pyiceberg_filter
        ):
            iceberg_table_filter = try_convert_pyarrow_predicate(pyarrow_predicate)

        if verbose:
            pyarrow_predicate_display = (
                "Some(<redacted>)" if pyarrow_predicate is not None else "None"
            )
            iceberg_table_filter_display = (
                "Some(<redacted>)" if iceberg_table_filter is not None else "None"
            )

            eprint(
                "IcebergScanResolver: to_dataset_scan(): "
                f"snapshot ID: {self.snapshot_id}, "
                f"from snapshot ID exclusive: {self.from_snapshot_id_exclusive}, "
                f"to snapshot ID inclusive: {self.to_snapshot_id_inclusive}, "
                f"limit: {limit}, "
                f"projection: {projection}, "
                f"filter_columns: {filter_columns}, "
                f"statistics_columns: {statistics_columns}, "
                f"pyarrow_predicate: {pyarrow_predicate_display}, "
                f"iceberg_table_filter: {iceberg_table_filter_display}, "
                f"self.use_metadata_statistics: {self.use_metadata_statistics}"
            )

        verbose_print_sensitive(
            lambda: (
                f"IcebergScanResolver: to_dataset_scan(): {pyarrow_predicate = }, {iceberg_table_filter = }"
            )
        )

        tbl = self.table.get()

        if verbose:
            eprint(
                "IcebergScanResolver: to_dataset_scan(): "
                f"tbl.metadata.current_snapshot_id: {tbl.metadata.current_snapshot_id}"
            )

        snapshot_id = self.snapshot_id
        is_incremental = (
            self.from_snapshot_id_exclusive is not None
            or self.to_snapshot_id_inclusive is not None
        )
        schema_id = None

        if snapshot_id is not None:
            snapshot = tbl.snapshot_by_id(snapshot_id)

            if snapshot is None:
                msg = f"iceberg snapshot ID not found: {snapshot_id}"
                raise ValueError(msg)

            schema_id = snapshot.schema_id

            if schema_id is None:
                msg = (
                    f"IcebergScanResolver: requested snapshot {snapshot_id} "
                    "did not contain a schema ID"
                )
                raise ValueError(msg)

            iceberg_schema = tbl.schemas()[schema_id]
            snapshot_id_key = f"{snapshot.snapshot_id}"
        else:
            iceberg_schema = tbl.schema()
            schema_id = tbl.metadata.current_schema_id

            if (
                self.schema_at_creation is not None
                and self.schema_at_creation.as_struct() != iceberg_schema.as_struct()
            ):
                msg = (
                    "iceberg: the table schema changed after the scan was created "
                    f"(schema ID {self.schema_at_creation.schema_id} -> {schema_id}); "
                    "create a new scan"
                )
                raise ComputeError(msg)

            current_snapshot_id = (
                v.snapshot_id if (v := tbl.current_snapshot()) is not None else None
            )
            resolved_end_snapshot_id = (
                self.to_snapshot_id_inclusive
                if self.to_snapshot_id_inclusive is not None
                else current_snapshot_id
            )
            snapshot_id_key = (
                f"incremental:{self.from_snapshot_id_exclusive}:"
                f"{resolved_end_snapshot_id}:schema:{schema_id}"
                if is_incremental
                else f"{current_snapshot_id or ''}"
            )

        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == snapshot_id_key
        ):
            if verbose:
                eprint(
                    "IcebergScanResolver: to_dataset_scan(): early return "
                    f"({snapshot_id_key = })"
                )

            return None

        # Take from parameter first then envvar
        reader_override = self.reader_override or os.getenv(
            "POLARS_ICEBERG_READER_OVERRIDE"
        )

        if reader_override and reader_override not in ["native", "pyiceberg"]:
            msg = (
                "iceberg: unknown value for reader_override: "
                f"'{reader_override}', expected one of ('native', 'pyiceberg')"
            )
            raise ValueError(msg)

        if reader_override != "pyiceberg" and use_plugin_planner():
            if verbose:
                eprint(
                    "IcebergScanResolver: to_dataset_scan(): "
                    "plan with the polars_iceberg plugin"
                )

            try:
                lf = plugin_scan(
                    tbl,
                    snapshot_id=self.snapshot_id,
                    from_snapshot_id_exclusive=self.from_snapshot_id_exclusive,
                    to_snapshot_id_inclusive=self.to_snapshot_id_inclusive,
                    projection=projection,
                    filter_columns=filter_columns,
                    statistics_columns=statistics_columns,
                    iceberg_table_filter=iceberg_table_filter,
                    limit=limit,
                    use_metadata_statistics=self.use_metadata_statistics,
                    fast_deletion_count=self.fast_deletion_count,
                    user_storage_options=self.table.iceberg_storage_properties,
                )
            except NotImplementedError as e:
                # A table feature the plugin does not support (e.g. equality deletes),
                # which PyIceberg can plan.
                if plugin_planner_required():
                    raise

                if verbose:
                    eprint(
                        "IcebergScanResolver: to_dataset_scan(): "
                        f"plugin planner unsupported, plan with PyIceberg: {e}"
                    )
            else:
                return _PluginIcebergScanData(lf=lf, snapshot_id_key=snapshot_id_key)

        if iceberg_table_filter is not None and schema_id != (
            tbl.metadata.current_schema_id
        ):
            iceberg_table_filter = filter_for_scan_schema(
                iceberg_table_filter, iceberg_schema, tbl.schema()
            )

        if iceberg_table_filter is not None:
            iceberg_table_filter = filter_for_truncate_overflow(
                iceberg_table_filter, tbl
            )

        fallback_reason = (
            "forced reader_override='pyiceberg'"
            if reader_override == "pyiceberg"
            else None
        )

        selected_fields = ("*",) if projection is None else tuple(projection)

        projected_iceberg_schema = (
            iceberg_schema
            if selected_fields == ("*",)
            else iceberg_schema.select(*selected_fields)
        )

        initial_defaults = {
            x: value
            for x in projected_iceberg_schema.field_ids
            if (
                value := extract_field_initial_default(
                    projected_iceberg_schema.find_field(x)
                )
            )
            is not None
        }

        sources = []
        source_sizes = []
        missing_field_defaults = IdentityTransformedPartitionValuesBuilder(
            tbl,
            projected_iceberg_schema,
        )
        # Statistics of columns that are not filtered on are best effort.
        best_effort_statistics_columns = [
            c for c in statistics_columns or [] if c not in (filter_columns or [])
        ]
        statistics_loader: IcebergStatisticsLoader | None = (
            IcebergStatisticsLoader(
                tbl,
                iceberg_schema.select(
                    *(filter_columns or []), *best_effort_statistics_columns
                ),
                best_effort_columns=best_effort_statistics_columns,
            )
            if self.use_metadata_statistics
            and (filter_columns is not None or best_effort_statistics_columns)
            else None
        )
        position_delete_files: dict[int, list[str]] = {}
        deletion_vectors: dict[int, str] = {}
        # Deleted row counts of the deletion vectors of each Puffin file, by data file.
        puffin_deletion_vectors: dict[str, dict[str, int] | None] = {}
        total_physical_rows: int = 0
        total_deleted_rows: int = 0
        total_position_delete_files = 0
        total_deletion_vectors = 0

        if reader_override != "pyiceberg" and not fallback_reason:
            from pyiceberg.manifest import DataFileContent, FileFormat

            if verbose:
                eprint("IcebergScanResolver: to_dataset_scan(): begin path expansion")

            start_time = perf_counter()

            scan = _new_pyiceberg_scan(
                tbl,
                snapshot_id=snapshot_id,
                from_snapshot_id_exclusive=self.from_snapshot_id_exclusive,
                to_snapshot_id_inclusive=self.to_snapshot_id_inclusive,
                limit=limit,
                selected_fields=selected_fields,
            )

            if iceberg_table_filter is not None:
                scan = scan.filter(
                    filter_with_nan_ordering(iceberg_table_filter, iceberg_schema)
                )

            for i, file_info in enumerate(scan.plan_files()):
                if file_info.file.file_format != FileFormat.PARQUET:
                    fallback_reason = (
                        f"non-parquet format: {file_info.file.file_format}"
                    )
                    break

                if file_info.delete_files:
                    position_delete_files[i] = []
                    position_delete_num_rows = 0
                    deletion_vector_num_rows = 0

                    for deletion_file in file_info.delete_files:
                        if deletion_file.content != DataFileContent.POSITION_DELETES:
                            fallback_reason = (
                                "unsupported deletion file type: "
                                f"{deletion_file.content}"
                            )
                            break

                        match deletion_file.file_format:
                            case FileFormat.PARQUET:
                                # The native reader reads position delete files of
                                # one data file only. This also keeps the deleted row
                                # count exact: the rows of a delete file scoped to a
                                # partition may belong to data files that are no
                                # longer live.
                                referenced = _referenced_data_file(deletion_file)

                                if referenced is None:
                                    fallback_reason = (
                                        "position delete file not limited to one "
                                        f"data file: {deletion_file.file_path}"
                                    )
                                    break

                                if referenced != file_info.file.file_path:
                                    # PyIceberg associates a delete file without
                                    # `file_path` bounds with every data file of its
                                    # partition.
                                    continue

                                position_delete_files[i].append(deletion_file.file_path)
                                position_delete_num_rows += deletion_file.record_count

                            case FileFormat.PUFFIN:
                                # PyIceberg associates a deletion vector without
                                # `file_path` bounds (as Iceberg Java writes them)
                                # with every data file of its partition, and does not
                                # read `referenced_data_file`. The Puffin footer
                                # says which data files it holds deletes of.
                                if (
                                    deletion_file.file_path
                                    not in puffin_deletion_vectors
                                ):
                                    puffin_deletion_vectors[deletion_file.file_path] = (
                                        _read_puffin_deletion_vector_counts(
                                            tbl.io,
                                            deletion_file.file_path,
                                            deletion_file.file_size_in_bytes,
                                        )
                                    )

                                if (
                                    counts := puffin_deletion_vectors[
                                        deletion_file.file_path
                                    ]
                                ) is None:
                                    fallback_reason = (
                                        "unsupported Puffin footer: "
                                        f"{deletion_file.file_path}"
                                    )
                                    break

                                num_rows = counts.get(
                                    _normalize_windows_iceberg_file_uri(
                                        file_info.file.file_path
                                    )
                                )

                                # Read by Polars, which does not accept PyIceberg's
                                # Windows `file://C:/` URIs.
                                deletion_vector_path = (
                                    _normalize_windows_iceberg_file_uri(
                                        deletion_file.file_path
                                    )
                                )

                                if num_rows is None or (
                                    deletion_vectors.get(i) == deletion_vector_path
                                ):
                                    # Not of this data file, or a deletion vector of
                                    # this data file in the same Puffin file.
                                    continue

                                if i in deletion_vectors:
                                    fallback_reason = "multiple deletion vectors associated with one data file"
                                    break

                                deletion_vectors[i] = deletion_vector_path
                                deletion_vector_num_rows += num_rows

                            case x:
                                fallback_reason = (
                                    f"unsupported deletion file format: {x}"
                                )
                                break

                    if i in deletion_vectors:
                        total_deleted_rows += deletion_vector_num_rows
                        total_deletion_vectors += 1
                        del position_delete_files[i]
                    elif not position_delete_files[i]:
                        del position_delete_files[i]
                    else:
                        total_deleted_rows += position_delete_num_rows
                        total_position_delete_files += len(position_delete_files[i])

                if fallback_reason:
                    break

                missing_field_defaults.push_partition_values(
                    current_index=i,
                    partition_spec_id=file_info.file.spec_id,
                    partition_values=file_info.file.partition,
                )

                if statistics_loader is not None:
                    statistics_loader.push_file_statistics(file_info.file)

                total_physical_rows += file_info.file.record_count

                sources.append(
                    _normalize_windows_iceberg_file_uri(file_info.file.file_path)
                )
                source_sizes.append(file_info.file.file_size_in_bytes)

            if verbose:
                elapsed = perf_counter() - start_time
                eprint(
                    "IcebergScanResolver: to_dataset_scan(): "
                    f"finish path expansion ({elapsed:.3f}s)"
                )

                if isinstance(scan.io, CachingFileIO):
                    eprint(
                        "IcebergScanResolver: to_dataset_scan(): "
                        "metadata file cache: "
                        f"hits: {scan.io.stats.hits}, "
                        f"misses: {scan.io.stats.misses}, "
                        f"cached bytes: {scan.io.cache.total_bytes}"
                    )

        if not fallback_reason:
            if verbose:
                eprint(
                    "IcebergScanResolver: to_dataset_scan(): "
                    f"native scan_parquet(): "
                    f"num_sources: {len(sources)}, "
                    f"snapshot ID: {snapshot_id}, "
                    f"schema ID: {schema_id}, "
                    f"num_position_delete_files: {total_position_delete_files}, "
                    f"num_deletion_vectors: {total_deletion_vectors}"
                )

            # The arrow schema returned by `schema_to_pyarrow` will contain
            # 'PARQUET:field_id'
            column_mapping = schema_to_pyarrow(iceberg_schema)

            identity_transformed_values = missing_field_defaults.finish()

            min_max_statistics = (
                statistics_loader.finish(len(sources), identity_transformed_values)
                if statistics_loader is not None
                else None
            )

            storage_options = (
                _convert_iceberg_to_object_store_storage_options(
                    self.table.iceberg_storage_properties
                )
                if self.table.iceberg_storage_properties is not None
                else None
            )

            return _NativeIcebergScanData(
                sources=sources,
                source_sizes=source_sizes,
                projected_iceberg_schema=projected_iceberg_schema,
                column_mapping=column_mapping,
                default_values=(identity_transformed_values, initial_defaults),
                position_delete_files=position_delete_files,
                deletion_vectors=deletion_vectors,
                min_max_statistics=min_max_statistics,
                statistics_loader=statistics_loader,
                storage_options=storage_options,
                row_count=(
                    (total_physical_rows, total_deleted_rows)
                    if (
                        self.use_metadata_statistics
                        and (self.fast_deletion_count or total_deleted_rows == 0)
                    )
                    else None
                ),
                snapshot_id_key=snapshot_id_key,
            )

        elif reader_override == "native":
            msg = f"iceberg reader_override='native' failed: {fallback_reason}"
            raise ComputeError(msg)

        if verbose:
            eprint(
                "IcebergScanResolver: to_dataset_scan(): "
                f"fallback to python[pyiceberg] scan: {fallback_reason}"
            )

        import polars.io.iceberg._utils

        func = partial(
            polars.io.iceberg._utils._scan_pyarrow_dataset_impl,
            # The scan is reused while the snapshot is unchanged, but PyIceberg
            # replaces the metadata of `tbl` on e.g. schema updates.
            copy.copy(tbl),
            snapshot_id=snapshot_id,
            from_snapshot_id_exclusive=self.from_snapshot_id_exclusive,
            to_snapshot_id_inclusive=self.to_snapshot_id_inclusive,
            n_rows=limit,
            with_columns=projection,
            iceberg_table_filter=(
                filter_for_pyiceberg_reader(iceberg_table_filter, iceberg_schema)
                if iceberg_table_filter is not None
                else None
            ),
        )

        arrow_schema = schema_to_pyarrow(tbl.schema())

        lf = pl.LazyFrame._scan_python_function(
            arrow_schema,
            func,
            pyarrow=True,
            is_pure=True,
        )

        return _PyIcebergScanData(lf=lf, snapshot_id_key=snapshot_id_key)


class _ResolvedScanDataBase(ABC):
    @abstractmethod
    def to_lazyframe(self) -> pl.LazyFrame: ...


@dataclass(kw_only=True)
class _NativeIcebergScanData(_ResolvedScanDataBase):
    """Resolved parameters for a native Iceberg scan."""

    sources: list[str]
    source_sizes: list[int]
    projected_iceberg_schema: pyiceberg.schema.Schema
    column_mapping: pa.Schema
    default_values: tuple[dict[int, pl.Series | str], dict[int, pl.Series]]
    position_delete_files: dict[int, list[str]]
    deletion_vectors: dict[int, str]
    min_max_statistics: pl.DataFrame | None
    # This is here for test purposes, as the `min_max_statistics` on this
    # dataclass can contain coalesced values from `default_values`. A test may
    # access the statistics loader directly to inspect the values before
    # coalescing.
    statistics_loader: IcebergStatisticsLoader | None
    storage_options: StorageOptionsDict | None
    # (physical, deleted)
    row_count: tuple[int, int] | None
    snapshot_id_key: str

    def to_lazyframe(self) -> pl.LazyFrame:
        from polars.io.parquet.functions import scan_parquet

        return scan_parquet(
            self.sources,
            glob=False,
            cast_options=ScanCastOptions._default_iceberg(),
            missing_columns="insert",
            extra_columns="ignore",
            storage_options=self.storage_options,
            _column_mapping=("iceberg-column-mapping", self.column_mapping),
            _default_values=("iceberg", self.default_values),
            _deletion_files=(
                "iceberg",
                (self.position_delete_files, self.deletion_vectors),
            ),
            _table_statistics=self.min_max_statistics,
            _row_count=self.row_count,
            _source_sizes=self.source_sizes,
        )


@dataclass(kw_only=True)
class _PyIcebergScanData(_ResolvedScanDataBase):
    """Resolved parameters for reading via PyIceberg."""

    # We're not interested in inspecting anything for the pyiceberg scan, so
    # this class is just a wrapper.
    lf: pl.LazyFrame
    snapshot_id_key: str

    def to_lazyframe(self) -> pl.LazyFrame:
        return self.lf


@dataclass(kw_only=True)
class _PluginIcebergScanData(_ResolvedScanDataBase):
    """Native Iceberg scan planned by the `polars_iceberg` plugin."""

    lf: pl.LazyFrame
    snapshot_id_key: str

    def to_lazyframe(self) -> pl.LazyFrame:
        return self.lf


def _read_puffin_deletion_vector_counts(
    io: Any, path: str, file_size: int
) -> dict[str, int] | None:
    """
    Deleted row counts of the deletion vectors of a Puffin file, by data file.

    Only the footer is read. Returns `None` if the footer is not supported.
    """
    import json

    # Footer: magic (4), payload, payload size (4, LE), flags (4), magic (4).
    with io.new_input(path).open() as f:
        f.seek(file_size - 12)
        tail = f.read(12)
        payload_size = int.from_bytes(tail[:4], "little", signed=True)
        # Flag bit 0: compressed footer payload.
        if len(tail) != 12 or tail[8:] != b"PFA1" or payload_size < 0 or tail[4] & 1:
            return None

        f.seek(file_size - 12 - payload_size)
        footer = json.loads(f.read(payload_size))

    try:
        return {
            _normalize_windows_iceberg_file_uri(
                blob["properties"]["referenced-data-file"]
            ): int(blob["properties"]["cardinality"])
            for blob in footer["blobs"]
            if blob["type"] == "deletion-vector-v1"
        }
    except (KeyError, TypeError, ValueError):
        return None


# Reserved field ID of `file_path` in position delete files.
_DELETE_FILE_PATH_FIELD_ID = 2147483546


def _referenced_data_file(deletion_file: Any) -> str | None:
    """The data file a position delete file holds deletes of, if it is only one."""
    if (path := getattr(deletion_file, "referenced_data_file", None)) is not None:
        return path

    lower = (deletion_file.lower_bounds or {}).get(_DELETE_FILE_PATH_FIELD_ID)
    upper = (deletion_file.upper_bounds or {}).get(_DELETE_FILE_PATH_FIELD_ID)
    if lower is None or lower != upper:
        return None
    try:
        return lower.decode() if isinstance(lower, bytes) else str(lower)
    except UnicodeDecodeError:
        return None


def _redact_dict_values(obj: Any) -> Any:
    return (
        dict.fromkeys(obj.keys(), "REDACTED")
        if isinstance(obj, dict)
        else f"<{type(obj).__name__} object>"
        if obj is not None
        else "None"
    )


def _convert_iceberg_to_object_store_storage_options(
    iceberg_storage_properties: dict[str, str],
) -> dict[str, str]:
    storage_options = {}

    # Allow-list for HDFS
    # See https://py.iceberg.apache.org/configuration/#hdfs
    HDFS_KEY_PREFIX = "hdfs."

    for k, v in iceberg_storage_properties.items():
        if (
            translated_key := ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP.get(k)
        ) is not None:
            storage_options[translated_key] = _convert_iceberg_property_value(
                translated_key, v
            )
        elif "." not in k or k.startswith(HDFS_KEY_PREFIX):
            # Pass-through non-Iceberg config keys, as they may be native config
            # keys. We identify Iceberg keys by checking for a dot - from
            # observation nearly all Iceberg config keys contain dots, whereas
            # native config keys do not contain them.
            storage_options[k] = v

        # Otherwise, unknown keys are ignored / not passed. This is to avoid
        # interfering with credential provider auto-init, which bails on
        # unknown keys.

    return storage_options


def _convert_iceberg_property_value(object_store_key: str, value: Any) -> Any:
    """Iceberg FileIO property value → object store config value."""
    # PyIceberg timeouts are (fractional) seconds, whereas object store parses
    # durations with a unit.
    if object_store_key in {"connect_timeout", "timeout"}:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return value
        return f"{round(seconds * 1000)}ms"

    return value


# https://py.iceberg.apache.org/configuration/#fileio
# This does not contain all keys - some have no object-store equivalent.
ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP: Final[dict[str, str]] = {
    # S3
    "s3.endpoint": "aws_endpoint_url",
    "s3.access-key-id": "aws_access_key_id",
    "s3.secret-access-key": "aws_secret_access_key",
    "s3.session-token": "aws_session_token",
    "s3.region": "aws_region",
    "s3.proxy-uri": "proxy_url",
    "s3.connect-timeout": "connect_timeout",
    "s3.request-timeout": "timeout",
    "s3.force-virtual-addressing": "aws_virtual_hosted_style_request",
    # Azure
    "adls.account-name": "azure_storage_account_name",
    "adls.account-key": "azure_storage_account_key",
    "adls.sas-token": "azure_storage_sas_key",
    "adls.tenant-id": "azure_storage_tenant_id",
    "adls.client-id": "azure_storage_client_id",
    "adls.client-secret": "azure_storage_client_secret",
    "adls.account-host": "azure_storage_authority_host",
    "adls.token": "azure_storage_token",
    # Google storage
    "gcs.oauth2.token": "bearer_token",
    # HuggingFace
    "hf.token": "token",
}

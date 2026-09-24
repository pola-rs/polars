from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
    from collections.abc import Callable, Iterable

    import pyarrow as pa
    import pyiceberg.catalog
    import pyiceberg.table
    from pyiceberg.io.pyarrow import (
        DataFileStatistics,
        MetricModeTypes,
        StatisticsCollector,
    )
    from pyiceberg.manifest import DataFile
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table import CommitTableResponse, Transaction
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.sorting import SortOrder
    from pyiceberg.table.update import TableRequirement, TableUpdate
    from pyiceberg.transforms import Transform
    from pyiceberg.typedef import Record
    from pyiceberg.types import IcebergType

    import polars as pl
    from polars._plr import PyLazyFrame
    from polars._typing import ParquetCompression, StorageOptionsDict


_IcebergSinkedFile = tuple[str, int, int, bytes]

_SINK_UUID_PROPERTY = "polars.sink-uuid"


class _AlreadyCommitted(Exception):
    """A snapshot tagged with this sink's uuid already landed.

    Deliberately not a `CommitFailedException` or `ValidationException`, so pyiceberg
    does not clean up this transaction's manifests when it propagates.
    """


def _already_committed(metadata: TableMetadata, sink_uuid: str) -> bool:
    return any(
        s.summary is not None and s.summary.get(_SINK_UUID_PROPERTY) == sink_uuid
        for s in metadata.snapshots
    )


class _SinkCatalog:
    """Delegates to `catalog`, but recovers the outcome of a failed `commit_table`.

    pyiceberg deletes a transaction's manifests when a commit fails, and does not
    always check first whether the commit landed (empty overwrite producers, zero
    retries, the final attempt). Resolving the outcome here, before pyiceberg sees
    the failure, keeps a landed snapshot's manifests.
    """

    def __init__(
        self, catalog: pyiceberg.catalog.Catalog, sink_uuid: str, *, verbose: bool
    ) -> None:
        self._catalog = catalog
        self._sink_uuid = sink_uuid
        self._verbose = verbose
        self._snapshot_ids: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._catalog, name)

    def commit_table(
        self,
        table: pyiceberg.table.Table,
        requirements: tuple[TableRequirement, ...],
        updates: tuple[TableUpdate, ...],
    ) -> CommitTableResponse:
        from pyiceberg.exceptions import CommitStateUnknownException
        from pyiceberg.table import CommitTableResponse
        from pyiceberg.table.update import AddSnapshotUpdate

        self._snapshot_ids.update(
            u.snapshot.snapshot_id for u in updates if isinstance(u, AddSnapshotUpdate)
        )

        try:
            return self._catalog.commit_table(table, requirements, updates)
        except Exception as e:
            try:
                current = self._catalog.load_table(table.name())
            except Exception:
                msg = "could not verify the outcome of a failed Iceberg commit"
                raise CommitStateUnknownException(msg) from e

            metadata = current.metadata
            if any(s.snapshot_id in self._snapshot_ids for s in metadata.snapshots):
                if self._verbose:
                    eprint("IcebergSinkState[commit]: recovered lost commit response")
                return CommitTableResponse(
                    metadata=metadata,
                    metadata_location=current.metadata_location,  # type: ignore[call-arg]
                )

            if _already_committed(metadata, self._sink_uuid):
                raise _AlreadyCommitted from e

            raise


def _sink_transaction(table: pyiceberg.table.Table, sink_uuid: str) -> Transaction:
    from pyiceberg.table import Transaction

    class _SinkTransaction(Transaction):
        # pyiceberg calls this after refreshing the table for a retry, before it
        # writes new manifests. A twin that lands after this check moves the branch
        # ref, so the next commit fails and `_SinkCatalog` finds the twin.
        def _rebuild_snapshot_updates(self) -> None:
            if _already_committed(self._table.metadata, sink_uuid):
                raise _AlreadyCommitted
            super()._rebuild_snapshot_updates()

    return _SinkTransaction(table)


def _nested_partition_source_ids(schema: Schema, spec: PartitionSpec) -> set[int]:
    return {
        field.source_id
        for field in spec.fields
        if schema.accessor_for_field(field.source_id).inner is not None
    }


def _iceberg_source_expr(schema: Schema, source_id: int) -> pl.Expr:
    from pyiceberg.types import StructType

    import polars as pl

    accessor = schema.accessor_for_field(source_id)
    source_field = schema.fields[accessor.position]
    expr = pl.col(source_field.name)

    while accessor.inner is not None:
        accessor = accessor.inner
        source_type = source_field.field_type
        if not isinstance(source_type, StructType):
            msg = f"Iceberg source field {source_id} has non-struct parent"
            raise TypeError(msg)
        source_field = source_type.fields[accessor.position]
        expr = expr.struct.field(source_field.name)

    return expr


def _infer_partition_from_statistics(
    statistics: DataFileStatistics, spec: PartitionSpec, schema: Schema
) -> Record:
    from pyiceberg.partitioning import partition_record_value
    from pyiceberg.typedef import Record

    partition_values: list[Any] = []
    for field in spec.fields:
        aggregate = statistics.column_aggregates.get(field.source_id)
        if aggregate is None:
            partition_values.append(None)
            continue

        source_type = schema.find_field(field.source_id).field_type
        transform = field.transform.transform(source_type)
        lower_value = transform(
            partition_record_value(field, aggregate.current_min, schema)
        )
        upper_value = transform(
            partition_record_value(field, aggregate.current_max, schema)
        )
        # A file can contain different source values in one transformed partition.
        if lower_value != upper_value:
            msg = (
                "Cannot infer partition value from Parquet metadata for partition "
                f"field '{field.name}': {lower_value=}, {upper_value=}"
            )
            raise ValueError(msg)
        partition_values.append(lower_value)

    return Record(*partition_values)


def _temporal_transform_expr(
    expr: pl.Expr, source_type: IcebergType, transform: Transform[Any, Any]
) -> pl.Expr:
    from pyiceberg.transforms import DayTransform, MonthTransform, YearTransform

    import polars as pl

    if type(source_type).__name__ in {"TimestamptzType", "TimestamptzNanoType"}:
        expr = expr.dt.convert_time_zone("UTC")

    if isinstance(transform, YearTransform):
        return expr.dt.year() - 1970
    if isinstance(transform, MonthTransform):
        return (expr.dt.year() - 1970) * 12 + expr.dt.month() - 1
    if isinstance(transform, DayTransform):
        return expr.cast(pl.Date).cast(pl.Int32)
    return expr.dt.epoch("us") // 3_600_000_000


def _data_files_from_sink_metadata(
    table_metadata: TableMetadata,
    sinked_files: list[_IcebergSinkedFile],
    nested_source_ids: set[int],
    sort_order_id: int | None,
) -> Iterable[DataFile]:
    from pyiceberg.io.pyarrow import (
        MetricModeTypes,
        MetricsMode,
        compute_statistics_plan,
        parquet_path_to_id_mapping,
    )
    from pyiceberg.utils.concurrent import ExecutorFactory

    schema = table_metadata.schema()
    statistics_plan = compute_statistics_plan(schema, table_metadata.properties)
    nested_metrics_modes = {}
    for source_id in nested_source_ids:
        nested_metrics_modes[source_id] = statistics_plan[source_id].mode.type
        # Nested bounds are needed temporarily to infer the partition value.
        statistics_plan[source_id] = replace(
            statistics_plan[source_id],
            mode=MetricsMode(MetricModeTypes.FULL),
        )
    parquet_column_mapping = parquet_path_to_id_mapping(schema)

    executor = ExecutorFactory.get_or_create()
    futures = [
        executor.submit(
            _data_file_from_sink_metadata,
            table_metadata,
            sinked_file,
            statistics_plan,
            parquet_column_mapping,
            nested_metrics_modes,
            sort_order_id,
        )
        for sinked_file in sinked_files
    ]
    return [future.result() for future in futures]


def _data_file_from_sink_metadata(
    table_metadata: TableMetadata,
    sinked_file: _IcebergSinkedFile,
    statistics_plan: dict[int, StatisticsCollector],
    parquet_column_mapping: dict[str, int],
    nested_metrics_modes: dict[int, MetricModeTypes],
    sort_order_id: int | None,
) -> DataFile:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import (
        MetricModeTypes,
        _check_pyarrow_schema_compatible,
        data_file_statistics_from_parquet_metadata,
    )
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat

    schema = table_metadata.schema()
    file_path, num_rows, num_bytes, parquet_metadata_bytes = sinked_file
    parquet_metadata = pq.read_metadata(pa.BufferReader(parquet_metadata_bytes))

    if parquet_metadata.num_rows != num_rows:
        msg = (
            f"native sink row count {num_rows} does not match Parquet metadata "
            f"row count {parquet_metadata.num_rows} for '{file_path}'"
        )
        raise ValueError(msg)

    _check_pyarrow_schema_compatible(schema, parquet_metadata.schema.to_arrow_schema())
    statistics = data_file_statistics_from_parquet_metadata(
        parquet_metadata=parquet_metadata,
        stats_columns=statistics_plan,
        parquet_column_mapping=parquet_column_mapping,
    )
    partition = _infer_partition_from_statistics(
        statistics, table_metadata.spec(), schema
    )
    serialized_statistics = statistics.to_serialized_dict()
    for source_id, metrics_mode in nested_metrics_modes.items():
        serialized_statistics["lower_bounds"].pop(source_id, None)
        serialized_statistics["upper_bounds"].pop(source_id, None)
        if metrics_mode is MetricModeTypes.NONE:
            serialized_statistics["value_counts"].pop(source_id, None)
            serialized_statistics["null_value_counts"].pop(source_id, None)
            serialized_statistics["nan_value_counts"].pop(source_id, None)

    data_file_args = {
        "content": DataFileContent.DATA,
        "file_path": file_path,
        "file_format": FileFormat.PARQUET,
        "partition": partition,
        "file_size_in_bytes": num_bytes,
        "sort_order_id": sort_order_id,
        "spec_id": table_metadata.default_spec_id,
        "equality_ids": None,
        "key_metadata": None,
        **serialized_statistics,
    }
    factory = getattr(DataFile, "from_args", None)
    if factory is None:
        return DataFile(**data_file_args)
    return factory(**data_file_args)


def _add_files(
    transaction: Transaction,
    sinked_files: list[_IcebergSinkedFile],
    snapshot_properties: dict[str, str],
    sort_order_id: int | None,
) -> None:
    from pyiceberg.table import TableProperties

    table_metadata = transaction.table_metadata
    nested_source_ids = _nested_partition_source_ids(
        table_metadata.schema(), table_metadata.spec()
    )
    if table_metadata.name_mapping() is None:
        transaction.set_properties(
            {
                TableProperties.DEFAULT_NAME_MAPPING: table_metadata.schema().name_mapping.model_dump_json()
            }
        )

    with transaction.update_snapshot(
        snapshot_properties=snapshot_properties
    ).fast_append() as append_files:
        for data_file in _data_files_from_sink_metadata(
            table_metadata,
            sinked_files,
            nested_source_ids,
            sort_order_id,
        ):
            append_files.append_data_file(data_file)


def _partition_key_exprs(
    table: pyiceberg.table.Table, source_schema: pa.Schema | None = None
) -> list[pl.Expr] | None:
    spec = table.spec()

    if not spec.fields:
        return None

    from pyiceberg.io.pyarrow import MetricModeTypes, compute_statistics_plan
    from pyiceberg.transforms import (
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )
    from pyiceberg.types import BinaryType, IntegerType, LongType, StringType

    schema = table.schema()
    nested_source_ids = _nested_partition_source_ids(schema, spec)
    statistics_plan = compute_statistics_plan(schema, table.metadata.properties)
    bounds_metrics_modes = {MetricModeTypes.TRUNCATE, MetricModeTypes.FULL}
    reserved_names = {field.name for field in schema.fields}
    if source_schema is not None:
        reserved_names.update(source_schema.names)
    exprs: list[pl.Expr] = []

    for field in spec.fields:
        source_field = schema.find_field(field.source_id)
        statistics = statistics_plan.get(field.source_id)
        if field.source_id not in nested_source_ids and (
            statistics is None or statistics.mode.type not in bounds_metrics_modes
        ):
            source_name = schema.find_column_name(field.source_id)
            metrics_mode = (
                statistics.mode.type.value if statistics is not None else "unavailable"
            )
            msg = (
                "sink to Iceberg table with partition field "
                f"'{field.name}' on source column '{source_name}' with "
                f"'{metrics_mode}' metrics; partition value inference requires "
                "lower and upper bounds"
            )
            raise NotImplementedError(msg)

        source_type = source_field.field_type
        transform = field.transform
        expr = _iceberg_source_expr(schema, field.source_id)

        if isinstance(transform, IdentityTransform):
            pass
        elif isinstance(
            transform, (YearTransform, MonthTransform, DayTransform, HourTransform)
        ):
            expr = _temporal_transform_expr(expr, source_type, transform)
        elif isinstance(transform, TruncateTransform):
            if isinstance(source_type, (IntegerType, LongType)):
                expr = expr - expr % transform.width
            elif isinstance(source_type, StringType):
                expr = expr.str.slice(0, transform.width)
            elif isinstance(source_type, BinaryType):
                expr = expr.bin.slice(0, transform.width)
            else:
                msg = (
                    "sink to Iceberg table with "
                    f"'{transform}' partition transform on '{source_type}'"
                )
                raise NotImplementedError(msg)
        else:
            msg = f"sink to Iceberg table with '{transform}' partition transform"
            raise NotImplementedError(msg)

        key_name = f"__POLARS_ICEBERG_PARTITION_{field.field_id}"
        while key_name in reserved_names:
            key_name += "_"
        reserved_names.add(key_name)
        exprs.append(expr.alias(key_name))

    return exprs


class _SortTransform:
    """Apply an Iceberg sort transform without pickling its cached callable."""

    def __init__(
        self, transform: Transform[Any, Any], source_type: IcebergType
    ) -> None:
        self.transform = transform
        self.source_type = source_type
        self.resolved: NoPickleOption[
            tuple[Callable[[pa.Array], pa.Array], pa.DataType]
        ] = NoPickleOption()

    def __call__(self, series: pl.Series) -> pl.Series:
        import polars as pl

        if (resolved := self.resolved.get()) is None:
            from pyiceberg.io.pyarrow import schema_to_pyarrow

            resolved = (
                self.transform.pyarrow_transform(self.source_type),
                schema_to_pyarrow(self.source_type),
            )
            self.resolved.set(resolved)

        transform, dtype = resolved
        return pl.Series(series.name, transform(series.to_arrow().cast(dtype)))


def _sort_key_exprs(
    schema: Schema, sort_order: SortOrder, input_schema: pl.Schema
) -> tuple[list[pl.Expr], list[bool], list[bool]]:
    from pyiceberg.table.sorting import NullOrder, SortDirection
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        VoidTransform,
        YearTransform,
    )
    from pyiceberg.types import (
        BinaryType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        StructType,
    )

    import polars as pl

    exprs: list[pl.Expr] = []
    descending: list[bool] = []
    nulls_last: list[bool] = []
    sort_order.check_compatible(schema)

    for field in sort_order.fields:
        accessor = schema.accessor_for_field(field.source_id)
        source_field = schema.fields[accessor.position]
        input_type = input_schema.get(source_field.name)
        while input_type is not None and accessor.inner is not None:
            accessor = accessor.inner
            source_struct = source_field.field_type
            assert isinstance(source_struct, StructType)
            source_field = source_struct.fields[accessor.position]
            input_type = (
                pl.Schema(dict(input_type)).get(source_field.name)
                if isinstance(input_type, pl.Struct)
                else None
            )
        if input_type is None or input_type == pl.Null:
            continue

        source_type = schema.find_field(field.source_id).field_type
        transform = field.transform
        expr = _iceberg_source_expr(schema, field.source_id)

        if isinstance(transform, IdentityTransform):
            if isinstance(source_type, (FloatType, DoubleType)):
                value = expr.cast(pl.Float64)
                bits = value.reinterpret(dtype=pl.Int64)
                # Iceberg distinguishes signed zero and places negative NaNs first.
                bits = (
                    pl.when(value.is_nan())
                    .then(
                        pl.when(bits < 0).then(-(1 << 51)).otherwise(0x7FF8000000000000)
                    )
                    .otherwise(bits)
                )
                expr = bits ^ pl.when(bits < 0).then((1 << 63) - 1).otherwise(0)
        elif isinstance(
            transform, (YearTransform, MonthTransform, DayTransform, HourTransform)
        ):
            expr = _temporal_transform_expr(expr, source_type, transform)
        elif isinstance(transform, TruncateTransform):
            if isinstance(source_type, (IntegerType, LongType, DecimalType)):
                # The quotient preserves ordering and ties without truncation overflow.
                expr = expr.to_physical() // transform.width
            elif isinstance(source_type, StringType):
                expr = expr.str.slice(0, transform.width)
            elif isinstance(source_type, BinaryType):
                expr = expr.bin.slice(0, transform.width)
            else:
                msg = f"Iceberg sort transform '{transform}' on '{source_type}'"
                raise NotImplementedError(msg)
        elif isinstance(transform, BucketTransform):
            expr = expr.map_batches(
                _SortTransform(transform, source_type),
                return_dtype=pl.Int32,
                is_elementwise=True,
            )
        elif isinstance(transform, VoidTransform):
            continue
        else:
            msg = f"Iceberg sort transform '{transform}'"
            raise NotImplementedError(msg)

        exprs.append(expr)
        descending.append(field.direction is SortDirection.DESC)
        nulls_last.append(field.null_order is NullOrder.NULLS_LAST)

    return exprs, descending, nulls_last


@dataclass(kw_only=True)
class IcebergSinkState:
    py_catalog_class_module: str
    py_catalog_class_qualname: str

    catalog_name: str
    catalog_properties: dict[str, str]

    table_name: str
    mode: Literal["append", "overwrite"]
    schema_mode: Literal["merge", "overwrite"] | None
    snapshot_properties: dict[str, str]
    iceberg_storage_properties: StorageOptionsDict
    compression: ParquetCompression
    compression_level: int | None
    row_group_size: int | None
    maintain_order: bool

    sink_uuid_str: str

    table_: NoPickleOption[pyiceberg.table.Table]
    source_schema: pa.Schema | None
    sort_order_id: int | None
    commit_result_df: NoPickleOption[pl.DataFrame]

    @staticmethod
    def new(
        target: str | pyiceberg.table.Table,
        *,
        mode: Literal["append", "overwrite"] = "append",
        schema_mode: Literal["merge", "overwrite"] | None = None,
        snapshot_properties: dict[str, str] | None = None,
        catalog: pyiceberg.catalog.Catalog | IcebergCatalogConfig | None = None,
        storage_options: StorageOptionsDict | None = None,
        compression: ParquetCompression = "zstd",
        compression_level: int | None = None,
        row_group_size: int | None = None,
        maintain_order: bool = True,
    ) -> IcebergSinkState:
        if schema_mode == "overwrite" and mode != "overwrite":
            msg = "schema_mode='overwrite' requires mode='overwrite'"
            raise ValueError(msg)

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
            schema_mode=schema_mode,
            snapshot_properties=snapshot_properties or {},
            iceberg_storage_properties=storage_options or {},
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            maintain_order=maintain_order,
            sink_uuid_str=gen_uuid_v7().hex(),
            table_=NoPickleOption(target if not isinstance(target, str) else None),
            source_schema=None,
            sort_order_id=None,
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
        if self.schema_mode is not None:
            self.source_schema = lf.collect_schema().to_arrow()
        return wrap_ldf(lf._ldf.sink_iceberg(self))

    def _get_source_schema(self) -> pa.Schema:
        assert self.source_schema is not None
        return self.source_schema

    def _update_schema(self, transaction: Transaction) -> None:
        if self.schema_mode == "merge":
            with transaction.update_schema() as update:
                update.union_by_name(self._get_source_schema())
        elif self.schema_mode == "overwrite":
            with transaction.update_schema(allow_incompatible_changes=True) as update:
                update.set_identifier_fields()
                for field in transaction.table_metadata.schema().fields:
                    update.delete_column(field.name)

            with transaction.update_schema(allow_incompatible_changes=True) as update:
                update.union_by_name(self._get_source_schema())

    def _schemas_for_write(
        self, table: pyiceberg.table.Table
    ) -> tuple[Schema, pa.Schema]:
        from pyiceberg.io.pyarrow import pyarrow_to_schema, schema_to_pyarrow

        if self.schema_mode is None:
            return table.schema(), schema_to_pyarrow(table.schema())

        transaction = table.transaction()
        self._update_schema(transaction)
        evolved_schema = transaction.table_metadata.schema()
        source_schema = pyarrow_to_schema(
            self._get_source_schema(), name_mapping=evolved_schema.name_mapping
        )
        return evolved_schema, schema_to_pyarrow(source_schema)

    def _attach_resolved_sink(self, plf: PyLazyFrame) -> PyLazyFrame:
        from pyiceberg.table import TableProperties
        from pyiceberg.utils.properties import property_as_bool, property_as_int

        import polars as pl

        table = self.table()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        if self.schema_mode == "overwrite" and table.spec().fields:
            msg = "schema_mode='overwrite' is not supported for partitioned Iceberg tables"
            raise NotImplementedError(msg)

        partition_key_exprs = _partition_key_exprs(table, self.source_schema)

        sort_order = table.sort_order()
        if sort_order.fields and self.schema_mode == "overwrite":
            msg = "schema_mode='overwrite' is not supported for Iceberg tables with a sort order"
            raise NotImplementedError(msg)

        if location_provider_impl := table_properties.get(
            TableProperties.WRITE_PY_LOCATION_PROVIDER_IMPL
        ):
            msg = (
                "sink to Iceberg table with custom location provider"
                f" '{location_provider_impl}'"
            )
            raise NotImplementedError(msg)

        object_storage_enabled = property_as_bool(
            table_properties,
            TableProperties.OBJECT_STORE_ENABLED,
            TableProperties.OBJECT_STORE_ENABLED_DEFAULT,
        )
        object_storage_partitioned_paths = (
            property_as_bool(
                table_properties,
                TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS,
                TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS_DEFAULT,
            )
            if object_storage_enabled
            else None
        )

        schema, arrow_schema = self._schemas_for_write(table)
        lf = wrap_ldf(plf)
        self.sort_order_id = None
        if sort_order.fields:
            exprs, descending, nulls_last = _sort_key_exprs(
                schema, sort_order, lf.collect_schema()
            )
            if exprs:
                lf = lf.sort(
                    exprs,
                    descending=descending,
                    nulls_last=nulls_last,
                    maintain_order=self.maintain_order,
                )
            self.sort_order_id = sort_order.order_id

        approximate_bytes_per_file = 2 * 1024 * 1024 * 1024

        if v := property_as_int(
            properties=table_metadata.properties,
            property_name=TableProperties.WRITE_TARGET_FILE_SIZE_BYTES,
        ):
            estimated_compression_ratio = 4
            approximate_bytes_per_file = min(
                estimated_compression_ratio * v, (1 << 64) - 1
            )

        return lf.sink_parquet(
            pl.PartitionBy(
                _normalize_windows_iceberg_file_uri(
                    self.sink_base_path(object_storage_enabled=object_storage_enabled)
                ),
                file_path_provider=PlIcebergPathProviderConfig(
                    object_storage_partitioned_paths=object_storage_partitioned_paths
                ),
                key=partition_key_exprs,
                include_key=False if partition_key_exprs is not None else None,
                approximate_bytes_per_file=approximate_bytes_per_file,
            ),
            arrow_schema=arrow_schema,
            compression=self.compression,
            compression_level=self.compression_level,
            row_group_size=self.row_group_size,
            maintain_order=True if sort_order.fields else self.maintain_order,
            storage_options=self._get_converted_storage_options(),
            lazy=True,
        )._ldf

    def commit(self, sinked_files: list[_IcebergSinkedFile]) -> pl.DataFrame:
        """Commit the sinked files to the table.

        A re-execution of the same sink invocation (same `sink_uuid_str`) is skipped
        while the snapshot it tagged is retained. It commits again once that snapshot
        has expired, or if the table is rolled back while the re-execution is in
        flight: no catalog requirement can detect the latter.
        """
        import polars as pl
        import polars._utils.logging

        function_start_instant = perf_counter()
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(f"IcebergSinkState[commit]: mode: '{self.mode}'")

        table = self.table()

        if _already_committed(table.metadata, self.sink_uuid_str):
            if verbose:
                eprint("IcebergSinkState[commit]: already committed, skipping")
        else:
            self._commit(table, sinked_files, verbose=verbose)

        self.commit_result_df.set(
            pl.DataFrame(
                {"metadata_path": table.metadata_location},
                schema={"metadata_path": pl.String},
                height=1,
            )
        )

        if verbose:
            total_elapsed = perf_counter() - function_start_instant
            eprint(
                f"IcebergSinkState[commit]: finished, total elapsed time: {total_elapsed:.3f}s"
            )

        return self.commit_result_df.get()  # type: ignore[return-value]

    def _commit(
        self,
        table: pyiceberg.table.Table,
        sinked_files: list[_IcebergSinkedFile],
        *,
        verbose: bool,
    ) -> None:
        from pyiceberg.table import Table

        original_metadata_location = table.metadata_location
        snapshot_properties = self.snapshot_properties | {
            _SINK_UUID_PROPERTY: self.sink_uuid_str
        }

        if sys.platform == "win32":
            sinked_files = [
                (
                    f"file://{path[8:]}" if path.startswith("file:///") else path,
                    num_rows,
                    num_bytes,
                    parquet_metadata,
                )
                for path, num_rows, num_bytes, parquet_metadata in sinked_files
            ]

        wrapped = Table(
            table.name(),
            table.metadata,
            table.metadata_location,
            table.io,
            _SinkCatalog(table.catalog, self.sink_uuid_str, verbose=verbose),  # type: ignore[arg-type]
            table.config,
        )

        try:
            with _sink_transaction(wrapped, self.sink_uuid_str) as tx:
                if (
                    self.sort_order_id is not None
                    and tx.table_metadata.sort_order_by_id(self.sort_order_id) is None
                ):
                    msg = f"Iceberg sort order {self.sort_order_id} is no longer available"
                    raise ValueError(msg)
                self._update_schema(tx)

                if self.mode == "overwrite":
                    from pyiceberg.expressions import AlwaysTrue

                    tx.delete(AlwaysTrue(), snapshot_properties=snapshot_properties)

                if verbose:
                    eprint("IcebergSinkState[commit]: begin add_files")

                start_instant = perf_counter()

                _add_files(
                    tx,
                    sinked_files,
                    snapshot_properties,
                    self.sort_order_id,
                )

                if verbose:
                    elapsed = perf_counter() - start_instant
                    eprint(
                        f"IcebergSinkState[commit]: finish add_files ({elapsed:.3f}s)"
                    )
                    eprint("IcebergSinkState[commit]: begin transaction commit")

                start_instant = perf_counter()
        except _AlreadyCommitted:
            if verbose:
                eprint("IcebergSinkState[commit]: already committed, skipping")
            table.refresh()
            return

        if verbose:
            elapsed = perf_counter() - start_instant
            eprint(
                f"IcebergSinkState[commit]: finish transaction commit ({elapsed:.3f}s)"
            )

        table.metadata = wrapped.metadata
        table.metadata_location = wrapped.metadata_location
        table.io = wrapped.io
        table.config = wrapped.config

        assert table.metadata_location != original_metadata_location

    def sink_base_path(self, *, object_storage_enabled: bool) -> str:
        from pyiceberg.table import TableProperties

        table = self.table()
        table_metadata = table.metadata
        table_properties = table_metadata.properties

        sink_base_path = (
            path.rstrip("/")
            if (path := table_properties.get(TableProperties.WRITE_DATA_PATH))
            else f"{table_metadata.location.rstrip('/')}/data"
        )

        if object_storage_enabled:
            return f"{sink_base_path}/"

        return f"{sink_base_path}/{self.sink_uuid_str}/"


@dataclass(frozen=True, kw_only=True)
class PlIcebergPathProviderConfig(_InternalPlPathProviderConfig):
    pl_path_provider_id: ClassVar[str] = "iceberg"
    extension: ClassVar[Literal["parquet"]] = "parquet"
    object_storage_partitioned_paths: bool | None = None

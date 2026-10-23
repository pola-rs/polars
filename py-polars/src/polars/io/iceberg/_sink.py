from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from polars._utils.logging import eprint
from polars._utils.wrap import wrap_ldf
from polars.exceptions import ComputeError
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    _convert_iceberg_to_object_store_storage_options,
)
from polars.io.iceberg._utils import (
    _is_null_pushdown_safe,
    _normalize_windows_iceberg_file_uri,
    predicate_to_pyiceberg_delete_filter,
)
from polars.io.partition import _InternalPlPathProviderConfig

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import gen_uuid_v7

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa
    import pyiceberg.catalog
    import pyiceberg.table
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io.pyarrow import StatisticsCollector
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
    from polars._typing import (
        ParquetCompression,
        PolarsDataType,
        StorageOptionsDict,
    )


# `(path, num_rows, num_bytes, parquet_metadata, partition_keys)`
_IcebergSinkedFile = tuple[str, int, int, bytes, "pl.DataFrame | None"]

_SINK_UUID_PROPERTY = "polars.sink-uuid"


class _AlreadyCommitted(Exception):
    """A snapshot tagged with this sink's uuid already landed.

    Deliberately not a `CommitFailedException` or `ValidationException`, so pyiceberg
    does not clean up this transaction's manifests when it propagates.
    """

    def __init__(self, table: pyiceberg.table.Table) -> None:
        self.table = table


def _create_namespace_if_not_exists(
    catalog: pyiceberg.catalog.Catalog,
    namespace: tuple[str, ...],
    *,
    create_parents: bool = True,
) -> None:
    """Create `namespace`, and any missing parent namespaces."""
    try:
        # Tolerates a concurrent create of the same namespace.
        catalog.create_namespace_if_not_exists(namespace)
    except Exception:
        # The create may have landed with its response lost.
        with contextlib.suppress(Exception):
            if catalog.namespace_exists(namespace):
                return

        # The parent namespace may be missing. Catalogs report this differently
        # (e.g. REST raises a `RESTError` rather than `NoSuchNamespaceError`), so
        # check for it directly.
        parent_missing = False
        if create_parents and len(namespace) > 1:
            with contextlib.suppress(Exception):
                parent_missing = not catalog.namespace_exists(namespace[:-1])

        if not parent_missing:
            raise
    else:
        return

    _create_namespace_if_not_exists(catalog, namespace[:-1])
    _create_namespace_if_not_exists(catalog, namespace, create_parents=False)


def _copy_table_state(src: pyiceberg.table.Table, dst: pyiceberg.table.Table) -> None:
    # Same check as `Table.refresh`: a dropped and recreated table must not take
    # over the caller's table.
    if src.metadata.table_uuid != dst.metadata.table_uuid:
        msg = (
            "Table UUID does not match: "
            f"current={dst.metadata.table_uuid} != refreshed={src.metadata.table_uuid}"
        )
        raise ValueError(msg)
    dst.metadata = src.metadata
    dst.metadata_location = src.metadata_location
    dst.io = src.io
    dst.config = src.config


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
                raise _AlreadyCommitted(current) from e

            raise


def _sink_transaction(table: pyiceberg.table.Table, sink_uuid: str) -> Transaction:
    from pyiceberg.table import Transaction

    class _SinkTransaction(Transaction):  # type: ignore[misc]
        # pyiceberg calls this after refreshing the table for a retry, before it
        # writes new manifests. A twin that lands after this check moves the branch
        # ref, so the next commit fails and `_SinkCatalog` finds the twin.
        def _rebuild_snapshot_updates(self) -> None:
            if _already_committed(self._table.metadata, sink_uuid):
                raise _AlreadyCommitted(self._table)
            super()._rebuild_snapshot_updates()

    return _SinkTransaction(table)


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


def _partition_record(
    partition_keys: pl.DataFrame | None, spec: PartitionSpec, schema: Schema
) -> Record:
    """
    The partition of a sinked file, from its partition keys (`_partition_key_exprs`).

    Parquet statistics cannot give it: they are truncated for long values, absent for
    NaN, and do not distinguish signed zeros.
    """
    import struct

    from pyiceberg.partitioning import partition_record_value
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.typedef import Record
    from pyiceberg.types import DoubleType, FloatType

    if not spec.fields:
        return Record()

    if partition_keys is None or partition_keys.shape != (1, len(spec.fields)):
        msg = (
            "Iceberg sink: partition keys of a file do not match the partition spec: "
            f"{None if partition_keys is None else partition_keys.schema}"
        )
        raise ComputeError(msg)

    partition_values: list[Any] = []
    for field, value in zip(spec.fields, partition_keys.row(0), strict=True):
        source_type = schema.find_field(field.source_id).field_type
        if value is None or not isinstance(field.transform, IdentityTransform):
            # Transforms give partition values.
            pass
        elif isinstance(source_type, (FloatType, DoubleType)):
            # The key holds the bits, to keep signed zeros apart.
            value = struct.unpack("<d", struct.pack("<q", value))[0]
        elif not isinstance(value, int):
            # Temporal keys are already in the partition representation.
            value = partition_record_value(field, value, schema)
        partition_values.append(value)

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


def _bucket_transform_expr(
    expr: pl.Expr, source_type: IcebergType, num_buckets: int
) -> pl.Expr | None:
    """Iceberg `bucket[num_buckets]` transform, or None for an unsupported type."""
    from pyiceberg.types import (
        BinaryType,
        DateType,
        DecimalType,
        FixedType,
        IntegerType,
        LongType,
        StringType,
        TimeType,
        UUIDType,
    )

    import polars as pl

    # The hash depends on the representation, so first cast to the source type.
    if isinstance(source_type, (IntegerType, LongType)):
        expr = expr.cast(pl.Int64)
    elif isinstance(source_type, DateType):
        expr = expr.cast(pl.Date)
    elif isinstance(source_type, TimeType):
        expr = expr.cast(pl.Time)
    elif isinstance(source_type, DecimalType):
        expr = expr.cast(pl.Decimal(source_type.precision, source_type.scale))
    elif isinstance(source_type, StringType):
        expr = expr.cast(pl.String)
    elif isinstance(source_type, (BinaryType, FixedType, UUIDType)):
        expr = expr.cast(pl.Binary)
    elif type(source_type).__name__ not in {
        "TimestampType",
        "TimestamptzType",
        "TimestampNanoType",
        "TimestamptzNanoType",
    }:
        # Timestamps of any time unit are hashed as microseconds since the epoch.
        return None

    return (expr._mm3_hash() & 0x7FFFFFFF) % num_buckets


def _data_files_from_sink_metadata(
    table_metadata: TableMetadata,
    sinked_files: list[_IcebergSinkedFile],
    sort_order_id: int | None,
) -> Iterable[DataFile]:
    from pyiceberg.io.pyarrow import (
        compute_statistics_plan,
        parquet_path_to_id_mapping,
    )
    from pyiceberg.utils.concurrent import ExecutorFactory

    schema = table_metadata.schema()
    statistics_plan = compute_statistics_plan(schema, table_metadata.properties)
    parquet_column_mapping = parquet_path_to_id_mapping(schema)

    executor = ExecutorFactory.get_or_create()
    futures = [
        executor.submit(
            _data_file_from_sink_metadata,
            table_metadata,
            sinked_file,
            statistics_plan,
            parquet_column_mapping,
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
    sort_order_id: int | None,
) -> DataFile:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.io.pyarrow import (
        _check_pyarrow_schema_compatible,
        data_file_statistics_from_parquet_metadata,
    )
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat

    schema = table_metadata.schema()
    file_path, num_rows, num_bytes, parquet_metadata_bytes, partition_keys = sinked_file
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
    partition = _partition_record(partition_keys, table_metadata.spec(), schema)
    serialized_statistics = statistics.to_serialized_dict()

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


def _data_files_for_commit(
    transaction: Transaction,
    sinked_files: list[_IcebergSinkedFile],
    sort_order_id: int | None,
) -> list[DataFile]:
    return list(
        _data_files_from_sink_metadata(
            transaction.table_metadata,
            sinked_files,
            sort_order_id,
        )
    )


def _dynamic_overwrite_filter(
    table_metadata: TableMetadata, data_files: list[DataFile]
) -> BooleanExpression:
    """Match the rows of the partitions that `data_files` are written to."""
    from pyiceberg.expressions import AlwaysFalse, And, EqualTo, IsNull, Or, Reference

    spec = table_metadata.spec()
    schema = table_metadata.schema()
    _check_dynamic_overwrite_spec(spec, schema)
    names = [schema.find_column_name(field.source_id) for field in spec.fields]

    partition_filters: list[BooleanExpression] = []
    for partition in {data_file.partition for data_file in data_files}:
        predicates: list[BooleanExpression] = []
        for i, name in enumerate(names):
            assert name is not None
            value = partition[i]
            if value is None:
                predicates.append(IsNull(Reference(name)))  # type: ignore[call-arg]
            else:
                predicates.append(EqualTo(Reference(name), value))  # type: ignore[call-arg]
        partition_filters.append(
            And(*predicates) if len(predicates) > 1 else predicates[0]
        )

    if not partition_filters:
        return AlwaysFalse()
    if len(partition_filters) == 1:
        return partition_filters[0]
    return Or(*partition_filters)


def _check_dynamic_overwrite_spec(spec: PartitionSpec, schema: Schema) -> None:
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType, FloatType

    if spec.is_unpartitioned():
        msg = "partition_overwrite='dynamic' requires a partitioned Iceberg table"
        raise ValueError(msg)
    for field in spec.fields:
        if not isinstance(field.transform, IdentityTransform):
            msg = (
                "partition_overwrite='dynamic' only supports identity partition "
                f"transforms, got {field.transform} for partition field '{field.name}'"
            )
            raise NotImplementedError(msg)
        source_type = schema.find_field(field.source_id).field_type
        # NaN partition values cannot be matched with `EqualTo`.
        if isinstance(source_type, (FloatType, DoubleType)):
            msg = (
                "partition_overwrite='dynamic' does not support floating-point "
                f"partition source columns (partition field '{field.name}')"
            )
            raise NotImplementedError(msg)
        # PyIceberg has no nanosecond literals.
        if type(source_type).__name__ in {"TimestampNanoType", "TimestamptzNanoType"}:
            msg = (
                "partition_overwrite='dynamic' does not support nanosecond "
                f"timestamp partition source columns (partition field '{field.name}')"
            )
            raise NotImplementedError(msg)
        # A null optional ancestor makes the source column null, which
        # `IsNull` does not match.
        source_name = schema.find_column_name(field.source_id)
        assert source_name is not None
        if not _is_null_pushdown_safe(source_name.split("."), schema):
            msg = (
                "partition_overwrite='dynamic' does not support partition source "
                f"columns nested within an optional struct (partition field "
                f"'{field.name}')"
            )
            raise NotImplementedError(msg)


def _add_files(
    transaction: Transaction,
    data_files: list[DataFile],
    snapshot_properties: dict[str, str],
) -> None:
    from pyiceberg.table import TableProperties

    table_metadata = transaction.table_metadata
    if table_metadata.name_mapping() is None:
        transaction.set_properties(
            {
                TableProperties.DEFAULT_NAME_MAPPING: table_metadata.schema().name_mapping.model_dump_json()
            }
        )

    with transaction.update_snapshot(
        snapshot_properties=snapshot_properties
    ).fast_append() as append_files:
        for data_file in data_files:
            append_files.append_data_file(data_file)


def _partition_key_exprs(
    spec: PartitionSpec, schema: Schema, source_schema: pa.Schema | None = None
) -> list[pl.Expr] | None:
    """Partition keys of the data, written with `schema`."""
    if not spec.fields:
        return None

    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )
    from pyiceberg.types import (
        BinaryType,
        DateType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        TimeType,
    )

    import polars as pl

    reserved_names = {field.name for field in schema.fields}
    if source_schema is not None:
        reserved_names.update(source_schema.names)
    exprs: list[pl.Expr] = []

    for field in spec.fields:
        source_field = schema.find_field(field.source_id)
        source_type = source_field.field_type
        transform = field.transform
        expr = _iceberg_source_expr(schema, field.source_id)

        # The keys are the partition values (see `_partition_record`).
        if isinstance(transform, IdentityTransform):
            if isinstance(source_type, (FloatType, DoubleType)):
                # Iceberg keeps signed zeros apart, Polars' grouping does not. NaNs
                # are one partition.
                value = expr.cast(pl.Float64)
                expr = (
                    pl.when(value.is_nan())
                    .then(pl.lit(0x7FF8000000000000, pl.Int64))
                    .otherwise(value.reinterpret(dtype=pl.Int64))
                )
            elif isinstance(source_type, DateType):
                expr = expr.cast(pl.Date).to_physical()
            elif isinstance(source_type, TimeType):
                expr = expr.cast(pl.Time).to_physical() // 1000
            elif type(source_type).__name__ in {
                "TimestampType",
                "TimestamptzType",
            }:
                expr = expr.dt.cast_time_unit("us").to_physical()
            elif type(source_type).__name__ in {
                "TimestampNanoType",
                "TimestamptzNanoType",
            }:
                expr = expr.dt.cast_time_unit("ns").to_physical()
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
        elif isinstance(transform, BucketTransform):
            bucket = _bucket_transform_expr(expr, source_type, transform.num_buckets)
            if bucket is None:
                msg = (
                    "sink to Iceberg table with "
                    f"'{transform}' partition transform on '{source_type}'"
                )
                raise NotImplementedError(msg)
            expr = bucket
        else:
            msg = f"sink to Iceberg table with '{transform}' partition transform"
            raise NotImplementedError(msg)

        key_name = f"__POLARS_ICEBERG_PARTITION_{field.field_id}"
        while key_name in reserved_names:
            key_name += "_"
        reserved_names.add(key_name)
        exprs.append(expr.alias(key_name))

    return exprs


def _create_table_schema(
    schema: pl.Schema,
) -> tuple[Schema, dict[str, pl.DataType]]:
    """Iceberg schema of a table created from `schema`.

    Also returns the casts that make the data match the table schema, all of which
    are lossless. Raises if a column has no lossless Iceberg equivalent.
    """
    from itertools import count

    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        ListType,
        LongType,
        NestedField,
        StringType,
        StructType,
        TimestampType,
        TimestamptzType,
        TimeType,
    )

    import polars as pl

    field_ids = count(1)
    unsupported: list[str] = []

    def convert(
        dtype: PolarsDataType, path: str
    ) -> tuple[IcebergType, pl.DataType] | None:
        if dtype == pl.Boolean:
            return BooleanType(), pl.Boolean()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16):
            return IntegerType(), pl.Int32()
        if dtype in (pl.Int64, pl.UInt32):
            return LongType(), pl.Int64()
        if dtype in (pl.Float16, pl.Float32):
            return FloatType(), pl.Float32()
        if dtype == pl.Float64:
            return DoubleType(), pl.Float64()
        if isinstance(dtype, pl.Decimal):
            return DecimalType(dtype.precision, dtype.scale), dtype
        if dtype in (pl.String, pl.Categorical, pl.Enum):
            return StringType(), pl.String()
        if dtype == pl.Binary:
            return BinaryType(), pl.Binary()
        if dtype == pl.Date:
            return DateType(), pl.Date()
        if dtype == pl.Time:
            # Microsecond precision, like any Iceberg time column.
            return TimeType(), pl.Time()
        if isinstance(dtype, pl.Datetime):
            if dtype.time_unit == "ns":
                unsupported.append(
                    f"'{path}' ({dtype}): Iceberg timestamps have microsecond "
                    "precision, cast to Datetime('us')"
                )
                return None
            # A time zone is only a display setting: the instants are preserved.
            if dtype.time_zone is None:
                return TimestampType(), pl.Datetime("us")
            return TimestamptzType(), pl.Datetime("us", "UTC")
        if isinstance(dtype, (pl.List, pl.Array)):
            element_id = next(field_ids)
            if (inner := convert(dtype.inner, f"{path}[]")) is None:
                return None
            element_type, element_dtype = inner
            return (
                ListType(element_id, element_type, element_required=False),
                pl.List(element_dtype),
            )
        if isinstance(dtype, pl.Struct):
            fields, struct_fields = [], []
            for field in dtype.fields:
                field_id = next(field_ids)
                if (inner := convert(field.dtype, f"{path}.{field.name}")) is None:
                    continue
                fields.append(NestedField(field_id, field.name, inner[0]))
                struct_fields.append(pl.Field(field.name, inner[1]))
            if len(struct_fields) < len(dtype.fields):
                return None
            return StructType(*fields), pl.Struct(struct_fields)

        unsupported.append(f"'{path}' ({dtype})")
        return None

    fields, casts = [], {}
    for name, dtype in schema.items():
        field_id = next(field_ids)
        if (converted := convert(dtype, name)) is None:
            continue
        iceberg_type, cast_dtype = converted
        fields.append(NestedField(field_id, name, iceberg_type))
        if cast_dtype != dtype:
            casts[name] = cast_dtype

    if unsupported:
        msg = (
            "create_if_not_exists: columns without a lossless Iceberg type: "
            + ", ".join(unsupported)
        )
        raise TypeError(msg)

    return Schema(*fields), casts


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
            bucket = _bucket_transform_expr(expr, source_type, transform.num_buckets)
            if bucket is None:
                msg = f"Iceberg sort transform '{transform}' on '{source_type}'"
                raise NotImplementedError(msg)
            expr = bucket
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
    # Rows deleted by `mode="overwrite"`. None deletes all rows.
    # A `pl.Expr` is converted when the sink is resolved.
    overwrite_filter: BooleanExpression | pl.Expr | None
    dynamic_partition_overwrite: bool
    create_if_not_exists: bool
    create_namespace_if_not_exists: bool
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
    # The partition spec of the partition keys.
    partition_spec_id: int | None
    commit_result_df: NoPickleOption[pl.DataFrame]

    @staticmethod
    def new(
        target: str | pyiceberg.table.Table,
        *,
        mode: Literal["append", "overwrite"] = "append",
        schema_mode: Literal["merge", "overwrite"] | None = None,
        overwrite_filter: pl.Expr | BooleanExpression | str | None = None,
        partition_overwrite: Literal["dynamic"] | None = None,
        create_if_not_exists: bool = False,
        create_namespace_if_not_exists: bool = False,
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

        if partition_overwrite not in (None, "dynamic"):
            msg = f"invalid partition_overwrite: {partition_overwrite!r}"
            raise ValueError(msg)

        if overwrite_filter is not None or partition_overwrite is not None:
            if mode != "overwrite":
                msg = (
                    "overwrite_filter and partition_overwrite require mode='overwrite'"
                )
                raise ValueError(msg)
            if overwrite_filter is not None and partition_overwrite is not None:
                msg = "overwrite_filter and partition_overwrite are mutually exclusive"
                raise ValueError(msg)
            if schema_mode == "overwrite":
                msg = (
                    "schema_mode='overwrite' cannot be combined with overwrite_filter "
                    "or partition_overwrite"
                )
                raise ValueError(msg)

        if isinstance(overwrite_filter, str):
            from pyiceberg.expressions.parser import parse

            overwrite_filter = parse(overwrite_filter)

        import polars as pl

        if isinstance(overwrite_filter, pl.Expr) and not isinstance(target, str):
            # Fail early when the table is available.
            overwrite_filter = predicate_to_pyiceberg_delete_filter(
                overwrite_filter, target
            )

        if create_if_not_exists and not isinstance(target, str):
            msg = "create_if_not_exists=True requires a table identifier as `target`"
            raise TypeError(msg)

        if create_namespace_if_not_exists and not create_if_not_exists:
            msg = (
                "create_namespace_if_not_exists=True requires create_if_not_exists=True"
            )
            raise ValueError(msg)

        catalog_config = (
            (
                IcebergCatalogConfig._from_api_parameter_or_environment_default(
                    catalog,
                    fn_name="sink_iceberg",
                )[0]
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
            overwrite_filter=overwrite_filter,
            dynamic_partition_overwrite=partition_overwrite == "dynamic",
            create_if_not_exists=create_if_not_exists,
            create_namespace_if_not_exists=create_namespace_if_not_exists,
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
            partition_spec_id=None,
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
            self.table_.set(self._load_or_create_table(catalog))

        return self.table_.get()  # type: ignore[return-value]

    def _load_or_create_table(
        self, catalog: pyiceberg.catalog.Catalog
    ) -> pyiceberg.table.Table:
        from pyiceberg.exceptions import NoSuchTableError

        # Load first: this also runs at commit time, possibly in another process,
        # where the table normally exists already.
        try:
            return catalog.load_table(self.table_name)
        except NoSuchTableError:
            if not self.create_if_not_exists:
                raise

        try:
            return self._create_table(catalog)
        except Exception:
            # The create may have landed with its response lost.
            try:
                return catalog.load_table(self.table_name)
            except NoSuchTableError:
                pass
            raise

    def _create_table(
        self, catalog: pyiceberg.catalog.Catalog
    ) -> pyiceberg.table.Table:
        from pyiceberg.exceptions import NoSuchNamespaceError

        # Tolerates a concurrent create of the same table.
        try:
            return catalog.create_table_if_not_exists(
                self.table_name, schema=self._get_source_schema()
            )
        except NoSuchNamespaceError:
            if not self.create_namespace_if_not_exists:
                raise

        _create_namespace_if_not_exists(
            catalog, catalog.namespace_from(self.table_name)
        )

        return catalog.create_table_if_not_exists(
            self.table_name, schema=self._get_source_schema()
        )

    def _get_converted_storage_options(self) -> dict[str, str]:
        return _convert_iceberg_to_object_store_storage_options(
            self.iceberg_storage_properties
        )

    def attach_sink(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        if self.create_if_not_exists:
            from pyiceberg.io.pyarrow import schema_to_pyarrow

            # Cast the data to match a table created from it, which can only have
            # Iceberg types.
            table_schema, casts = _create_table_schema(lf.collect_schema())
            if casts:
                lf = lf.cast(casts)  # type: ignore[arg-type]
            self.source_schema = schema_to_pyarrow(
                table_schema, include_field_ids=False
            )
        elif self.schema_mode is not None:
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

        sort_order = table.sort_order()
        if sort_order.fields and self.schema_mode == "overwrite":
            msg = "schema_mode='overwrite' is not supported for Iceberg tables with a sort order"
            raise NotImplementedError(msg)

        # Under schema evolution, the keys are of the data as written.
        schema, arrow_schema = self._schemas_for_write(table)
        partition_key_exprs = _partition_key_exprs(
            table.spec(), schema, self.source_schema
        )
        self.partition_spec_id = table.spec().spec_id

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

        # Fail before writing any data.
        if self.dynamic_partition_overwrite:
            _check_dynamic_overwrite_spec(table.spec(), schema)
        if isinstance(self.overwrite_filter, pl.Expr):
            self.overwrite_filter = predicate_to_pyiceberg_delete_filter(
                self.overwrite_filter, table
            )
        elif self.overwrite_filter is not None:
            from pyiceberg.expressions.visitors import bind

            bind(schema, self.overwrite_filter, case_sensitive=True)

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

        if _already_committed(table.metadata, self.sink_uuid_str):
            if verbose:
                eprint("IcebergSinkState[commit]: already committed, skipping")
            return

        original_metadata_location = table.metadata_location
        snapshot_properties = self.snapshot_properties | {
            _SINK_UUID_PROPERTY: self.sink_uuid_str
        }

        if sys.platform == "win32":
            sinked_files = [
                (
                    f"file://{path[8:]}" if path.startswith("file:///") else path,
                    *rest,
                )
                for path, *rest in sinked_files
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
                if (
                    self.partition_spec_id is not None
                    and tx.table_metadata.default_spec_id != self.partition_spec_id
                ):
                    msg = (
                        f"Iceberg partition spec changed during the sink "
                        f"({self.partition_spec_id} -> "
                        f"{tx.table_metadata.default_spec_id})"
                    )
                    raise ValueError(msg)
                self._update_schema(tx)

                data_files = _data_files_for_commit(
                    tx, sinked_files, self.sort_order_id
                )

                if self.mode == "overwrite":
                    from pyiceberg.expressions import AlwaysFalse, AlwaysTrue

                    if self.dynamic_partition_overwrite:
                        delete_filter = _dynamic_overwrite_filter(
                            tx.table_metadata, data_files
                        )
                    elif self.overwrite_filter is not None:
                        # Converted from a `pl.Expr` when the sink was resolved.
                        delete_filter = self.overwrite_filter  # type: ignore[assignment]
                    else:
                        delete_filter = AlwaysTrue()

                    if verbose:
                        kind = (
                            "dynamic partitions"
                            if self.dynamic_partition_overwrite
                            else "filter"
                            if self.overwrite_filter is not None
                            else "all rows"
                        )
                        eprint(f"IcebergSinkState[commit]: overwrite: {kind}")

                    if delete_filter != AlwaysFalse():
                        tx.delete(
                            delete_filter, snapshot_properties=snapshot_properties
                        )

                if verbose:
                    eprint("IcebergSinkState[commit]: begin add_files")

                start_instant = perf_counter()

                _add_files(tx, data_files, snapshot_properties)

                if verbose:
                    elapsed = perf_counter() - start_instant
                    eprint(
                        f"IcebergSinkState[commit]: finish add_files ({elapsed:.3f}s)"
                    )
                    eprint("IcebergSinkState[commit]: begin transaction commit")

                start_instant = perf_counter()
        except _AlreadyCommitted as e:
            if verbose:
                eprint("IcebergSinkState[commit]: already committed, skipping")
            _copy_table_state(e.table, table)
            return

        if verbose:
            elapsed = perf_counter() - start_instant
            eprint(
                f"IcebergSinkState[commit]: finish transaction commit ({elapsed:.3f}s)"
            )

        _copy_table_state(wrapped, table)

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

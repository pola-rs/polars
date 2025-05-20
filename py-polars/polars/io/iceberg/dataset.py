from __future__ import annotations

import datetime
import os
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import polars._reexport as pl
from polars._utils.logging import eprint, verbose
from polars.exceptions import ComputeError
from polars.io.iceberg._utils import _scan_pyarrow_dataset_impl

if TYPE_CHECKING:
    import contextlib

    import pyarrow as pa
    from pyiceberg.partitioning import PartitionField
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import Table

    from polars._typing import IcebergWriteMode

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyLazyFrame


class IcebergDataset:
    """Dataset interface for PyIceberg."""

    def __init__(
        self,
        source: str | Table,
        *,
        snapshot_id: int | None = None,
        iceberg_storage_properties: dict[str, Any] | None = None,
        reader_override: Literal["native", "pyiceberg"] | None,
    ) -> None:
        self._metadata_path = None
        self._table = None
        self._snapshot_id = snapshot_id
        self._iceberg_storage_properties = iceberg_storage_properties
        self._reader_override: Literal["native", "pyiceberg"] | None = reader_override

        # Accept either a path or a table object. The one we don't have is
        # lazily initialized when needed.

        if isinstance(source, str):
            self._metadata_path = source
        else:
            self._table = source

    #
    # PythonDatasetProvider interface functions
    #

    def reader_name(self) -> str:
        """Name of the reader."""
        return "iceberg"

    def schema(self) -> pa.schema:
        """Fetch the schema of the table."""
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        return schema_to_pyarrow(self.table().schema())

    def to_dataset_scan(
        self,
        *,
        limit: int | None = None,
        projection: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Construct a LazyFrame scan."""
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"limit: {limit}, "
                f"projection: {projection}"
            )

        tbl = self.table()

        selected_fields = ("*",) if projection is None else tuple(projection)

        snapshot_id = self._snapshot_id

        if snapshot_id is not None:
            if tbl.snapshot_by_id(snapshot_id) is None:
                msg = f"iceberg snapshot ID not found: {snapshot_id}"
                raise ValueError(msg)

        # Take from parameter first then envvar
        reader_override = self._reader_override or os.getenv(
            "POLARS_ICEBERG_READER_OVERRIDE"
        )

        if reader_override and reader_override not in ["native", "pyiceberg"]:
            msg = (
                "iceberg: unknown value for reader_override: "
                f"'{reader_override}', expected one of ('native', 'pyiceberg')"
            )
            raise ValueError(msg)

        # Try native scan
        fallback_reason = (
            "forced reader_override='pyiceberg'"
            if reader_override == "pyiceberg"
            # TODO: Enable native scans by default after we have type casting support,
            # currently it may fail if the dataset has changed types.
            else "native scans disabled by default"
            if reader_override != "native"
            else None
        )

        sources = []
        deletion_files: dict[int, list[str]] = {}

        if reader_override != "pyiceberg" and not fallback_reason:
            from pyiceberg.manifest import DataFileContent, FileFormat

            if verbose:
                eprint("IcebergDataset: to_dataset_scan(): begin path expansion")

            start_time = perf_counter()

            scan = tbl.scan(
                snapshot_id=snapshot_id, limit=limit, selected_fields=selected_fields
            )

            total_deletion_files = 0

            for i, file_info in enumerate(scan.plan_files()):
                if file_info.file.file_format != FileFormat.PARQUET:
                    fallback_reason = (
                        f"non-parquet format: {file_info.file.file_format}"
                    )
                    break

                if file_info.delete_files:
                    deletion_files[i] = []

                    for deletion_file in file_info.delete_files:
                        if deletion_file.content != DataFileContent.POSITION_DELETES:
                            fallback_reason = (
                                "unsupported deletion file type: "
                                f"{deletion_file.content}"
                            )
                            break

                        if deletion_file.file_format != FileFormat.PARQUET:
                            fallback_reason = (
                                "unsupported deletion file format: "
                                f"{deletion_file.file_format}"
                            )
                            break

                        deletion_files[i].append(deletion_file.file_path)
                        total_deletion_files += 1

                if fallback_reason:
                    break

                sources.append(file_info.file.file_path)

            if verbose:
                elapsed = perf_counter() - start_time
                eprint(
                    "IcebergDataset: to_dataset_scan(): "
                    f"finish path expansion ({elapsed:.3f}s)"
                )

        if not fallback_reason:
            from polars.io.parquet.functions import scan_parquet

            if verbose:
                eprint(
                    "IcebergDataset: to_dataset_scan(): "
                    f"native scan_parquet() ({len(sources)} sources), "
                    f"deletion files: {total_deletion_files} files, "
                    f"{len(deletion_files)} sources"
                )

            return scan_parquet(
                sources,
                missing_columns="insert",
                extra_columns="ignore",
                _deletion_files=("iceberg-position-delete", deletion_files),
            )

        elif reader_override == "native":
            msg = f"iceberg reader_override='native' failed: {fallback_reason}"
            raise ComputeError(msg)

        if verbose:
            eprint(
                "IcebergDataset: to_dataset_scan(): "
                f"fallback to python[pyiceberg] scan: {fallback_reason}"
            )

        func = partial(
            _scan_pyarrow_dataset_impl,
            tbl,
            snapshot_id=snapshot_id,
            n_rows=limit,
            with_columns=projection,
        )

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        arrow_schema = schema_to_pyarrow(tbl.schema())

        lf = pl.LazyFrame._scan_python_function(arrow_schema, func, pyarrow=True)

        return lf

    def _to_dataset_sink(
        self,
        lf: PyLazyFrame,
        *,
        mode: IcebergWriteMode,
    ) -> pl.LazyFrame:
        return self.to_dataset_sink(pl.LazyFrame._from_pyldf(lf), mode=mode)

    def to_dataset_sink(
        self, lf: pl.LazyFrame, *, mode: IcebergWriteMode
    ) -> pl.LazyFrame:
        """Write a LazyFrame into the Iceberg dataset."""
        import uuid
        from pathlib import Path

        import pyarrow as pa
        import pyiceberg.transforms as ts
        from pyiceberg.expressions import AlwaysTrue
        from pyiceberg.manifest import DataFile
        from pyiceberg.typedef import Record as IcebergRecord

        from polars import Boolean, col, lit

        def _partition_field_to_partition_expr(
            field: PartitionField, schema: IcebergSchema
        ) -> pl.Expr:
            part: pl.Expr = col(schema.find_field(field.source_id).name)

            if isinstance(field.transform, ts.IdentityTransform):
                pass
            elif isinstance(field.transform, ts.YearTransform):
                part = part.dt.truncate("1y").dt.date()
            elif isinstance(field.transform, ts.MonthTransform):
                part = part.dt.truncate("1m").dt.date()
            elif isinstance(field.transform, ts.DayTransform):
                part = part.dt.truncate("1d").dt.date()
            elif isinstance(field.transform, ts.HourTransform):
                part = part.dt.truncate("1h")
            elif isinstance(field.transform, ts.BucketTransform):
                msg = "BucketTransform not yet implemented."
                raise NotImplementedError(msg)
            elif isinstance(field.transform, ts.TruncateTransform):
                msg = "TruncateTransform not yet implemented."
                raise NotImplementedError(msg)
            else:
                msg = f"{field.transform} not implemented. Is this is a new transform?"
                raise NotImplementedError(msg)

            return part

        def _to_partition_representation(value: Any) -> Any:
            if value is None:
                return None

            if isinstance(value, datetime.datetime):
                # Convert to microseconds since epoch
                return (value - datetime.datetime(1970, 1, 1)) // datetime.timedelta(
                    microseconds=1
                )
            elif isinstance(value, datetime.date):
                # Convert to days since epoch
                return (value - datetime.date(1970, 1, 1)) // datetime.timedelta(days=1)
            elif isinstance(value, datetime.time):
                # Convert to microseconds since midnight
                return (
                    value.hour * 60 * 60 + value.minute * 60 + value.second
                ) * 1_000_000 + value.microsecond
            elif isinstance(value, uuid.UUID):
                return str(value)
            else:
                return value

        def make_iceberg_record(
            partition_values: dict[str, Any] | None,
        ) -> IcebergRecord:
            from pyiceberg.typedef import Record as IcebergRecord

            if partition_values:
                iceberg_part_vals = {
                    k: _to_partition_representation(v)
                    for k, v in partition_values.items()
                }
                return IcebergRecord(**iceberg_part_vals)
            else:
                return IcebergRecord()

        def field_to_parquet_overwrites(field: pa.Field) -> ParquetFieldOverwrites:
            field_id = field.metadata.pop(b"PARQUET:field_id")
            field_id = int(field_id)

            unsupported_types = [
                "MapType",
                "ExtensionType",
                "UnionType",
                "DictionaryType",
                "SparseUnionType",
                "DenseUnionType",
                "Decimal256",
                "ListViewType",
                "LargeListViewType",
                "FixedSizeBinaryType",
                "BaseExtensionType",
                "PyExtensionType",
                "UnknownExtensionType",
                "JsonType",
                "Bool8Type",
                "UuidType",
                "OpaqueType",
                "RunEndEncodedType",
            ]
            is_unsupported_type = any(
                hasattr(pa, t) and isinstance(field.type, getattr(pa, t))
                for t in unsupported_types
            )
            if is_unsupported_type:
                msg = f"arrow `{field.type!r}` is not supported in polars Iceberg sinks"
                raise NotImplementedError(msg)

            children: (
                None | ParquetFieldOverwrites | dict[str, ParquetFieldOverwrites]
            ) = None
            if isinstance(field.type, pa.StructType):
                children = {
                    f.name: field_to_parquet_overwrites(f) for f in field.type.fields
                }
            elif isinstance(field.type, (pa.LargeListType, pa.ListType)):
                children = field_to_parquet_overwrites(field.type.value_field)

            return ParquetFieldOverwrites(
                name=field.name,
                field_id=field_id,
                required=not field.nullable,
                children=children,
            )

        from pyiceberg.io.pyarrow import schema_to_pyarrow

        table = self.table()
        location_provider = self.table().location_provider()
        iceberg_schema = table.schema()
        pyarrow_schema = schema_to_pyarrow(iceberg_schema)

        if len(table.sort_order().fields) > 0:
            msg = "Iceberg with sort orders"
            raise NotImplementedError(msg)

        polars_schema = pl.DataFrame(pyarrow_schema.empty_table()).schema

        spec_id = table.metadata.partition_specs[0].spec_id
        partition_exprs = [
            _partition_field_to_partition_expr(p, iceberg_schema).alias(p.name)
            for p in table.metadata.partition_specs[0].fields
        ]
        if len(partition_exprs) == 0:
            partition_exprs = [
                lit(False, dtype=Boolean()).alias("__POLARS_DUMMY_PARTITION")
            ]

        lf = lf.match_to_schema(polars_schema)

        from polars.io import PartitionByKey
        from polars.io.parquet import ParquetFieldOverwrites

        # Create mapping to map column name to unique integer
        field_overwrites = [
            field_to_parquet_overwrites(field) for field in pyarrow_schema
        ]
        field_ids = [f.field_id for f in field_overwrites]

        def _file_path_cb(_ctx: Any) -> str:
            name = f"{uuid.uuid4()}.parquet"
            path = location_provider.new_data_location(name, None)
            if path.startswith("file://"):
                Path(path[len("file://") :]).parent.mkdir(parents=True, exist_ok=True)
            offset = path[len(location_provider.table_location) :]
            if offset.startswith("/"):
                offset = offset[1:]
            return offset

        def _finish_callback(df: pl.DataFrame) -> None:
            with table.transaction() as tx:
                if mode == "overwrite":
                    if table.current_snapshot is not None:
                        tx.delete(delete_filter=AlwaysTrue())
                elif mode == "append":
                    pass
                else:
                    msg = "mode is required to be in {'overwrite', 'append'}"
                    raise ValueError(msg)

                from pyiceberg.io.pyarrow import DataFileStatistics
                from pyiceberg.manifest import DataFileContent
                from pyiceberg.manifest import FileFormat as IcebergFileFormat

                with tx._append_snapshot_producer({}) as append_files:
                    for row in df.iter_rows():
                        path = row[0]
                        num_rows = row[1]
                        file_size = row[2]
                        keys = row[3]
                        column_statistics = row[4:]

                        k = {
                            k: _to_partition_representation(v) for k, v in keys.items()
                        }

                        props = {
                            "content": DataFileContent.DATA,
                            "file_path": path,
                            "file_format": IcebergFileFormat.PARQUET,
                            "partition": IcebergRecord(**k),
                            "file_size_in_bytes": file_size,
                            "sort_order_id": None,
                            "spec_id": spec_id,
                            "equality_ids": None,
                            "key_metadata": None,
                            "record_count": num_rows,
                        }

                        null_value_counts = {
                            field_ids[i]: s["null_count"]
                            for i, s in enumerate(column_statistics)
                        }
                        nan_value_counts = {
                            field_ids[i]: s["nan_count"]
                            for i, s in enumerate(column_statistics)
                        }

                        statistics = DataFileStatistics(
                            record_count=num_rows,
                            column_sizes={},  # TODO: be part of statistics
                            value_counts={},  # TODO: be part of statistics
                            null_value_counts=null_value_counts,
                            nan_value_counts=nan_value_counts,
                            column_aggregates={},  # TODO: be part of statistics
                            split_offsets={},  # TODO: be part of statistics
                        )

                        data_file = DataFile(
                            **{**props, **statistics.to_serialized_dict()}
                        )
                        append_files.append_data_file(data_file)

        return lf.sink_parquet(
            PartitionByKey(
                location_provider.table_location,
                file_path=_file_path_cb,
                by=partition_exprs,
                include_key=False,
                finish_callback=_finish_callback,
            ),
            field_overwrites=field_overwrites,
            lazy=True,
        )

    #
    # Accessors
    #

    def metadata_path(self) -> str:
        """Fetch the metadata path."""
        if self._metadata_path is None:
            if self._table is None:
                msg = "impl error: both metadata_path and table are None"
                raise ValueError(msg)

            self._metadata_path = self.table().metadata_location

        return self._metadata_path

    def table(self) -> Table:
        """Fetch the PyIceberg Table object."""
        if self._table is None:
            if self._metadata_path is None:
                msg = "impl error: both metadata_path and table are None"
                raise ValueError(msg)

            if verbose():
                eprint(f"IcebergDataset: construct table from {self._metadata_path = }")

            from pyiceberg.table import StaticTable

            self._table = StaticTable.from_metadata(
                metadata_location=self._metadata_path,
                properties=self._iceberg_storage_properties or {},
            )

        return self._table

    #
    # Serialization functions
    #
    # We don't serialize the iceberg table object - the remote machine should
    # use their own permissions to reconstruct the table object from the path.
    #

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "metadata_path": self.metadata_path(),
            "snapshot_id": self._snapshot_id,
            "iceberg_storage_properties": self._iceberg_storage_properties,
            "reader_override": self._reader_override,
        }

        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = state["snapshot_id"]
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            reader_override = state["reader_override"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: '{snapshot_id}', "
                f"iceberg_storage_properties: {keys_repr}, "
                f"reader_override: {reader_override}"
            )

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if verbose():
            path_repr = state["metadata_path"]
            snapshot_id = state["snapshot_id"]
            keys_repr = _redact_dict_values(state["iceberg_storage_properties"])
            reader_override = state["reader_override"]

            eprint(
                "IcebergDataset: getstate(): "
                f"path: '{path_repr}', "
                f"snapshot_id: '{snapshot_id}', "
                f"iceberg_storage_properties: {keys_repr}, "
                f"reader_override: {reader_override}"
            )

        IcebergDataset.__init__(
            self,
            state["metadata_path"],
            snapshot_id=state["snapshot_id"],
            iceberg_storage_properties=state["iceberg_storage_properties"],
            reader_override=state["reader_override"],
        )


def _redact_dict_values(obj: Any) -> Any:
    return (
        {k: "REDACTED" for k in obj.keys()}  # noqa: SIM118
        if isinstance(obj, dict)
        else f"<{type(obj).__name__} object>"
        if obj is not None
        else "None"
    )

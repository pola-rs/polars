from __future__ import annotations

import contextlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlparse

from polars._utils.logging import eprint, verbose
from polars._utils.wrap import wrap_ldf
from polars.exceptions import ComputeError, SchemaError
from polars.io.cloud._utils import POLARS_STORAGE_CONFIG_KEYS, NoPickleOption
from polars.io.delta._utils import (
    _check_if_delta_available,
    _resolve_delta_lake_uri,
)
from polars.io.iceberg._sink import PlIcebergPathProviderConfig
from polars.io.iceberg._utils import _normalize_windows_iceberg_file_uri

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars._plr import _delta_file_stats, gen_uuid_v7

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import deltalake
    from deltalake.transaction import AddAction

    import polars as pl
    from polars._plr import PyLazyFrame
    from polars._typing import PolarsDataType, StorageOptionsDict
    from polars.io.cloud.credential_provider._builder import (
        CredentialProviderBuilder,
    )
    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderFunction,
    )


# `(path, num_rows, num_bytes, parquet_metadata, partition_keys)`
_DeltaSinkedFile = tuple[str, int, int, bytes, "pl.DataFrame | None"]

DeltaSinkMode = Literal["error", "append", "overwrite", "ignore"]

# Recorded in the commit info, to find the version of a commit that landed.
_SINK_UUID_PROPERTY = "polars.sink-uuid"

# Writer features that this sink honors, and that deltalake commits to. A table with
# any other writer feature is rejected before any file is written.
#
# - appendOnly: `mode="overwrite"` is rejected.
# - invariants: only nullability is checked, so a column with an SQL invariant is
#   rejected.
# - changeDataFeed: the add and remove actions describe the change of an append or
#   overwrite.
# - deletionVectors: nothing to do for new files.
#
# domainMetadata and vacuumProtocolCheck need nothing for new files either, but
# deltalake 1.6.6 refuses to commit to tables with them.
_ALLOWED_WRITER_FEATURES = frozenset(
    {
        "appendOnly",
        "invariants",
        "timestampNtz",
        "changeDataFeed",
        "deletionVectors",
    }
)

# Writer features that only constrain variant columns, so they are allowed when the
# table has none. deltalake 1.6.4 to 1.6.6 adds variantType to tables without
# variant columns.
_VARIANT_WRITER_FEATURES = frozenset({"variantType", "variantType-preview"})

# Same default as delta-rs and Spark.
_DEFAULT_NUM_INDEXED_COLS = 32

_UNIX_EPOCH = datetime(1970, 1, 1)


def _delta_dtype(dtype: PolarsDataType) -> PolarsDataType:
    """The Polars type of `dtype` when written to Delta Lake.

    Delta has no unsigned integers, no nanosecond timestamps and no time zones other
    than UTC, so these are cast to the closest Delta type.
    """
    import polars as pl

    if isinstance(dtype, pl.UInt8):
        return pl.Int16()
    if isinstance(dtype, pl.UInt16):
        return pl.Int32()
    if isinstance(dtype, pl.UInt32):
        return pl.Int64()
    if isinstance(dtype, (pl.UInt64, pl.UInt128, pl.Int128)):
        msg = f"Delta Lake has no integer type that can hold {dtype}"
        raise TypeError(msg)
    if isinstance(dtype, (pl.Time, pl.Duration, pl.Null, pl.Object)):
        msg = f"Delta Lake has no type for {dtype}"
        raise TypeError(msg)
    if isinstance(dtype, pl.Float16):
        return pl.Float32()
    if isinstance(dtype, pl.Decimal) and (dtype.precision or 38) > 38:
        msg = f"Delta Lake decimals have a precision of at most 38, got {dtype}"
        raise TypeError(msg)
    if isinstance(dtype, pl.Datetime):
        # A time zone is converted to UTC, keeping the instant.
        return pl.Datetime("us", "UTC" if dtype.time_zone is not None else None)
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return pl.String()
    if isinstance(dtype, (pl.List, pl.Array)):
        return pl.List(_delta_dtype(dtype.inner))
    if isinstance(dtype, pl.Struct):
        return pl.Struct({f.name: _delta_dtype(f.dtype) for f in dtype.fields})
    return dtype


def _is_delta_type(dtype: PolarsDataType) -> bool:
    """Whether data of `dtype` is written to Delta Lake without a cast."""
    try:
        return _delta_dtype(dtype) == dtype
    except TypeError:
        return False


def _check_deltalake_apis() -> None:
    """Raise if the installed deltalake lacks an API that the sink commits with."""
    import deltalake

    if missing := _missing_deltalake_apis():
        msg = (
            f"DeltaSinkState requires a newer deltalake (installed: "
            f"{deltalake.__version__}), which has: {', '.join(missing)}"
        )
        raise ImportError(msg)


def _missing_deltalake_apis() -> list[str]:
    """The APIs that the sink commits with and the installed deltalake lacks."""
    import inspect

    import deltalake
    import deltalake.transaction

    return [
        name
        for name, available in [
            (
                "DeltaTable.create_write_transaction",
                hasattr(deltalake.DeltaTable, "create_write_transaction"),
            ),
            (
                "DeltaTable.transaction_version",
                hasattr(deltalake.DeltaTable, "transaction_version"),
            ),
            (
                "transaction.create_table_with_add_actions",
                hasattr(deltalake.transaction, "create_table_with_add_actions"),
            ),
            ("transaction.AddAction", hasattr(deltalake.transaction, "AddAction")),
            ("transaction.Transaction", hasattr(deltalake.transaction, "Transaction")),
            (
                "CommitProperties(app_transactions=...)",
                hasattr(deltalake.transaction, "CommitProperties")
                and "app_transactions"
                in inspect.signature(deltalake.transaction.CommitProperties).parameters,
            ),
        ]
        if not available
    ]


def _check_writable(table: deltalake.DeltaTable, mode: DeltaSinkMode) -> None:
    """Raise if the sink cannot write to `table` without breaking its features."""
    protocol = table.protocol()
    writer_version = protocol.min_writer_version

    if writer_version > 7:
        msg = f"Delta table requires unsupported writer version {writer_version}"
        raise NotImplementedError(msg)

    if writer_version == 7:
        unsupported = set(protocol.writer_features or ()) - _ALLOWED_WRITER_FEATURES
        if not any(_contains_variant(f.type) for f in table.schema().fields):
            unsupported -= _VARIANT_WRITER_FEATURES
        if unsupported:
            msg = (
                "sink to Delta table with unsupported writer features: "
                f"{sorted(unsupported)}"
            )
            raise NotImplementedError(msg)

    # Legacy writer versions 2-6 imply features that are only active when
    # configured, so check the configuration and the schema as well.
    configuration = table.metadata().configuration

    if constraints := sorted(
        k for k in configuration if k.startswith("delta.constraints.")
    ):
        msg = f"sink to Delta table with CHECK constraints: {constraints}"
        raise NotImplementedError(msg)

    if configuration.get("delta.columnMapping.mode", "none") != "none":
        msg = "sink to Delta table with column mapping"
        raise NotImplementedError(msg)

    if configuration.get("delta.enableRowTracking", "false").lower() == "true":
        msg = "sink to Delta table with row tracking"
        raise NotImplementedError(msg)

    for name, metadata in _iter_field_metadata(table.schema().fields, ""):
        for key in metadata:
            if key == "delta.generationExpression":
                msg = f"sink to Delta table with generated column '{name}'"
                raise NotImplementedError(msg)
            if key.startswith("delta.identity."):
                msg = f"sink to Delta table with identity column '{name}'"
                raise NotImplementedError(msg)
            if key == "delta.invariants":
                msg = f"sink to Delta table with an invariant on column '{name}'"
                raise NotImplementedError(msg)

    if (
        mode == "overwrite"
        and configuration.get("delta.appendOnly", "false").lower() == "true"
    ):
        msg = "cannot overwrite Delta table: it is append-only ('delta.appendOnly')"
        raise ValueError(msg)


def _iter_field_metadata(
    fields: list[deltalake.Field], prefix: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The metadata of every field in `fields`, including nested fields."""
    for field in fields:
        name = f"{prefix}{field.name}"
        yield name, field.metadata
        yield from _iter_type_field_metadata(field.type, f"{name}.")


def _iter_type_field_metadata(
    dtype: Any, prefix: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    type_name = getattr(dtype, "type", None)
    if type_name == "struct":
        yield from _iter_field_metadata(dtype.fields, prefix)
    elif type_name == "array":
        yield from _iter_type_field_metadata(dtype.element_type, prefix)
    elif type_name == "map":
        yield from _iter_type_field_metadata(dtype.key_type, prefix)
        yield from _iter_type_field_metadata(dtype.value_type, prefix)


def _contains_variant(dtype: Any) -> bool:
    """Whether a Delta type is a variant or has one nested inside."""
    type_name = getattr(dtype, "type", None)
    if type_name == "variant":
        return True
    if type_name == "struct":
        return any(_contains_variant(f.type) for f in dtype.fields)
    if type_name == "array":
        return _contains_variant(dtype.element_type)
    if type_name == "map":
        return _contains_variant(dtype.key_type) or _contains_variant(dtype.value_type)
    return False


def _strip_field_metadata(schema: deltalake.Schema) -> str:
    """`schema` as Delta schema JSON, without the metadata of any field.

    Field metadata, e.g. a column comment, doesn't affect the data files.
    """

    def strip(dtype: Any) -> None:
        if not isinstance(dtype, dict):
            return
        for field in dtype.get("fields", ()):
            field["metadata"] = {}
            strip(field["type"])
        for key in ("elementType", "keyType", "valueType"):
            strip(dtype.get(key))

    schema_json = json.loads(schema.to_json())
    strip(schema_json)
    return json.dumps(schema_json)


def _table_schema(table: deltalake.DeltaTable) -> str:
    """The schema of `table`, as Delta schema JSON without field metadata."""
    return _strip_field_metadata(table.schema())


def _polars_schema(schema_json: str) -> pl.Schema:
    """The Polars schema of data with the Delta schema `schema_json`."""
    from deltalake import Schema

    import polars as pl

    return pl.Schema(Schema.from_json(schema_json).to_arrow())


def _file_schema(schema_json: str, partition_columns: list[str]) -> deltalake.Schema:
    """The schema of the data files: `schema_json` without the partition columns."""
    from deltalake import Schema

    schema = json.loads(schema_json)
    schema["fields"] = [
        f for f in schema["fields"] if f["name"] not in partition_columns
    ]
    return Schema.from_json(json.dumps(schema))


def _non_nullable_leaf_paths(schema: deltalake.Schema) -> list[tuple[str, ...]]:
    """Parquet paths of the leaves that are non-nullable along their whole path.

    The null count of other leaves also counts nulls of a nullable parent.
    """
    from deltalake.schema import ArrayType, MapType, StructType

    out: list[tuple[str, ...]] = []

    def visit(field: deltalake.Field, path: tuple[str, ...]) -> None:
        if field.nullable:
            return
        if isinstance(field.type, StructType):
            for child in field.type.fields:
                visit(child, (*path, child.name))
        elif not isinstance(field.type, (ArrayType, MapType)):
            out.append(path)

    for field in schema.fields:
        visit(field, (field.name,))
    return out


def _stats_leaf_paths(
    schema: deltalake.Schema, configuration: Mapping[str, str | None]
) -> set[tuple[str, ...]]:
    """The leaf columns of a data file to collect statistics for.

    `delta.dataSkippingStatsColumns` lists the columns by name, where a struct
    selects all of its fields. Otherwise `delta.dataSkippingNumIndexedCols` selects
    the first columns of the table, counting top-level columns like delta-rs, or
    all columns when it is -1.

    Statistics are only collected for leaves that are not in a list or map.
    """
    from deltalake.schema import ArrayType, MapType, StructType

    leaves: list[tuple[str, ...]] = []

    def visit(dtype: deltalake.schema.DataType, path: tuple[str, ...]) -> None:
        if isinstance(dtype, StructType):
            for child in dtype.fields:
                visit(child.type, (*path, child.name))
        elif not isinstance(dtype, (ArrayType, MapType)) and dtype.type != "binary":
            leaves.append(path)

    if stats_columns := configuration.get("delta.dataSkippingStatsColumns"):
        selected = [
            tuple(part.strip("`") for part in name.strip().split("."))
            for name in stats_columns.split(",")
            if name.strip()
        ]
        for field in schema.fields:
            visit(field.type, (field.name,))
        return {
            leaf
            for leaf in leaves
            if any(leaf[: len(prefix)] == prefix for prefix in selected)
        }

    num_indexed_cols = int(
        configuration.get("delta.dataSkippingNumIndexedCols")
        or _DEFAULT_NUM_INDEXED_COLS
    )
    fields = list(schema.fields)
    if num_indexed_cols >= 0:
        fields = fields[:num_indexed_cols]
    for field in fields:
        visit(field.type, (field.name,))
    return set(leaves)


def _format_float(value: float, *, f32: bool) -> str:
    """A float as Rust's `Display` writes it, e.g. `0.1`, `2`, `-0` or `NaN`.

    That is the shortest representation that round-trips, without an exponent.
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = repr(value)
    if f32:
        # The shortest representation that round-trips through a 32-bit float.
        bits = struct.pack("<f", value)
        for digits in range(1, 10):
            candidate = f"{value:.{digits}g}"
            if struct.pack("<f", float(candidate)) == bits:
                text = candidate
                break
    out = format(Decimal(text), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if out == "0" and math.copysign(1.0, value) < 0:
        out = "-0"
    return out


def _partition_value(series: pl.Series) -> str | None:
    """The value of a height-1 partition key, as a Delta partition value string.

    For example `2026-11-05` for a date, `2026-11-05 12:00:00.000000` for a
    timestamp (in UTC with a time zone), or `true` for a boolean.
    """
    import polars as pl

    dtype = series.dtype
    value = series[0]
    if value is None:
        return None
    if isinstance(dtype, pl.String):
        return value
    if isinstance(dtype, pl.Boolean):
        return "true" if value else "false"
    if dtype.is_integer():
        return str(value)
    if isinstance(dtype, (pl.Float32, pl.Float64)):
        return _format_float(value, f32=isinstance(dtype, pl.Float32))
    if isinstance(dtype, pl.Decimal):
        return f"{value:.{dtype.scale}f}"
    if isinstance(dtype, pl.Date):
        return value.isoformat()
    if isinstance(dtype, pl.Datetime):
        micros = series.dt.cast_time_unit("us").to_physical()[0]
        value = _UNIX_EPOCH + timedelta(microseconds=micros)
        return f"{value:%Y-%m-%d %H:%M:%S.%f}"
    if isinstance(dtype, pl.Binary):
        # The Delta protocol escapes every byte, e.g. `\u0001ÿ`.
        return "".join(f"\\u{b:04X}" for b in value)

    msg = f"unsupported Delta partition column type: {dtype}"
    raise TypeError(msg)


def _relative_data_path(file_path: str, sink_uuid: str) -> str:
    """The path of a sinked file relative to the table root, URL-encoded.

    The sink writes the files to `<table>/<sink_uuid>/`, so the relative path starts
    at the sink directory. This avoids matching the table prefix, which differs in
    form (e.g. `file://`, a trailing slash, Windows drive letters).
    """
    normalized = file_path.replace("\\", "/")
    marker = f"/{sink_uuid}/"
    idx = normalized.rfind(marker)
    if idx < 0:
        msg = f"Delta sink: file '{file_path}' is not in the sink directory"
        raise ComputeError(msg)
    relative = normalized[idx + 1 :]
    return quote(relative, safe="/")


def _is_local_uri(uri: str) -> bool:
    scheme = urlparse(uri).scheme
    # A single letter is a Windows drive.
    return scheme in ("", "file") or len(scheme) == 1


def _local_path(uri: str) -> str:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = parsed.path
        if os.name == "nt" and path.startswith("/") and path[2:3] == ":":
            path = path[1:]
        return path
    return uri


@dataclass
class _LoadedTable:
    """A sink's table as loaded in this process, and what is known about the sink.

    Held in a `NoPickleOption`, so a sink state in another process loads the table
    again and assumes nothing.
    """

    # None if the table didn't exist.
    table: deltalake.DeltaTable | None
    # `_native_sink_state` created the sink for `write_delta` or `sink_delta`, and
    # its commit hasn't started. No commit of the sink can exist yet, so the checks
    # for one are skipped. Its files are written with the credentials in the storage
    # options, which these functions resolved.
    routed: bool = False
    # `_native_sink_state` loaded the table just before the sink is resolved, so
    # resolving uses it without reloading it.
    fresh: bool = False


@dataclass(kw_only=True)
class DeltaSinkState:
    """Writes Parquet files with the native sink, then commits them to a Delta table.

    `commit()` can run in another process, and can be called again for the same sink
    (e.g. after a lost response): it commits the files at most once.

    With a `DeltaTable` as target, the table object is reused to load the table, but
    its storage options are not known here. The data files are written with
    `storage_options` only, and a commit in another process loads the table by its
    URI. `LazyFrame.sink_delta` has the same limitation.
    """

    table_uri: str
    mode: DeltaSinkMode
    # Includes the credentials of the credential provider.
    storage_options: dict[str, str]

    sink_uuid_str: str

    # Partition columns of the table when the sink was resolved, or None before.
    partition_columns: list[str] | None

    table_: NoPickleOption[_LoadedTable]
    # Schema of the table, or of the data for a new table, as Delta schema JSON
    # without field metadata. Set when the sink is resolved.
    source_schema: str | None
    commit_result_df: NoPickleOption[pl.DataFrame]

    @staticmethod
    def new(
        target: str | Path | deltalake.DeltaTable,
        *,
        mode: DeltaSinkMode = "error",
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | CredentialProviderBuilder
        | Literal["auto"]
        | None = "auto",
    ) -> DeltaSinkState:
        _check_if_delta_available()
        _check_deltalake_apis()

        from deltalake import DeltaTable

        from polars.io.cloud.credential_provider._builder import (
            _init_credential_provider_builder,
        )
        from polars.io.cloud.credential_provider._providers import (
            _get_credentials_from_provider_expiry_aware,
        )

        if mode not in ("error", "append", "overwrite", "ignore"):
            msg = f"invalid Delta sink mode: {mode!r}"
            raise ValueError(msg)

        table: DeltaTable | None = None
        if isinstance(target, DeltaTable):
            if credential_provider is not None and credential_provider != "auto":
                msg = "cannot use credential_provider when passing a DeltaTable object"
                raise ValueError(msg)
            table = target
            table_uri = target.table_uri
            credential_provider_builder = None
        else:
            table_uri = _resolve_delta_lake_uri(str(target), strict=False)
            credential_provider_builder = _init_credential_provider_builder(
                credential_provider, table_uri, storage_options, "sink_delta"
            )

        credential_provider_creds = {}

        if credential_provider_builder and (
            provider := credential_provider_builder.build_credential_provider()
        ):
            credential_provider_creds = (
                _get_credentials_from_provider_expiry_aware(provider) or {}
            )

        merged_storage_options = {
            k: str(v)
            for k, v in {**(storage_options or {}), **credential_provider_creds}.items()
            if v is not None
        }

        return DeltaSinkState(
            table_uri=table_uri,
            mode=mode,
            storage_options=merged_storage_options,
            sink_uuid_str=gen_uuid_v7().hex(),
            partition_columns=None,
            table_=NoPickleOption(None if table is None else _LoadedTable(table)),
            source_schema=None,
            commit_result_df=NoPickleOption(),
        )

    def _delta_storage_options(self) -> dict[str, str]:
        # deltalake rejects the Polars-specific keys.
        return {
            k: v
            for k, v in self.storage_options.items()
            if k not in POLARS_STORAGE_CONFIG_KEYS
        }

    def _load_table(self, *, refresh: bool = True) -> deltalake.DeltaTable | None:
        """Load the latest version of the table, or None if it does not exist.

        Reuses the table object of an earlier load, or the one passed to `new()`,
        which may have storage options of its own. With `refresh=False`, the result
        of an earlier load is returned as it is.
        """
        from deltalake import DeltaTable
        from deltalake.exceptions import TableNotFoundError

        loaded = self.table_.get()
        if loaded is not None:
            if not refresh:
                return loaded.table
            if loaded.table is not None:
                loaded.table.update_incremental()
                return loaded.table

        try:
            table = DeltaTable(
                self.table_uri, storage_options=self._delta_storage_options()
            )
        except TableNotFoundError:
            table = None

        if loaded is None:
            self.table_.set(_LoadedTable(table))
        else:
            loaded.table = table
        return table

    def _deltalake_write_reason(self, schema: pl.Schema) -> str | None:
        """Why deltalake should write data of `schema` instead of this sink, or None.

        This sink rejects data whose types differ from the table's, while deltalake
        casts it. Writes that this sink rejects for other reasons are also left to
        deltalake, which raises its own error or supports them.
        """
        table = self._load_table()
        if table is None:
            return None

        if self.mode in ("error", "ignore"):
            # deltalake raises its own error, or writes nothing.
            return f"mode is {self.mode!r} and the table exists"

        try:
            _check_writable(table, self.mode)
        except (NotImplementedError, ValueError) as e:
            return str(e)

        expected = _polars_schema(_table_schema(table))
        if set(schema.names()) != set(expected.names()):
            return "the data has other columns than the table"
        for name, dtype in expected.items():
            if schema[name] != dtype:
                return f"column {name!r} is {schema[name]}, and the table's is {dtype}"
        return None

    def attach_sink(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return wrap_ldf(lf._ldf.sink_delta(self))

    def sink_base_path(self) -> str:
        """The directory of this sink's data files."""
        return f"{self.table_uri.rstrip('/')}/{self.sink_uuid_str}/"

    def _attach_resolved_sink(self, plf: PyLazyFrame) -> PyLazyFrame:
        import polars as pl

        lf = wrap_ldf(plf)
        input_schema = lf.collect_schema()

        loaded = self.table_.get()
        fresh = loaded is not None and loaded.fresh
        routed = loaded is not None and loaded.routed
        if loaded is not None:
            loaded.fresh = False
        table = self._load_table(refresh=not fresh)
        # The files of a `write_delta` or `sink_delta` sink are written with the
        # credentials that it resolved into the storage options, like the commit,
        # rather than resolving them again.
        credential_provider: Literal["auto"] | None = None if routed else "auto"

        if table is not None:
            # Write no data files when there is nothing to commit: for `"ignore"`,
            # or when this sink already committed, e.g. when a plan that ran is
            # resolved again. A keyed sink of no rows writes no files.
            if self.mode == "ignore" or (not routed and self._landed(table)):
                self.partition_columns = table.metadata().partition_columns
                self.source_schema = _table_schema(table)
                return (
                    pl.LazyFrame(schema={"_": pl.Int8})
                    .sink_parquet(
                        pl.PartitionBy(
                            _normalize_windows_iceberg_file_uri(self.sink_base_path()),
                            file_path_provider=PlIcebergPathProviderConfig(),
                            key=[pl.col("_")],
                            include_key=False,
                        ),
                        storage_options=self.storage_options,
                        credential_provider=credential_provider,
                        lazy=True,
                    )
                    ._ldf
                )

            if self.mode == "error":
                msg = f"Delta table already exists at '{self.table_uri}'"
                raise FileExistsError(msg)

        casts = {
            name: target
            for name, dtype in input_schema.items()
            if (target := _delta_dtype(dtype)) != dtype
        }
        if casts:
            lf = lf.cast(casts)  # type: ignore[arg-type]
            input_schema = lf.collect_schema()

        if table is not None:
            _check_writable(table, self.mode)

            partition_columns = table.metadata().partition_columns
            table_schema = _table_schema(table)
            expected = _polars_schema(table_schema)

            if set(input_schema.names()) != set(expected.names()) or any(
                input_schema[name] != dtype for name, dtype in expected.items()
            ):
                msg = (
                    "schema of the data does not match the Delta table schema\n"
                    f"data: {input_schema}\n"
                    f"table: {expected}"
                )
                raise SchemaError(msg)

            lf = lf.select(expected.names())
        else:
            from deltalake import Schema

            # A new table is not partitioned.
            partition_columns = []
            table_schema = _strip_field_metadata(Schema.from_arrow(input_schema))

        self.partition_columns = partition_columns
        self.source_schema = table_schema

        return lf.sink_parquet(
            pl.PartitionBy(
                _normalize_windows_iceberg_file_uri(self.sink_base_path()),
                file_path_provider=PlIcebergPathProviderConfig(),
                key=[pl.col(c) for c in partition_columns] or None,
                include_key=False if partition_columns else None,
                approximate_bytes_per_file=2 * 1024 * 1024 * 1024,
            ),
            # The Parquet sink takes large strings, binaries and lists.
            arrow_schema=_file_schema(table_schema, partition_columns).to_arrow(
                as_large_types=True
            ),
            storage_options=self.storage_options,
            credential_provider=credential_provider,
            lazy=True,
        )._ldf

    def _add_actions(
        self,
        sinked_files: list[_DeltaSinkedFile],
        configuration: Mapping[str, str | None],
    ) -> list[AddAction]:
        from deltalake import Schema
        from deltalake.transaction import AddAction

        assert self.source_schema is not None
        assert self.partition_columns is not None

        partition_columns = self.partition_columns
        file_schema = _file_schema(self.source_schema, partition_columns)

        stats_leaves = _stats_leaf_paths(file_schema, configuration)
        non_nullable_leaves = _non_nullable_leaf_paths(file_schema)
        non_nullable_partitions = {
            field.name
            for field in Schema.from_json(self.source_schema).fields
            if field.name in partition_columns and not field.nullable
        }

        modification_time = int(time.time() * 1000)
        actions = []

        for file_path, num_rows, num_bytes, metadata_bytes, keys in sinked_files:
            metadata_num_rows, stats_json, null_counts = _delta_file_stats(
                metadata_bytes, list(stats_leaves)
            )

            if metadata_num_rows != num_rows:
                msg = (
                    f"native sink row count {num_rows} does not match Parquet "
                    f"metadata row count {metadata_num_rows} for '{file_path}'"
                )
                raise ComputeError(msg)

            leaf_null_counts = dict(null_counts)
            for leaf in non_nullable_leaves:
                if num_rows > 0 and leaf_null_counts.get(leaf) != 0:
                    msg = (
                        f"Delta sink: non-nullable column '{'.'.join(leaf)}' has nulls "
                        f"(or no null count) in '{file_path}'"
                    )
                    raise ComputeError(msg)

            partition_values: dict[str, str | None] = {}
            if partition_columns:
                if (
                    keys is None
                    or keys.columns != partition_columns
                    or keys.height != 1
                ):
                    msg = (
                        "Delta sink: partition keys of a file do not match the "
                        f"partition columns {partition_columns}: "
                        f"{None if keys is None else keys.schema}"
                    )
                    raise ComputeError(msg)
                for name in partition_columns:
                    value = _partition_value(keys.get_column(name))
                    if value is None and name in non_nullable_partitions:
                        msg = f"Delta sink: non-nullable partition column '{name}' has nulls"
                        raise ComputeError(msg)
                    partition_values[name] = value

            actions.append(
                AddAction(
                    path=_relative_data_path(file_path, self.sink_uuid_str),
                    size=num_bytes,
                    partition_values=partition_values,
                    modification_time=modification_time,
                    data_change=True,
                    stats=stats_json,
                )
            )

        return actions

    def commit(self, sinked_files: list[_DeltaSinkedFile]) -> pl.DataFrame:
        """Commit the sinked files to the table.

        Another commit of the same sink (same `sink_uuid_str`) is skipped: the
        commit records the sink uuid as an application transaction, which Delta
        keeps until `delta.setTransactionRetentionDuration` expires it.

        Returns the version that the commit created. It is null when the commit was
        skipped (`mode="ignore"`), when the version is no longer in the history, or
        for a sink of `write_delta` or `sink_delta`, which don't report it.
        """
        import polars as pl
        import polars._utils.logging

        function_start_instant = perf_counter()
        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(f"DeltaSinkState[commit]: mode: '{self.mode}'")

        version = self._commit(sinked_files, verbose=verbose)

        self.commit_result_df.set(
            pl.DataFrame({"version": [version]}, schema={"version": pl.Int64})
        )

        if verbose:
            total_elapsed = perf_counter() - function_start_instant
            eprint(
                f"DeltaSinkState[commit]: finished, total elapsed time: {total_elapsed:.3f}s"
            )

        return self.commit_result_df.get()  # type: ignore[return-value]

    def _commit(
        self, sinked_files: list[_DeltaSinkedFile], *, verbose: bool
    ) -> int | None:
        from deltalake import Schema
        from deltalake.transaction import (
            CommitProperties,
            Transaction,
            create_table_with_add_actions,
        )

        # The commit may run in another process than `new()`.
        _check_deltalake_apis()

        loaded = self.table_.get()
        routed = loaded is not None and loaded.routed
        if routed:
            assert loaded is not None
            loaded.routed = False
            # Commit on top of the table as it was resolved: deltalake checks the
            # commits since then for conflicts. Except for an overwrite of an
            # existing table: deltalake doesn't check for files appended since, so
            # the overwrite wouldn't remove them.
            table = self._load_table(
                refresh=self.mode == "overwrite" and loaded.table is not None
            )
        else:
            table = self._load_table()

        if table is not None and not routed and self._landed(table):
            if verbose:
                eprint("DeltaSinkState[commit]: already committed, skipping")
            return self._landed_version(table)

        if self.source_schema is None or self.partition_columns is None:
            msg = "Delta sink was not resolved before the commit"
            raise ComputeError(msg)

        if table is not None and self.mode in ("error", "ignore"):
            if self.mode == "ignore":
                if verbose:
                    eprint("DeltaSinkState[commit]: table exists, ignoring")
                return None
            # The table was created after the sink was resolved.
            msg = f"Delta table already exists at '{self.table_uri}'"
            raise FileExistsError(msg)

        commit_properties = CommitProperties(
            custom_metadata={_SINK_UUID_PROPERTY: self.sink_uuid_str},
            app_transactions=[Transaction(self.sink_uuid_str, 0)],
        )

        try:
            if table is None:
                actions = self._add_actions(sinked_files, {})
                if verbose:
                    eprint("DeltaSinkState[commit]: begin create table")
                try:
                    create_table_with_add_actions(
                        self.table_uri,
                        Schema.from_json(self.source_schema),
                        actions,
                        mode="error",
                        partition_by=self.partition_columns or None,
                        storage_options=self._delta_storage_options(),
                        commit_properties=commit_properties,
                    )
                except Exception:
                    # Another writer may have created the table meanwhile.
                    table = self._load_table()
                    if table is None or self._landed(table):
                        raise
                    if self.mode in ("error", "ignore"):
                        raise
                    if verbose:
                        eprint(
                            "DeltaSinkState[commit]: table created concurrently, "
                            "committing to it"
                        )
                else:
                    return 0

            self._check_unchanged(table)
            actions = self._add_actions(sinked_files, table.metadata().configuration)
            if verbose:
                eprint(
                    f"DeltaSinkState[commit]: begin write transaction ({len(actions)} files)"
                )
            read_version = table.version()
            table.create_write_transaction(
                actions,  # type: ignore[arg-type]
                "overwrite" if self.mode == "overwrite" else "append",
                table.schema(),
                partition_by=self.partition_columns or None,
                commit_properties=commit_properties,
            )
            if routed:
                # Not looked up: `write_delta` and `sink_delta` don't report it.
                return None
            return self._committed_version(table, read_version)
        except Exception as e:
            # The commit may have landed, e.g. when a post-commit hook failed, or a
            # concurrent retry of this sink committed first.
            try:
                current = self._load_table()
            except Exception as load_error:
                msg = (
                    "could not verify the outcome of a failed Delta commit; "
                    "the commit state is unknown"
                )
                raise ComputeError(msg) from load_error
            if current is not None and self._landed(current):
                if verbose:
                    eprint("DeltaSinkState[commit]: recovered landed commit")
                return self._landed_version(current)
            raise e from None

    def _landed(self, table: deltalake.DeltaTable) -> bool:
        return table.transaction_version(self.sink_uuid_str) is not None

    def _landed_version(
        self, table: deltalake.DeltaTable, limit: int | None = None
    ) -> int | None:
        """The version this sink committed, from the commit info in the history.

        Reads the commit info of the latest `limit` versions, or of all versions.
        """
        with contextlib.suppress(Exception):
            for entry in table.history(limit):
                if entry.get(_SINK_UUID_PROPERTY) == self.sink_uuid_str:
                    return entry.get("version")
        return None

    def _committed_version(
        self, table: deltalake.DeltaTable, read_version: int
    ) -> int | None:
        """The version this sink just committed on top of `read_version`.

        deltalake doesn't return it, and other commits may have landed before it.
        Only the versions after `read_version` are read: if there is one, it is the
        sink's.
        """
        table.update_incremental()
        new_versions = table.version() - read_version
        if new_versions == 1:
            return table.version()
        return self._landed_version(table, limit=new_versions)

    def _check_unchanged(self, table: deltalake.DeltaTable) -> None:
        """Raise if the table changed in a way that conflicts with the written files."""
        assert self.source_schema is not None

        partition_columns = table.metadata().partition_columns
        if partition_columns != self.partition_columns:
            msg = (
                "Delta table partition columns changed during the sink: "
                f"{self.partition_columns} -> {partition_columns}"
            )
            raise SchemaError(msg)

        table_schema = _table_schema(table)
        if json.loads(table_schema) != json.loads(self.source_schema):
            msg = (
                "Delta table schema changed during the sink\n"
                f"written: {self.source_schema}\n"
                f"table: {table_schema}"
            )
            raise SchemaError(msg)

        _check_writable(table, self.mode)

    def remove_written_files(self) -> str | None:
        """Delete the data files of a sink whose commit did not land, best effort.

        Only local tables are supported. Returns the directory where data files may
        remain, or None if they were removed.
        """
        import shutil

        location = self.sink_base_path()
        try:
            table = self._load_table()
            if table is not None and self._landed(table):
                return location
        except Exception:
            return location

        if not _is_local_uri(location):
            return location
        try:
            shutil.rmtree(_local_path(location), ignore_errors=False)
        except FileNotFoundError:
            pass
        except Exception:
            return location
        return None


def _native_sink_state(
    target: str | deltalake.DeltaTable,
    *,
    mode: str,
    storage_options: Mapping[str, Any] | None,
    delta_write_options: Mapping[str, Any] | None,
    schema: pl.Schema,
) -> DeltaSinkState | None:
    """A sink state for `write_delta` and `sink_delta`, or None to write with deltalake.

    The native sink takes only data that needs no cast. deltalake casts with arrow-rs,
    whose results differ from Polars casts, e.g. in rounding and in parsing strings.
    deltalake also writes when there are `delta_write_options`, or table features
    that the sink doesn't support.

    Parameters
    ----------
    target
        The table URI, resolved, or a DeltaTable.
    mode
        The write mode, passed to the sink state or to deltalake.
    storage_options
        Storage options that include the credentials of the credential provider.
    delta_write_options
        Keyword arguments for `deltalake.write_deltalake`.
    schema
        The schema of the data.
    """
    from deltalake import DeltaTable

    state = None
    reason: str | None
    if mode not in ("error", "append", "overwrite", "ignore"):
        reason = f"mode is {mode!r}"
    elif delta_write_options:
        reason = f"delta_write_options are given: {sorted(delta_write_options)}"
    elif missing := _missing_deltalake_apis():
        reason = f"deltalake lacks {', '.join(missing)}"
    elif isinstance(target, DeltaTable) and not _is_local_uri(target.table_uri):
        # The sink would write the data files without the table's storage options.
        reason = "the target is a DeltaTable of a remote table"
    elif casts := [name for name, dtype in schema.items() if not _is_delta_type(dtype)]:
        reason = f"column {casts[0]!r} of type {schema[casts[0]]} needs a cast"
    else:
        state = DeltaSinkState.new(
            target,
            mode=mode,  # type: ignore[arg-type]
            storage_options=dict(storage_options) if storage_options else None,
            credential_provider=None,
        )
        reason = state._deltalake_write_reason(schema)

    if reason is None:
        assert state is not None
        # The check above loaded the table, for a sink created just now.
        loaded = state.table_.get()
        assert loaded is not None
        loaded.routed = loaded.fresh = True
        return state
    if verbose():
        eprint(f"Delta write uses deltalake, as {reason}")
    return None

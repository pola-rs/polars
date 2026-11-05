# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import copy
import gzip
import io
import itertools
import json
import math
import os
import pickle
import re
import struct
import sys
import uuid
import warnings
import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal as D
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import pyiceberg
import pyiceberg.exceptions
import pyiceberg.table
import pytest
from pyiceberg.expressions import literal
from pyiceberg.partitioning import (
    BucketTransform,
    DayTransform,
    HourTransform,
    IdentityTransform,
    MonthTransform,
    PartitionField,
    PartitionSpec,
    TruncateTransform,
    VoidTransform,
    YearTransform,
)
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.table import StaticTable
from pyiceberg.table.sorting import NullOrder, SortDirection, SortField, SortOrder
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampNanoType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)

import polars as pl
from polars._utils.various import parse_version
from polars.io._expand_paths import _expand_paths
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergScanResolver,
    IcebergScanTableSerializer,
    IcebergTableWrap,
    _NativeIcebergScanData,
)
from polars.io.iceberg._sink import (
    _SINK_UUID_PROPERTY,
    IcebergSinkState,
    PlIcebergPathProviderConfig,
)
from polars.io.iceberg._utils import (
    _convert_predicate,
    _new_pyiceberg_scan,
    _normalize_windows_iceberg_file_uri,
    _to_ast,
    try_convert_pyarrow_predicate,
)
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.io.conftest import normalize_path_separator_pl
from tests.unit.io.test_scan_row_deletion import write_position_deletes  # noqa: F401


@pytest.fixture(autouse=True)
def _pyiceberg_planner(plmonkeypatch: PlMonkeyPatch) -> None:
    # Tests the PyIceberg planner, independent of whether `polars_iceberg` is installed
    # (`test_iceberg_plugin_suite.py` re-runs these tests with the plugin).
    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", "pyiceberg")


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pyiceberg.table.snapshots import Snapshot

    from polars._typing import EngineType
    from tests.conftest import PlMonkeyPatch
    from tests.unit.io.test_scan_row_deletion import (
        WritePositionDeletes,
    )

    # Mypy does not understand the constructors and we can't construct the inputs
    # explicitly since they are abstract base classes.
    AlwaysFalse: Any
    AlwaysTrue: Any
    And: Any
    EqualTo: Any
    GreaterThan: Any
    GreaterThanOrEqual: Any
    In: Any
    IsNaN: Any
    IsNull: Any
    LessThan: Any
    LessThanOrEqual: Any
    Not: Any
    NotEqualTo: Any
    NotNaN: Any
    Or: Any
    Reference: Any
else:
    from pyiceberg.expressions import (
        AlwaysFalse,
        AlwaysTrue,
        And,
        EqualTo,
        GreaterThan,
        GreaterThanOrEqual,
        In,
        IsNaN,
        IsNull,
        LessThan,
        LessThanOrEqual,
        Not,
        NotEqualTo,
        NotNaN,
        Or,
        Reference,
    )


with warnings.catch_warnings():
    # Upstream issue at https://github.com/apache/iceberg-python/issues/2648.
    warnings.simplefilter("ignore", pydantic.warnings.PydanticDeprecatedSince212)
    # Upstream issue at https://github.com/apache/iceberg-python/issues/2849.
    warnings.simplefilter("ignore", DeprecationWarning)
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.io.pyarrow import schema_to_pyarrow


@pytest.fixture(autouse=True)
def sqlcatalog_uses_null_pool_connections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # note: SqlCatalog is sqlite-backed, and pyiceberg builds its engine with
    # a SingletonThreadPool that can raise a ResourceWarning on test shutdown;
    # we ensure it uses a NullPool instead, resolving the connection "leak"
    import pyiceberg.catalog.sql as pyiceberg_sql
    from sqlalchemy.pool import NullPool

    tmp_path = tmp_path_factory.mktemp("iceberg_catalogs")
    original_create_engine = pyiceberg_sql.create_engine
    counter = itertools.count()

    def create_engine(url: str, **kwargs: Any) -> Any:
        if url == "sqlite:///:memory:":
            # need to use a file-backed catalog, otherwise NullPool connections
            # will be working with different in-memory databases on each call
            url = f"sqlite:///{tmp_path}/catalog_{next(counter)}.db"

        kwargs["poolclass"] = NullPool
        return original_create_engine(url, **kwargs)

    monkeypatch.setattr(pyiceberg_sql, "create_engine", create_engine)


def new_iceberg_scan_resolver(
    source: str | pyiceberg.table.Table,
) -> IcebergScanResolver:
    return IcebergScanResolver(
        table=IcebergTableWrap(
            table_=NoPickleOption(
                source if isinstance(source, pyiceberg.table.Table) else None
            ),
            table_descriptor_=source
            if not isinstance(source, pyiceberg.table.Table)
            else None,
            serializer=IcebergScanTableSerializer(),
            iceberg_storage_properties=None,
        ),
        snapshot_id=None,
        from_snapshot_id_exclusive=None,
        to_snapshot_id_inclusive=None,
        reader_override=None,
        use_metadata_statistics=True,
        fast_deletion_count=False,
        use_pyiceberg_filter=True,
    )


def resolve_scan(
    tbl: pyiceberg.table.Table, **kwargs: Any
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """
    Resolve the scan as the engine does, returning it and its `unified_scan_args`.

    Unlike `_to_dataset_scan_impl()`, this uses the plugin planner when re-run by
    `test_iceberg_plugin_suite.py`.
    """
    resolved = new_iceberg_scan_resolver(tbl).to_dataset_scan(**kwargs)
    assert resolved is not None
    lf, _ = resolved

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'json' serialization format")
        plan = json.loads(lf.serialize(format="json"))

    return lf, plan["Scan"]["unified_scan_args"]


def resolved_table_statistics(
    tbl: pyiceberg.table.Table, **kwargs: Any
) -> pl.DataFrame | None:
    """Min/max statistics passed to the resolved scan."""
    _, scan_args = resolve_scan(tbl, **kwargs)
    statistics = scan_args["table_statistics"]
    return None if statistics is None else pl.read_ipc_stream(bytes(statistics))


def new_iceberg_table(
    tmp_path: Path,
    *,
    schema: IcebergSchema,
    partition_spec: PartitionSpec | None = None,
    sort_order: SortOrder | None = None,
    name: str = "table",
    properties: dict[str, str] | None = None,
) -> tuple[pyiceberg.table.Table, SqlCatalog]:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    namespace = uuid.uuid4().bytes.hex()
    catalog.create_namespace(namespace)

    create_table_kwargs: dict[str, Any] = {"properties": properties or {}}
    if partition_spec is not None:
        create_table_kwargs["partition_spec"] = partition_spec
    if sort_order is not None:
        create_table_kwargs["sort_order"] = sort_order

    return catalog.create_table(
        (namespace, name), schema, **create_table_kwargs
    ), catalog


def nested_field_by_path(
    field_path: list[str], field_ids: Iterator[int]
) -> NestedField:
    """Build a (possibly nested) `NestedField` wrapping a `LongType` leaf.

    E.g. `["mydict", "age"]` -> a `mydict` struct field containing `age`;
    `["a", "b", "c"]` -> `a` containing `b` containing `c`.
    """
    name = field_path[0]
    field_id = next(field_ids)
    if len(field_path) == 1:
        return NestedField(field_id, name, LongType())
    return NestedField(
        field_id,
        name,
        StructType(nested_field_by_path(field_path[1:], field_ids)),
        required=False,
    )


def nested_cell(field_path: list[str], value: Any) -> Any:
    """The row value for `field_path[0]`'s column matching `nested_field_by_path`."""
    if len(field_path) == 1:
        return value
    return {field_path[1]: nested_cell(field_path[1:], value)}


# PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
def format_file_uri_iceberg(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha()
        assert absolute_local_path[1] == ":"
        p = absolute_local_path.replace("\\", "/")
        return f"file://{p}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


@pytest.fixture
def iceberg_path(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")  # noqa: PTH211

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return format_file_uri_iceberg(f"{iceberg_path.resolve()}")


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only
class TestIcebergScanIO:
    """Test coverage for `iceberg` scan ops."""

    def test_scan_iceberg_plain(self, iceberg_path: str) -> None:
        q = pl.scan_iceberg(iceberg_path)
        assert len(q.collect()) == 3
        assert q.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id(self, iceberg_path: str) -> None:
        q = pl.scan_iceberg(iceberg_path, snapshot_id=7051579356916758811)
        assert len(q.collect()) == 3
        assert q.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id_not_found(self, iceberg_path: str) -> None:
        with pytest.raises(ValueError, match="snapshot ID not found"):
            pl.scan_iceberg(iceberg_path, snapshot_id=1234567890).collect()

    def test_scan_iceberg_filter_on_partition(self, iceberg_path: str) -> None:
        ts1 = datetime(2023, 3, 1, 18, 15)
        ts2 = datetime(2023, 3, 1, 19, 25)
        ts3 = datetime(2023, 3, 2, 22, 0)

        lf = pl.scan_iceberg(iceberg_path)

        res = lf.filter(pl.col("ts") >= ts2)
        assert len(res.collect()) == 2

        res = lf.filter(pl.col("ts") > ts2).select(pl.col("id"))
        assert res.collect().rows() == [(3,)]

        res = lf.filter(pl.col("ts") <= ts2).select("id", "ts")
        assert res.collect().rows(named=True) == [
            {"id": 1, "ts": ts1},
            {"id": 2, "ts": ts2},
        ]

        res = lf.filter(pl.col("ts") > ts3)
        assert len(res.collect()) == 0

        for constraint in (
            (pl.col("ts") == ts1) | (pl.col("ts") == ts3),
            pl.col("ts").is_in([ts1, ts3]),
        ):
            res = lf.filter(constraint).select("id")
            assert res.collect().rows() == [(1,), (3,)]

    def test_scan_iceberg_filter_on_column(self, iceberg_path: str) -> None:
        lf = pl.scan_iceberg(iceberg_path)
        res = lf.filter(pl.col("id") < 2)
        assert res.collect().rows() == [(1, "1", datetime(2023, 3, 1, 18, 15))]

        res = lf.filter(pl.col("id") == 2)
        assert res.collect().rows() == [(2, "2", datetime(2023, 3, 1, 19, 25))]

        res = lf.filter(pl.col("id").is_in([1, 3]))
        assert res.collect().rows() == [
            (1, "1", datetime(2023, 3, 1, 18, 15)),
            (3, "3", datetime(2023, 3, 2, 22, 0)),
        ]

        res = lf.filter(pl.col("id") != 2)
        assert res.collect().rows() == [
            (1, "1", datetime(2023, 3, 1, 18, 15)),
            (3, "3", datetime(2023, 3, 2, 22, 0)),
        ]

    def test_scan_iceberg_filter_starts_with(self, tmp_path: Path) -> None:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "s", StringType()))
        )
        pl.DataFrame({"s": ["apple", "banana", "berry"]}).write_iceberg(
            tbl, mode="append"
        )

        res = pl.scan_iceberg(tbl).filter(pl.col("s").str.starts_with("be"))
        assert res.collect()["s"].to_list() == ["berry"]

    def test_scan_iceberg_filter_is_nan(
        self,
        tmp_path: Path,
        plmonkeypatch: PlMonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        schema = IcebergSchema(NestedField(1, "value", DoubleType(), required=False))
        tbl, _ = new_iceberg_table(tmp_path, schema=schema)
        data = pl.DataFrame({"value": [None, 1.0, float("nan"), 2.0]})
        data.write_iceberg(tbl, mode="append")
        plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

        capfd.readouterr()
        res = pl.scan_iceberg(tbl).filter(pl.col("value").is_nan())
        [(value,)] = res.collect().rows()
        assert math.isnan(value)
        assert f"iceberg_table_filter = {IsNaN('value')!r}" in capfd.readouterr().err

        res = pl.scan_iceberg(tbl).filter(pl.col("value").is_not_nan())
        assert res.collect()["value"].to_list() == [1.0, 2.0]
        assert (
            f"iceberg_table_filter = {Not(IsNaN('value'))!r}" in capfd.readouterr().err
        )

    @pytest.mark.parametrize(
        "field_path",
        [
            ["mydict", "age"],
            ["a", "b", "c"],  # `struct_field_path` recurses - depth 2+.
        ],
    )
    def test_scan_iceberg_filter_struct_field(
        self, tmp_path: Path, field_path: list[str]
    ) -> None:
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(
                NestedField(1, "id", LongType()),
                nested_field_by_path(field_path, itertools.count(2)),
            ),
        )
        pl.DataFrame(
            {
                "id": [1, 2, 3],
                field_path[0]: [
                    nested_cell(field_path, 17),
                    nested_cell(field_path, 42),
                    None,
                ],
            }
        ).write_iceberg(tbl, mode="append")

        expr = pl.col(field_path[0])
        for name in field_path[1:]:
            expr = expr.struct.field(name)

        res = pl.scan_iceberg(tbl).filter(expr == 17).select("id")
        assert res.collect().rows() == [(1,)]

        res = pl.scan_iceberg(tbl).filter(expr.is_null()).select("id")
        assert res.collect().rows() == [(3,)]

    def test_scan_iceberg_filter_null_struct_with_required_child(
        self, tmp_path: Path
    ) -> None:
        # Null struct, required child: PyIceberg's BoundIsNull would wrongly
        # prune this, so we must decline to push it down.
        catalog = SqlCatalog(
            "default",
            uri="sqlite:///:memory:",
            warehouse=format_file_uri_iceberg(tmp_path),
        )
        catalog.create_namespace("ns")

        dtype = pa.struct([pa.field("x", pa.int64(), nullable=False)])
        data = pa.table({"s": pa.array([None], type=dtype)})
        tbl = catalog.create_table("ns.t", schema=data.schema)
        tbl.append(data)

        q = pl.scan_iceberg(tbl).filter(pl.col("s").struct.field("x").is_null())
        expected = [(None,)]

        assert q.collect().rows() == expected
        assert (
            q.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False)).rows()
            == expected
        )

    def test_scan_iceberg_filter_struct_field_unsanitizable_name(
        self,
        tmp_path: Path,
        plmonkeypatch: PlMonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        # Unsafe name must not be pushed down.
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(
                NestedField(1, "id", LongType()),
                NestedField(
                    2,
                    "mydict",
                    StructType(NestedField(3, "a!", LongType())),
                    required=False,
                ),
            ),
        )
        plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

        capfd.readouterr()
        pl.scan_iceberg(tbl).filter(pl.col("mydict").struct.field("a!") == 17).select(
            "id"
        ).explain()
        log = capfd.readouterr().err
        assert "pyarrow_predicate = None" in log
        assert "iceberg_table_filter = None" in log

    @pytest.mark.parametrize("method", ["is_nan", "is_not_nan"])
    def test_scan_iceberg_nan_decimal_rejected(
        self, tmp_path: Path, method: str
    ) -> None:
        # `is_nan`/`is_not_nan` on a Decimal column must raise, same as every
        # other Polars path - not be silently dropped by predicate pushdown.
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(NestedField(1, "value", DecimalType(10, 1))),
        )
        pl.DataFrame(
            {"value": [D("1.0"), D("2.0")]}, schema={"value": pl.Decimal(10, 1)}
        ).write_iceberg(tbl, mode="append")

        predicate = getattr(pl.col("value"), method)()
        with pytest.raises(
            pl.exceptions.InvalidOperationError,
            match=rf"`{method}` operation not supported for dtype `decimal",
        ):
            pl.scan_iceberg(tbl).filter(predicate).collect()

    def test_scan_iceberg_filter_starts_with_non_literal_prefix(
        self, tmp_path: Path
    ) -> None:
        # A column (not a literal) prefix cannot be pushed down; Polars must
        # still filter correctly afterwards.
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(
                NestedField(1, "s", StringType()),
                NestedField(2, "prefix", StringType()),
            ),
        )
        pl.DataFrame(
            {"s": ["apple", "banana", "berry"], "prefix": ["x", "b", "be"]}
        ).write_iceberg(tbl, mode="append")

        res = pl.scan_iceberg(tbl).filter(pl.col("s").str.starts_with(pl.col("prefix")))
        assert res.collect()["s"].to_list() == ["banana", "berry"]

    def test_scan_iceberg_filter_starts_with_null_prefix(self, tmp_path: Path) -> None:
        # A `None` prefix has no fixed value to push down; the row-wise
        # comparison (always null/false) must still be applied by Polars.
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "s", StringType()))
        )
        pl.DataFrame({"s": ["apple", "banana"]}).write_iceberg(tbl, mode="append")

        res = pl.scan_iceberg(tbl).filter(
            pl.col("s").str.starts_with(pl.lit(None, dtype=pl.String))
        )
        assert res.collect()["s"].to_list() == []

    def test_scan_iceberg_noteq_null_and_nan(self, tmp_path: Path) -> None:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "value", DoubleType()))
        )
        pl.DataFrame({"value": [1.0, None, float("nan")]}).write_iceberg(
            tbl, mode="append"
        )

        [(value,)] = (
            pl.scan_iceberg(tbl.metadata_location)
            .filter(pl.col("value") != 1.0)
            .collect()
            .rows()
        )
        assert math.isnan(value)

    def test_scan_iceberg_filter_is_in_empty(self, tmp_path: Path) -> None:
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(
                NestedField(1, "my_column", LongType(), required=False)
            ),
        )

        tbl.append(
            pa.table(
                {"my_column": [1, 2, 3, 4, 5]},
                schema=pa.schema([("my_column", pa.int64())]),
            )
        )

        result = pl.scan_iceberg(tbl).filter(pl.col("my_column").is_in([])).collect()

        assert result.is_empty()
        assert result.schema == {"my_column": pl.Int64}


@pytest.mark.ci_only
class TestIcebergExpressions:
    """Test coverage for `iceberg` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        expr = _to_ast("(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == IsNull("id")

    def test_is_not_null_expression(self) -> None:
        expr = _to_ast("~(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == Not(IsNull("id"))

    def test_isin_expression(self) -> None:
        expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
        assert _convert_predicate(expr) == In(
            "id", {literal(1), literal(2), literal(3)}
        )

    def test_parse_combined_expression(self) -> None:
        expr = _to_ast(
            "(((pa.compute.field('str') == '2') & (pa.compute.field('id') > 10)) | (pa.compute.field('id')).isin([1,2,3]))"
        )
        assert _convert_predicate(expr) == Or(
            left=And(
                left=EqualTo(term=Reference(name="str"), literal=literal("2")),
                right=GreaterThan(term="id", literal=literal(10)),
            ),
            right=In("id", {literal(1), literal(2), literal(3)}),
        )

    def test_parse_gt(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_noteq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') != '2023-08-08')")
        assert _convert_predicate(expr) == NotEqualTo("ts", "2023-08-08")

        assert try_convert_pyarrow_predicate(
            "(pa.compute.field('ts') != '2023-08-08')"
        ) == NotEqualTo("ts", "2023-08-08")

    def test_parse_ne_missing(self) -> None:
        expr = try_convert_pyarrow_predicate(
            "((pa.compute.field('ts') != '2023-08-08') | (pa.compute.field('ts')).is_null())"
        )
        assert expr == Or(NotEqualTo("ts", "2023-08-08"), IsNull("ts"))

    def test_parse_lt(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

    def test_parse_negative_literal(self) -> None:
        expr = _to_ast("(pa.compute.field('id') > -5)")
        assert _convert_predicate(expr) == GreaterThan("id", -5)

        expr = _to_ast("(pa.compute.field('id')).isin([-1,2])")
        assert _convert_predicate(expr) == In("id", {literal(-1), literal(2)})

        expr = _to_ast("(pa.compute.field('ts') < to_py_datetime(-123,'us'))")
        assert _convert_predicate(expr) == LessThan("ts", "1969-12-31T23:59:59.999877")

    def test_compare_boolean(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(True))")
        assert _convert_predicate(expr) == EqualTo("ts", True)

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(False))")
        assert _convert_predicate(expr) == EqualTo("ts", False)

    def test_bare_boolean_field(self) -> None:
        expr = try_convert_pyarrow_predicate("pa.compute.field('is_active')")
        assert expr == EqualTo("is_active", True)

    def test_bare_boolean_field_negated(self) -> None:
        expr = try_convert_pyarrow_predicate("~pa.compute.field('is_active')")
        assert expr == Not(EqualTo("is_active", True))

    def test_scalar_false_expression(self) -> None:
        expr = try_convert_pyarrow_predicate("pa.compute.scalar(False)")
        assert expr == AlwaysFalse()

    def test_scalar_true_expression(self) -> None:
        expr = try_convert_pyarrow_predicate("pa.compute.scalar(True)")
        assert expr == AlwaysTrue()

    def test_unconvertible_conjunct_is_dropped(self) -> None:
        # PyIceberg has no arithmetic, but dropping that conjunct only widens the
        # filter - the engine re-applies the full predicate after the scan.
        expr = try_convert_pyarrow_predicate(
            "((pa.compute.field('id') > 10) & ((pa.compute.field('id') * 2) > 4))"
        )
        assert expr == GreaterThan("id", 10)

        expr = try_convert_pyarrow_predicate(
            "(((pa.compute.field('id') * 2) > 4) & (pa.compute.field('id') > 10) "
            "& (pa.compute.field('id')).isin([1,2,3]))"
        )
        assert expr == And(
            GreaterThan("id", 10), In("id", {literal(1), literal(2), literal(3)})
        )

    def test_fully_unconvertible_predicate(self) -> None:
        expr = try_convert_pyarrow_predicate("((pa.compute.field('id') * 2) > 4)")
        assert expr is None

    def test_unparsable_predicate(self) -> None:
        # Not valid Python at all - nothing to convert.
        expr = try_convert_pyarrow_predicate("pa.compute.field('id') >")
        assert expr is None

    def test_convert_nested_struct_field_predicate(self) -> None:
        expr = try_convert_pyarrow_predicate(
            "(pa.compute.field('mydict', 'age') == 17)"
        )
        assert expr == EqualTo("mydict.age", 17)

        expr = try_convert_pyarrow_predicate(
            "(pa.compute.field('mydict', 'age')).is_null()"
        )
        assert expr == IsNull("mydict.age")

    def test_unconvertible_disjunct_is_not_dropped(self) -> None:
        # Unlike a conjunct, dropping one side of an `|` would narrow the filter.
        expr = try_convert_pyarrow_predicate(
            "((pa.compute.field('id') > 10) | ((pa.compute.field('id') * 2) > 4))"
        )
        assert expr is None


@pytest.mark.parametrize(
    ("predicate", "expected_factory"),
    [
        ("(pa.compute.field('value') == NaN)", lambda: IsNaN("value")),
        ("(pa.compute.field('value') != NaN)", lambda: NotNaN("value")),
        (
            "((pa.compute.field('value') != NaN) | "
            "(pa.compute.field('value')).is_null())",
            lambda: Or(NotNaN("value"), IsNull("value")),
        ),
        ("~(pa.compute.field('value') != NaN)", lambda: Not(NotNaN("value"))),
        ("(pa.compute.field('value') > NaN)", lambda: None),
        ("(pa.compute.field('value') <= NaN)", lambda: None),
        (
            "(pa.compute.field('value') == 'NaN')",
            lambda: EqualTo("value", "NaN"),
        ),
        ("(pa.compute.field('value')).is_nan()", lambda: IsNaN("value")),
        ("~(pa.compute.field('value')).is_nan()", lambda: Not(IsNaN("value"))),
    ],
)
def test_convert_nan_predicate(
    predicate: str, expected_factory: Callable[[], Any]
) -> None:
    assert try_convert_pyarrow_predicate(predicate) == expected_factory()


@dataclass(kw_only=True)
class _TableDataAllTypes:
    iceberg_schema: IcebergSchema
    arrow_schema: pa.Schema
    arrow_table: pa.Table
    polars_df: pl.DataFrame

    @staticmethod
    def new(*, exclude_uuid: bool = False) -> _TableDataAllTypes:
        include_decimal_and_fixed = parse_version(pyiceberg.__version__) >= (0, 10, 0)
        next_field_id = partial(next, itertools.count(1))

        iceberg_schema = IcebergSchema(
            NestedField(next_field_id(), "height_provider", IntegerType()),
            NestedField(next_field_id(), "BooleanType", BooleanType()),
            NestedField(next_field_id(), "IntegerType", IntegerType()),
            NestedField(next_field_id(), "LongType", LongType()),
            NestedField(next_field_id(), "FloatType", FloatType()),
            NestedField(next_field_id(), "DoubleType", DoubleType()),
            NestedField(next_field_id(), "DateType", DateType()),
            NestedField(next_field_id(), "TimeType", TimeType()),
            NestedField(next_field_id(), "TimestampType", TimestampType()),
            NestedField(next_field_id(), "TimestamptzType", TimestamptzType()),
            NestedField(next_field_id(), "StringType", StringType()),
            NestedField(next_field_id(), "BinaryType", BinaryType()),
            *(
                [
                    NestedField(next_field_id(), "DecimalType", DecimalType(18, 2)),
                    NestedField(next_field_id(), "FixedType", FixedType(1)),
                ]
                if include_decimal_and_fixed
                else []
            ),
            *(
                [NestedField(next_field_id(), "UUIDType", UUIDType())]
                if not exclude_uuid
                else []
            ),
        )

        arrow_table = pa.Table.from_pydict(
            {
                "height_provider": [1],
                "BooleanType": [True],
                "IntegerType": [1],
                "LongType": [1],
                "FloatType": [1.0],
                "DoubleType": [1.0],
                "DateType": [date(2025, 1, 1)],
                "TimeType": [time(11, 30)],
                "TimestampType": [datetime(2025, 1, 1)],
                "TimestamptzType": [datetime(2025, 1, 1)],
                "StringType": ["A"],
                "BinaryType": [b"A"],
                **(
                    {"DecimalType": [D("1.0")], "FixedType": [b"A"]}
                    if include_decimal_and_fixed
                    else {}
                ),
                **({"UUIDType": [b"0000111100001111"]} if not exclude_uuid else {}),
            },
            schema=schema_to_pyarrow(iceberg_schema, include_field_ids=False),
        )

        polars_df = pl.DataFrame(
            [
                pl.Series('height_provider', [1], dtype=pl.Int32),
                pl.Series('BooleanType', [True], dtype=pl.Boolean),
                pl.Series('IntegerType', [1], dtype=pl.Int32),
                pl.Series('LongType', [1], dtype=pl.Int64),
                pl.Series('FloatType', [1.0], dtype=pl.Float32),
                pl.Series('DoubleType', [1.0], dtype=pl.Float64),
                pl.Series('DateType', [date(2025, 1, 1)], dtype=pl.Date),
                pl.Series('TimeType', [time(11, 30)], dtype=pl.Time),
                pl.Series('TimestampType', [datetime(2025, 1, 1, 0, 0)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
                pl.Series('TimestamptzType', [datetime(2025, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))], dtype=pl.Datetime(time_unit='us', time_zone='UTC')),
                pl.Series('StringType', ['A'], dtype=pl.String),
                pl.Series('BinaryType', [b'A'], dtype=pl.Binary),
                *(
                    [
                        pl.Series('DecimalType', [D('1.00')], dtype=pl.Decimal(precision=18, scale=2)),
                        pl.Series('FixedType', [b'A'], dtype=pl.Binary),
                    ]
                    if include_decimal_and_fixed
                    else []
                ),
                *(
                    [pl.Series('UUIDType', [b"0000111100001111"], dtype=pl.Binary)]
                    if not exclude_uuid
                    else []
                )
            ]
        )  # fmt: skip

        arrow_schema = schema_to_pyarrow(iceberg_schema)

        return _TableDataAllTypes(
            iceberg_schema=iceberg_schema,
            arrow_schema=arrow_schema,
            arrow_table=arrow_table,
            polars_df=polars_df,
        )


@pytest.mark.write_disk
def test_iceberg_sink_parquet_arrow_schema_roundtrip_all_iceberg_types(
    tmp_path: Path,
) -> None:
    table_data = _TableDataAllTypes.new()
    iceberg_schema = table_data.iceberg_schema

    tbl, _ = new_iceberg_table(tmp_path, schema=iceberg_schema)

    data_file_path = str(tmp_path / "data.parquet")

    table_data.polars_df.lazy().sink_parquet(
        data_file_path,
        arrow_schema=table_data.arrow_schema,
    )

    tbl.add_files(
        [format_file_uri_iceberg(data_file_path)], check_duplicate_files=False
    )

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), table_data.polars_df)
    assert_frame_equal(pl.DataFrame(tbl.scan().to_arrow()), table_data.polars_df)


@pytest.mark.write_disk
def test_sink_iceberg_all_types(tmp_path: Path) -> None:
    table_data = _TableDataAllTypes.new()

    tbl, _ = new_iceberg_table(tmp_path, schema=table_data.iceberg_schema)

    original_md_path = tbl.metadata_location

    new_md_path = table_data.polars_df.lazy().sink_iceberg(tbl, mode="append").item()
    assert tbl.metadata_location != original_md_path
    assert new_md_path == tbl.metadata_location

    table_data.polars_df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(),
        pl.concat(2 * [table_data.polars_df]),
    )

    table_data.polars_df.lazy().sink_iceberg(tbl, mode="overwrite")

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(),
        table_data.polars_df,
    )


def _scan_sorted(tbl: pyiceberg.table.Table) -> pl.DataFrame:
    tbl.refresh()
    return pl.scan_iceberg(tbl).collect().sort(pl.all(), nulls_last=True)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "overwrite_filter",
    [
        "a != 2",
        pyiceberg.expressions.NotEqualTo("a", 2),  # type: ignore[call-arg, arg-type]
        pl.col("a") != 2,
    ],
)
def test_sink_iceberg_overwrite_filter(tmp_path: Path, overwrite_filter: Any) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "s", StringType())
        ),
    )
    pl.DataFrame({"a": [1, 2, None], "s": ["x", "y", "z"]}).lazy().sink_iceberg(
        tbl, mode="append"
    )
    snapshot_count = len(tbl.metadata.snapshots)

    pl.DataFrame({"a": [5], "s": ["new"]}).lazy().sink_iceberg(
        tbl, mode="overwrite", overwrite_filter=overwrite_filter
    )

    # Null does not match `a != 2`, same as a Polars filter.
    assert_frame_equal(
        _scan_sorted(tbl),
        pl.DataFrame({"a": [2, 5, None], "s": ["y", "new", "z"]}),
    )
    # The delete and the append are committed together.
    new_snapshots = tbl.metadata.snapshots[snapshot_count:]
    assert [s.summary.operation.value for s in new_snapshots] == [  # type: ignore[union-attr]
        "overwrite",
        "append",
    ]
    assert len({s.sequence_number for s in new_snapshots}) == 2
    assert tbl.metadata.snapshot_log[-1].snapshot_id == new_snapshots[-1].snapshot_id


def _overwrite_filter_table(
    tmp_path: Path,
) -> tuple[pyiceberg.table.Table, pl.DataFrame]:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "s", StringType()),
            NestedField(3, "d", DateType()),
            NestedField(4, "f", DoubleType()),
            NestedField(5, "t", TimestampType()),
            NestedField(6, "tz", TimestamptzType()),
            NestedField(7, "n", StructType(NestedField(8, "x", LongType()))),
        ),
    )
    t1, t2 = datetime(2026, 10, 1, 1, 2, 3, 456789), datetime(2026, 10, 2)
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, None, 1, 4],
            "s": ["x", "y", None, "xz", "y", "x"],
            "d": [date(2026, 10, 1), None, date(2026, 10, 2)] * 2,
            "f": [1.0, float("nan"), None, 2.0, 3.0, 4.0],
            "t": [t1, t2, None] * 2,
            "tz": pl.Series([t1, None, t2] * 2).dt.replace_time_zone("UTC"),
            "n": [{"x": 1}, None, {"x": None}] * 2,
        }
    )
    # Multiple files, so that both whole-file deletes and rewrites happen.
    for i in range(0, df.height, 2):
        df.slice(i, 2).lazy().sink_iceberg(tbl, mode="append")
    return tbl, df


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("a") == 1,
        pl.col("a") != 1,
        pl.col("a") > 1,
        pl.col("a").is_in([1, 3]),
        ~pl.col("a").is_in([1, 3]),
        pl.col("a").is_null(),
        pl.col("a").is_not_null(),
        pl.col("a").is_between(1, 3),
        (pl.col("a") == 1) | (pl.col("a") == 3),
        (pl.col("a") == 1) | pl.col("a").is_null(),
        (pl.col("a") == 1) & (pl.col("s") == "x"),
        (pl.col("a") > 1) & ~(pl.col("s") == "y"),
        pl.col("s").str.starts_with("x"),
        pl.col("d") == date(2026, 10, 1),
        pl.col("d") >= date(2026, 10, 2),
        (pl.col("d") == date(2026, 10, 1)) & pl.col("s").is_null(),
        # Nulls in one column must not decide the other side of `|`.
        (pl.col("a") == 1) | (pl.col("s") == "x"),
        (pl.col("a") != 1) | (pl.col("s") != "x"),
        ~((pl.col("a") == 1) & (pl.col("s") == "x")),
        ~((pl.col("a") > 2) | pl.col("d").is_null()),
        ~pl.col("s").str.starts_with("x") & (pl.col("d") != date(2026, 10, 2)),
        pl.col("t") == datetime(2026, 10, 1, 1, 2, 3, 456789),
        pl.col("t") > datetime(2026, 10, 1, 1, 2, 3, 456789),
        pl.col("tz")
        == datetime(2026, 10, 1, 1, 2, 3, 456789, tzinfo=zoneinfo.ZoneInfo("UTC")),
        pl.col("tz") < datetime(2026, 10, 2, tzinfo=zoneinfo.ZoneInfo("UTC")),
        pl.col("f") < 2.0,
        pl.col("f").is_nan(),
        ~pl.col("f").is_nan(),
        pl.col("f").is_null() | (pl.col("a") == 1),
    ],
)
def test_sink_iceberg_overwrite_filter_polars_semantics(
    tmp_path: Path, predicate: pl.Expr
) -> None:
    tbl, df = _overwrite_filter_table(tmp_path)

    df.clear().lazy().sink_iceberg(tbl, mode="overwrite", overwrite_filter=predicate)

    assert_frame_equal(
        _scan_sorted(tbl), df.remove(predicate).sort(pl.all(), nulls_last=True)
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("pass_table", [True, False])
@pytest.mark.parametrize(
    ("predicate", "match"),
    [
        (pl.col("f") > 1.0, "GreaterThan on floating-point field"),
        (pl.col("s").str.contains("x"), "unsupported expression"),
        (pl.col("a") + 1 == 2, "unsupported expression"),
        (pl.col("a") == pl.col("a"), "cannot convert predicate"),
        (pl.col("n").struct.field("x") == 1, "has an optional parent"),
        (pl.col("a"), "must be a single boolean expression"),
    ],
)
def test_sink_iceberg_overwrite_filter_polars_unsupported(
    tmp_path: Path, predicate: pl.Expr, match: str, pass_table: bool
) -> None:
    tbl, df = _overwrite_filter_table(tmp_path)
    files_before = set(tmp_path.rglob("*.parquet"))
    target: Any = tbl if pass_table else ".".join(tbl.name())

    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.lazy().sink_iceberg(
            target, mode="overwrite", overwrite_filter=predicate, catalog=tbl.catalog
        )

    # Rejected before any data is written.
    assert set(tmp_path.rglob("*.parquet")) == files_before
    assert_frame_equal(_scan_sorted(tbl), df.sort(pl.all(), nulls_last=True))


@pytest.mark.write_disk
def test_sink_iceberg_overwrite_filter_serialization(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    pl.DataFrame({"a": [1, 2]}).lazy().sink_iceberg(tbl, mode="append")

    state = IcebergSinkState.new(tbl, mode="overwrite", overwrite_filter="a = 1")
    lf = state.attach_sink(pl.LazyFrame({"a": [3]}))
    pl.LazyFrame.deserialize(io.BytesIO(lf.serialize())).collect()

    assert_frame_equal(_scan_sorted(tbl), pl.DataFrame({"a": [2, 3]}))


@pytest.mark.write_disk
def test_sink_iceberg_dynamic_partition_overwrite(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "p", StringType()),
            NestedField(2, "d", DateType()),
            NestedField(3, "v", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "p"),
            PartitionField(2, 1001, IdentityTransform(), "d"),
        ),
    )
    d1, d2 = date(2026, 10, 1), date(2026, 10, 2)
    pl.DataFrame(
        {
            "p": ["a", "a", "b", None, "b"],
            "d": [d1, d2, d1, d1, None],
            "v": [1, 2, 3, 4, 5],
        }
    ).lazy().sink_iceberg(tbl, mode="append")

    pl.DataFrame(
        {"p": ["a", None, "b", "c"], "d": [d1, d1, None, d2], "v": [10, 11, 12, 13]}
    ).lazy().sink_iceberg(tbl, mode="overwrite", partition_overwrite="dynamic")

    assert_frame_equal(
        _scan_sorted(tbl),
        pl.DataFrame(
            {
                "p": ["a", "a", "b", "b", "c", None],
                "d": [d1, d2, d1, None, d2, d1],
                "v": [10, 2, 3, 12, 13, 11],
            }
        ),
    )

    # No incoming data: nothing is deleted.
    pl.DataFrame(
        schema={"p": pl.String, "d": pl.Date, "v": pl.Int64}
    ).lazy().sink_iceberg(tbl, mode="overwrite", partition_overwrite="dynamic")
    assert _scan_sorted(tbl).height == 6


@pytest.mark.write_disk
def test_sink_iceberg_dynamic_partition_overwrite_nested_source(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(
                1,
                "s",
                StructType(NestedField(2, "x", LongType(), required=True)),
                required=True,
            ),
            NestedField(3, "v", LongType()),
        ),
        partition_spec=PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "x")),
    )
    df = pl.DataFrame({"s": [{"x": 1}, {"x": 2}], "v": [1, 2]})
    df.lazy().sink_iceberg(tbl, mode="append")

    pl.DataFrame({"s": [{"x": 2}], "v": [20]}).lazy().sink_iceberg(
        tbl, mode="overwrite", partition_overwrite="dynamic"
    )
    assert_frame_equal(
        _scan_sorted(tbl).select(pl.col("s").struct.field("x"), "v"),
        pl.DataFrame({"x": [1, 2], "v": [1, 20]}),
    )


@pytest.mark.write_disk
def test_sink_iceberg_create_if_not_exists(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("ns")

    df = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": ["x", None, "z"],
            "c": [[1], [], None],
            "d": [{"x": 1.0}, None, {"x": None}],
            "e": [date(2020, 1, 1), None, date(1970, 1, 1)],
            "f": [datetime(2020, 1, 1), None, None],
        }
    )

    with pytest.raises(pyiceberg.exceptions.NoSuchTableError):
        df.lazy().sink_iceberg("ns.t", mode="append", catalog=catalog)

    assert not catalog.table_exists("ns.t")

    df.lazy().sink_iceberg(
        "ns.t", mode="append", create_if_not_exists=True, catalog=catalog
    )
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("ns.t")).collect(), df)

    # Existing table is appended to.
    df.lazy().sink_iceberg(
        "ns.t", mode="append", create_if_not_exists=True, catalog=catalog
    )
    assert_frame_equal(
        pl.scan_iceberg(catalog.load_table("ns.t")).collect(), pl.concat([df, df])
    )

    tbl = catalog.load_table("ns.t")
    with pytest.raises(TypeError, match="requires a table identifier"):
        df.lazy().sink_iceberg(tbl, mode="append", create_if_not_exists=True)

    # The plan is resolved after a serialization roundtrip (e.g. on a remote
    # worker), where the table is created from the serialized schema.
    state = IcebergSinkState.new("ns.t2", create_if_not_exists=True, catalog=catalog)
    lf = state.attach_sink(df.lazy())
    pl.LazyFrame.deserialize(io.BytesIO(lf.serialize())).collect()
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("ns.t2")).collect(), df)


@pytest.mark.write_disk
def test_sink_iceberg_create_namespace_if_not_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )

    df = pl.DataFrame({"a": [1, 2]})

    with pytest.raises(pyiceberg.exceptions.NoSuchTableError):
        df.lazy().sink_iceberg("ns.t", mode="append", catalog=catalog)

    with pytest.raises(pyiceberg.exceptions.NoSuchNamespaceError):
        df.lazy().sink_iceberg(
            "ns.t", mode="append", create_if_not_exists=True, catalog=catalog
        )

    with pytest.raises(ValueError, match="requires create_if_not_exists=True"):
        df.lazy().sink_iceberg(
            "ns.t", mode="append", create_namespace_if_not_exists=True, catalog=catalog
        )

    assert not catalog.namespace_exists("ns")

    df.lazy().sink_iceberg(
        "ns.t",
        mode="append",
        create_if_not_exists=True,
        create_namespace_if_not_exists=True,
        catalog=catalog,
    )
    assert catalog.namespace_exists("ns")
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("ns.t")).collect(), df)

    # Missing parent namespaces are created first. Like the REST catalog, report a
    # missing parent as a `RESTError` rather than a `NoSuchNamespaceError`.
    create_namespace = SqlCatalog.create_namespace
    created = []

    def create_namespace_requires_parent(
        self: SqlCatalog, namespace: Any, *args: Any, **kwargs: Any
    ) -> Any:
        namespace = self.identifier_to_tuple(namespace)
        if len(namespace) > 1 and not self.namespace_exists(namespace[:-1]):
            raise pyiceberg.exceptions.RESTError(namespace[:-1])
        created.append(namespace)
        return create_namespace(self, namespace, *args, **kwargs)

    monkeypatch.setattr(
        SqlCatalog, "create_namespace", create_namespace_requires_parent
    )

    df.lazy().sink_iceberg(
        "a.b.t",
        mode="append",
        create_if_not_exists=True,
        create_namespace_if_not_exists=True,
        catalog=catalog,
    )
    assert created == [("a",), ("a", "b")]
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("a.b.t")).collect(), df)

    # The retried child create may land with its response lost.
    def create_namespace_requires_parent_lost_response(
        self: SqlCatalog, namespace: Any, *args: Any, **kwargs: Any
    ) -> Any:
        create_namespace_requires_parent(self, namespace, *args, **kwargs)
        if len(self.identifier_to_tuple(namespace)) > 1:
            msg = "lost response"
            raise ConnectionError(msg)

    monkeypatch.setattr(
        SqlCatalog, "create_namespace", create_namespace_requires_parent_lost_response
    )

    created.clear()
    df.lazy().sink_iceberg(
        "c.d.t",
        mode="append",
        create_if_not_exists=True,
        create_namespace_if_not_exists=True,
        catalog=catalog,
    )
    assert created == [("c",), ("c", "d")]
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("c.d.t")).collect(), df)

    # The namespace create may land with its response lost.
    def create_namespace_lost_response(
        self: SqlCatalog, *args: Any, **kwargs: Any
    ) -> Any:
        create_namespace(self, *args, **kwargs)
        msg = "lost response"
        raise ConnectionError(msg)

    monkeypatch.setattr(SqlCatalog, "create_namespace", create_namespace_lost_response)

    df.lazy().sink_iceberg(
        "ns2.t",
        mode="append",
        create_if_not_exists=True,
        create_namespace_if_not_exists=True,
        catalog=catalog,
    )
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("ns2.t")).collect(), df)

    # Errors are raised if the namespace was not created.
    def create_namespace_fails(self: SqlCatalog, *args: Any, **kwargs: Any) -> Any:
        msg = "create failed"
        raise ConnectionError(msg)

    monkeypatch.setattr(SqlCatalog, "create_namespace", create_namespace_fails)

    with pytest.raises(ConnectionError, match="create failed"):
        df.lazy().sink_iceberg(
            "ns3.t",
            mode="append",
            create_if_not_exists=True,
            create_namespace_if_not_exists=True,
            catalog=catalog,
        )


@pytest.mark.write_disk
def test_sink_iceberg_create_if_not_exists_lost_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("ns")

    create_table = SqlCatalog.create_table

    def create_table_lost_response(self: SqlCatalog, *args: Any, **kwargs: Any) -> Any:
        create_table(self, *args, **kwargs)
        msg = "lost response"
        raise ConnectionError(msg)

    monkeypatch.setattr(SqlCatalog, "create_table", create_table_lost_response)

    df = pl.DataFrame({"a": [1, 2]})
    df.lazy().sink_iceberg(
        "ns.t", mode="append", create_if_not_exists=True, catalog=catalog
    )
    assert_frame_equal(pl.scan_iceberg(catalog.load_table("ns.t")).collect(), df)

    # Errors are raised if the table was not created.
    def create_table_fails(self: SqlCatalog, *args: Any, **kwargs: Any) -> Any:
        msg = "create failed"
        raise ConnectionError(msg)

    monkeypatch.setattr(SqlCatalog, "create_table", create_table_fails)

    with pytest.raises(ConnectionError, match="create failed"):
        df.lazy().sink_iceberg(
            "ns.t2", mode="append", create_if_not_exists=True, catalog=catalog
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("schema_mode", [None, "merge"])
def test_sink_iceberg_create_if_not_exists_dtypes(
    tmp_path: Path, schema_mode: Literal["merge"] | None
) -> None:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("ns")

    dt = datetime(2020, 1, 1, 1, 2, 3, 456000)
    # (values, dtype, iceberg type, dtype read back)
    columns: dict[str, tuple[list[Any], pl.DataType, str, pl.DataType]] = {
        "bool": ([True], pl.Boolean(), "boolean", pl.Boolean()),
        "i8": ([-128], pl.Int8(), "int", pl.Int32()),
        "i16": ([-32768], pl.Int16(), "int", pl.Int32()),
        "i32": ([-(2**31)], pl.Int32(), "int", pl.Int32()),
        "i64": ([-(2**63)], pl.Int64(), "long", pl.Int64()),
        "u8": ([255], pl.UInt8(), "int", pl.Int32()),
        "u16": ([65535], pl.UInt16(), "int", pl.Int32()),
        "u32": ([2**32 - 1], pl.UInt32(), "long", pl.Int64()),
        "f16": ([1.5], pl.Float16(), "float", pl.Float32()),
        "f32": ([1.5], pl.Float32(), "float", pl.Float32()),
        "f64": ([1.5], pl.Float64(), "double", pl.Float64()),
        "dec": ([D("1.25")], pl.Decimal(38, 2), "decimal(38, 2)", pl.Decimal(38, 2)),
        "str": (["x"], pl.String(), "string", pl.String()),
        "cat": (["x"], pl.Categorical(), "string", pl.String()),
        "enum": (["x"], pl.Enum(["x", "y"]), "string", pl.String()),
        "bin": ([b"x"], pl.Binary(), "binary", pl.Binary()),
        "date": ([date(2020, 1, 1)], pl.Date(), "date", pl.Date()),
        "time": ([dt.time()], pl.Time(), "time", pl.Time()),
        "dt_ms": ([dt], pl.Datetime("ms"), "timestamp", pl.Datetime("us")),
        "dt_us": ([dt], pl.Datetime("us"), "timestamp", pl.Datetime("us")),
        "dt_tz": (
            [dt],
            pl.Datetime("ms", "Asia/Tokyo"),
            "timestamptz",
            pl.Datetime("us", "UTC"),
        ),
        "list": ([[1, None]], pl.List(pl.UInt8), "list<int>", pl.List(pl.Int32)),
        "arr": ([[1, 2]], pl.Array(pl.Int16, 2), "list<int>", pl.List(pl.Int32)),
        "struct": (
            [{"x": 1, "y": "a", "z": [[1]]}],
            pl.Struct(
                {
                    "x": pl.UInt32,
                    "y": pl.Enum(["a"]),
                    "z": pl.List(pl.Array(pl.Int8, 1)),
                }
            ),
            "struct<x: long, y: string, z: list<list<int>>>",
            pl.Struct({"x": pl.Int64, "y": pl.String, "z": pl.List(pl.List(pl.Int32))}),
        ),
    }

    df = pl.DataFrame(
        [
            pl.Series(name, values + [None], dtype=dtype)
            for name, (values, dtype, _, _) in columns.items()
        ]
    )
    expected = df.cast({name: c[3] for name, c in columns.items()})

    for _ in range(2):  # Created, then appended to.
        df.lazy().sink_iceberg(
            "ns.t",
            mode="append",
            schema_mode=schema_mode,
            create_if_not_exists=True,
            catalog=catalog,
        )

    tbl = catalog.load_table("ns.t")
    assert {
        f.name: re.sub(r"\d+: |optional ", "", str(f.field_type))
        for f in tbl.schema().fields
    } == {name: c[2] for name, c in columns.items()}

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), pl.concat([expected, expected]))


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("schema", "partition_spec", "error", "match"),
    [
        (
            IcebergSchema(NestedField(1, "a", LongType())),
            None,
            ValueError,
            "requires a partitioned",
        ),
        (
            IcebergSchema(NestedField(1, "a", DateType())),
            PartitionSpec(PartitionField(1, 1000, DayTransform(), "a_day")),
            NotImplementedError,
            "only supports identity",
        ),
        (
            IcebergSchema(NestedField(1, "a", DoubleType())),
            PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "a")),
            NotImplementedError,
            "floating-point",
        ),
        (
            IcebergSchema(
                NestedField(
                    1,
                    "s",
                    StructType(NestedField(2, "x", LongType(), required=True)),
                    required=False,
                )
            ),
            PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "x")),
            NotImplementedError,
            "nested within an optional struct",
        ),
    ],
)
def test_sink_iceberg_dynamic_partition_overwrite_unsupported(
    tmp_path: Path,
    schema: IcebergSchema,
    partition_spec: PartitionSpec | None,
    error: type[Exception],
    match: str,
) -> None:
    tbl, _ = new_iceberg_table(tmp_path, schema=schema, partition_spec=partition_spec)
    lf = pl.LazyFrame(schema=pl.Schema(schema.as_arrow()))
    with pytest.raises(error, match=match):
        lf.sink_iceberg(tbl, mode="overwrite", partition_overwrite="dynamic")
    # Rejected before any data is written.
    assert not list((tmp_path).rglob("*.parquet"))


def test_sink_iceberg_dynamic_partition_overwrite_timestamp_ns() -> None:
    from polars.io.iceberg._sink import _check_dynamic_overwrite_spec

    # Nanosecond timestamps require V3 tables, which PyIceberg cannot write yet.
    with pytest.raises(NotImplementedError, match="nanosecond"):
        _check_dynamic_overwrite_spec(
            PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "t")),
            IcebergSchema(NestedField(1, "t", TimestampNanoType())),
        )


@pytest.mark.write_disk
def test_sink_iceberg_overwrite_filter_validation(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    lf = pl.LazyFrame({"a": [1]})

    with pytest.raises(ValueError, match="require mode='overwrite'"):
        lf.sink_iceberg(tbl, mode="append", overwrite_filter="a = 1")
    with pytest.raises(ValueError, match="require mode='overwrite'"):
        lf.sink_iceberg(tbl, mode="append", partition_overwrite="dynamic")
    with pytest.raises(ValueError, match="mutually exclusive"):
        lf.sink_iceberg(
            tbl,
            mode="overwrite",
            overwrite_filter="a = 1",
            partition_overwrite="dynamic",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        lf.sink_iceberg(
            tbl, mode="overwrite", schema_mode="overwrite", overwrite_filter="a = 1"
        )
    with pytest.raises(ValueError, match="invalid partition_overwrite"):
        lf.sink_iceberg(tbl, mode="overwrite", partition_overwrite="static")  # type: ignore[arg-type]
    # Unknown columns fail when resolving the sink, before writing data.
    with pytest.raises(ValueError, match="Could not find field"):
        lf.sink_iceberg(tbl, mode="overwrite", overwrite_filter="b = 1")
    assert not list((tmp_path).rglob("*.parquet"))


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("dtype", "match"),
    [
        (pl.UInt64(), r"'c' \(UInt64\)"),
        (pl.Int128(), r"'c' \(Int128\)"),
        (
            pl.Datetime("ns"),
            r"'c' \(Datetime\(time_unit='ns'.*cast to Datetime\('us'\)",
        ),
        (pl.Duration("us"), r"'c' \(Duration"),
        (pl.Null(), r"'c' \(Null\)"),
        (pl.Object(), r"'c' \(Object\)"),
        (pl.List(pl.UInt64), r"'c\[\]' \(UInt64\)"),
        (pl.Struct({"x": pl.Int64, "y": pl.Duration("ms")}), r"'c.y' \(Duration"),
    ],
)
def test_sink_iceberg_create_if_not_exists_unsupported_dtypes(
    tmp_path: Path, dtype: pl.DataType, match: str
) -> None:
    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'iceberg_catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("ns")

    lf = pl.LazyFrame({"a": [1], "c": [None]}, schema={"a": pl.Int64, "c": dtype})
    with pytest.raises(TypeError, match=match):
        lf.sink_iceberg(
            "ns.t", mode="append", create_if_not_exists=True, catalog=catalog
        )

    assert not catalog.table_exists("ns.t")


_ICEBERG_TEMPORAL_SORT_TRANSFORMS: list[Any] = [
    YearTransform(),
    MonthTransform(),
    DayTransform(),
    BucketTransform(4),
]


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize(
    ("source_type", "transform", "values"),
    [
        (LongType(), IdentityTransform(), [3, -1, 2, 0, -1, 4]),
        (BooleanType(), IdentityTransform(), [True, False, True, False, True, False]),
        (StringType(), IdentityTransform(), ["é", "a", "😁", "aa", "中", "a"]),
        (BinaryType(), IdentityTransform(), [b"\xff", b"", b"\x00", b"a", b"aa", b"a"]),
        (
            DecimalType(20, 2),
            IdentityTransform(),
            [D("1.23"), D("-1.20"), D("0"), D("1.23"), D("2.01"), D("-3.20")],
        ),
        (LongType(), TruncateTransform(10), [19, -1, -11, 10, 0, -10]),
        (StringType(), TruncateTransform(2), ["abc", "aa", "😁ab", "a", "ab", "😁aa"]),
        (
            BinaryType(),
            TruncateTransform(2),
            [b"abc", b"aa", b"\xffab", b"a", b"ab", b"\xffaa"],
        ),
        (
            DecimalType(20, 2),
            TruncateTransform(10),
            [D("1.23"), D("-1.21"), D("-1.20"), D("0"), D("1.20"), D("-0.01")],
        ),
        (LongType(), BucketTransform(4), [34, -1, 0, 10, 35, -100]),
        (StringType(), BucketTransform(4), ["iceberg", "a", "", "😁", "aa", "a"]),
        (
            DecimalType(20, 2),
            BucketTransform(4),
            [D("1.23"), D("-1.21"), D("-1.20"), D("0"), D("1.20"), D("-0.01")],
        ),
        (
            BinaryType(),
            BucketTransform(4),
            [b"abc", b"aa", b"\xffab", b"a", b"ab", b"\xffaa"],
        ),
        (LongType(), VoidTransform(), [3, -1, 2, 0, -1, 4]),
        *[
            (
                DateType(),
                transform,
                [
                    date(1970, 1, 2),
                    date(1969, 12, 31),
                    date(2000, 2, 29),
                    date(1970, 1, 1),
                    date(1969, 1, 1),
                    date(2000, 1, 31),
                ],
            )
            for transform in _ICEBERG_TEMPORAL_SORT_TRANSFORMS
        ],
        *[
            (
                TimestampType(),
                transform,
                [
                    datetime(1970, 1, 1, 0, 1),
                    datetime(1969, 12, 31, 23, 59),
                    datetime(2000, 2, 29, 23),
                    datetime(1970, 1, 1),
                    datetime(1969, 1, 1),
                    datetime(2000, 1, 31),
                ],
            )
            for transform in [*_ICEBERG_TEMPORAL_SORT_TRANSFORMS, HourTransform()]
        ],
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_sort_order(
    tmp_path: Path,
    source_type: Any,
    transform: Any,
    values: list[Any],
    *,
    partitioned: bool,
    descending: bool,
    nulls_last: bool,
) -> None:
    values = [*values, None, None]
    ids = list(reversed(range(len(values))))
    df = pl.DataFrame(
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "value": pa.array(values, type=schema_to_pyarrow(source_type)),
                "partition": pa.array([i % 2 for i in ids], type=pa.int64()),
            }
        )
    )
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "value", source_type),
            NestedField(3, "partition", LongType()),
        ),
        partition_spec=(
            PartitionSpec(PartitionField(3, 1000, IdentityTransform(), "partition"))
            if partitioned
            else None
        ),
        sort_order=SortOrder(
            SortField(
                2,
                transform,
                direction=SortDirection.DESC if descending else SortDirection.ASC,
                null_order=NullOrder.NULLS_LAST
                if nulls_last
                else NullOrder.NULLS_FIRST,
            ),
            SortField(1, direction=SortDirection.ASC),
        ),
    )
    original_sort_order = tbl.sort_order()
    df.lazy().sink_iceberg(tbl, mode="append", maintain_order=False, row_group_size=2)
    tbl = catalog.load_table(tbl.name())
    assert tbl.sort_order() == original_sort_order
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), df, check_row_order=False)

    transform_value = transform.transform(source_type)
    for task in tbl.scan().plan_files():
        assert task.file.sort_order_id == original_sort_order.order_id
        with tbl.io.new_input(task.file.file_path).open() as file:
            actual = pq.read_table(file).column("id").to_pylist()
        partition_ids = [
            i for i in sorted(ids) if not partitioned or i % 2 == task.file.partition[0]
        ]
        keys = {i: transform_value(values[ids.index(i)]) for i in partition_ids}
        null_ids = [i for i in partition_ids if keys[i] is None]
        non_null_ids = sorted(
            (i for i in partition_ids if keys[i] is not None),
            key=lambda i: keys[i],
            reverse=descending,
        )
        expected = non_null_ids + null_ids if nulls_last else null_ids + non_null_ids
        assert actual == expected


@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_multiple_files(
    tmp_path: Path, *, maintain_order: bool, partitioned: bool
) -> None:
    from pyiceberg.table import TableProperties

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()), NestedField(2, "partition", LongType())
        ),
        partition_spec=(
            PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "partition"))
            if partitioned
            else None
        ),
        sort_order=SortOrder(SortField(1, direction=SortDirection.DESC)),
        properties={TableProperties.WRITE_TARGET_FILE_SIZE_BYTES: "1000"},
    )
    df = pl.DataFrame({"id": range(2000), "partition": [i % 3 for i in range(2000)]})
    df.sample(fraction=1, shuffle=True, seed=0).lazy().sink_iceberg(
        tbl, mode="append", maintain_order=maintain_order, row_group_size=17
    )
    tasks = list(tbl.scan().plan_files())
    assert len(tasks) > (3 if partitioned else 1)
    assert any(task.file.record_count > 17 for task in tasks)
    for task in tasks:
        assert task.file.sort_order_id == tbl.sort_order().order_id
        with tbl.io.new_input(task.file.file_path).open() as file:
            values = pq.read_table(file).column("id").to_pylist()
        assert values == sorted(values, reverse=True)
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), df, check_row_order=False)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_floats(
    tmp_path: Path, dtype: pl.DataType, *, descending: bool
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(
                2, "value", FloatType() if dtype == pl.Float32 else DoubleType()
            ),
        ),
        sort_order=SortOrder(
            SortField(
                2, direction=SortDirection.DESC if descending else SortDirection.ASC
            ),
            SortField(1),
        ),
    )
    ordered = [
        float("-nan"),
        float("-inf"),
        -1.0,
        -0.0,
        0.0,
        1.0,
        float("inf"),
        float("nan"),
    ]
    ids = [4, 7, 0, 3, 6, 1, 5, 2]
    df = pl.DataFrame(
        {"id": ids, "value": pl.Series([ordered[i] for i in ids], dtype=dtype)}
    )
    df.lazy().sink_iceberg(tbl, mode="append")
    [task] = tbl.scan().plan_files()
    assert task.file.sort_order_id == tbl.sort_order().order_id
    with tbl.io.new_input(task.file.file_path).open() as file:
        actual = pq.read_table(file).to_pydict()
    expected_ids = sorted(ids, reverse=descending)
    assert actual["id"] == expected_ids
    assert [struct.pack(">d", value) for value in actual["value"]] == [
        struct.pack(">d", ordered[i]) for i in expected_ids
    ]


def test_sink_iceberg_sort_key_exprs_are_serializable() -> None:
    # Sink-plan serialization happens before sort keys are built; test them directly.
    from polars.io.iceberg._sink import _sort_key_exprs

    exprs, _, _ = _sort_key_exprs(
        IcebergSchema(NestedField(1, "value", LongType())),
        SortOrder(SortField(1, BucketTransform(4))),
        pl.Schema({"value": pl.Int64}),
    )

    lf = pl.LazyFrame({"value": [1, 2, 3]}).select(exprs)
    roundtripped = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))

    assert_frame_equal(roundtripped.collect(), lf.collect())


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_nested_schema_merge(
    tmp_path: Path, *, missing: bool
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "payload", StructType(NestedField(3, "key", FloatType()))),
        ),
        sort_order=SortOrder(SortField(3), SortField(1, direction=SortDirection.DESC)),
    )
    payload = [{"key": 3.0}, None, {"key": 1.0}, {"key": 3.0}]
    df = pl.DataFrame({"id": [3, 1, 4, 2], "payload": payload})
    if missing:
        df = df.with_columns(pl.struct(pl.lit(1).alias("other")).alias("payload"))
    df.lazy().sink_iceberg(tbl, mode="append", schema_mode="merge")
    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as file:
        ids = pq.read_table(file).column("id").to_pylist()
    assert ids == ([4, 3, 2, 1] if missing else [1, 4, 3, 2])
    assert task.file.sort_order_id == tbl.sort_order().order_id


@pytest.mark.write_disk
def test_sink_iceberg_sort_order_used_at_write_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "id", LongType())),
        sort_order=SortOrder(SortField(1)),
    )
    original_order = tbl.sort_order()
    original_commit = IcebergSinkState.commit

    def commit(self: IcebergSinkState, sinked_files: Any) -> pl.DataFrame:
        with self.table().update_sort_order() as update:
            update.desc("id", IdentityTransform())
        return original_commit(self, sinked_files)

    monkeypatch.setattr(IcebergSinkState, "commit", commit)
    pl.LazyFrame({"id": [3, 1, 2]}).sink_iceberg(tbl, mode="append")
    tbl = catalog.load_table(tbl.name())
    assert tbl.sort_order().order_id != original_order.order_id
    [task] = tbl.scan().plan_files()
    assert task.file.sort_order_id == original_order.order_id
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("id").to_pylist() == [1, 2, 3]


@pytest.mark.write_disk
def test_sink_iceberg_sort_order_schema_overwrite_rejected(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "id", LongType())),
        sort_order=SortOrder(SortField(1)),
    )
    with pytest.raises(
        NotImplementedError, match=r"schema_mode='overwrite'.*sort order"
    ):
        pl.LazyFrame({"id": [1]}).sink_iceberg(
            tbl, mode="overwrite", schema_mode="overwrite"
        )
    assert tbl.current_snapshot() is None
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("format_version", [1, 2])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_append_and_overwrite(
    tmp_path: Path, format_version: int, *, partitioned: bool
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "id", LongType())),
        partition_spec=(
            PartitionSpec(PartitionField(1, 1000, TruncateTransform(10), "partition"))
            if partitioned
            else None
        ),
        sort_order=SortOrder(SortField(1)),
        properties={"format-version": str(format_version)},
    )
    pl.LazyFrame({"id": [3, 1, 2]}).sink_iceberg(tbl, mode="append")
    [old_task] = tbl.scan().plan_files()
    old_order = tbl.sort_order()
    with tbl.update_sort_order() as update:
        update.desc("id", IdentityTransform())
    new_order = tbl.sort_order()
    pl.LazyFrame({"id": [4, 6, 5]}).sink_iceberg(tbl, mode="append")
    tasks = list(tbl.scan().plan_files())
    assert len(tasks) == 2
    for task in tasks:
        existing = task.file.file_path == old_task.file.file_path
        assert task.file.sort_order_id == (
            old_order.order_id if existing else new_order.order_id
        )
        with tbl.io.new_input(task.file.file_path).open() as file:
            assert pq.read_table(file).column("id").to_pylist() == (
                [1, 2, 3] if existing else [6, 5, 4]
            )

    pl.LazyFrame({"id": [-3, -1, -2]}).sink_iceberg(tbl, mode="overwrite")
    [task] = tbl.scan().plan_files()
    assert task.file.sort_order_id == new_order.order_id
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("id").to_pylist() == [-1, -2, -3]
    assert Path(old_task.file.file_path.removeprefix("file://")).exists()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_null_and_missing_keys(
    tmp_path: Path, *, empty: bool, missing: bool
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()), NestedField(2, "value", TimestampType())
        ),
        sort_order=SortOrder(SortField(2, YearTransform())),
    )
    df = pl.DataFrame({"id": [3, 1, 2]})
    if not missing:
        df = df.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("value"))
    if empty:
        df = df.clear()
    df.lazy().sink_iceberg(tbl, mode="append", schema_mode="merge")
    for task in tbl.scan().plan_files():
        assert task.file.sort_order_id == tbl.sort_order().order_id
    expected = df.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("value"))
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), expected, check_row_order=False)


@pytest.mark.write_disk
def test_sink_iceberg_sort_order_stable_ties(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "value", StringType())),
        sort_order=SortOrder(SortField(1, TruncateTransform(2))),
    )
    pl.LazyFrame({"value": ["abc", "aba", "aac", "abb"]}).sink_iceberg(
        tbl, mode="append", maintain_order=True
    )
    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("value").to_pylist() == [
            "aac",
            "abc",
            "aba",
            "abb",
        ]


@pytest.mark.parametrize(
    "transform", [DayTransform(), HourTransform(), BucketTransform(16)]
)
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_timezone(tmp_path: Path, transform: Any) -> None:
    from polars.io.iceberg._sink import _sort_key_exprs

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()), NestedField(2, "value", TimestamptzType())
        ),
        sort_order=SortOrder(SortField(2, transform), SortField(1)),
    )
    times = [
        datetime(
            2024, 11, 3, 1, 30, tzinfo=zoneinfo.ZoneInfo("America/New_York"), fold=1
        ),
        datetime(1969, 12, 31, 23, 59, tzinfo=zoneinfo.ZoneInfo("UTC")),
        datetime(
            2024, 11, 3, 1, 30, tzinfo=zoneinfo.ZoneInfo("America/New_York"), fold=0
        ),
        datetime(1969, 12, 31, 20, 0, tzinfo=zoneinfo.ZoneInfo("America/New_York")),
    ]
    df = pl.DataFrame({"id": range(len(times)), "value": times}).with_columns(
        pl.col("value").dt.convert_time_zone("America/New_York")
    )
    transform_value = transform.transform(TimestamptzType())
    expected = sorted(range(len(times)), key=lambda i: transform_value(times[i]))
    exprs, descending, nulls_last = _sort_key_exprs(
        tbl.schema(), tbl.sort_order(), df.schema
    )
    assert (
        df.sort(exprs, descending=descending, nulls_last=nulls_last)["id"].to_list()
        == expected
    )
    df.with_columns(pl.col("value").dt.convert_time_zone("UTC")).lazy().sink_iceberg(
        tbl, mode="append"
    )
    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("id").to_pylist() == expected


@pytest.mark.parametrize("source_type", [LongType(), DecimalType(38, 0)])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_truncate_extremes(
    tmp_path: Path, source_type: Any
) -> None:
    values = (
        [-(1 << 63), (1 << 63) - 1, 0, -(1 << 63) + 1]
        if isinstance(source_type, LongType)
        else [D("-" + "9" * 38), D("9" * 38), D(0), D("-" + "9" * 37 + "8")]
    )
    df = pl.DataFrame(
        pa.table({"value": pa.array(values, type=schema_to_pyarrow(source_type))})
    )
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "value", source_type)),
        sort_order=SortOrder(SortField(1, TruncateTransform(10))),
    )
    df.lazy().sink_iceberg(tbl, mode="append")
    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("value").to_pylist() == sorted(values)


@pytest.mark.write_disk
def test_sink_iceberg_sort_order_serialization(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "id", LongType())),
        sort_order=SortOrder(SortField(1, BucketTransform(16))),
    )
    state = IcebergSinkState.new(tbl)
    lf = state.attach_sink(pl.LazyFrame({"id": [3, 1, 2]}))
    pl.LazyFrame.deserialize(io.BytesIO(lf.serialize())).collect()
    tbl.refresh()
    [task] = tbl.scan().plan_files()
    assert task.file.sort_order_id == tbl.sort_order().order_id
    key = cast("Callable[[int], int]", BucketTransform(16).transform(LongType()))
    with tbl.io.new_input(task.file.file_path).open() as file:
        assert pq.read_table(file).column("id").to_pylist() == sorted(
            [3, 1, 2], key=key
        )


@pytest.mark.parametrize("source_type", [UUIDType(), FixedType(16)])
@pytest.mark.parametrize("transform", [IdentityTransform(), BucketTransform(16)])
@pytest.mark.write_disk
def test_sink_iceberg_sort_order_fixed_binary(
    tmp_path: Path, source_type: Any, transform: Any
) -> None:
    values = [b"\xff" * 16, b"a" * 16, b"\x00" * 16, b"b" * 16]
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "value", source_type)),
        sort_order=SortOrder(SortField(1, transform)),
    )
    pl.LazyFrame({"value": values}).sink_iceberg(tbl, mode="append")
    expected = sorted(values, key=transform.transform(source_type))
    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as file:
        actual = pq.read_table(file).column("value").to_pylist()
    assert [v.bytes if isinstance(v, uuid.UUID) else v for v in actual] == expected


@pytest.mark.write_disk
def test_sink_iceberg_sort_order_unknown_transform_rejected(tmp_path: Path) -> None:
    from pyiceberg.exceptions import ValidationError
    from pyiceberg.transforms import UnknownTransform

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "id", LongType())),
        sort_order=SortOrder(SortField(1, UnknownTransform("future"))),
    )
    with pytest.raises(ValidationError, match="Invalid source field"):
        pl.LazyFrame({"id": [1]}).sink_iceberg(tbl, mode="append")
    assert tbl.current_snapshot() is None
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.write_disk
def test_sink_iceberg_uses_native_parquet_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    file_io_type = type(tbl.io)
    original_new_input = file_io_type.new_input

    def new_input(self: Any, location: str) -> Any:
        if "/data/" in location:
            msg = f"unexpected data file read: {location}"
            raise AssertionError(msg)
        return original_new_input(self, location)

    monkeypatch.setattr(file_io_type, "new_input", new_input)

    pl.LazyFrame({"a": [1, 2, 3]}).sink_iceberg(tbl, mode="append")

    [task] = tbl.scan().plan_files()
    assert task.file.record_count == 3
    assert (
        task.file.file_size_in_bytes
        == Path(task.file.file_path.removeprefix("file://")).stat().st_size
    )


@pytest.mark.write_disk
def test_sink_iceberg_parquet_writer_options(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )

    pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).sink_iceberg(
        tbl,
        mode="append",
        compression="gzip",
        compression_level=1,
        row_group_size=2,
        maintain_order=False,
    )

    [task] = tbl.scan().plan_files()
    with tbl.io.new_input(task.file.file_path).open() as input_file:
        metadata = pq.read_metadata(input_file)

    assert metadata.num_row_groups == 3
    assert metadata.row_group(0).column(0).compression == "GZIP"
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("a"),
        pl.DataFrame({"a": [1, 2, 3, 4, 5]}),
    )


@pytest.mark.parametrize(
    ("iceberg_type", "transform", "values"),
    [
        pytest.param(
            StringType(),
            IdentityTransform(),
            ["a", "a", "b"],
            id="identity",
        ),
        pytest.param(
            DateType(),
            YearTransform(),
            [date(1969, 12, 31), date(2024, 1, 1), date(2024, 12, 31)],
            id="year",
        ),
        pytest.param(
            DateType(),
            MonthTransform(),
            [date(2024, 1, 1), date(2024, 1, 31), date(2024, 2, 1)],
            id="month",
        ),
        pytest.param(
            DateType(),
            DayTransform(),
            [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
            id="day",
        ),
        pytest.param(
            TimestampType(),
            HourTransform(),
            [
                datetime(2024, 1, 1, 1, 15),
                datetime(2024, 1, 1, 1, 45),
                datetime(2024, 1, 1, 2),
            ],
            id="hour",
        ),
        pytest.param(
            LongType(),
            TruncateTransform(10),
            [-11, -1, 0, 9, 10],
            id="truncate-integer",
        ),
        pytest.param(
            StringType(),
            TruncateTransform(2),
            ["éclair", "école", "abc"],
            id="truncate-string",
        ),
        pytest.param(
            BinaryType(),
            TruncateTransform(2),
            [b"\xffabc", b"\xfede", b"\xffabc"],
            id="truncate-binary",
        ),
        pytest.param(
            TimestamptzType(),
            YearTransform(),
            [
                datetime(2024, 12, 31, 23, tzinfo=zoneinfo.ZoneInfo("UTC")),
                datetime(2025, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo("UTC")),
            ],
            id="year-timestamptz-utc-boundary",
        ),
        pytest.param(
            LongType(),
            BucketTransform(4),
            [34, -1, 0, 10, 35, -100, 2**63 - 1, -(2**63)],
            id="bucket-long",
        ),
        pytest.param(
            IntegerType(),
            BucketTransform(3),
            pl.Series([34, -1, 0, 2**31 - 1, -(2**31)], dtype=pl.Int32),
            id="bucket-integer",
        ),
        pytest.param(
            StringType(),
            BucketTransform(4),
            ["iceberg", "a", "", "😁", "aa"],
            id="bucket-string",
        ),
        pytest.param(
            BinaryType(),
            BucketTransform(4),
            [b"\x00\x01\x02\x03", b"", b"\xff", b"abcde"],
            id="bucket-binary",
        ),
        pytest.param(
            DecimalType(9, 2),
            BucketTransform(4),
            pl.Series(
                [
                    D("14.20"),
                    D("0"),
                    D("-0.01"),
                    D("1.28"),
                    D("-1.29"),
                    D("9999999.99"),
                ],
                dtype=pl.Decimal(9, 2),
            ),
            id="bucket-decimal",
        ),
        pytest.param(
            DateType(),
            BucketTransform(4),
            [date(2017, 11, 16), date(1969, 12, 31), date(1970, 1, 1)],
            id="bucket-date",
        ),
        pytest.param(
            TimeType(),
            BucketTransform(4),
            [time(22, 31, 8), time(0), time(23, 59, 59, 999999)],
            id="bucket-time",
        ),
        pytest.param(
            TimestampType(),
            BucketTransform(4),
            [datetime(2017, 11, 16, 22, 31, 8), datetime(1969, 12, 31, 23, 59, 59)],
            id="bucket-timestamp",
        ),
        pytest.param(
            TimestamptzType(),
            BucketTransform(4),
            [
                datetime(2017, 11, 16, 22, 31, 8, tzinfo=zoneinfo.ZoneInfo("UTC")),
                datetime(1969, 12, 31, 23, 59, tzinfo=zoneinfo.ZoneInfo("UTC")),
            ],
            id="bucket-timestamptz",
        ),
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_partitioned_transforms(
    tmp_path: Path,
    iceberg_type: Any,
    transform: Any,
    values: list[Any] | pl.Series,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "partition_col", iceberg_type),
            NestedField(2, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, transform, "partition_col_transformed")
        ),
    )
    df = pl.DataFrame({"partition_col": values, "value": range(len(values))})

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    partition_of = transform.transform(iceberg_type)
    expected_partitions = [partition_of(value) for value in values]
    tasks = list(tbl.scan().plan_files())
    assert {task.file.partition[0] for task in tasks} == set(expected_partitions)
    # Every row is in the file of its partition.
    for task in tasks:
        with tbl.io.new_input(task.file.file_path).open() as file:
            rows = pq.read_table(file).column("value").to_pylist()
        assert {expected_partitions[i] for i in rows} == {task.file.partition[0]}


@pytest.mark.parametrize(
    ("iceberg_type", "values"),
    [
        pytest.param(UUIDType(), [uuid.UUID(int=i).bytes for i in range(8)], id="uuid"),
        pytest.param(FixedType(3), [bytes([i, i, i]) for i in range(8)], id="fixed"),
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_partitioned_bucket_fixed_binary(
    tmp_path: Path, iceberg_type: Any, values: list[bytes]
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "partition_col", iceberg_type),
            NestedField(2, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, BucketTransform(4), "partition_col_bucket")
        ),
    )
    df = pl.DataFrame(
        {"partition_col": values, "value": range(len(values))},
        schema={"partition_col": pl.Binary, "value": pl.Int64},
    )
    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    bucket = BucketTransform(4).transform(iceberg_type)
    expected_partitions = [bucket(v) for v in values]
    tasks = list(tbl.scan().plan_files())
    assert {task.file.partition[0] for task in tasks} == set(expected_partitions)
    assert len(tasks) > 1
    for task in tasks:
        with tbl.io.new_input(task.file.file_path).open() as file:
            rows = pq.read_table(file).column("value").to_pylist()
        assert {expected_partitions[i] for i in rows} == {task.file.partition[0]}


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_bucket(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "nested", StructType(NestedField(3, "id", LongType()))),
            NestedField(2, "s", StringType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(3, 1000, BucketTransform(8), "id_bucket"),
            PartitionField(2, 1001, BucketTransform(3), "s_bucket"),
        ),
    )
    n = 1000
    df = pl.DataFrame(
        {
            "nested": pl.int_range(n, eager=True).to_frame("id").to_struct(),
            "s": pl.int_range(n, eager=True).cast(pl.String),
        }
    ).with_columns(pl.col("nested").struct.with_fields(pl.field("id") * 7919 - 9))

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), df, check_row_order=False)

    id_bucket = BucketTransform(8).transform(LongType())
    s_bucket = BucketTransform(3).transform(StringType())
    tasks = list(tbl.scan().plan_files())
    assert len(tasks) == 24
    for task in tasks:
        with tbl.io.new_input(task.file.file_path).open() as file:
            file_df = pl.from_arrow(pq.read_table(file))
        assert isinstance(file_df, pl.DataFrame)
        assert {
            (id_bucket(row["nested"]["id"]), s_bucket(row["s"]))
            for row in file_df.iter_rows(named=True)
        } == {(task.file.partition[0], task.file.partition[1])}

    # PyIceberg prunes files by their bucket.
    some_id = df["nested"].struct.field("id")[123]
    tasks = list(tbl.scan(row_filter=f"nested.id == {some_id}").plan_files())
    assert len(tasks) == 3
    assert {task.file.partition[0] for task in tasks} == {id_bucket(some_id)}
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg")
        .filter(pl.col("nested").struct.field("id") == some_id)
        .collect(),
        df[123],
    )


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_bucket_schema_merge(tmp_path: Path) -> None:
    # The partition keys are of the data as written, with the merged schema.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "c", DecimalType(9, 2))),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, BucketTransform(4), "c_bucket")
        ),
    )
    # Values that only fit the widened decimal(18, 2).
    values = [D("1234567890.12"), D("-1234567890.12"), D("1.23")]
    pl.LazyFrame({"c": pl.Series(values, dtype=pl.Decimal(18, 2))}).sink_iceberg(
        tbl, mode="append", schema_mode="merge"
    )

    tbl.refresh()
    assert tbl.schema().find_field(1).field_type == DecimalType(18, 2)
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("c"),
        pl.DataFrame({"c": pl.Series(values, dtype=pl.Decimal(18, 2))}).sort("c"),
    )
    bucket = BucketTransform(4).transform(DecimalType(18, 2))
    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {
        bucket(v) for v in values
    }


def test_sink_iceberg_bucket_timestamp_ns() -> None:
    from pyiceberg.types import TimestamptzNanoType

    from polars.io.iceberg._sink import _bucket_transform_expr

    nanos = [0, 1, 999, 1000, -1, -999, -1000, -1001, 1510871468123456789]
    for iceberg_type, time_zone in [
        (TimestampNanoType(), None),
        (TimestamptzNanoType(), "UTC"),
    ]:
        bucket = BucketTransform(16).transform(iceberg_type)
        s = pl.Series("a", nanos).cast(pl.Datetime("ns", time_zone))
        expr = _bucket_transform_expr(pl.col("a"), iceberg_type, 16)
        assert expr is not None
        assert s.to_frame().select(expr).to_series().to_list() == [
            bucket(v) for v in nanos
        ]


@pytest.mark.write_disk
def test_sink_iceberg_compound_partition_paths_are_unique(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "first", StringType()),
            NestedField(2, "second", StringType()),
            NestedField(3, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "first"),
            PartitionField(2, 1001, IdentityTransform(), "second"),
        ),
    )
    df = pl.DataFrame(
        {
            "first": ["a", "a\x01"],
            "second": ["\x01b", "b"],
            "value": [1, 2],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    file_paths = [task.file.file_path for task in tbl.scan().plan_files()]
    assert len(file_paths) == 2
    assert len(set(file_paths)) == 2


@pytest.mark.write_disk
def test_sink_iceberg_partition_key_name_collision(tmp_path: Path) -> None:
    partition_key_name = "__POLARS_ICEBERG_PARTITION_1000"
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "partition_col", LongType()),
            NestedField(2, partition_key_name, LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "partition_col")
        ),
    )
    df = pl.DataFrame(
        {
            "partition_col": [1, 1, 2],
            partition_key_name: [10, 20, 30],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort(partition_key_name),
        df,
    )


@pytest.mark.write_disk
def test_sink_iceberg_partition_key_name_collision_schema_merge(
    tmp_path: Path,
) -> None:
    partition_key_name = "__POLARS_ICEBERG_PARTITION_1000"
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "partition_col", LongType())),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "partition_col")
        ),
    )
    df = pl.DataFrame(
        {
            "partition_col": [1, 1, 2],
            partition_key_name: [10, 20, 30],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append", schema_mode="merge")

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort(partition_key_name),
        df,
    )


@pytest.mark.parametrize("metrics_mode", ["none", "counts"])
@pytest.mark.parametrize("column_override", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_partitioned_without_bounds_metrics(
    tmp_path: Path, *, metrics_mode: str, column_override: bool
) -> None:
    # Partition values come from the partition keys, not from the file statistics.
    from pyiceberg.table import TableProperties

    property_name = (
        f"{TableProperties.METRICS_MODE_COLUMN_CONF_PREFIX}.partition_col"
        if column_override
        else TableProperties.DEFAULT_WRITE_METRICS_MODE
    )
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "partition_col", LongType()),
            NestedField(2, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "partition_col")
        ),
        properties={property_name: metrics_mode},
    )

    df = pl.DataFrame({"partition_col": [7, 7, 8], "value": [1, 2, 3]})
    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {7, 8}


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_allows_bounds_metrics_override(
    tmp_path: Path,
) -> None:
    from pyiceberg.table import TableProperties

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "partition_col", LongType()),
            NestedField(2, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "partition_col")
        ),
        properties={
            TableProperties.DEFAULT_WRITE_METRICS_MODE: "none",
            f"{TableProperties.METRICS_MODE_COLUMN_CONF_PREFIX}.partition_col": "full",
        },
    )
    df = pl.DataFrame({"partition_col": [7, 7, 8], "value": [1, 2, 3]})

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {7, 8}


@pytest.mark.write_disk
@pytest.mark.parametrize("nested_metrics_mode", [None, "none"])
def test_sink_iceberg_partitioned_nested_source_name_collision(
    tmp_path: Path, nested_metrics_mode: str | None
) -> None:
    from pyiceberg.table import TableProperties

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(
                2,
                "b",
                StructType(NestedField(3, "a", LongType())),
            ),
            NestedField(4, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(3, 1000, IdentityTransform(), "nested_a")
        ),
        properties=(
            {
                f"{TableProperties.METRICS_MODE_COLUMN_CONF_PREFIX}.b.a": nested_metrics_mode
            }
            if nested_metrics_mode is not None
            else None
        ),
    )
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": [{"a": 10}, {"a": 11}, {"a": 10}, {"a": 11}, None],
            "value": [1, 2, 3, 4, 5],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    data_files = [task.file for task in tbl.scan().plan_files()]
    top_level_source_id = tbl.schema().find_field("a").field_id
    nested_source_id = tbl.schema().find_field("b.a").field_id
    assert {data_file.partition[0] for data_file in data_files} == {10, 11, None}
    assert all(
        top_level_source_id in (data_file.lower_bounds or {})
        for data_file in data_files
    )
    assert all(
        nested_source_id not in (data_file.lower_bounds or {})
        for data_file in data_files
    )
    assert all(
        nested_source_id not in (data_file.upper_bounds or {})
        for data_file in data_files
    )
    if nested_metrics_mode == "none":
        assert all(
            nested_source_id not in (data_file.value_counts or {})
            for data_file in data_files
        )
        assert all(
            nested_source_id not in (data_file.null_value_counts or {})
            for data_file in data_files
        )
    else:
        assert all(
            nested_source_id in (data_file.value_counts or {})
            for data_file in data_files
        )


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_deeply_nested_source_transform(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(
                1,
                "outer",
                StructType(
                    NestedField(
                        2,
                        "inner",
                        StructType(NestedField(3, "value", LongType())),
                    )
                ),
            ),
            NestedField(4, "row", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(3, 1000, TruncateTransform(10), "value_truncated")
        ),
    )
    df = pl.DataFrame(
        {
            "outer": [
                {"inner": {"value": 11}},
                {"inner": {"value": 12}},
                {"inner": {"value": 21}},
                {"inner": {"value": 22}},
            ],
            "row": [1, 2, 3, 4],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("row"), df)
    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {10, 20}


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_nested_source_overwrite(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(
                1,
                "nested",
                StructType(NestedField(2, "partition_col", LongType())),
            ),
            NestedField(3, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(2, 1000, IdentityTransform(), "partition_col")
        ),
    )
    pl.LazyFrame(
        {
            "nested": [{"partition_col": 1}, {"partition_col": 2}],
            "value": [1, 2],
        }
    ).sink_iceberg(tbl, mode="append")
    replacement = pl.DataFrame(
        {
            "nested": [{"partition_col": 3}, {"partition_col": 3}],
            "value": [3, 4],
        }
    )

    replacement.lazy().sink_iceberg(tbl, mode="overwrite")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), replacement)
    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {3}


@pytest.mark.parametrize(
    ("iceberg_type", "transform", "values"),
    [
        pytest.param(LongType(), VoidTransform(), [1, 2], id="void"),
        pytest.param(DoubleType(), BucketTransform(4), [1.0, 2.0], id="bucket-double"),
        pytest.param(
            DecimalType(10, 2),
            TruncateTransform(10),
            [D("1.23"), D("11.23")],
            id="truncate-decimal",
        ),
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_partitioned_unsupported_transform(
    tmp_path: Path,
    iceberg_type: Any,
    transform: Any,
    values: list[Any],
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", iceberg_type)),
        partition_spec=PartitionSpec(PartitionField(1, 1000, transform, "part")),
    )

    with pytest.raises(NotImplementedError, match="partition transform"):
        pl.LazyFrame({"a": values}).sink_iceberg(tbl, mode="append")


@pytest.mark.write_disk
def test_sink_iceberg_schema_merge(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", IntegerType()),
            NestedField(2, "existing", StringType()),
        ),
    )
    original_schema = tbl.schema()

    pl.DataFrame(
        {
            "id": pl.Series([1], dtype=pl.Int32),
            "existing": ["old"],
        }
    ).lazy().sink_iceberg(tbl, mode="append")

    pl.DataFrame(
        {
            "id": pl.Series([2], dtype=pl.Int64),
            "new": ["new"],
        }
    ).lazy().sink_iceberg(tbl, mode="append", schema_mode="merge")

    schema = tbl.schema()
    assert schema.find_field("id").field_id == original_schema.find_field("id").field_id
    assert schema.find_field("id").field_type == LongType()
    assert (
        schema.find_field("existing").field_id
        == original_schema.find_field("existing").field_id
    )
    assert schema.find_field("new").field_id > original_schema.highest_field_id
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("id"),
        pl.DataFrame(
            {
                "id": pl.Series([1, 2], dtype=pl.Int64),
                "existing": ["old", None],
                "new": [None, "new"],
            }
        ),
    )


@pytest.mark.write_disk
def test_sink_iceberg_schema_merge_nested(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(
                1,
                "payload",
                StructType(
                    NestedField(2, "id", LongType()),
                    NestedField(3, "existing", StringType()),
                ),
            ),
        ),
    )
    pl.DataFrame(
        {
            "payload": pl.Series(
                [{"id": 1, "existing": "old"}],
                dtype=pl.Struct({"id": pl.Int64, "existing": pl.String}),
            )
        }
    ).lazy().sink_iceberg(tbl, mode="append")

    pl.DataFrame(
        {
            "payload": pl.Series(
                [{"id": 2, "new": "new"}],
                dtype=pl.Struct({"id": pl.Int64, "new": pl.String}),
            )
        }
    ).lazy().sink_iceberg(tbl, mode="append", schema_mode="merge")

    assert tbl.schema().find_field("payload.id").field_id == 2
    assert tbl.schema().find_field("payload.existing").field_id == 3
    assert tbl.schema().find_field("payload.new").field_id == 4
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("payload"),
        pl.DataFrame(
            {
                "payload": pl.Series(
                    [
                        {"id": 1, "existing": "old", "new": None},
                        {"id": 2, "existing": None, "new": "new"},
                    ],
                    dtype=pl.Struct(
                        {
                            "id": pl.Int64,
                            "existing": pl.String,
                            "new": pl.String,
                        }
                    ),
                )
            }
        ),
    )


@pytest.mark.write_disk
def test_sink_iceberg_schema_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "old", StringType()),
        ),
    )
    pl.LazyFrame({"id": [1], "old": ["old"]}).sink_iceberg(tbl, mode="append")

    commit_table = tbl.catalog.commit_table
    commit_count = 0

    def count_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal commit_count
        commit_count += 1
        return commit_table(*args, **kwargs)

    monkeypatch.setattr(tbl.catalog, "commit_table", count_commit)

    replacement = pl.DataFrame(
        {
            "id": ["new"],
            "new": pl.Series(
                [{"value": 2}],
                dtype=pl.Struct({"value": pl.Int64}),
            ),
        }
    ).lazy()
    replacement.sink_iceberg(
        tbl,
        mode="overwrite",
        schema_mode="overwrite",
    )

    schema = tbl.schema()
    assert commit_count == 1
    assert [field.name for field in schema.fields] == ["id", "new"]
    assert schema.find_field("id").field_type == StringType()
    assert schema.find_field("new").field_type.is_struct
    assert schema.find_field("new.value").field_type == LongType()
    assert all(field.field_id > 2 for field in schema.fields)
    assert schema.find_field("new.value").field_id > 2
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), replacement.collect())

    with pytest.raises(
        ValueError,
        match="schema_mode='overwrite' requires mode='overwrite'",
    ):
        replacement.sink_iceberg(
            tbl,
            mode="append",
            schema_mode="overwrite",
        )


@pytest.mark.write_disk
def test_sink_iceberg_partitioned_schema_overwrite_unsupported(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "a")),
    )

    with pytest.raises(
        NotImplementedError,
        match="schema_mode='overwrite' is not supported for partitioned Iceberg tables",
    ):
        pl.LazyFrame({"a": [1]}).sink_iceberg(
            tbl,
            mode="overwrite",
            schema_mode="overwrite",
        )

    assert tbl.current_snapshot() is None


@pytest.mark.write_disk
def test_sink_iceberg_snapshot_properties(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
    )
    snapshot_properties = {"application": "polars", "run-id": "run-123"}

    def snapshot_ids() -> set[int]:
        return {snapshot.snapshot_id for snapshot in tbl.snapshots()}

    def assert_snapshot_properties(new_snapshot_ids: set[int]) -> None:
        snapshots = [
            snapshot
            for snapshot in tbl.snapshots()
            if snapshot.snapshot_id in new_snapshot_ids
        ]
        assert snapshots
        for snapshot in snapshots:
            assert snapshot.summary is not None
            summary_properties = snapshot.summary.additional_properties
            assert snapshot_properties.items() <= summary_properties.items()

    before_append = snapshot_ids()
    pl.LazyFrame({"a": [1, 2]}).sink_iceberg(
        tbl,
        mode="append",
        snapshot_properties=snapshot_properties,
    )
    assert_snapshot_properties(snapshot_ids() - before_append)

    before_overwrite = snapshot_ids()
    pl.LazyFrame({"a": [3]}).sink_iceberg(
        tbl,
        mode="overwrite",
        snapshot_properties=snapshot_properties,
    )
    assert_snapshot_properties(snapshot_ids() - before_overwrite)


def _sink_uncommitted(
    monkeypatch: pytest.MonkeyPatch,
    tbl: pyiceberg.table.Table,
    lf: pl.LazyFrame | None = None,
    *,
    mode: Literal["append", "overwrite"] = "append",
) -> tuple[IcebergSinkState, list[Any]]:
    if lf is None:
        lf = pl.LazyFrame({"a": [42]})
    captured = []

    def commit(self: IcebergSinkState, sinked_files: Any) -> pl.DataFrame:
        captured.append((self, sinked_files))
        return pl.DataFrame({"metadata_path": [""]})

    with monkeypatch.context() as m:
        m.setattr(IcebergSinkState, "commit", commit)
        lf.sink_iceberg(tbl, mode=mode)

    [(state, sinked_files)] = captured
    return state, sinked_files


# A re-execution of the same sink invocation, as polars-cloud does when a
# committing worker dies. Pickling drops the loaded table.
def _twin(state: IcebergSinkState) -> IcebergSinkState:
    twin: IcebergSinkState = pickle.loads(pickle.dumps(state))
    assert twin.table_.get() is None
    return twin


def _tagged_snapshots(
    tbl: pyiceberg.table.Table, state: IcebergSinkState
) -> list[Snapshot]:
    tbl.refresh()
    return [
        s
        for s in tbl.snapshots()
        if s.summary is not None
        and s.summary.get(_SINK_UUID_PROPERTY) == state.sink_uuid_str
    ]


# The current snapshot, with its manifest list and manifests still on disk.
def _landed_snapshot(tbl: pyiceberg.table.Table) -> Snapshot:
    tbl.refresh()
    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    assert tbl.io.new_input(snapshot.manifest_list).exists()
    for manifest in snapshot.manifests(tbl.io):
        assert tbl.io.new_input(manifest.manifest_path).exists()
    return snapshot


def _foreign_append(tbl: pyiceberg.table.Table, values: list[int]) -> None:
    pl.LazyFrame({"a": values}).sink_iceberg(
        tbl.catalog.load_table(tbl.name()), mode="append"
    )


def _rows(tbl: pyiceberg.table.Table) -> list[int]:
    tbl.refresh()
    return sorted(pl.scan_iceberg(tbl).collect()["a"].to_list())


def _count_commit_table(
    monkeypatch: pytest.MonkeyPatch, catalog: pyiceberg.catalog.Catalog
) -> Mock:
    commit_table = Mock(wraps=catalog.commit_table)
    monkeypatch.setattr(catalog, "commit_table", commit_table)
    return commit_table


@pytest.mark.write_disk
def test_sink_iceberg_commit_twin_from_stale_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Without retries, pyiceberg never reaches its refresh, so only the recovery
    # at the catalog call can find the twin.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties={"commit.retry.num-retries": "0"},
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    twin = _twin(state)
    twin.table()
    state.commit(sinked_files)
    landed = _landed_snapshot(tbl)

    calls = _count_commit_table(monkeypatch, twin.table().catalog)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    result = twin.commit(sinked_files)

    assert calls.call_count == 1
    assert (
        "IcebergSinkState[commit]: already committed, skipping"
        in capfd.readouterr().err
    )
    assert _landed_snapshot(tbl) == landed
    assert _tagged_snapshots(tbl, state) == [landed]
    assert _rows(tbl) == [42]
    assert result.item() == tbl.metadata_location


@pytest.mark.write_disk
def test_sink_iceberg_commit_twin_lands_during_retry_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    twin = _twin(state)
    twin_catalog = twin.table().catalog

    commit_table = twin_catalog.commit_table
    calls = 0

    def commit_table_after_foreign_append(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            _foreign_append(tbl, [7])
        return commit_table(*args)

    monkeypatch.setattr(twin_catalog, "commit_table", commit_table_after_foreign_append)

    sleep = pyiceberg.table.time.sleep
    slept = False

    def sleep_while_original_commits(seconds: float) -> None:
        nonlocal slept
        if not slept:
            slept = True
            state.commit(sinked_files)
        sleep(0)

    monkeypatch.setattr(pyiceberg.table.time, "sleep", sleep_while_original_commits)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    twin.commit(sinked_files)

    # Only the foreign append fails the first commit, so the twin is found by the
    # check after pyiceberg's retry refresh, without a second commit attempt.
    assert slept
    assert calls == 1
    assert (
        "IcebergSinkState[commit]: already committed, skipping"
        in capfd.readouterr().err
    )
    assert _tagged_snapshots(tbl, state) == [_landed_snapshot(tbl)]
    assert _rows(tbl) == [7, 42]


@pytest.mark.write_disk
def test_sink_iceberg_commit_updates_caller_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The commit goes through a copy of the table with a wrapped catalog.
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )

    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    assert state.table() is tbl
    result = state.commit(sinked_files)
    current = catalog.load_table(tbl.name()).metadata_location
    assert tbl.metadata_location == current
    assert result.item() == current

    # An "already committed" skip also updates it.
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    _twin(state).commit(sinked_files)
    result = state.commit(sinked_files)
    current = catalog.load_table(tbl.name()).metadata_location
    assert tbl.metadata_location == current
    assert result.item() == current
    assert _rows(tbl) == [42, 42]


@pytest.mark.parametrize(
    ("mode", "properties"),
    [
        pytest.param("append", {}, id="append"),
        pytest.param("append", {"commit.retry.num-retries": "0"}, id="no-retries"),
        pytest.param(
            "overwrite",
            {},
            id="overwrite-empty-table",
            marks=pytest.mark.filterwarnings(
                "ignore:Delete operation did not match any records"
            ),
        ),
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_commit_recovers_lost_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    mode: Literal["append", "overwrite"],
    properties: dict[str, str],
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties=properties,
    )
    commit_table = tbl.catalog.commit_table

    def commit_table_lose_response(*args: Any) -> Any:
        commit_table(*args)
        msg = "lost response"
        raise pyiceberg.exceptions.CommitFailedException(msg)

    monkeypatch.setattr(tbl.catalog, "commit_table", commit_table_lose_response)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl, mode=mode)
    state.commit(sinked_files)

    assert (
        "IcebergSinkState[commit]: recovered lost commit response"
        in capfd.readouterr().err
    )
    _landed_snapshot(tbl)
    assert len(tbl.snapshots()) == 1
    assert _rows(tbl) == [42]


@pytest.mark.write_disk
def test_sink_iceberg_commit_unverifiable_outcome_keeps_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    commit_table = catalog.commit_table
    load_table = catalog.load_table

    def commit_table_lose_response(*args: Any) -> Any:
        commit_table(*args)
        monkeypatch.setattr(catalog, "load_table", load_table_fails)
        msg = "lost response"
        raise pyiceberg.exceptions.CommitFailedException(msg)

    def load_table_fails(*args: Any) -> Any:
        monkeypatch.setattr(catalog, "load_table", load_table)
        msg = "catalog unavailable"
        raise ConnectionError(msg)

    monkeypatch.setattr(catalog, "commit_table", commit_table_lose_response)

    with pytest.raises(pyiceberg.exceptions.CommitStateUnknownException):
        state.commit(sinked_files)

    _landed_snapshot(tbl)
    assert len(tbl.snapshots()) == 1
    assert _rows(tbl) == [42]


@pytest.mark.write_disk
def test_sink_iceberg_commit_twin_in_recreated_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    original_uuid = tbl.metadata.table_uuid
    original_metadata_location = tbl.metadata_location
    catalog.drop_table(tbl.name())
    catalog.create_table(tbl.name(), tbl.schema())
    _twin(state).commit(sinked_files)

    with pytest.raises(ValueError, match="Table UUID does not match"):
        state.commit(sinked_files)

    assert tbl.metadata.table_uuid == original_uuid
    assert tbl.metadata_location == original_metadata_location


@pytest.mark.write_disk
def test_sink_iceberg_commit_sequential_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    state.commit(sinked_files)
    landed = _landed_snapshot(tbl)

    twin = _twin(state)
    calls = _count_commit_table(monkeypatch, twin.table().catalog)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    twin.commit(sinked_files)

    assert calls.call_count == 0
    assert (
        "IcebergSinkState[commit]: already committed, skipping"
        in capfd.readouterr().err
    )
    assert _landed_snapshot(tbl) == landed
    assert _tagged_snapshots(tbl, state) == [landed]
    assert _rows(tbl) == [42]


@pytest.mark.write_disk
def test_sink_iceberg_commit_empty_overwrite_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    pl.LazyFrame({"a": [1]}).sink_iceberg(tbl, mode="append")
    state, sinked_files = _sink_uncommitted(
        monkeypatch,
        tbl,
        pl.LazyFrame({"a": []}, schema={"a": pl.Int64}),
        mode="overwrite",
    )
    state.commit(sinked_files)
    assert _rows(tbl) == []
    _foreign_append(tbl, [7])

    _twin(state).commit(sinked_files)

    assert _rows(tbl) == [7]


@pytest.mark.write_disk
def test_sink_iceberg_commit_replay_after_expired_middle_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    state.commit(sinked_files)
    _foreign_append(tbl, [1])
    tbl.refresh()
    middle = tbl.current_snapshot()
    assert middle is not None
    _foreign_append(tbl, [2])
    tbl.refresh()
    tbl.maintenance.expire_snapshots().by_id(middle.snapshot_id).commit()
    assert tbl.snapshot_by_id(middle.snapshot_id) is None

    _twin(state).commit(sinked_files)

    assert len(_tagged_snapshots(tbl, state)) == 1
    assert _rows(tbl) == [1, 2, 42]


@pytest.mark.write_disk
def test_sink_iceberg_commit_replay_after_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    pl.LazyFrame({"a": [1]}).sink_iceberg(tbl, mode="append")
    parent = tbl.current_snapshot()
    assert parent is not None
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    state.commit(sinked_files)
    tbl.manage_snapshots().rollback_to_snapshot(parent.snapshot_id).commit()
    assert _rows(tbl) == [1]

    _twin(state).commit(sinked_files)

    assert len(_tagged_snapshots(tbl, state)) == 1
    assert _rows(tbl) == [1]


@pytest.mark.write_disk
def test_sink_iceberg_commit_retries_after_concurrent_foreign_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties={"commit.retry.min-wait-ms": "0"},
    )
    state, sinked_files = _sink_uncommitted(monkeypatch, tbl)
    _foreign_append(tbl, [7])
    calls = _count_commit_table(monkeypatch, catalog)

    state.commit(sinked_files)

    assert calls.call_count == 2
    assert _rows(tbl) == [7, 42]


@pytest.mark.write_disk
def test_sink_iceberg_rerun_appends_again(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    lf = pl.LazyFrame({"a": [42]})
    lf.sink_iceberg(tbl, mode="append")
    lf.sink_iceberg(tbl, mode="append")

    assert _rows(tbl) == [42, 42]
    uuids = {s.summary.get(_SINK_UUID_PROPERTY) for s in tbl.snapshots()}  # type: ignore[union-attr]
    assert len(uuids) == 2
    assert None not in uuids


@pytest.mark.parametrize(
    ("partitioned_paths", "custom_data_path"),
    [
        pytest.param(None, False, id="defaults"),
        pytest.param(False, False, id="partitioned-paths-false"),
        pytest.param(True, True, id="custom-data-path-trailing-slash"),
    ],
)
@pytest.mark.write_disk
def test_sink_iceberg_object_storage(
    tmp_path: Path,
    *,
    partitioned_paths: bool | None,
    custom_data_path: bool,
) -> None:
    from pyiceberg.table import TableProperties
    from pyiceberg.table.locations import load_location_provider

    properties = {TableProperties.OBJECT_STORE_ENABLED: "true"}
    if partitioned_paths is not None:
        properties[TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS] = (
            "true" if partitioned_paths else "false"
        )
    if custom_data_path:
        data_path = format_file_uri_iceberg(tmp_path / "custom-data")
        properties[TableProperties.WRITE_DATA_PATH] = f"{data_path}/"

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties=properties,
    )

    effective_partitioned_paths = (
        TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS_DEFAULT
        if partitioned_paths is None
        else partitioned_paths
    )
    location_provider = load_location_provider(
        tbl.metadata.location, tbl.metadata.properties
    )

    def assert_data_file_paths_match_location_provider() -> list[str]:
        file_paths = [task.file.file_path for task in tbl.scan().plan_files()]
        assert len(file_paths) == len(set(file_paths))

        for file_path in file_paths:
            final_path_component = file_path.rsplit("/", 1)[-1]
            file_name = (
                final_path_component
                if effective_partitioned_paths
                else final_path_component.split("-", 1)[1]
            )
            assert file_path == location_provider.new_data_location(file_name)

        return file_paths

    pl.LazyFrame({"a": [1, 2]}).sink_iceberg(tbl, mode="append")
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("a"), pl.DataFrame({"a": [1, 2]})
    )
    first_append_paths = assert_data_file_paths_match_location_provider()
    assert len(first_append_paths) == 1

    pl.LazyFrame({"a": [3]}).sink_iceberg(tbl, mode="append")
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("a"), pl.DataFrame({"a": [1, 2, 3]})
    )
    second_append_paths = assert_data_file_paths_match_location_provider()
    assert len(second_append_paths) == 2
    assert set(first_append_paths) < set(second_append_paths)

    pl.LazyFrame({"a": [4]}).sink_iceberg(tbl, mode="overwrite")
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), pl.DataFrame({"a": [4]}))
    overwrite_paths = assert_data_file_paths_match_location_provider()
    assert len(overwrite_paths) == 1
    assert set(overwrite_paths).isdisjoint(second_append_paths)


@pytest.mark.parametrize("partitioned_paths", [False, True])
@pytest.mark.write_disk
def test_sink_iceberg_partitioned_object_storage(
    tmp_path: Path, *, partitioned_paths: bool
) -> None:
    from pyiceberg.table import TableProperties

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(
                2,
                "nested",
                StructType(NestedField(3, "b", LongType())),
            ),
            NestedField(4, "value", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "a"),
            PartitionField(3, 1001, TruncateTransform(10), "b_truncated"),
        ),
        properties={
            TableProperties.OBJECT_STORE_ENABLED: "true",
            TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS: (
                "true" if partitioned_paths else "false"
            ),
        },
    )
    df = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "nested": [{"b": 11}, {"b": 12}, {"b": 21}],
            "value": [1, 2, 3],
        }
    )

    df.lazy().sink_iceberg(tbl, mode="append")

    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("value"), df)
    tasks = list(tbl.scan().plan_files())
    assert {(task.file.partition[0], task.file.partition[1]) for task in tasks} == {
        (1, 10),
        (2, 20),
    }
    file_paths = [task.file.file_path for task in tasks]
    assert len(file_paths) == 2
    assert len(file_paths) == len(set(file_paths))


@pytest.mark.write_disk
def test_sink_iceberg_object_storage_property_update(tmp_path: Path) -> None:
    from pyiceberg.table import TableProperties
    from pyiceberg.table.locations import load_location_provider

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties={TableProperties.OBJECT_STORE_ENABLED: "true"},
    )

    pl.LazyFrame({"a": [1]}).sink_iceberg(tbl, mode="append")
    [first_path] = [task.file.file_path for task in tbl.scan().plan_files()]
    first_file_name = first_path.rsplit("/", 1)[-1]
    initial_location_provider = load_location_provider(
        tbl.metadata.location, tbl.metadata.properties
    )
    assert first_path == initial_location_provider.new_data_location(first_file_name)

    with tbl.transaction() as tx:
        tx.set_properties(
            {TableProperties.WRITE_OBJECT_STORE_PARTITIONED_PATHS: "false"}
        )

    pl.LazyFrame({"a": [2]}).sink_iceberg(tbl, mode="append")
    file_paths = {task.file.file_path for task in tbl.scan().plan_files()}
    assert len(file_paths) == 2
    assert first_path in file_paths
    [second_path] = file_paths - {first_path}
    second_file_name = second_path.rsplit("/", 1)[-1].split("-", 1)[1]
    updated_location_provider = load_location_provider(
        tbl.metadata.location, tbl.metadata.properties
    )
    assert second_path == updated_location_provider.new_data_location(second_file_name)

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("a"), pl.DataFrame({"a": [1, 2]})
    )


@pytest.mark.write_disk
def test_sink_iceberg_raises_on_static_table(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )

    err_cx = pytest.raises(
        TypeError,
        match=r"cannot sink to static Iceberg table.*class.*StaticTable.*class.*NoopCatalog",
    )

    with err_cx:
        IcebergSinkState.new(StaticTable.from_metadata(tbl.metadata_location))

    with err_cx:
        pl.LazyFrame({"a": 1}).sink_iceberg(
            StaticTable.from_metadata(tbl.metadata_location), mode="append"
        )

    with pytest.raises(TypeError, match="cannot use NoopCatalog with sink_iceberg"):
        pl.LazyFrame({"a": 1}).sink_iceberg(
            "namespace.table",
            catalog=StaticTable.from_metadata(tbl.metadata_location).catalog,
            mode="append",
        )


@pytest.mark.write_disk
def test_sink_iceberg_pickle(tmp_path: Path) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "a")),
    )

    sink_state = IcebergSinkState.new(tbl)
    sink_q = sink_state.attach_sink(pl.LazyFrame({"a": 1}))
    sink_q.collect()
    rdf = sink_state.commit_result_df.get()
    assert rdf is not None
    new_md_path = rdf.item(0, "metadata_path")

    tbl = sink_state.table()

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), pl.DataFrame({"a": 1}))
    assert new_md_path == tbl.metadata_location

    snapshot_properties = {"run-id": "pickled-run"}
    sink_state = IcebergSinkState.new(
        tbl,
        schema_mode="merge",
        snapshot_properties=snapshot_properties,
    )
    sink_q = sink_state.attach_sink(pl.LazyFrame({"a": 2, "b": "x"}))
    sink_q = pickle.loads(pickle.dumps(sink_q))
    sink_q.collect()

    new_tbl = catalog.load_table(tbl.name())
    new_md_path = new_tbl.metadata_location

    assert new_tbl.metadata_location != tbl.metadata_location

    tbl = new_tbl

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(),
        pl.DataFrame({"a": [2, 1], "b": ["x", None]}),
    )
    current_snapshot = tbl.current_snapshot()
    assert current_snapshot is not None
    assert current_snapshot.summary is not None
    summary_properties = current_snapshot.summary.additional_properties
    assert snapshot_properties.items() <= summary_properties.items()

    assert new_md_path == tbl.metadata_location


@pytest.mark.write_disk
def test_sink_iceberg__attach_resolved_sink_deferred_to_ir_resolution(
    tmp_path: Path,
) -> None:
    sink_state = IcebergSinkState.new(
        "x.x",
        catalog=SqlCatalog(
            "default",
            uri="sqlite:///:memory:",
            warehouse=format_file_uri_iceberg(tmp_path / "catalog"),
        ),
    )
    q = sink_state.attach_sink(pl.LazyFrame())

    with pytest.raises(pyiceberg.exceptions.NoSuchTableError):
        q.collect()


@pytest.mark.write_disk
def test_sink_iceberg_internal_path_provider(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [None, ""]})

    df.lazy().sink_parquet(
        pl.PartitionBy(
            tmp_path / "keyed",
            file_path_provider=PlIcebergPathProviderConfig(),
            key="a",
        )
    )

    assert_frame_equal(
        pl.scan_parquet(tmp_path / "keyed").sort("*").collect(),
        df,
    )

    assert (
        _expand_paths(tmp_path / "keyed")
        .with_columns(
            part_prefix=pl.col("path").str.extract(r"/\w{32}(\w{32})00000000\.parquet$")
        )
        .select(
            (pl.col("part_prefix").len() == 2)
            & (pl.col("part_prefix").null_count() == 0)
            & (pl.col("part_prefix").n_unique() == 1)
        )
        .collect()
        .item()
    )

    df.lazy().sink_parquet(
        pl.PartitionBy(
            tmp_path / "max-rows",
            file_path_provider=PlIcebergPathProviderConfig(
                object_storage_partitioned_paths=True
            ),
            max_rows_per_file=1,
        )
    )

    assert_frame_equal(
        pl.scan_parquet(tmp_path / "max-rows").sort("*").collect(),
        df,
    )

    assert (
        _expand_paths(tmp_path / "max-rows/**/*.parquet")
        .with_columns(
            part_prefix=pl.col("path").str.extract(
                r"/[01]{4}/[01]{4}/[01]{4}/[01]{8}/(\w{32})0000000[01]\.parquet$"
            ),
        )
        .select(
            (pl.col("part_prefix").len() == 2)
            & (pl.col("part_prefix").null_count() == 0)
            & (pl.col("part_prefix").n_unique() == 1)
        )
        .collect()
        .item()
    )


@pytest.mark.write_disk
def test_iceberg_dataset_does_not_pickle_table_object(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4]},
        schema={"row_index": pl.Int32},
    )

    df.write_iceberg(tbl, mode="append")

    dataset = new_iceberg_scan_resolver(tbl)

    assert dataset.table.table_.get() is not None
    dataset = pickle.loads(pickle.dumps(dataset))
    assert dataset.table.table_.get() is None

    assert_frame_equal(dataset.to_dataset_scan()[0].collect(), df)


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings("ignore:Delete operation did not match any records")
@pytest.mark.filterwarnings(
    "ignore:Iceberg does not have a dictionary type. <class 'pyarrow.lib.DictionaryType'> will be inferred as large_string on read."
)
def test_write_iceberg(df: pl.DataFrame, tmp_path: Path) -> None:
    # time64[ns] type is currently not supported in pyiceberg.
    # https://github.com/apache/iceberg-python/issues/1169
    df = df.drop("time", "cat", "enum")

    # in-memory catalog
    catalog = SqlCatalog(
        "default", uri="sqlite:///:memory:", warehouse=format_file_uri_iceberg(tmp_path)
    )
    catalog.create_namespace("foo")
    table = catalog.create_table(
        "foo.bar",
        schema=df.to_arrow().schema,
    )

    df.write_iceberg(table, mode="overwrite")
    actual = pl.scan_iceberg(table).collect()

    assert_frame_equal(df, actual)

    # append on top of already written data, expecting twice the data
    df.write_iceberg(table, mode="append")
    # double the `df` by vertically stacking the dataframe on top of itself
    expected = df.vstack(df)
    actual = pl.scan_iceberg(table).collect()
    assert_frame_equal(expected, actual, check_dtypes=False)


@pytest.mark.write_disk
def test_write_iceberg_required_columns(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default", uri="sqlite:///:memory:", warehouse=format_file_uri_iceberg(tmp_path)
    )
    catalog.create_namespace("ns")
    tbl = catalog.create_table(
        "ns.t",
        IcebergSchema(
            NestedField(1, "id", LongType(), required=True),
            NestedField(2, "v", LongType(), required=False),
        ),
    )

    pl.DataFrame({"id": [1, 2], "v": [None, 2]}).write_iceberg(tbl, mode="append")
    # the null is outside the slice
    pl.DataFrame({"id": [None, 3], "v": [3, 3]}).slice(1).write_iceberg(
        tbl, mode="append"
    )
    pl.DataFrame({"id": [4]}).write_iceberg(tbl, mode="append")
    with pytest.raises(pl.exceptions.SchemaError, match="column 'id' contains nulls"):
        pl.DataFrame({"id": [None, 5]}).write_iceberg(tbl, mode="append")

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect().sort("id"),
        pl.DataFrame({"id": [1, 2, 3, 4], "v": [None, 2, 3, None]}),
    )

    pl.DataFrame({"id": [6], "v": [6]}).write_iceberg(tbl, mode="overwrite")
    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(), pl.DataFrame({"id": [6], "v": [6]})
    )


@pytest.mark.write_disk
def test_scan_iceberg_row_index_renamed(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
            NestedField(2, "file_path", StringType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4], "file_path": None},
        schema={"row_index": pl.Int32, "file_path": pl.String},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.rename_column("row_index", "row_index_in_file")
        sch.rename_column("file_path", "file_path_in_file")

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]
    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={
            "row_index_in_file": pl.Int32,
            "file_path_in_file": pl.String,
        },
        _column_mapping=(
            "iceberg-column-mapping",
            new_iceberg_scan_resolver(tbl).table.arrow_schema(),
        ),
        include_file_paths="file_path",
        row_index_name="row_index",
        row_index_offset=3,
    )

    assert_frame_equal(
        q.collect().with_columns(normalize_path_separator_pl(pl.col("file_path"))),
        pl.DataFrame(
            {
                "row_index": [3, 4, 5, 6, 7],
                "row_index_in_file": [0, 1, 2, 3, 4],
                "file_path_in_file": None,
                "file_path": file_paths[0],
            },
            schema={
                "row_index": pl.get_index_type(),
                "row_index_in_file": pl.Int32,
                "file_path_in_file": pl.String,
                "file_path": pl.String,
            },
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_table_name(tmp_path: Path) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
        ),
    )

    pl.DataFrame({"a": 1}).lazy().sink_iceberg(tbl, mode="append")
    pl.DataFrame({"a": 2}).lazy().sink_iceberg(
        ".".join(tbl.name()),
        mode="append",
        catalog=catalog,
    )

    tbl = catalog.load_table(tbl.name())

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(),
        pl.DataFrame({"a": [2, 1]}),
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl.metadata_location).collect(),
        pl.DataFrame({"a": [2, 1]}),
    )

    assert_frame_equal(
        pl.scan_iceberg(".".join(tbl.name()), catalog=catalog).collect(),
        pl.DataFrame({"a": [2, 1]}),
    )


def test_scan_iceberg_rejects_snapshot_id_with_incremental_range() -> None:
    with pytest.raises(ValueError, match="cannot combine `snapshot_id`"):
        pl.scan_iceberg(
            "/tmp/metadata.json",
            snapshot_id=1,
            from_snapshot_id_exclusive=2,
        )


@pytest.mark.write_disk
def test_new_pyiceberg_scan_forwards_incremental_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
    )
    sentinel = object()
    captured: dict[str, Any] = {}

    def incremental_append_scan(table: Any, **kwargs: Any) -> object:
        assert table is tbl
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        pyiceberg.table.Table,
        "incremental_append_scan",
        incremental_append_scan,
    )

    result = _new_pyiceberg_scan(
        tbl,
        snapshot_id=None,
        from_snapshot_id_exclusive=10,
        to_snapshot_id_inclusive=20,
        selected_fields=("a",),
        limit=3,
    )

    assert result is sentinel
    assert captured == {
        "from_snapshot_id_exclusive": 10,
        "to_snapshot_id_inclusive": 20,
        "selected_fields": ("a",),
        "limit": 3,
    }


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["native", "pyiceberg"])
def test_scan_iceberg_incremental_append_range(
    tmp_path: Path,
    reader_override: str,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
    )

    tbl.append(pa.table({"a": [1]}))
    first_snapshot = tbl.current_snapshot()
    assert first_snapshot is not None

    tbl.append(pa.table({"a": [2, 3]}))
    second_snapshot = tbl.current_snapshot()
    assert second_snapshot is not None

    query = (
        pl.scan_iceberg(
            tbl,
            from_snapshot_id_exclusive=first_snapshot.snapshot_id,
            to_snapshot_id_inclusive=second_snapshot.snapshot_id,
            reader_override=reader_override,  # type: ignore[arg-type]
        )
        .filter(pl.col("a") > 2)
        .limit(1)
    )
    assert_frame_equal(query.collect(), pl.DataFrame({"a": [3]}))

    tbl.append(pa.table({"a": [4]}))
    assert_frame_equal(query.collect(), pl.DataFrame({"a": [3]}))


@pytest.mark.write_disk
def test_scan_iceberg_incremental_append_tracks_current_snapshot(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
    )

    tbl.append(pa.table({"a": [1]}))
    first_snapshot = tbl.current_snapshot()
    assert first_snapshot is not None

    query = pl.scan_iceberg(
        tbl,
        from_snapshot_id_exclusive=first_snapshot.snapshot_id,
    )

    tbl.append(pa.table({"a": [2]}))
    assert_frame_equal(query.collect(), pl.DataFrame({"a": [2]}))

    tbl.append(pa.table({"a": [3]}))
    assert_frame_equal(query.collect().sort("a"), pl.DataFrame({"a": [2, 3]}))


@pytest.mark.write_disk
def test_scan_iceberg_polars_storage_options_keys(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
            NestedField(2, "file_path", StringType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4], "file_path": None},
        schema={"row_index": pl.Int32, "file_path": pl.String},
    ).write_iceberg(tbl, mode="append")

    capfd.readouterr()

    pl.scan_iceberg(
        tbl,
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 3,
            "retry_timeout_ms": 9873,
            "retry_init_backoff_ms": 9874,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
        },
    ).collect()

    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
max_retries: Some(3), \
retry_timeout: Some(9.873s), \
retry_init_backoff: Some(9.874s), \
retry_max_backoff: Some(9.875s), \
retry_base_multiplier: Some(TotalOrdWrap(3.14159)) }"""
        in capture
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["pyiceberg", "native"])
def test_scan_iceberg_collect_without_version_scans_latest(
    tmp_path: Path,
    reader_override: str,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )

    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", LongType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    q = pl.scan_iceberg(tbl, reader_override=reader_override)  # type: ignore[arg-type]

    assert_frame_equal(q.collect(), pl.DataFrame(schema={"a": pl.Int64}))

    pl.DataFrame({"a": 1}).write_iceberg(tbl, mode="append")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1}))

    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    snapshot_id = snapshot.snapshot_id

    q_with_id = pl.scan_iceberg(
        tbl,
        reader_override=reader_override,  # type: ignore[arg-type]
        snapshot_id=snapshot_id,
    )

    assert_frame_equal(q_with_id.collect(), pl.DataFrame({"a": 1}))

    pl.DataFrame({"a": 2}).write_iceberg(tbl, mode="append")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": [2, 1]}))

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    assert_frame_equal(q_with_id.collect(), pl.DataFrame({"a": 1}))

    assert (
        "IcebergScanResolver: to_dataset_scan(): early return (snapshot_id_key = "
        in capfd.readouterr().err
    )


@pytest.mark.write_disk
def test_scan_iceberg_extra_columns(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", IntegerType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"a": [0, 1, 2, 3, 4]},
        schema={"a": pl.Int32},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column("a")
        sch.add_column("a", IntegerType())

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]

    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Int32},
        _column_mapping=(
            "iceberg-column-mapping",
            new_iceberg_scan_resolver(tbl).table.arrow_schema(),
        ),
    )

    # The original column is considered an extra column despite having the same
    # name as the physical ID does not match.

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="extra column in file outside of expected schema: a",
    ):
        q.collect()

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Int32},
        _column_mapping=(
            "iceberg-column-mapping",
            new_iceberg_scan_resolver(tbl).table.arrow_schema(),
        ),
        extra_columns="ignore",
        missing_columns="insert",
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [None, None, None, None, None],
            },
            schema={"a": pl.Int32},
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_extra_struct_fields(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", StructType(NestedField(1, "a", IntegerType()))),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"a": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]},
        schema={"a": pl.Struct({"a": pl.Int32})},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column(("a", "a"))
        sch.add_column(("a", "a"), IntegerType())

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]

    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Struct({"a": pl.Int32})},
        _column_mapping=(
            "iceberg-column-mapping",
            new_iceberg_scan_resolver(tbl).table.arrow_schema(),
        ),
    )

    # The original column is considered an extra column despite having the same
    # name as the physical ID does not match.

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="encountered extra struct field: a",
    ):
        q.collect()

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Struct({"a": pl.Int32})},
        _column_mapping=(
            "iceberg-column-mapping",
            new_iceberg_scan_resolver(tbl).table.arrow_schema(),
        ),
        cast_options=pl.ScanCastOptions(
            extra_struct_fields="ignore", missing_struct_fields="insert"
        ),
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [
                    {"a": None},
                    {"a": None},
                    {"a": None},
                    {"a": None},
                    {"a": None},
                ],
            },
            schema={"a": pl.Struct({"a": pl.Int32})},
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_filter_on_struct_field_28476(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "order_id", StringType()),
            NestedField(
                2,
                "order_details",
                StructType(NestedField(3, "qid", StringType())),
            ),
        ),
    )
    df = pl.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_details": [{"qid": "X1"}, {"qid": "X2"}, {"qid": "X3"}],
        },
        schema={
            "order_id": pl.String,
            "order_details": pl.Struct({"qid": pl.String}),
        },
    )
    df.write_iceberg(tbl, mode="append")

    q = pl.scan_iceberg(tbl).filter(pl.col("order_details").struct.field("qid") == "X2")
    expected = df.slice(1, 1)

    assert_frame_equal(q.collect(), expected)
    assert_frame_equal(
        q.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False)),
        expected,
    )


@pytest.mark.write_disk
def test_scan_iceberg_column_deletion(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", StructType(NestedField(0, "inner", StringType())))
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame({"a": [{"inner": "A"}]}).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column("a").add_column(
            "a", StructType(NestedField(0, "inner", StringType()))
        )

    pl.DataFrame({"a": [{"inner": "A"}]}).write_iceberg(tbl, mode="append")

    expect = pl.DataFrame({"a": [{"inner": "A"}, None]})

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )


@pytest.mark.write_disk
def test_scan_iceberg_nested_column_cast_deletion_rename(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=StructType(
                        NestedField(
                            field_id=next_field_id(),
                            name="field_1",
                            field_type=MapType(
                                key_id=next_field_id(),
                                key_type=ListType(
                                    element_id=next_field_id(), element=TimestampType(), element_required=False
                                ),
                                value_id=next_field_id(),
                                value_type=ListType(
                                    element_id=next_field_id(), element=IntegerType(), element_required=False
                                ),
                                value_required=False,
                            ),
                            required=False,
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=StringType(), required=False
                        ),
                    ),
                    element_required=False,
                ),
                required=False,
            ),
            NestedField(field_id=next_field_id(), name="column_2", field_type=StringType(), required=False),
            NestedField(
                field_id=next_field_id(),
                name="column_3",
                field_type=MapType(
                    key_id=next_field_id(),
                    key_type=StructType(
                        NestedField(
                            field_id=next_field_id(), name="field_1", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=IntegerType(), required=False
                        ),
                    ),
                    value_id=next_field_id(),
                    value_type=StructType(
                        NestedField(
                            field_id=next_field_id(), name="field_1", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=IntegerType(), required=False
                        ),
                    ),
                    value_required=False,
                ),
                required=False,
            ),
        ),
    )  # fmt: skip

    tbl = catalog.load_table("namespace.table")

    df_dict = {
        "column_1": [
            [
                {
                    "field_1": [
                        {"key": [datetime(2025, 1, 1), None], "value": [1, 2, None]},
                        {"key": [datetime(2025, 1, 2), None], "value": None},
                    ],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [
                {
                    "field_1": [{"key": [datetime(2025, 1, 1), None], "value": None}],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [{"field_1": [], "field_2": None, "field_3": None}],
            [None],
            [],
        ],
        "column_2": ["1", "2", "3", "4", None],
        "column_3": [
            [
                {
                    "key": {"field_1": 1, "field_2": 2, "field_3": 3},
                    "value": {"field_1": 7, "field_2": 8, "field_3": 9},
                }
            ],
            [
                {
                    "key": {"field_1": 1, "field_2": 2, "field_3": 3},
                    "value": {"field_1": 7, "field_2": 8, "field_3": 9},
                }
            ],
            [
                {
                    "key": {"field_1": None, "field_2": None, "field_3": None},
                    "value": {"field_1": None, "field_2": None, "field_3": None},
                }
            ],
            [
                {
                    "key": {"field_1": None, "field_2": None, "field_3": None},
                    "value": None,
                }
            ],
            [],
        ],
    }

    df = pl.DataFrame(
        df_dict,
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.Map(pl.List(pl.Datetime("us")), pl.List(pl.Int32)),
                        "field_2": pl.Int32,
                        "field_3": pl.String,
                    }
                )
            ),
            "column_2": pl.String,
            "column_3": pl.Map(
                pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
                pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
            ),
        },
    )  # fmt: skip

    # The Iceberg table schema has a `Map` type, whereas the polars DataFrame
    # stores `list[struct{..}]` - directly using `write_iceberg()` causes the
    # following error:
    # * ValueError: PyArrow table contains more columns:
    #   column_1.element.field_1.element
    # We workaround this by constructing a pyarrow table an arrow schema.
    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(
                        pa.struct(
                            [
                                (
                                    "field_1",
                                    pa.map_(pa.large_list(pa.timestamp("us")), pa.large_list(pa.int32())),
                                ),
                                ("field_2", pa.int32()),
                                ("field_3", pa.string()),
                            ]
                        )
                    ),
                ),
                ("column_2", pa.string()),
                (
                    "column_3",
                    pa.map_(
                        pa.struct([("field_1", pa.int32()), ("field_2", pa.int32()), ("field_3", pa.int32())]),
                        pa.struct([("field_1", pa.int32()), ("field_2", pa.int32()), ("field_3", pa.int32())]),
                    ),
                ),
            ]
        ),
    )  # fmt: skip

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)

    # Change schema
    # Note: Iceberg doesn't allow modifying the "key" part of the Map type.

    # Promote types
    with tbl.update_schema() as sch:
        sch.update_column(("column_1", "field_2"), LongType())
        sch.update_column(("column_3", "value", "field_1"), LongType())
        sch.update_column(("column_3", "value", "field_2"), LongType())
        sch.update_column(("column_3", "value", "field_3"), LongType())

    # Delete/Rename:
    # * Delete `*_2` fields
    # * Rename:
    #   * `{x}_1` -> `{x}_2`
    #   * `{x}_3` -> `{x}_1`
    #     * And move the field position to 1st

    # Delete `*_2` fields/columns.
    with tbl.update_schema() as sch:
        sch.delete_column("column_2")
        sch.delete_column(("column_3", "value", "field_2"))
        sch.delete_column(("column_1", "field_2"))

    # Shift nested fields in `column_1`
    with tbl.update_schema() as sch:
        sch.rename_column(("column_1", "field_1"), "field_2")

    with tbl.update_schema() as sch:
        sch.rename_column(("column_1", "field_3"), "field_1")

    with tbl.update_schema() as sch:
        sch.move_first(("column_1", "field_1"))

    # Shift nested fields in `column_2`
    with tbl.update_schema() as sch:
        sch.rename_column(("column_3", "value", "field_1"), "field_2")

    with tbl.update_schema() as sch:
        sch.rename_column(("column_3", "value", "field_3"), "field_1")

    with tbl.update_schema() as sch:
        sch.move_first(("column_3", "value", "field_1"))

    # Shift top-level columns
    with tbl.update_schema() as sch:
        sch.rename_column("column_1", "column_2")

    with tbl.update_schema() as sch:
        sch.rename_column("column_3", "column_1")

    with tbl.update_schema() as sch:
        sch.move_first("column_1")

    expect = pl.DataFrame(
        {
            "column_2": [
                [
                    {
                        "field_2": [
                            {"key": [datetime(2025, 1, 1, 0, 0), None], "value": [1, 2, None]},
                            {"key": [datetime(2025, 1, 2), None], "value": None},
                        ],
                        "field_1": "F3",
                    }
                ],
                [{"field_2": [{"key": [datetime(2025, 1, 1, 0, 0), None], "value": None}], "field_1": "F3"}],
                [{"field_2": [], "field_1": None}],
                [None],
                [],
            ],
            "column_1": [
                [{"key": {"field_1": 1, "field_2": 2, "field_3": 3}, "value": {"field_2": 7, "field_1": 9}}],
                [{"key": {"field_1": 1, "field_2": 2, "field_3": 3}, "value": {"field_2": 7, "field_1": 9}}],
                [
                    {
                        "key": {"field_1": None, "field_2": None, "field_3": None},
                        "value": {"field_2": None, "field_1": None},
                    }
                ],
                [{"key": {"field_1": None, "field_2": None, "field_3": None}, "value": None}],
                [],
            ],
        },
        schema={
            "column_1": pl.Map(
                pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
                pl.Struct({"field_1": pl.Int64, "field_2": pl.Int64}),
            ),
            "column_2": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.String,
                        "field_2": pl.Map(
                            pl.List(pl.Datetime(time_unit="us", time_zone=None)),
                            pl.List(pl.Int32),
                        ),
                    }
                )
            ),
        },
    )  # fmt: skip

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), expect
    )
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)


@pytest.mark.write_disk
@pytest.mark.xfail(
    reason="""\
[Upstream Issue]
PyIceberg writes NULL as empty lists into the Parquet file.
* Issue on Polars repo - https://github.com/pola-rs/polars/issues/23715
* Issue on PyIceberg repo - https://github.com/apache/iceberg-python/issues/2246
"""
)
def test_scan_iceberg_nulls_multiple_nesting(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=StructType(
                        NestedField(
                            field_id=next_field_id(),
                            name="field_1",
                            field_type=ListType(
                                element_id=next_field_id(),
                                element=StructType(
                                    NestedField(field_id=next_field_id(), name="key", field_type=ListType(
                                        element_id=next_field_id(),
                                        element=TimestampType(),
                                        element_required=False,
                                    ), required=True),
                                    NestedField(field_id=next_field_id(), name="value", field_type=ListType(
                                        element_id=next_field_id(),
                                        element=IntegerType(),
                                        element_required=False,
                                    ), required=False),
                                ),
                                element_required=False
                            ),
                            required=False,
                        ),
                        NestedField(field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False),
                        NestedField(field_id=next_field_id(), name="field_3", field_type=StringType(), required=False),
                    ),
                    element_required=False,
                ),
                required=False,
            ),
        ),
    )  # fmt: skip

    tbl = catalog.load_table("namespace.table")

    df_dict = {
        "column_1": [
            [
                {
                    "field_1": [
                        {"key": [datetime(2025, 1, 1), None], "value": [1, 2, None]}
                    ],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [
                {
                    "field_1": [{"key": [datetime(2025, 1, 1), None], "value": None}],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [{"field_1": None, "field_2": None, "field_3": None}],
            [None],
            None,
        ],
    }

    df = pl.DataFrame(
        df_dict,
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.List(
                            pl.Struct(
                                {
                                    "key": pl.List(pl.Datetime("us")),
                                    "value": pl.List(pl.Int32),
                                }
                            )
                        ),
                        "field_2": pl.Int32,
                        "field_3": pl.String,
                    }
                )
            ),
        },
    )

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(
                        pa.struct(
                            [
                                (
                                    "field_1",
                                    pa.large_list(
                                        pa.struct(
                                            [
                                                pa.field(
                                                    "key",
                                                    pa.large_list(pa.timestamp("us")),
                                                    nullable=False,
                                                ),
                                                ("value", pa.large_list(pa.int32())),
                                            ]
                                        )
                                    ),
                                ),
                                ("field_2", pa.int32()),
                                ("field_3", pa.string()),
                            ]
                        )
                    ),
                )
            ]
        ),
    )

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)


@pytest.mark.write_disk
def test_scan_iceberg_nulls_nested(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=IntegerType(),
                    element_required=False,
                ),
                required=False,
            ),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {
            "column_1": [
                [1, 2],
                [None],
                None,
            ],
        },
        schema={
            "column_1": pl.List(pl.Int32),
        },
    )

    df_dict = df.to_dict(as_series=False)

    assert_frame_equal(pl.DataFrame(df_dict, schema=df.schema), df)

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(pa.int32()),
                )
            ]
        ),
    )

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)


@pytest.mark.write_disk
@pytest.mark.parametrize("use_pyiceberg_filter", [True, False])
def test_scan_iceberg_parquet_prefilter_with_column_mapping(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    use_pyiceberg_filter: bool,
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=StringType(),
                required=False,
            ),
            NestedField(
                field_id=next_field_id(),
                name="column_2",
                field_type=IntegerType(),
                required=False,
            ),
            NestedField(
                field_id=next_field_id(),
                name="column_3",
                field_type=StringType(),
                required=False,
            ),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {
            "column_1": ["A", "B", "C", "D", "E", "F"],
            "column_2": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int32),
            "column_3": ["P", "Q", "R", "S", "T", "U"],
        }
    )

    df.slice(0, 3).write_iceberg(tbl, mode="append")
    df.slice(3).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.update_column("column_2", LongType())

    with tbl.update_schema() as sch:
        sch.delete_column("column_1")

    with tbl.update_schema() as sch:
        sch.rename_column("column_3", "column_1")

    with tbl.update_schema() as sch:
        sch.rename_column("column_2", "column_3")

    with tbl.update_schema() as sch:
        sch.move_first("column_1")

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect().sort("column_3"),
        pl.DataFrame(
            {
                "column_1": ["P", "Q", "R", "S", "T", "U"],
                "column_3": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
            }
        ),
    )

    expect = pl.DataFrame(
        {
            "column_1": ["T"],
            "column_3": pl.Series([5], dtype=pl.Int64),
        }
    )

    # PyIceberg 0.12.0 handles schema evolution during filter evaluation.
    q = pl.scan_iceberg(
        tbl, reader_override="native", use_pyiceberg_filter=use_pyiceberg_filter
    ).filter(pl.col("column_3") == 5)

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        cx.setenv("POLARS_FORCE_EMPTY_READER_CAPABILITIES", "0")
        capfd.readouterr()
        out = q.collect()
        capture = capfd.readouterr().err

    assert_frame_equal(out, expect)

    if use_pyiceberg_filter:
        # Skipped from pyiceberg, we don't see the file at all.
        assert "[MultiScanTaskInit]: 1 source" in capture
        return

    # First file
    assert "Source filter mask initialization via table statistics" in capture
    assert "Predicate pushdown allows skipping 1 / 2 files" in capture
    # Second file
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 1 / 1 row groups" in capture
    )
    assert (
        "[ParquetFileReader]: Pre-filtered decode enabled (1 live [1 column predicates, 0 rest], 1 non-live)"
        in capture
    )


# Note: This test also generally covers primitive type round-tripping.
@pytest.mark.parametrize(
    "exclude_uuid",
    [True, False] if parse_version(pyiceberg.__version__) >= (0, 10, 0) else [True],
)
@pytest.mark.write_disk
def test_fill_missing_fields_with_identity_partition_values(
    exclude_uuid: bool, tmp_path: Path
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    table_data = _TableDataAllTypes.new(exclude_uuid=exclude_uuid)
    iceberg_schema = table_data.iceberg_schema
    arrow_tbl = table_data.arrow_table

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            # We have this to offset the indices
            PartitionField(
                iceberg_schema.fields[0].field_id, 0, BucketTransform(32), "bucket"
            ),
            *(
                PartitionField(field.field_id, 0, IdentityTransform(), field.name)
                for field in iceberg_schema.fields[1:]
            ),
        ),
    )

    if not exclude_uuid:
        # Note: If this starts working one day we can include it in tests.
        with pytest.raises(
            pa.ArrowNotImplementedError,
            match=r"Keys of type extension<arrow\.uuid>",
        ):
            tbl.append(arrow_tbl)

        return

    tbl.append(arrow_tbl)

    expect = table_data.polars_df

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    dfiles = [*tbl.scan().plan_files()]

    assert len(dfiles) == 1

    p = dfiles[0].file.file_path.removeprefix("file://")

    # Drop every column except 'height_provider'
    pq.write_table(
        pa.Table.from_pydict(
            {"height_provider": [1]},
            schema=schema_to_pyarrow(iceberg_schema.select("height_provider")),
        ),
        p,
    )

    out = pl.DataFrame(tbl.scan().to_arrow())

    assert_frame_equal(
        out.select(pl.col(c).cast(dt) for c, dt in expect.schema.items()),
        expect,
    )

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)


@pytest.mark.write_disk
def test_fill_missing_fields_with_identity_partition_values_nested(
    tmp_path: Path,
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(next_field_id(), "height_provider", IntegerType()),
        NestedField(
            next_field_id(),
            "struct_1",
            StructType(
                NestedField(
                    next_field_id(),
                    "struct_2",
                    StructType(NestedField(2001, "field_1", LongType())),
                )
            ),
        ),
    )

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            PartitionField(2001, 0, IdentityTransform(), "field_1")
        ),
    )

    pl.DataFrame(
        {"height_provider": [0], "struct_1": [{"struct_2": {"field_1": 300}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    expect = pl.DataFrame(
        [
            pl.Series("height_provider", [0], dtype=pl.Int32),
            pl.Series(
                "struct_1",
                [{"struct_2": {"field_1": 300}}],
                dtype=pl.Struct({"struct_2": pl.Struct({"field_1": pl.Int64})}),
            ),
        ]
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    # Note: We will still match even if the partition field is renamed, since it still
    # has the same source field ID.
    with tbl.update_spec() as spu:
        spu.rename_field("field_1", "AAA")

    pl.DataFrame(
        {"height_provider": [None], "struct_1": [{"struct_2": {"field_1": 301}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    with tbl.update_spec() as spu:
        spu.remove_field("AAA")

    pl.DataFrame(
        {"height_provider": [None], "struct_1": [{"struct_2": {"field_1": 302}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    for i, data_file in enumerate(tbl.scan().plan_files()):
        p = data_file.file.file_path.removeprefix("file://")

        pq.write_table(
            pa.Table.from_pydict(
                {"height_provider": [i]},
                schema=schema_to_pyarrow(iceberg_schema.select("height_provider")),
            ),
            p,
        )

    # Deleting partitions only takes effect for newly added files.
    expect = pl.DataFrame(
        [
            pl.Series("height_provider", [0, 1, 2], dtype=pl.Int32),
            pl.Series(
                "struct_1",
                [
                    None,
                    {"struct_2": {"field_1": 301}},
                    {"struct_2": {"field_1": 300}},
                ],
                dtype=pl.Struct({"struct_2": pl.Struct({"field_1": pl.Int64})}),
            ),
        ]
    )

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").select("struct_1").collect(),
        expect.select("struct_1"),
    )


@pytest.mark.write_disk
def test_scan_iceberg_min_max_statistics_filter(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import datetime

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    test_decimal_and_fixed = parse_version(pyiceberg.__version__) >= (0, 10, 0)

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(next_field_id(), "height_provider", IntegerType()),
        NestedField(next_field_id(), "BooleanType", BooleanType()),
        NestedField(next_field_id(), "IntegerType", IntegerType()),
        NestedField(next_field_id(), "LongType", LongType()),
        NestedField(next_field_id(), "FloatType", FloatType()),
        NestedField(next_field_id(), "DoubleType", DoubleType()),
        NestedField(next_field_id(), "DateType", DateType()),
        NestedField(next_field_id(), "TimeType", TimeType()),
        NestedField(next_field_id(), "TimestampType", TimestampType()),
        NestedField(next_field_id(), "TimestamptzType", TimestamptzType()),
        NestedField(next_field_id(), "StringType", StringType()),
        NestedField(next_field_id(), "BinaryType", BinaryType()),
        *(
            [
                NestedField(next_field_id(), "DecimalType", DecimalType(18, 2)),
                NestedField(
                    next_field_id(), "DecimalTypeLargeValue", DecimalType(38, 0)
                ),
                NestedField(
                    next_field_id(), "DecimalTypeLargeNegativeValue", DecimalType(38, 0)
                ),
                NestedField(next_field_id(), "FixedType", FixedType(1)),
            ]
            if test_decimal_and_fixed
            else []
        ),
    )

    pl_schema = pl.Schema(
        {
            "height_provider": pl.Int32(),
            "BooleanType": pl.Boolean(),
            "IntegerType": pl.Int32(),
            "LongType": pl.Int64(),
            "FloatType": pl.Float32(),
            "DoubleType": pl.Float64(),
            "DateType": pl.Date(),
            "TimeType": pl.Time(),
            "TimestampType": pl.Datetime(time_unit="us", time_zone=None),
            "TimestamptzType": pl.Datetime(time_unit="us", time_zone="UTC"),
            "StringType": pl.String(),
            "BinaryType": pl.Binary(),
            "DecimalType": pl.Decimal(precision=18, scale=2),
            "DecimalTypeLargeValue": pl.Decimal(precision=38, scale=0),
            "DecimalTypeLargeNegativeValue": pl.Decimal(precision=38, scale=0),
            "FixedType": pl.Binary(),
        }
    )

    df_dict = {
        "height_provider": [1],
        "BooleanType": [True],
        "IntegerType": [1],
        "LongType": [1],
        "FloatType": [1.0],
        "DoubleType": [1.0],
        "DateType": [datetime.date(2025, 1, 1)],
        "TimeType": [datetime.time(11, 30)],
        "TimestampType": [datetime.datetime(2025, 1, 1)],
        "TimestamptzType": [datetime.datetime(2025, 1, 1)],
        "StringType": ["A"],
        "BinaryType": [b"A"],
        **(
            {
                "DecimalType": [D("1.00")],
                # This helps ensure loads are done with the correct endianness.
                "DecimalTypeLargeValue": [D("73377733337777733333377777773333333377")],
                "DecimalTypeLargeNegativeValue": [
                    D("-73377733337777733333377777773333333377")
                ],
                "FixedType": [b"A"],
            }
            if test_decimal_and_fixed
            else {}
        ),
    }

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=schema_to_pyarrow(iceberg_schema, include_field_ids=False),
    )

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            # We have this to offset the indices
            PartitionField(
                iceberg_schema.fields[0].field_id, 0, BucketTransform(32), "bucket"
            ),
            *(
                PartitionField(field.field_id, 0, IdentityTransform(), field.name)
                for field in iceberg_schema.fields[1:]
            ),
        ),
    )

    tbl.append(arrow_tbl)

    expect = pl.DataFrame(df_dict, schema=pl_schema)

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    # Begin inspecting statistics

    assert resolved_table_statistics(tbl) is None

    min_max_statistics = resolved_table_statistics(
        tbl, filter_columns=["height_provider"]
    )
    assert min_max_statistics is not None

    min_max_values = min_max_statistics.with_columns(
        pl.all().cast(pl.String)
    ).transpose(include_header=True)

    assert_frame_equal(
        min_max_values,
        pl.DataFrame(
            [
                ("len", "1"),
                ("height_provider_nc", "0"),
                ("height_provider_min", "1"),
                ("height_provider_max", "1"),
            ],
            orient="row",
            schema=min_max_values.schema,
        ),
    )

    scan_data = new_iceberg_scan_resolver(tbl)._to_dataset_scan_impl(
        filter_columns=pl_schema.names()
    )

    assert isinstance(scan_data, _NativeIcebergScanData)
    assert scan_data.statistics_loader is not None

    non_coalesced_min_max_values = (
        scan_data.statistics_loader.finish(len(scan_data.sources), {})
        .with_columns(pl.all().cast(pl.String))
        .transpose(include_header=True)
    )

    assert_frame_equal(
        non_coalesced_min_max_values,
        pl.DataFrame(
            [
                ("len", "1"),
                ("height_provider_nc", "0"),
                ("height_provider_min", "1"),
                ("height_provider_max", "1"),
                ("BooleanType_nc", "0"),
                ("BooleanType_min", "true"),
                ("BooleanType_max", "true"),
                ("IntegerType_nc", "0"),
                ("IntegerType_min", "1"),
                ("IntegerType_max", "1"),
                ("LongType_nc", "0"),
                ("LongType_min", "1"),
                ("LongType_max", "1"),
                ("FloatType_nc", "0"),
                ("FloatType_min", None),
                ("FloatType_max", None),
                ("DoubleType_nc", "0"),
                ("DoubleType_min", None),
                ("DoubleType_max", None),
                ("DateType_nc", "0"),
                ("DateType_min", "2025-01-01"),
                ("DateType_max", "2025-01-01"),
                ("TimeType_nc", "0"),
                ("TimeType_min", "11:30:00"),
                ("TimeType_max", "11:30:00"),
                ("TimestampType_nc", "0"),
                ("TimestampType_min", "2025-01-01 00:00:00.000000"),
                ("TimestampType_max", "2025-01-01 00:00:00.000000"),
                ("TimestamptzType_nc", "0"),
                ("TimestamptzType_min", "2025-01-01 00:00:00.000000+00:00"),
                ("TimestamptzType_max", "2025-01-01 00:00:00.000000+00:00"),
                ("StringType_nc", "0"),
                ("StringType_min", "A"),
                ("StringType_max", "A"),
                ("BinaryType_nc", "0"),
                ("BinaryType_min", "A"),
                ("BinaryType_max", "A"),
                ("DecimalType_nc", "0"),
                ("DecimalType_min", "1.00"),
                ("DecimalType_max", "1.00"),
                ("DecimalTypeLargeValue_nc", "0"),
                ("DecimalTypeLargeValue_min", "73377733337777733333377777773333333377"),
                ("DecimalTypeLargeValue_max", "73377733337777733333377777773333333377"),
                ("DecimalTypeLargeNegativeValue_nc", "0"),
                (
                    "DecimalTypeLargeNegativeValue_min",
                    "-73377733337777733333377777773333333377",
                ),
                (
                    "DecimalTypeLargeNegativeValue_max",
                    "-73377733337777733333377777773333333377",
                ),
                ("FixedType_nc", "0"),
                ("FixedType_min", "A"),
                ("FixedType_max", "A"),
            ],
            orient="row",
            schema=non_coalesced_min_max_values.schema,
        ),
    )

    # The PyIceberg planner's statistics before coalescing (above) are internal to it,
    # the coalesced statistics are those of the resolved scan.
    min_max_statistics = resolved_table_statistics(
        tbl, filter_columns=pl_schema.names()
    )
    assert min_max_statistics is not None

    coalesced_min_max_values = min_max_statistics.with_columns(
        pl.all().cast(pl.String)
    ).transpose(include_header=True)

    coalesced_ne_non_coalesced = pl.concat(
        [
            non_coalesced_min_max_values.select(
                pl.struct(pl.all()).alias("non_coalesced")
            ),
            coalesced_min_max_values.select(pl.struct(pl.all()).alias("coalesced")),
        ],
        how="horizontal",
    ).filter(pl.first() != pl.last())

    # Float statistics are available after coalescing from an identity partition field.
    assert_frame_equal(
        coalesced_ne_non_coalesced,
        pl.DataFrame(
            [
                (
                    {"column": "FloatType_min", "column_0": None},
                    {"column": "FloatType_min", "column_0": "1.0"},
                ),
                (
                    {"column": "FloatType_max", "column_0": None},
                    {"column": "FloatType_max", "column_0": "1.0"},
                ),
                (
                    {"column": "DoubleType_min", "column_0": None},
                    {"column": "DoubleType_min", "column_0": "1.0"},
                ),
                (
                    {"column": "DoubleType_max", "column_0": None},
                    {"column": "DoubleType_max", "column_0": "1.0"},
                ),
            ],
            orient="row",
            schema=coalesced_ne_non_coalesced.schema,
        ),
    )

    dfiles = [x.file.file_path for x in tbl.scan().plan_files()]
    assert len(dfiles) == 1

    Path(dfiles[0].removeprefix("file://")).unlink()

    expect_file_not_found_err = pytest.raises(
        OSError,
        match=(
            "The system cannot find the file specified"
            if sys.platform == "win32"
            else "No such file or directory"
        ),
    )

    with expect_file_not_found_err:
        pl.scan_iceberg(tbl, reader_override="native").collect()

    iceberg_table_filter_seen = False

    def ensure_filter_skips_file(filter_expr: pl.Expr) -> None:
        nonlocal iceberg_table_filter_seen

        with plmonkeypatch.context() as cx:
            cx.setenv("POLARS_VERBOSE", "1")
            capfd.readouterr()

            assert_frame_equal(
                pl.scan_iceberg(tbl, reader_override="native").filter(filter_expr),
                pl.LazyFrame(schema=pl_schema),
            )

            capture = capfd.readouterr().err

            if "iceberg_table_filter: Some(<redacted>)" in capture:
                assert "allows skipping 0 / 0 files" in capture
                assert (
                    "apply_scan_predicate_to_scan_ir: PredicateFileSkip { no_residual_predicate: false, original_len: 0 }"
                    in capture
                )

                # Scanning with pyiceberg can also skip the file if the predicate
                # can be converted.
                assert_frame_equal(
                    pl.scan_iceberg(tbl, reader_override="pyiceberg").filter(
                        filter_expr
                    ),
                    pl.LazyFrame(schema=pl_schema),
                )

                iceberg_table_filter_seen = True
            else:
                assert "allows skipping 1 / 1 files" in capture
                assert (
                    "apply_scan_predicate_to_scan_ir: PredicateFileSkip { no_residual_predicate: false, original_len: 1 }"
                    in capture
                )

            capfd.readouterr()

            assert_frame_equal(
                pl.scan_iceberg(tbl, reader_override="native")
                .with_row_index()
                .filter(filter_expr),
                pl.LazyFrame(schema=pl_schema).with_row_index(),
            )

            capture = capfd.readouterr().err

            assert "iceberg_table_filter: Some(<redacted>)" not in capture

    # Check different operators
    ensure_filter_skips_file(pl.col("IntegerType") > 1)
    ensure_filter_skips_file(pl.col("IntegerType") != 1)
    ensure_filter_skips_file(pl.col("IntegerType").is_in([0]))

    # Ensure `use_metadata_statistics=False` does not skip based on statistics
    with expect_file_not_found_err:
        pl.scan_iceberg(
            tbl,
            reader_override="native",
            use_metadata_statistics=False,
        ).filter(pl.col("IntegerType") > 1).collect()

    with expect_file_not_found_err:
        pickle.loads(
            pickle.dumps(
                pl.scan_iceberg(
                    tbl,
                    reader_override="native",
                    use_metadata_statistics=False,
                ).filter(pl.col("IntegerType") > 1)
            )
        ).collect()

    # Check different types
    ensure_filter_skips_file(pl.col("BooleanType") < True)
    ensure_filter_skips_file(pl.col("IntegerType") < 1)
    ensure_filter_skips_file(pl.col("LongType") < 1)
    ensure_filter_skips_file(pl.col("FloatType") < 1.0)
    ensure_filter_skips_file(pl.col("DoubleType") < 1.0)
    ensure_filter_skips_file(pl.col("DateType") < datetime.date(2025, 1, 1))
    ensure_filter_skips_file(pl.col("TimeType") < datetime.time(11, 30))
    ensure_filter_skips_file(pl.col("TimestampType") < datetime.datetime(2025, 1, 1))
    ensure_filter_skips_file(
        pl.col("TimestamptzType")
        < pl.lit(datetime.datetime(2025, 1, 1), dtype=pl.Datetime("ms", "UTC"))
    )
    ensure_filter_skips_file(pl.col("StringType") < "A")
    ensure_filter_skips_file(pl.col("BinaryType") < b"A")
    ensure_filter_skips_file(pl.col("DecimalType") < D("1.00"))
    ensure_filter_skips_file(
        pl.col("DecimalTypeLargeValue") < D("73377733337777733333377777773333333377")
    )
    ensure_filter_skips_file(
        pl.col("DecimalTypeLargeNegativeValue")
        < D("-73377733337777733333377777773333333377")
    )
    ensure_filter_skips_file(pl.col("FixedType") < b"A")

    # Check row index. It should have a null_count statistic column of 0.
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native")
        .with_row_index()
        .filter(pl.col("index").is_null()),
        pl.LazyFrame(schema={"index": pl.get_index_type(), **pl_schema}),
    )

    assert iceberg_table_filter_seen


def test_import_decimal_from_iceberg_binary_repr_sign_extend_29449() -> None:
    from polars._plr import PySeries
    from polars._utils.wrap import wrap_s

    # Iceberg stores the unscaled value as a two's-complement big-endian integer
    # using the minimum number of bytes, so the leading bytes must be sign-extended.
    assert_series_equal(
        wrap_s(
            PySeries._import_decimal_from_iceberg_binary_repr(
                bytes_list=[b"\xf0\xc9\xa7", b"\x04\xe2", b"\x00", b"\xff", None],
                precision=7,
                scale=2,
            )
        ),
        pl.Series(
            [D("-9969.53"), D("12.50"), D("0.00"), D("-0.01"), None],
            dtype=pl.Decimal(precision=7, scale=2),
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_negative_decimal_statistics_29449(tmp_path: Path) -> None:
    dtype = pl.Decimal(precision=7, scale=2)

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "profit", DecimalType(7, 2), required=False)
        ),
    )

    tbl.append(
        pa.table({"profit": pa.array([D("-9969.53"), D("12.50")], pa.decimal128(7, 2))})
    )

    expect = pl.DataFrame(
        {"profit": pl.Series([D("-9969.53"), D("12.50")], dtype=dtype)}
    )

    for reader_override in ["native", "pyiceberg"]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override).collect(),  # type: ignore[arg-type]
            expect,
        )

        # The lower bound is the 3-byte two's-complement `f0c9a7`. Decoding it as
        # unsigned used to raise a precision error here.
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override)  # type: ignore[arg-type]
            .filter(pl.col("profit") > 0)
            .collect(),
            expect.filter(pl.col("profit") > 0),
        )

    min_max_statistics = resolved_table_statistics(tbl, filter_columns=["profit"])
    assert min_max_statistics is not None
    assert_frame_equal(
        min_max_statistics.select("profit_min", "profit_max"),
        pl.DataFrame(
            {
                "profit_min": pl.Series([D("-9969.53")], dtype=dtype),
                "profit_max": pl.Series([D("12.50")], dtype=dtype),
            }
        ),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("x") == D("-5.00"),
        pl.col("x") < 0,
        pl.col("x") <= D("-5.00"),
    ],
)
def test_scan_iceberg_equal_negative_decimal_29462(
    tmp_path: Path, predicate: pl.Expr
) -> None:
    values = [D("-5.00")] * 3
    dtype = pl.Decimal(precision=5, scale=2)

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", DecimalType(5, 2), required=False)),
    )
    tbl.append(pa.table({"x": pa.array(values, pa.decimal128(5, 2))}))

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native")  # type: ignore[arg-type]
        .filter(predicate)
        .collect(),
        pl.DataFrame({"x": pl.Series(values, dtype=dtype)}),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["native", "pyiceberg"])
def test_scan_iceberg_decimal_precision_promotion(
    tmp_path: Path, reader_override: Any
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "d", DecimalType(9, 2)),
        ),
    )

    tbl.append(
        pl.DataFrame(
            {"a": [1], "d": pl.Series([D("1.00")], dtype=pl.Decimal(9, 2))}
        ).to_arrow()
    )

    with tbl.update_schema() as u:
        u.update_column("d", DecimalType(10, 2))

    tbl = tbl.catalog.load_table(tbl.name())

    tbl.append(
        pl.DataFrame(
            {"a": [2], "d": pl.Series([D("12345678.90")], dtype=pl.Decimal(10, 2))}
        ).to_arrow()
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override=reader_override).collect(),
        pl.DataFrame(
            {
                "a": [1, 2],
                "d": pl.Series([D("1.00"), D("12345678.90")], dtype=pl.Decimal(10, 2)),
            }
        ),
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_scan_iceberg_categorical_24140(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(
            next_field_id(),
            "values",
            StringType(),
        ),
    )

    tbl = catalog.create_table("namespace.table", iceberg_schema)

    df = pl.DataFrame(
        {"values": "A"},
        schema={"values": pl.Categorical()},
    )

    arrow_tbl = df.to_arrow()

    arrow_type = arrow_tbl.schema.field("values").type
    assert arrow_type.index_type == pa.uint32()
    assert arrow_type.value_type == pa.large_string()

    tbl.append(arrow_tbl)

    expect = pl.DataFrame({"values": "A"}, schema={"values": pl.String})

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["native", "pyiceberg"])
def test_scan_iceberg_fast_count(tmp_path: Path, reader_override: Any) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(NestedField(1, "a", LongType())),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame({"a": [0, 1, 2, 3, 4]}).write_iceberg(tbl, mode="append")

    assert (
        pl.scan_iceberg(
            tbl, reader_override=reader_override, use_metadata_statistics=True
        )
        .select(pl.len())
        .collect()
        .item()
        == 5
    )

    assert (
        pl.scan_iceberg(
            tbl, reader_override=reader_override, use_metadata_statistics=True
        )
        .filter(pl.col("a") <= 2)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )
    assert (
        pl.scan_iceberg(
            tbl, reader_override=reader_override, use_metadata_statistics=True
        )
        .head(3)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    assert (
        pl.scan_iceberg(
            tbl, reader_override=reader_override, use_metadata_statistics=True
        )
        .slice(1, 3)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    dfiles = [*tbl.scan().plan_files()]

    assert len(dfiles) == 1

    p = dfiles[0].file.file_path.removeprefix("file://")

    # Overwrite the data file with one that has a different number of rows
    pq.write_table(
        pa.Table.from_pydict(
            {"a": [0, 1, 2]},
            schema=schema_to_pyarrow(tbl.schema()),
        ),
        p,
    )

    if reader_override == "pyiceberg":
        return

    # `use_metadata_statistics=False` should disable sourcing the row count from
    # Iceberg metadata.

    assert (
        pl.scan_iceberg(
            tbl,
            reader_override=reader_override,
            use_metadata_statistics=False,
        )
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    assert (
        pickle.loads(
            pickle.dumps(
                pl.scan_iceberg(
                    tbl, reader_override=reader_override, use_metadata_statistics=False
                ).select(pl.len())
            )
        )
        .collect()
        .item()
        == 3
    )

    Path(p).unlink()

    with pytest.raises(
        OSError,
        match=(
            "The system cannot find the file specified"
            if sys.platform == "win32"
            else "No such file or directory"
        ),
    ):
        pl.scan_iceberg(tbl, reader_override=reader_override).collect()

    # `select(len())` should be able to return the result from the Iceberg metadata
    # without looking at the underlying data files.
    assert (
        pl.scan_iceberg(tbl, reader_override=reader_override)
        .select(pl.len())
        .collect()
        .item()
        == 5
    )


@pytest.mark.write_disk
def test_scan_iceberg_passes_source_sizes(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
    )
    expected = pl.DataFrame({"a": [1, 2, 3]})
    tbl.append(expected.to_arrow())

    lf, scan_args = resolve_scan(tbl)
    assert scan_args["source_sizes"] == [
        task.file.file_size_in_bytes for task in tbl.scan().plan_files()
    ]
    assert_frame_equal(lf.collect(), expected)


def test_scan_iceberg_idxsize_limit() -> None:
    if isinstance(pl.get_index_type(), pl.UInt64):
        assert (
            pl.scan_parquet([b""], schema={}, _row_count=(1 << 32, 0))
            .select(pl.len())
            .collect()
            .item()
            == 1 << 32
        )

        return

    f = io.BytesIO()

    pl.DataFrame({"x": 1}).write_parquet(f)

    q = pl.scan_parquet([f.getvalue()], schema={"x": pl.Int64}, _row_count=(1 << 32, 0))

    assert_frame_equal(q.collect(), pl.DataFrame({"x": 1}))

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"row count \(4294967296\) exceeded maximum supported of 4294967295.*Consider installing 'polars\[rt64\]'.",
    ):
        q.select(pl.len()).collect()


@pytest.mark.write_disk
def test_iceberg_filter_bool_26474(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "test", uri="sqlite:///:memory:", warehouse=format_file_uri_iceberg(tmp_path)
    )
    catalog.create_namespace("default")
    tbl = catalog.create_table(
        "default.test",
        IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "foo", BooleanType()),
        ),
    )

    schema = {"id": pl.Int64, "foo": pl.Boolean}

    dfs = [
        pl.DataFrame({"id": [1], "foo": [True]}, schema=schema),
        pl.DataFrame({"id": [2], "foo": [False]}, schema=schema),
        pl.DataFrame({"id": [3], "foo": [None]}, schema=schema),
    ]

    for df in dfs:
        df.write_iceberg(tbl, mode="append")

    assert sum(1 for _ in tbl.scan().plan_files()) == 3

    dfs_concat = pl.concat(dfs)

    for predicate in [
        pl.col("foo"),
        ~pl.col("foo"),
        pl.col("foo") & pl.col("foo"),
        pl.col("foo") | pl.col("foo"),
        pl.col("foo") ^ pl.col("foo"),
        pl.col("foo") & ~pl.col("foo"),
        pl.col("foo") | ~pl.col("foo"),
        pl.col("foo") ^ pl.col("foo"),
        pl.col("foo") & pl.col("foo") | pl.col("foo"),
        pl.col("foo") | pl.col("foo") & pl.col("foo"),
        pl.col("foo") == True,  # noqa: E712
        pl.col("foo") == False,  # noqa: E712
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            dfs_concat.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["native", "pyiceberg"])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize(
    ("predicate", "expected_filter_factory"),
    [
        (pl.col("value") == float("nan"), lambda: IsNaN("value")),
        (pl.col("value") != float("nan"), lambda: NotNaN("value")),
        (
            pl.col("value").eq_missing(float("nan")),
            lambda: And(IsNaN("value"), Not(IsNull("value"))),
        ),
        (
            pl.col("value").ne_missing(float("nan")),
            lambda: Or(NotNaN("value"), IsNull("value")),
        ),
        (pl.col("value") > float("nan"), lambda: None),
    ],
)
def test_scan_iceberg_nan_comparisons(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    reader_override: Literal["native", "pyiceberg"],
    dtype: pl.DataType,
    predicate: pl.Expr,
    expected_filter_factory: Callable[[], Any | None],
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(
                2,
                "value",
                FloatType() if dtype == pl.Float32 else DoubleType(),
                required=False,
            ),
        ),
    )
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "value": pl.Series([None, float("nan"), 1.0, 2.0], dtype=dtype),
        }
    )
    df.write_iceberg(tbl, mode="append")

    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")
    capfd.readouterr()
    actual = (
        pl.scan_iceberg(tbl, reader_override=reader_override)
        .filter(predicate)
        .collect()
    )
    capture = capfd.readouterr().err

    assert_frame_equal(actual, df.filter(predicate), check_row_order=False)
    assert f"iceberg_table_filter = {expected_filter_factory()!r}" in capture


@pytest.mark.write_disk
def test_scan_iceberg_partial_and_pushdown(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")
    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", DoubleType()),
        ),
    )
    tbl = catalog.load_table("namespace.table")
    pl.DataFrame(
        {"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]},
    ).write_iceberg(tbl, mode="append")

    # a > 1 is convertible; cast is not
    q = pl.scan_iceberg(tbl).filter(
        (pl.col("a") > 1) & (pl.col("b").cast(pl.Int64) > 0)
    )

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    # Verify: partial predicate was pushed
    assert "pyarrow_predicate = " in capture
    assert "pa.compute.field('a') > 1" in capture
    # Verify: correctness
    assert len(result) == 2
    assert result["a"].to_list() == [2, 3]

    # Arithmetic lowers to a pyarrow expression but has no PyIceberg equivalent,
    # so only the other conjunct makes it into the table filter.
    q = pl.scan_iceberg(tbl).filter((pl.col("a") > 1) & (pl.col("b") * 2 > 40.0))

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    assert "pa.compute.field('b') *" in capture
    assert result["a"].to_list() == [3]


@pytest.mark.write_disk
def test_scan_iceberg_is_in_pushdown(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")
    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", StringType()),
        ),
    )
    tbl = catalog.load_table("namespace.table")
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_iceberg(
        tbl, mode="append"
    )

    q = pl.scan_iceberg(tbl).filter(pl.col("b").is_in(["x", "z"]))

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    # Verify: `is_in` is lowered into the pyarrow predicate
    assert 'isin(["x","z"])' in capture
    # Verify: correctness
    assert result["a"].to_list() == [1, 3]


@pytest.mark.write_disk
def test_scan_iceberg_is_in_pushdown_thousands_separator(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")
    catalog.create_table(
        "namespace.table", IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl = catalog.load_table("namespace.table")
    for v in [1000, 2000, 5]:
        pl.DataFrame({"a": [v]}).write_iceberg(tbl, mode="append")

    # The pushed-down list must not be rendered as `[1,000,2,000]`.
    with pl.Config(thousands_separator=","):
        result = pl.scan_iceberg(tbl).filter(pl.col("a").is_in([1000, 2000])).collect()

    assert sorted(result["a"].to_list()) == [1000, 2000]


@pytest.mark.write_disk
def test_scan_iceberg_row_estimate(
    tmp_path: Path,
    write_position_deletes: WritePositionDeletes,  # noqa: F811
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(NestedField(1, "a", LongType())),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame({"a": [0, 1, 2, 3, 4]}).write_iceberg(tbl, mode="append")

    q = pl.scan_iceberg(tbl, use_metadata_statistics=True)
    plan = q.explain()

    assert "ESTIMATED ROWS: 5" in plan

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]

    deletion_files = (  # type: ignore[var-annotated]
        "iceberg",
        (
            {
                0: [
                    write_position_deletes(pl.Series([1, 2])),
                ],
            },
            {},
        ),
    )

    q = pl.scan_parquet(
        file_paths,
        _deletion_files=deletion_files,  # type: ignore[arg-type]
        _row_count=(5, 2),
    )
    plan = q.explain()

    assert "ESTIMATED ROWS: 3" in plan
    assert q.select(pl.len()).collect().item() == 3
    assert q.collect().height == 3


def test_convert_iceberg_storage_options_dot_filtering() -> None:
    from polars.io.iceberg._dataset import (
        _convert_iceberg_to_object_store_storage_options,
    )

    result = _convert_iceberg_to_object_store_storage_options(
        {
            "user": "foo",
            "hdfs.host": "localhost",
            "hdfs.port": "9005",
            # "dfs.replication": "3",
            # "write.parquet.compression": "zstd",
            "other.foo": "bar",
        }
    )

    # keep
    assert result["user"] == "foo"
    assert result["hdfs.host"] == "localhost"
    assert result["hdfs.port"] == "9005"

    # drop
    assert "dfs.replication" not in result
    assert "write.parquet.compression" not in result
    assert "other.foo" not in result


@pytest.mark.write_disk
def test_scan_iceberg_v3_field_initial_default(tmp_path: Path) -> None:
    table, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "height_provider", IntegerType()))
    )

    pl.LazyFrame(
        {"height_provider": 1},
        schema={"height_provider": pl.Int32},
    ).sink_iceberg(table, mode="append")

    md_path = Path(
        table.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())

    md_object["format-version"] = 3
    md_object["schemas"][-1]["fields"] += [
{ "id": 102, "name": "BooleanType", "required": False, "type": "boolean", "initial-default": True },
{ "id": 103, "name": "IntegerType", "required": False, "type": "int", "initial-default": 1 },
{ "id": 104, "name": "LongType", "required": False, "type": "long", "initial-default": 1 },
{ "id": 105, "name": "FloatType", "required": False, "type": "float", "initial-default": 1.0 },
{ "id": 106, "name": "DoubleType", "required": False, "type": "double", "initial-default": 1.0 },
{ "id": 107, "name": "DateType", "required": False, "type": "date", "initial-default": "2025-01-01" },
{ "id": 108, "name": "TimeType", "required": False, "type": "time", "initial-default": "11:30:00" },
{ "id": 109, "name": "TimestampType", "required": False, "type": "timestamp", "initial-default": "2025-01-01T00:00:00" },
{ "id": 110, "name": "TimestamptzType", "required": False, "type": "timestamptz", "initial-default": "2025-01-01T00:00:00+00:00" },
{ "id": 111, "name": "StringType", "required": False, "type": "string", "initial-default": "A" },
{ "id": 112, "name": "BinaryType", "required": False, "type": "binary", "initial-default": "41" },
{ "id": 113, "name": "DecimalType", "required": False, "type": "decimal(18, 2)", "initial-default": "1.00" },
{ "id": 114, "name": "FixedType", "required": False, "type": "fixed[1]", "initial-default": "41" },
{ "id": 115, "name": "UUIDType", "required": False, "type": "uuid", "initial-default": "30303030-3131-3131-3030-303031313131" }
]  # fmt: skip

    md_path.write_text(json.dumps(md_object))
    table = catalog.load_table(table.name())
    expect = _TableDataAllTypes.new().polars_df

    assert_frame_equal(
        pl.scan_iceberg(table).collect(),
        expect,
    )


def test_scan_iceberg_self_join_28465(tmp_path: Path) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "x", LongType()))
    )

    pl.DataFrame({"x": 1}).write_iceberg(table, mode="append")

    lf = pl.scan_iceberg(table)
    q = lf.join(lf, on="x")

    assert_frame_equal(q.collect(), pl.DataFrame({"x": 1}))


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize("query_kind", ["projection", "predicate"])
def test_scan_iceberg_branch_caches_29345(
    tmp_path: Path, engine: EngineType, query_kind: str
) -> None:
    table, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "http_method", StringType()),
            NestedField(3, "endpoint", StringType()),
        ),
    )
    pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "http_method": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "endpoint": ["/a", None, "/c", None, "/e"],
        }
    ).write_iceberg(table, mode="append")

    lf = pl.scan_iceberg(table, reader_override="pyiceberg")
    if query_kind == "projection":
        left = lf.filter(pl.col("id") > 0)
        right = lf.filter(pl.col("endpoint").is_not_null()).select("id").limit(2)
        q = left.join(right, on="id", how="semi").select("id", "http_method")
        expected = pl.DataFrame({"id": [1, 3], "http_method": ["GET", "PUT"]})
    else:
        low = lf.filter(pl.col("id") <= 2).select("id")
        high = lf.filter(pl.col("id") >= 4).select("id")
        q = pl.concat([low, high])
        expected = pl.DataFrame({"id": [1, 2, 4, 5]})

    for _ in range(2):
        assert_frame_equal(q.collect(engine=engine), expected, check_row_order=False)


def test_scan_iceberg_schema_change_24498(
    tmp_path: Path,
) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )

    df1 = pl.DataFrame([{"a": 1}])
    df1.write_iceberg(table, mode="append")

    df2 = df1.with_columns(b=pl.col("a") * 2)

    with table.update_schema() as update_schema:
        update_schema.union_by_name(df2.to_arrow().schema)

    df2.write_iceberg(table, mode="overwrite")

    assert pl.scan_iceberg(
        table,
        snapshot_id=table.snapshots()[0].snapshot_id,
    ).collect_schema() == {"a": pl.Int64}

    assert pl.scan_iceberg(
        table,
        snapshot_id=table.snapshots()[1].snapshot_id,
    ).collect_schema() == {"a": pl.Int64, "b": pl.Int64}

    assert pl.scan_iceberg(
        table,
        snapshot_id=table.snapshots()[2].snapshot_id,
    ).collect_schema() == {"a": pl.Int64, "b": pl.Int64}


@pytest.mark.write_disk
def test_scan_iceberg_renamed_column_with_pruned_metadata(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Pruning must preserve renamed columns, which Iceberg maps by field ID.
    from polars._plr import PyLazyFrame
    from polars._utils.wrap import wrap_ldf

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")
    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "old", IntegerType()),
            NestedField(2, "other", IntegerType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")
    pl.DataFrame(
        {"old": [1, 2, 3], "other": [4, 5, 6]},
        schema={"old": pl.Int32, "other": pl.Int32},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.rename_column("old", "new")

    plmonkeypatch.setenv("POLARS_PRUNE_PARQUET_METADATA", "1")

    lf = wrap_ldf(
        PyLazyFrame.new_from_dataset_object(
            new_iceberg_scan_resolver(tbl), resolve_heavy_sources=4
        )
    )

    assert lf.select("new").collect().to_series().to_list() == [1, 2, 3]


def _count_opens(
    monkeypatch: pytest.MonkeyPatch, tbl: pyiceberg.table.Table, suffix: str = ".avro"
) -> list[str]:
    from polars.io.iceberg._cache import CachingFileIO

    assert not isinstance(tbl.io, CachingFileIO)

    file_io_type = type(tbl.io)
    original_new_input = file_io_type.new_input
    opened: list[str] = []

    def new_input(self: Any, location: str) -> Any:
        if location.endswith(suffix):
            opened.append(location)
        return original_new_input(self, location)

    monkeypatch.setattr(file_io_type, "new_input", new_input)
    return opened


@pytest.mark.write_disk
def test_scan_iceberg_metadata_file_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.iceberg._cache import CachingFileIO, reset_metadata_file_cache

    reset_metadata_file_cache()

    try:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
        )
        for i in range(3):
            pl.DataFrame({"a": [i]}).write_iceberg(tbl, mode="append")

        opened = _count_opens(monkeypatch, tbl)
        expected = pl.DataFrame({"a": [0, 1, 2]})

        assert_frame_equal(
            pl.scan_iceberg(tbl).collect(), expected, check_row_order=False
        )
        first_scan = set(opened)
        assert first_scan
        assert not isinstance(tbl.io, CachingFileIO)

        opened.clear()
        assert_frame_equal(
            pl.scan_iceberg(tbl).collect(), expected, check_row_order=False
        )
        assert opened == []

        # After a commit, files read by earlier scans are not fetched again.
        pl.DataFrame({"a": [3]}).write_iceberg(tbl, mode="append")
        opened.clear()
        assert_frame_equal(
            pl.scan_iceberg(tbl).collect(),
            pl.DataFrame({"a": [0, 1, 2, 3]}),
            check_row_order=False,
        )
        assert first_scan.isdisjoint(opened)
    finally:
        reset_metadata_file_cache()


@pytest.mark.write_disk
def test_scan_iceberg_metadata_file_cache_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from polars.io.iceberg._cache import reset_metadata_file_cache

    plmonkeypatch.setenv("POLARS_ICEBERG_METADATA_FILE_CACHE_MB", "0")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    reset_metadata_file_cache()

    try:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
        )
        pl.DataFrame({"a": [1]}).write_iceberg(tbl, mode="append")

        opened = _count_opens(monkeypatch, tbl)

        assert pl.scan_iceberg(tbl).collect().item() == 1
        first_scan = len(opened)
        assert first_scan > 0

        opened.clear()
        assert pl.scan_iceberg(tbl).collect().item() == 1
        assert len(opened) == first_scan
        assert "metadata file cache: bypassed: disabled" in capfd.readouterr().err
    finally:
        reset_metadata_file_cache()


def test_iceberg_metadata_file_cache_eviction() -> None:
    from polars.io.iceberg._cache import CacheStats, IcebergMetadataFileCache

    cache = IcebergMetadataFileCache(max_bytes=10)
    stats = CacheStats()

    cache.put("a", b"123456", stats)
    cache.put("b", b"123456", stats)
    assert cache.get("a") is None
    assert cache.get("b") == b"123456"
    assert stats.evicted == 1

    # Entries larger than the budget are not stored.
    cache.put("c", b"12345678901", stats)
    assert cache.get("c") is None
    assert stats.too_large == 1

    fetches: list[str] = []

    def fetch() -> bytes:
        fetches.append("fetch")
        return b"1234"

    assert cache.get_or_fetch("d", fetch, stats) == b"1234"
    assert cache.get_or_fetch("d", fetch, stats) == b"1234"
    assert len(fetches) == 1
    assert (stats.hits, stats.misses) == (1, 1)
    # The miss on `d` evicted `b`.
    assert stats.evicted == 2

    def fetch_fail() -> bytes:
        raise OSError

    # A failed fetch is not cached.
    with pytest.raises(OSError):
        cache.get_or_fetch("e", fetch_fail, stats)
    assert cache.get_or_fetch("e", fetch, stats) == b"1234"
    assert len(fetches) == 2


_UUID = "0b9d4f5e-1c2a-4b3d-8e7f-6a5b4c3d2e1f"


@pytest.mark.parametrize(
    ("location", "cacheable"),
    [
        (f"s3://bucket/tbl/metadata/snap-1-1-{_UUID}.avro", True),
        (f"s3://bucket/{_UUID}/metadata/snap-1-1-2.avro", False),
        (f"s3://bucket/tbl/metadata/00001-{_UUID}.metadata.json", True),
        (f"s3://bucket/tbl/metadata/00001-{_UUID}.gz.metadata.json", True),
        (f"s3://bucket/{_UUID}/metadata/v1.metadata.json", False),
        (f"C:\\warehouse\\tbl\\metadata\\snap-1-1-{_UUID}.avro", True),
        (f"C:\\warehouse\\{_UUID}\\metadata\\v1.metadata.json", False),
        (f"tbl/metadata/00001-{_UUID}.metadata.json", False),
        (f"file:tbl/metadata/snap-1-1-{_UUID}.avro", False),
    ],
)
def test_iceberg_metadata_file_cacheable(location: str, cacheable: bool) -> None:
    from polars.io.iceberg._cache import _is_cacheable_metadata_file

    assert _is_cacheable_metadata_file(location) is cacheable


@pytest.mark.write_disk
def test_iceberg_load_static_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO
    from pyiceberg.table import StaticTable

    from polars.io.iceberg._cache import load_static_table, reset_metadata_file_cache

    reset_metadata_file_cache()

    try:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
        )
        location = tbl.metadata_location
        properties = {"s3.region": "eu-central-1"}

        # Matches `StaticTable.from_metadata`.
        expected = StaticTable.from_metadata(location, properties=properties)
        first, stats = load_static_table(location, properties)
        assert type(first) is type(expected)
        assert first.name() == expected.name()
        assert first.metadata_location == expected.metadata_location
        assert first.metadata == expected.metadata
        assert type(first.io) is type(expected.io)
        assert first.io.properties == expected.io.properties
        assert type(first.catalog) is type(expected.catalog)
        assert (stats.hits, stats.misses) == (0, 1)

        # Later loads read the cached file and build their own table and FileIO.
        second, stats = load_static_table(location, properties)
        assert (stats.hits, stats.misses) == (1, 0)
        assert second is not first
        assert second.io is not first.io
        assert second.metadata == first.metadata

        # Relative paths resolve against the working directory.
        monkeypatch.chdir(tmp_path)
        relative = os.path.relpath(location.removeprefix("file://"))
        _, stats = load_static_table(relative, properties)
        assert stats.bypass == "relative path"

        # A FileIO that writes to its properties, as adlfs does, changes neither
        # the caller's properties nor the cache scope.
        original_new_input = PyArrowFileIO.new_input

        def new_input(self: Any, location: str) -> Any:
            self.properties.setdefault("adls.account-name", "onelake")
            return original_new_input(self, location)

        monkeypatch.setattr(PyArrowFileIO, "new_input", new_input)
        reset_metadata_file_cache()
        load_static_table(location, properties)
        _, stats = load_static_table(location, properties)
        assert (stats.hits, stats.misses) == (1, 0)
        assert properties == {"s3.region": "eu-central-1"}
    finally:
        reset_metadata_file_cache()


def test_iceberg_metadata_file_cache_invalid_size(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    from polars.io.iceberg._cache import (
        get_metadata_file_cache,
        reset_metadata_file_cache,
    )

    try:
        for value in ("256MiB", "-1"):
            plmonkeypatch.setenv("POLARS_ICEBERG_METADATA_FILE_CACHE_MB", value)
            reset_metadata_file_cache()
            with pytest.raises(
                ValueError, match="POLARS_ICEBERG_METADATA_FILE_CACHE_MB"
            ):
                get_metadata_file_cache()

        plmonkeypatch.setenv("POLARS_ICEBERG_METADATA_FILE_CACHE_MB", "5")
        reset_metadata_file_cache()
        assert get_metadata_file_cache().max_bytes == 5 * 2**20
    finally:
        reset_metadata_file_cache()


@pytest.mark.write_disk
def test_iceberg_caching_file_io_scope(tmp_path: Path) -> None:
    from pyiceberg.catalog.rest.auth import NoopAuthManager
    from pyiceberg.io.pyarrow import PyArrowFileIO

    from polars.io.iceberg._cache import CachingFileIO, IcebergMetadataFileCache

    cache = IcebergMetadataFileCache(max_bytes=1024)

    manifest = tmp_path / f"{uuid.uuid4()}-m0.avro"
    manifest.write_bytes(b"data")
    no_uuid = tmp_path / "manifest.avro"
    no_uuid.write_bytes(b"data")

    # (hits, misses) of one read.
    miss, hit, bypass = (0, 1), (1, 0), (0, 0)

    def read(
        properties: dict[str, Any], path: Path = manifest, file_io_cls: Any = None
    ) -> tuple[int, int]:
        file_io = CachingFileIO((file_io_cls or PyArrowFileIO)(properties), cache)
        with file_io.new_input(format_file_uri_iceberg(path)).open() as f:
            assert f.read() == b"data"
        return file_io.stats.hits, file_io.stats.misses

    assert read({"s3.access-key-id": "a"}) == miss

    # Table metadata pointers do not change the scope.
    pointers = {"metadata_location": "v2", "previous_metadata_location": "v1"}
    assert read({"s3.access-key-id": "a", **pointers}) == hit

    # Entries are not shared across credentials.
    assert read({"s3.access-key-id": "b"}) == miss

    # Values that are not plain bypass the cache.
    class Redacted(str):
        def __repr__(self) -> str:
            return "'***'"

    class CustomAuthManager(NoopAuthManager):  # type: ignore[misc]
        pass

    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    for opaque in (
        {"s3.secret-access-key": Redacted("x")},
        {"auth": cyclic},
        {"auth": {"type": "basic", "basic": {"password": Redacted("x")}}},
        {"auth": {1: "x"}},
        {"auth": [[0] * 200] * 200},
        {"auth.manager": CustomAuthManager()},
    ):
        assert read({"s3.access-key-id": "a", **opaque}) == bypass

    # Nested settings are scoped by content.
    basic_a = {"auth": {"type": "basic", "basic": {"username": "a", "password": "x"}}}
    basic_b = {"auth": {"type": "basic", "basic": {"username": "b", "password": "y"}}}
    assert read({"s3.access-key-id": "a", **basic_a}) == miss
    assert read({"s3.access-key-id": "a", **basic_a}) == hit
    assert read({"s3.access-key-id": "a", **basic_b}) == miss

    # PyIceberg's built-in REST auth managers do not change the scope.
    for m in (NoopAuthManager(), NoopAuthManager()):
        assert read({"s3.access-key-id": "a", "auth.manager": m}) == hit

    # FileIO classes other than PyIceberg's built-in ones bypass the cache.
    class CustomFileIO(PyArrowFileIO):  # type: ignore[misc]
        pass

    assert read({"s3.access-key-id": "a"}, file_io_cls=CustomFileIO) == bypass

    # File names without a UUID are not cached.
    assert read({"s3.access-key-id": "a"}, no_uuid) == bypass
    assert len(cache) == 4


@pytest.mark.write_disk
def test_iceberg_caching_file_io_copy_pickle(tmp_path: Path) -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    from polars.io.iceberg._cache import CachingFileIO, IcebergMetadataFileCache

    manifest = tmp_path / f"{uuid.uuid4()}-m0.avro"
    manifest.write_bytes(b"data")

    file_io = CachingFileIO(PyArrowFileIO(), IcebergMetadataFileCache(1024))

    for copied in (
        copy.copy(file_io),
        copy.deepcopy(file_io),
        pickle.loads(pickle.dumps(file_io)),
    ):
        input_file = copied.new_input(format_file_uri_iceberg(manifest))
        with input_file.open(False) as f:
            assert f.read() == b"data"


def test_iceberg_metadata_file_cache_wraps_once() -> None:
    from types import SimpleNamespace

    from pyiceberg.io.pyarrow import PyArrowFileIO

    from polars.io.iceberg._cache import CachingFileIO, with_metadata_file_cache

    scan = SimpleNamespace(io=PyArrowFileIO())
    with_metadata_file_cache(scan)
    wrapped = scan.io
    assert isinstance(wrapped, CachingFileIO)

    with_metadata_file_cache(scan)
    assert scan.io is wrapped


@pytest.mark.write_disk
def test_scan_iceberg_metadata_file_cache_incremental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.iceberg._cache import reset_metadata_file_cache

    reset_metadata_file_cache()

    try:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
        )
        for i in range(3):
            pl.DataFrame({"a": [i]}).write_iceberg(tbl, mode="append")

        snapshots = tbl.snapshots()
        opened = _count_opens(monkeypatch, tbl)
        expected = pl.DataFrame({"a": [1, 2]})

        def scan() -> pl.DataFrame:
            return pl.scan_iceberg(
                tbl,
                from_snapshot_id_exclusive=snapshots[0].snapshot_id,
                to_snapshot_id_inclusive=snapshots[-1].snapshot_id,
            ).collect()

        assert_frame_equal(scan(), expected, check_row_order=False)
        assert opened

        opened.clear()
        assert_frame_equal(scan(), expected, check_row_order=False)
        assert opened == []
    finally:
        reset_metadata_file_cache()


@pytest.mark.write_disk
def test_scan_iceberg_reuses_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyiceberg.catalog.sql import SqlCatalog

    import polars.io.iceberg._dataset as dataset_module

    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    pl.DataFrame({"a": [1, 2]}).write_iceberg(tbl, mode="append")
    name = ".".join(tbl.name())
    expected = pl.DataFrame({"a": [1, 2]})

    constructed = 0
    original_init = SqlCatalog.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal constructed
        constructed += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SqlCatalog, "__init__", init)

    def scan_twice(catalog: Any) -> None:
        lfs = [pl.scan_iceberg(name, catalog=catalog) for _ in range(2)]
        assert_frame_equal(
            pl.concat(lfs).collect(),
            pl.concat([expected, expected]),
            check_row_order=False,
        )

    # The passed catalog is used for table loads, also by several scans in one query.
    scan_twice(catalog)
    assert constructed == 0

    # The catalog is not pickled; the config is the fallback.
    lf = pickle.loads(pickle.dumps(pl.scan_iceberg(name, catalog=catalog)))
    assert_frame_equal(lf.collect(), expected, check_row_order=False)
    assert constructed == 1

    # Subclasses are used directly as well.
    class SubCatalog(SqlCatalog):  # type: ignore[misc]
        pass

    sub = SubCatalog(catalog.name, **catalog.properties)
    constructed = 0
    scan_twice(sub)
    assert constructed == 0

    # Other catalog classes get a new catalog per scan.
    monkeypatch.setattr(dataset_module, "_REUSABLE_CATALOG_CLASSES", frozenset())
    scan_twice(catalog)
    assert constructed == 2


@pytest.mark.write_disk
def test_scan_iceberg_catalog_descriptor_without_instance(tmp_path: Path) -> None:
    from polars.io.iceberg._dataset import (
        IcebergCatalogConfig,
        IcebergCatalogTableDescriptor,
    )

    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    pl.DataFrame({"a": [1]}).write_iceberg(tbl, mode="append")

    descriptor = IcebergCatalogTableDescriptor(
        table_identifier=".".join(tbl.name()),
        catalog_config=IcebergCatalogConfig.from_catalog(catalog),
    )
    # Descriptors pickled by earlier versions have no `catalog_`.
    del descriptor.catalog_

    wrap = IcebergTableWrap(
        table_=NoPickleOption(),
        table_descriptor_=descriptor,
        serializer=IcebergScanTableSerializer(),
        iceberg_storage_properties=None,
    )
    assert (
        wrap.get().metadata_location == catalog.load_table(tbl.name()).metadata_location
    )


@pytest.mark.write_disk
def test_scan_iceberg_row_index_filter_without_statistics(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl.append(pl.DataFrame({"a": [1, 2, 3]}).to_arrow())

    assert_frame_equal(
        pl.scan_iceberg(tbl, use_metadata_statistics=False)
        .with_row_index()
        .filter(pl.col("index") > 0)
        .collect(),
        pl.DataFrame(
            {"index": pl.Series([1, 2], dtype=pl.get_index_type()), "a": [2, 3]}
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_metadata_path_file_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from polars.io.iceberg._cache import reset_metadata_file_cache

    reset_metadata_file_cache()
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    def metadata_file_reads() -> list[str]:
        # The metadata file cache read of each scan, up to the hit and miss counts.
        prefix = "IcebergTableWrap: metadata file cache: "
        return [
            ", ".join(line.removeprefix(prefix).split(", ")[:2])
            for line in capfd.readouterr().err.splitlines()
            if line.startswith(prefix)
        ]

    try:
        tbl, _ = new_iceberg_table(
            tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
        )
        pl.DataFrame({"a": [1]}).write_iceberg(tbl, mode="append")
        opened = _count_opens(monkeypatch, tbl, ".metadata.json")
        capfd.readouterr()

        for _ in range(2):
            assert pl.scan_iceberg(tbl.metadata_location).collect().item() == 1
        assert len(opened) == 1
        assert metadata_file_reads() == [
            "hits: 0, misses: 1",
            "hits: 1, misses: 0",
        ]

        metadata = Path(tbl.metadata_location.removeprefix("file://")).read_bytes()

        # Compressed metadata files are cached as well.
        gz = tmp_path / f"00002-{uuid.uuid4()}.gz.metadata.json"
        gz.write_bytes(gzip.compress(metadata))
        opened.clear()
        for _ in range(2):
            assert pl.scan_iceberg(format_file_uri_iceberg(gz)).collect().item() == 1
        assert len(opened) == 1
        assert metadata_file_reads() == [
            "hits: 0, misses: 1",
            "hits: 1, misses: 0",
        ]

        # Metadata file names without a write-time UUID are read on every scan.
        copied = tmp_path / "v1.metadata.json"
        copied.write_bytes(metadata)
        opened.clear()
        for _ in range(2):
            path = format_file_uri_iceberg(copied)
            assert pl.scan_iceberg(path).collect().item() == 1
        assert len(opened) == 2
        assert metadata_file_reads() == ["bypassed: no UUID in file name"] * 2

        # So is every metadata file when the cache is disabled.
        plmonkeypatch.setenv("POLARS_ICEBERG_METADATA_FILE_CACHE_MB", "0")
        reset_metadata_file_cache()
        opened.clear()
        for _ in range(2):
            assert pl.scan_iceberg(tbl.metadata_location).collect().item() == 1
        assert len(opened) == 2
        assert metadata_file_reads() == ["bypassed: disabled"] * 2
    finally:
        reset_metadata_file_cache()


def join_structure(lf: pl.LazyFrame) -> list[str]:
    return [
        line.strip()
        for line in lf.explain().splitlines()
        if "JOIN" in line or "ON:" in line or "SCAN" in line
    ]


def small_build_side(*keys: int, key: str) -> pl.LazyFrame:
    # Only a filtered build side publishes a runtime filter.
    lf = pl.LazyFrame({key: list(keys), "e": list(range(len(keys)))})
    return lf.filter(pl.col("e") >= 0)


def write_iceberg_tables(
    tmp_path: Path, frames: dict[str, pl.DataFrame]
) -> tuple[dict[str, pyiceberg.table.Table], dict[str, list[str]]]:
    """Write each frame to its own table; return the tables and their data files."""
    iceberg_types = {pl.Int64: LongType(), pl.Boolean: BooleanType()}
    tables = {}
    for name, df in frames.items():
        fields = (
            NestedField(i + 1, column, iceberg_types[dtype])  # type: ignore[index]
            for i, (column, dtype) in enumerate(df.schema.items())
        )
        tbl, _ = new_iceberg_table(tmp_path, schema=IcebergSchema(*fields), name=name)
        df.write_iceberg(tbl, mode="append")
        tables[name] = tbl

    files = {
        name: [
            _normalize_windows_iceberg_file_uri(f.file.file_path)
            for f in tbl.scan().plan_files()
        ]
        for name, tbl in tables.items()
    }
    return tables, files


@pytest.mark.write_disk
def test_scan_iceberg_join_order_matches_parquet(tmp_path: Path) -> None:
    n = 1000
    tables, files = write_iceberg_tables(
        tmp_path,
        {
            "a": pl.DataFrame({"k": [i % 5 for i in range(n)], "x": range(n)}),
            "b": pl.DataFrame({"k": [i % 5 for i in range(n)], "y": range(n)}),
            "d": pl.DataFrame({"k": range(5), "flag": [True] + 4 * [False]}),
        },
    )

    def query(scan: Callable[[str], pl.LazyFrame]) -> pl.LazyFrame:
        return (
            scan("a")
            .join(scan("b"), on="k")
            .join(scan("d").filter(pl.col("flag")), on="k")
        )

    iceberg = query(lambda name: pl.scan_iceberg(tables[name]))
    parquet = query(lambda name: pl.scan_parquet(files[name]))

    structure = join_structure(iceberg)
    assert structure == join_structure(parquet)

    # The filtered dimension is joined before the other fact.
    plan = "\n".join(structure)
    assert plan.index(files["d"][0]) < plan.index(files["b"][0])


@pytest.mark.write_disk
def test_scan_iceberg_join_order_with_a_filter_above_a_shared_scan(
    tmp_path: Path,
) -> None:
    n = 1000
    tables, files = write_iceberg_tables(
        tmp_path,
        {
            "a": pl.DataFrame(
                {
                    "k": [i % 5 for i in range(n)],
                    "dk": range(n),
                    "ek": range(n),
                    "y": range(n),
                }
            ),
            "b": pl.DataFrame({"k": [i % 5 for i in range(n)]}),
            "d": pl.DataFrame(
                {"dk": range(n), "year": [1900 + i // 5 for i in range(n)]}
            ),
            "e": pl.DataFrame({"ek": range(n), "flag": [i % 20 for i in range(n)]}),
        },
    )

    def query(scan: Callable[[str], pl.LazyFrame]) -> pl.LazyFrame:
        # `d` is read twice, so its filter stays above the shared scan instead of
        # being pushed into it.
        d = scan("d")
        return (
            scan("a")
            .join(scan("b"), on="k")
            .join(d.filter(pl.col("year") == 1999), on="dk")
            .join(scan("e").filter(pl.col("flag") == 0), on="ek")
            .join(d.select(y="dk"), on="y")
        )

    iceberg = query(lambda name: pl.scan_iceberg(tables[name]))
    parquet = query(lambda name: pl.scan_parquet(files[name]))
    assert 'FILTER col("year") == 1999' in iceberg.explain()

    structure = join_structure(iceberg)
    assert structure == join_structure(parquet)

    # The statistics of `year` show the filter keeps far fewer rows than the one on
    # `flag`.
    plan = "\n".join(structure)
    assert plan.index(files["d"][0]) < plan.index(files["e"][0])


@pytest.mark.write_disk
def test_scan_iceberg_cache_decision_matches_parquet_29822(tmp_path: Path) -> None:
    rows = 20_000
    tables, files = write_iceberg_tables(
        tmp_path,
        {
            "fact": pl.DataFrame(
                {
                    "key": [i % 100 for i in range(rows)],
                    "grp": [i % 7 for i in range(rows)],
                    "val": range(rows),
                }
            ),
            "dim": pl.DataFrame({"key": range(100), "name": range(100)}),
        },
    )

    def query(scan: Callable[[str], pl.LazyFrame]) -> pl.LazyFrame:
        # Each branch filters the shared subplan on its own, so the caches are only
        # kept if the statistics show that sharing the subplan is cheaper.
        base = (
            scan("fact")
            .join(scan("dim"), on="key")
            .group_by("grp", "name")
            .agg(pl.col("val").sum())
        )
        return pl.concat(
            [base.filter(pl.col("grp") != i).select("name", "val") for i in range(8)]
        )

    iceberg = query(lambda name: pl.scan_iceberg(tables[name]))
    parquet = query(lambda name: pl.scan_parquet(files[name]))

    assert parquet.explain().count("CACHE[id:") == 8
    assert iceberg.explain().count("CACHE[id:") == 8
    assert_frame_equal(iceberg.collect(), parquet.collect(), check_row_order=False)


@pytest.mark.write_disk
def test_scan_iceberg_statistics_do_not_read_filtered_out_metadata(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "k", LongType()), NestedField(2, "v", LongType())
        ),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "k")),
    )
    for k in [1, 2]:
        pl.DataFrame({"k": [k] * 10, "v": range(10)}).write_iceberg(tbl, mode="append")
    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    # Only a read of partition `k = 2` needs this manifest.
    for manifest in snapshot.manifests(tbl.io):
        entries = manifest.fetch_manifest_entry(tbl.io)
        if all(e.data_file.partition[0] == 2 for e in entries):
            Path(manifest.manifest_path.removeprefix("file://")).unlink()

    scan = pl.scan_iceberg(tbl)
    k1 = scan.filter(pl.col("k") == 1)
    # Different filters over a shared scan, so the cache decision asks for estimates.
    q = pl.concat([k1.filter(pl.col("v") < 5), k1.filter(pl.col("v") >= 5)])
    assert q.collect(engine="in-memory").height == 10
    assert q.collect(engine="streaming").height == 10

    sql = "SELECT k FROM o WHERE s < (SELECT SUM(v) FROM i WHERE i.k = o.k AND i.k = 1)"
    ctx = pl.SQLContext(o=pl.DataFrame({"k": [1], "s": [0]}), i=scan)
    assert ctx.execute(sql).collect().height == 1


@pytest.mark.write_disk
def test_scan_iceberg_shared_scan_reads_one_snapshot(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "x", LongType())), name="a"
    )
    pl.DataFrame({"x": [1]}).write_iceberg(tbl, mode="append")
    scan = pl.scan_iceberg(tbl)

    original = IcebergScanResolver.to_dataset_scan
    calls = 0

    def to_dataset_scan(self: IcebergScanResolver, **kwargs: Any) -> Any:
        nonlocal calls
        resolved = original(self, **kwargs)
        calls += 1
        if calls == 1:
            # A writer commits between two reads of the same scan.
            pl.DataFrame({"x": [2]}).write_iceberg(tbl, mode="append")
        return resolved

    with patch.object(IcebergScanResolver, "to_dataset_scan", to_dataset_scan):
        out = pl.concat([scan, scan]).collect(engine="streaming")

    # Both reads of one scan see the same snapshot.
    assert out["x"].to_list() == [1, 1]


@pytest.mark.write_disk
def test_scan_iceberg_sql_correlated_aggregate_restricted_like_parquet(
    tmp_path: Path,
) -> None:
    tables, files = write_iceberg_tables(
        tmp_path,
        {
            "o": pl.DataFrame({"k": range(10), "s": range(10)}),
            "i": pl.DataFrame({"k": [i % 20 for i in range(100)], "v": range(100)}),
        },
    )
    query = "SELECT k FROM o WHERE s < (SELECT SUM(v) FROM i WHERE i.k = o.k)"

    def plan_and_result(
        scan: Callable[[str], pl.LazyFrame],
    ) -> tuple[str, pl.DataFrame]:
        lf = pl.SQLContext({name: scan(name) for name in tables}).execute(query)
        return lf.explain(), lf.collect()

    iceberg_plan, iceberg = plan_and_result(lambda n: pl.scan_iceberg(tables[n]))
    parquet_plan, parquet = plan_and_result(lambda n: pl.scan_parquet(files[n]))

    # The inner table is over twice the outer one, so the aggregate is restricted to
    # the outer keys.
    assert "SEMI JOIN" in parquet_plan
    assert "SEMI JOIN" in iceberg_plan
    assert_frame_equal(iceberg, parquet, check_row_order=False)


@pytest.mark.write_disk
def test_scan_iceberg_join_on_full_range_decimal_keys(tmp_path: Path) -> None:
    big = 10**38 - 1
    df = pl.from_arrow(
        pa.table({"k": pa.array([D(-big), D(big)], pa.decimal128(38, 0))})
    )
    assert isinstance(df, pl.DataFrame)

    scans = []
    for name in "abc":
        tbl, _ = new_iceberg_table(
            tmp_path,
            schema=IcebergSchema(NestedField(1, "k", DecimalType(38, 0))),
            name=name,
        )
        df.write_iceberg(tbl, mode="append")
        scans.append(pl.scan_iceberg(tbl))

    a, b, c = scans
    q = a.join(b, on="k").join(c, on="k")
    assert_frame_equal(q.collect(), df, check_row_order=False)


@pytest.mark.write_disk
def test_scan_iceberg_join_planning_reads_no_data_files(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", LongType())
        ),
    )
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).write_iceberg(tbl, mode="append")

    for f in tbl.scan().plan_files():
        Path(f.file.file_path.removeprefix("file://")).unlink()

    lf = pl.scan_iceberg(tbl)
    plan = lf.join(lf.select("a", c="b"), on="a").explain()
    assert "ESTIMATED ROWS: 3" in plan


@pytest.mark.write_disk
def test_scan_iceberg_fallback_join_keeps_projection(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", LongType()),
            NestedField(3, "c", LongType()),
        ),
    )
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).write_iceberg(
        tbl, mode="append"
    )

    q = (
        pl.scan_iceberg(tbl, reader_override="pyiceberg")
        .select("a", "b")
        .join(pl.LazyFrame({"a": [2, 3]}), on="a")
    )

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    calls = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("IcebergScanResolver: to_dataset_scan(): snapshot ID")
    ]

    # The Python scan is built with the final projection.
    assert "projection: ['a', 'b']" in calls[-1]
    assert_frame_equal(
        out, pl.DataFrame({"a": [2, 3], "b": [5, 6]}), check_row_order=False
    )


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.write_disk
def test_scan_iceberg_runtime_join_filter(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    renamed: bool,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "x", LongType()), NestedField(2, "f", LongType())
        ),
    )
    # One file per append.
    for i in range(3):
        pl.DataFrame(
            {"x": range(100 * i, 100 * i + 100), "f": range(100)}
        ).write_iceberg(tbl, mode="append")

    # The table statistics hold `f`. They only hold `x` if the join key has its name.
    probe = pl.scan_iceberg(tbl).filter(pl.col("f") >= 0)
    key = "k" if renamed else "x"
    q = probe.rename({"x": key}).join(small_build_side(150, 160, key=key), on=key)
    assert 'col("x").dynamic_predicate()' in q.explain(engine="streaming")

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    # The reader prunes row groups only with its capabilities.
    plmonkeypatch.setenv("POLARS_FORCE_EMPTY_READER_CAPABILITIES", "0")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err

    assert err.count("reading 0 / 1 row groups") == 2
    assert err.count("reading 1 / 1 row groups") == 1
    assert_frame_equal(
        out,
        pl.DataFrame({key: [150, 160], "f": [50, 60], "e": [0, 1]}),
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_scan_iceberg_filter_identity_partition_decimal_widened(
    tmp_path: Path,
) -> None:

    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DecimalType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "d", DecimalType(9, 2))),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "d")),
    )
    tbl.append(
        pl.DataFrame(
            {"d": [D("1.00"), D("2.50")]}, schema={"d": pl.Decimal(9, 2)}
        ).to_arrow()
    )
    with tbl.update_schema() as update:
        update.update_column("d", DecimalType(18, 2))
    tbl = tbl.catalog.load_table(tbl.name())

    predicate = pl.col("d") == pl.lit(D("1.00"), pl.Decimal(18, 2))
    assert_frame_equal(
        pl.scan_iceberg(tbl).filter(predicate).collect(),
        pl.DataFrame({"d": [D("1.00")]}, schema={"d": pl.Decimal(18, 2)}),
    )


@pytest.mark.write_disk
def test_scan_iceberg_time_travel_filter_after_schema_change(
    tmp_path: Path,
) -> None:

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", StringType())
        ),
    )
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tbl.append(df.to_arrow())
    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    snapshot_id = snapshot.snapshot_id
    # `b` is dropped; `a` is renamed, and another column takes its name.
    with tbl.update_schema() as update:
        update.delete_column("b")
    with tbl.update_schema() as update:
        update.rename_column("a", "a_old")
    with tbl.update_schema() as update:
        update.add_column("a", StringType())
    tbl = tbl.catalog.load_table(tbl.name())

    for predicate in [
        pl.col("b") == "x",
        pl.col("a") == 2,
        (pl.col("a") > 1) & (pl.col("b") != "z"),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, snapshot_id=snapshot_id).filter(predicate).collect(),
            df.filter(predicate),
        )


@pytest.mark.write_disk
def test_scan_iceberg_filter_float_nan_ordering(
    tmp_path: Path,
) -> None:

    from pyiceberg.types import DoubleType

    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "f", DoubleType()))
    )
    # One file with NaN and regular values, one with only NaN.
    tbl.append(pl.DataFrame({"f": [1.0, float("nan")]}).to_arrow())
    tbl.append(pl.DataFrame({"f": [float("nan")]}).to_arrow())
    df = pl.DataFrame({"f": [1.0, float("nan"), float("nan")]})

    # In Polars, NaN is greater than all other values.
    for predicate in [
        pl.col("f") > 1.5,
        pl.col("f") >= 1.5,
        ~(pl.col("f") < 1.5),
        ~(pl.col("f") <= 1.5),
        (pl.col("f") > 1.5) | (pl.col("f") == 0.0),
        pl.col("f") < 1.5,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
def test_scan_iceberg_reader_filter_float_nan(
    tmp_path: Path, reader_override: Literal["pyiceberg"] | None
) -> None:
    from pyiceberg.types import DoubleType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "x", DoubleType()), NestedField(2, "i", LongType())
        ),
    )
    df = pl.DataFrame(
        {"x": [1.0, float("nan"), 3.0, None, 0.0, -0.0], "i": [1, 2, 3, 4, 5, 6]}
    )
    tbl.append(df.to_arrow())

    for predicate in [
        pl.col("x") > 2.0,
        pl.col("x") != 1.0,
        ~(pl.col("x") <= 1.0),
        pl.col("x").is_nan(),
        ~pl.col("x").is_in([1.0]),
        (pl.col("x") > 2.0) & (pl.col("i") > 1),
        pl.col("x") < 2.0,
        pl.col("x") < float("nan"),
        pl.col("x").is_in([3.0]),
        # PyArrow `is_in` distinguishes -0.0 from 0.0.
        pl.col("x").is_in([0.0, 5.0]),
        pl.col("x").is_in([-0.0, 5.0]),
        pl.col("x") == 0.0,
        (pl.col("x") <= 1.0) | (pl.col("i") == 2),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override)
            .filter(predicate)
            .collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
def test_scan_iceberg_initial_default_identity_partition_added_later(
    tmp_path: Path,
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl.append(pl.DataFrame({"a": [1, 2]}).to_arrow())

    md_path = Path(
        tbl.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())
    md_object["format-version"] = 3
    md_object["schemas"][-1]["fields"].append(
        {
            "id": 2,
            "name": "region",
            "required": False,
            "type": "string",
            "initial-default": "EU",
        }
    )
    md_object["last-column-id"] = 2
    # Partition by `region` after the files were written: those files are in a
    # spec without the identity field.
    md_object["partition-specs"].append(
        {
            "spec-id": 1,
            "fields": [
                {
                    "source-id": 2,
                    "field-id": 1000,
                    "name": "region",
                    "transform": "identity",
                }
            ],
        }
    )
    md_object["default-spec-id"] = 1
    md_object["last-partition-id"] = 1000
    md_path.write_text(json.dumps(md_object))
    tbl = catalog.load_table(tbl.name())

    expect = pl.DataFrame({"a": [1, 2], "region": ["EU", "EU"]})
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), expect)
    assert_frame_equal(
        pl.scan_iceberg(tbl).filter(pl.col("region") == "EU").collect(), expect
    )


@pytest.mark.write_disk
def test_scan_iceberg_initial_default_null_identity_partition_value(
    tmp_path: Path,
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "old", StringType())
        ),
    )
    tbl.append(pa.table({"a": [1], "old": ["x"]}, schema=tbl.schema().as_arrow()))
    with tbl.update_spec() as update:
        update.add_identity("old")
    tbl = catalog.load_table(tbl.name())
    tbl.append(pa.table({"a": [2], "old": [None]}, schema=tbl.schema().as_arrow()))

    md_path = Path(
        tbl.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())
    md_object["format-version"] = 3
    # Replace `old` by a new field `region`, which no file has. Its identity
    # partition (moved over) is null for the second file.
    md_object["schemas"][-1]["fields"][1] = {
        "id": 3,
        "name": "region",
        "required": False,
        "type": "string",
        "initial-default": "EU",
    }
    md_object["last-column-id"] = 3
    for spec in md_object["partition-specs"]:
        for field in spec["fields"]:
            field["source-id"] = 3
            field["name"] = "region"
    md_path.write_text(json.dumps(md_object))
    tbl = catalog.load_table(tbl.name())

    # The file of the unpartitioned spec takes the `initial-default`, and the
    # file with a null identity partition value is null.
    expect = pl.DataFrame({"a": [1, 2], "region": ["EU", None]})
    assert_frame_equal(pl.scan_iceberg(tbl).collect().sort("a"), expect)
    for predicate in [pl.col("region").is_null(), pl.col("region") == "EU"]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            expect.filter(predicate),
        )


@pytest.mark.write_disk
def test_scan_iceberg_filter_struct_fields_swapped(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(
                1,
                "st",
                StructType(
                    NestedField(2, "x", IntegerType()),
                    NestedField(3, "y", IntegerType()),
                ),
            )
        ),
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"st": {"x": 1, "y": None}}, {"st": {"x": 2, "y": None}}],
            schema=tbl.schema().as_arrow(),
        )
    )
    # Swap the names: the file's null counts of `x` are those of the table's `y`.
    with tbl.update_schema() as update:
        update.rename_column("st.x", "tmp")
    with tbl.update_schema() as update:
        update.rename_column("st.y", "x")
    with tbl.update_schema() as update:
        update.rename_column("st.tmp", "y")

    df = pl.DataFrame(
        {"st": [{"y": 1, "x": None}, {"y": 2, "x": None}]},
        schema={"st": pl.Struct({"y": pl.Int32, "x": pl.Int32})},
    )
    f = pl.col("st").struct.field
    for predicate in [
        f("x").is_null(),
        f("y").is_not_null(),
        f("y") == 1,
        f("x").is_not_null(),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(), df.filter(predicate)
        )


@pytest.mark.write_disk
def test_scan_iceberg_filter_float_identity_partition_changed(tmp_path: Path) -> None:
    from pyiceberg.types import DoubleType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "f", DoubleType()), NestedField(2, "i", LongType())
        ),
    )
    tbl.append(pl.DataFrame({"f": [1.0], "i": [1]}).to_arrow())
    with tbl.update_spec() as update:
        update.add_identity("f")
    tbl.append(pl.DataFrame({"f": [100.0], "i": [2]}).to_arrow())
    with tbl.update_spec() as update:
        update.remove_field("f")
    tbl.append(pl.DataFrame({"f": [200.0], "i": [3]}).to_arrow())

    df = pl.DataFrame({"f": [1.0, 100.0, 200.0], "i": [1, 2, 3]})
    for predicate in [pl.col("f") > 50, (pl.col("f") == 1.0) | (pl.col("i") == 3)]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_scan_iceberg_reader_pyiceberg_negative_slice(
    tmp_path: Path, engine: EngineType
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    for i in range(3):
        tbl.append(pl.DataFrame({"a": [i]}).to_arrow())

    lf = pl.scan_iceberg(tbl, reader_override="pyiceberg")
    df = lf.collect()
    assert df.height == 3

    assert_frame_equal(lf.tail(2).collect(engine=engine), df.tail(2))
    assert_frame_equal(lf.slice(-2, 1).collect(engine=engine), df.slice(-2, 1))
    assert lf.tail(2).select(pl.len()).collect(engine=engine).item() == 2


@pytest.mark.write_disk
def test_scan_iceberg_reader_pyiceberg_filter_nested_columns(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(
                2, "m", MapType(3, StringType(), 4, LongType(), value_required=False)
            ),
            NestedField(5, "li", ListType(6, LongType(), element_required=False)),
        ),
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"a": 1, "m": None, "li": None}, {"a": 2, "m": [("k", 1)], "li": [1]}],
            schema=tbl.schema().as_arrow(),
        )
    )
    df = pl.DataFrame(tbl.scan().to_arrow())

    for predicate in [
        pl.col("m").is_null(),
        pl.col("li").is_null(),
        pl.col("li").is_not_null() & (pl.col("a") > 0),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override="pyiceberg")
            .filter(predicate)
            .collect(),
            df.filter(predicate),
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
def test_scan_iceberg_filter_not_in_one_value_nulls_equal(
    tmp_path: Path, reader_override: Literal["pyiceberg"] | None
) -> None:
    # PyIceberg turns `In` of one value into `EqualTo`, whose negation drops nulls.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "id", LongType()), NestedField(2, "p", StringType())
        ),
    )
    df = pl.DataFrame({"id": [1, 2, 3], "p": ["a", None, "b"]})
    tbl.append(df.to_arrow())

    for predicate in [
        ~pl.col("p").is_in(["a"], nulls_equal=True),
        ~(pl.col("p").is_in(["a"], nulls_equal=True) & (pl.col("id") > 0)),
        pl.col("p") != "a",
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override)
            .filter(predicate)
            .collect(),
            df.filter(predicate),
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
@pytest.mark.parametrize("width", [3, 10])
def test_scan_iceberg_filter_truncate_partition_type_minimum(
    tmp_path: Path, reader_override: Literal["pyiceberg"] | None, width: int
) -> None:
    # The truncated value of the type's minimum wraps to a large positive value.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", IntegerType())),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, TruncateTransform(width), "x_trunc")
        ),
    )
    for v in [-(2**31), -1, 0, 15]:
        tbl.append(pl.DataFrame({"x": [v]}, schema={"x": pl.Int32}).to_arrow())
    with tbl.update_schema() as update:
        update.update_column("x", LongType())
    tbl.append(pl.DataFrame({"x": [-(2**63), 5]}).to_arrow())

    df = pl.DataFrame({"x": [-(2**31), -1, 0, 15, -(2**63), 5]})

    for predicate in [
        pl.col("x") < 0,
        pl.col("x") <= -5,
        pl.col("x") == -(2**31),
        pl.col("x") == -(2**63),
        pl.col("x").is_in([-(2**31), 15]),
        pl.col("x") >= 10,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override)
            .filter(predicate)
            .collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
def test_scan_iceberg_filter_categorical_literal(
    tmp_path: Path, reader_override: Literal["pyiceberg"] | None
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "s", StringType()))
    )
    df = pl.DataFrame({"s": ["a", "b"]})
    tbl.append(df.to_arrow())

    for dtype in [pl.Categorical(), pl.Enum(["a", "b"])]:
        assert_frame_equal(
            pl.scan_iceberg(tbl, reader_override=reader_override)
            .filter(pl.col("s") == pl.lit("a", dtype))
            .collect(),
            df.head(1),
        )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
def test_scan_iceberg_collect_after_schema_change(
    tmp_path: Path, reader_override: Literal["pyiceberg"] | None
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", LongType())
        ),
    )
    tbl.append(pl.DataFrame({"a": [1, 2], "b": [10, 20]}).to_arrow())
    first_snapshot_id = tbl.current_snapshot().snapshot_id  # type: ignore[union-attr]

    lf = pl.scan_iceberg(tbl, reader_override=reader_override)
    lf_snapshot = pl.scan_iceberg(
        tbl, snapshot_id=first_snapshot_id, reader_override=reader_override
    )
    expected = lf.collect()
    assert_frame_equal(lf_snapshot.collect(), expected)

    # Swap the column names, without a new snapshot. Scans are resolved by name, so
    # the scan must not mix the fields.
    with tbl.update_schema() as update:
        update.rename_column("a", "tmp")
    with tbl.update_schema() as update:
        update.rename_column("b", "a")
    with tbl.update_schema() as update:
        update.rename_column("tmp", "b")

    for q in [
        lf,
        lf.filter(pl.col("a") > 0),
        pl.concat([lf.filter(pl.col("a") > 0), lf.filter(pl.col("b") > 0)]),
    ]:
        with pytest.raises(pl.exceptions.ComputeError, match="table schema changed"):
            q.collect()

    assert_frame_equal(lf_snapshot.collect(), expected)
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override=reader_override).collect(),
        pl.DataFrame({"b": [1, 2], "a": [10, 20]}),
    )


@pytest.mark.write_disk
def test_scan_iceberg_filter_struct_column_added_later(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "id", LongType()))
    )
    tbl.append(pl.DataFrame({"id": [1]}).to_arrow())
    with tbl.update_schema() as update:
        update.add_column("st", StructType(NestedField(3, "x", LongType())))
    df = pl.DataFrame(
        {"id": [1, 2], "st": [None, {"x": 9}]},
        schema={"id": pl.Int64, "st": pl.Struct({"x": pl.Int64})},
    )
    tbl.append(df.tail(1).to_arrow())

    for predicate in [
        pl.col("st").struct.field("x") == 9,
        pl.col("st").is_null(),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            df.filter(predicate),
        )


def test_scan_iceberg_statistics_without_metrics() -> None:
    from types import SimpleNamespace

    from pyiceberg.types import LongType

    from polars.io.iceberg._utils import (
        IcebergColumnStatisticsLoader,
        LoadFromBytesImpl,
    )

    # The metric maps of data files are optional.
    loader = IcebergColumnStatisticsLoader(
        column_name="a",
        column_dtype=pl.Int64(),
        field_id=1,
        load_from_bytes_impl=LoadFromBytesImpl.init_for_field_type(
            LongType(), {LongType()}, pl.Int64()
        ),
        null_count=[],
        min_values=[],
        max_values=[],
    )
    loader.push_file_statistics(
        SimpleNamespace(  # type: ignore[arg-type]
            null_value_counts=None, lower_bounds=None, upper_bounds=None
        )
    )

    assert loader.finish(1, None).row(0) == (None, None, None)


@pytest.mark.write_disk
def test_scan_iceberg_v3_nested_field_initial_default(tmp_path: Path) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "s", StructType(NestedField(2, "a", LongType()))),
            NestedField(
                3,
                "ls",
                ListType(
                    4,
                    StructType(NestedField(5, "a", LongType())),
                    element_required=False,
                ),
            ),
            NestedField(
                6,
                "m",
                MapType(
                    7,
                    StringType(),
                    8,
                    StructType(NestedField(9, "a", LongType())),
                    value_required=False,
                ),
            ),
        ),
    )
    tbl.append(
        pa.Table.from_pylist(
            [{"s": {"a": 1}, "ls": [{"a": 1}, None], "m": [("k", {"a": 1})]}],
            schema=tbl.schema().as_arrow(),
        )
    )

    md_path = Path(
        tbl.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())
    md_object["format-version"] = 3
    fields = md_object["schemas"][-1]["fields"]

    def new_field(field_id: int, default: int) -> dict[str, Any]:
        return {
            "id": field_id,
            "name": "b",
            "required": False,
            "type": "long",
            "initial-default": default,
        }

    fields[0]["type"]["fields"].append(new_field(10, 5))
    fields[1]["type"]["element"]["fields"].append(new_field(11, 6))
    fields[2]["type"]["value"]["fields"].append(new_field(12, 7))
    md_object["last-column-id"] = 12
    md_path.write_text(json.dumps(md_object))
    tbl = catalog.load_table(tbl.name())

    struct = pl.Struct({"a": pl.Int64, "b": pl.Int64})
    expect = pl.DataFrame(
        {
            "s": [{"a": 1, "b": 5}],
            "ls": [[{"a": 1, "b": 6}, None]],
            "m": [{"k": {"a": 1, "b": 7}}],
        },
        schema={"s": struct, "ls": pl.List(struct), "m": pl.Map(pl.String, struct)},
    )

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), expect)
    assert_frame_equal(
        pl.scan_iceberg(tbl).filter(pl.col("s").struct.field("b") == 5).collect(),
        expect,
    )


@pytest.mark.write_disk
def test_scan_iceberg_reader_pyiceberg_time_travel_filter(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", StringType())
        ),
    )
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl.append(df.to_arrow())
    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    with tbl.update_schema() as update:
        update.delete_column("b")
    with tbl.update_schema() as update:
        update.rename_column("a", "c")
    with tbl.update_schema() as update:
        update.add_column("a", StringType())
    tbl = tbl.catalog.load_table(tbl.name())

    for predicate in [pl.col("a") >= 2, pl.col("b") == "x"]:
        assert_frame_equal(
            pl.scan_iceberg(
                tbl, snapshot_id=snapshot.snapshot_id, reader_override="pyiceberg"
            )
            .filter(predicate)
            .collect(),
            df.filter(predicate),
        )


@pytest.mark.write_disk
def test_scan_iceberg_time_travel_filter_identity_partition_before_promotion(
    tmp_path: Path,
) -> None:
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import IntegerType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "p", IntegerType())),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "p")),
    )
    tbl.append(pl.DataFrame({"p": [1, 2]}, schema={"p": pl.Int32}).to_arrow())
    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    with tbl.update_schema() as update:
        update.update_column("p", LongType())
    tbl = tbl.catalog.load_table(tbl.name())
    tbl.append(pl.DataFrame({"p": [3]}).to_arrow())

    assert_frame_equal(
        pl.scan_iceberg(tbl, snapshot_id=snapshot.snapshot_id)
        .filter(pl.col("p") == 1)
        .collect(),
        pl.DataFrame({"p": [1]}, schema={"p": pl.Int32}),
    )


@pytest.mark.write_disk
def test_scan_iceberg_reader_pyiceberg_incremental_len(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    snapshot_ids = []
    for i in range(3):
        tbl.append(pl.DataFrame({"a": [i, i]}).to_arrow())
        snapshot = tbl.current_snapshot()
        assert snapshot is not None
        snapshot_ids.append(snapshot.snapshot_id)

    lf = pl.scan_iceberg(
        tbl,
        reader_override="pyiceberg",
        from_snapshot_id_exclusive=snapshot_ids[0],
        to_snapshot_id_inclusive=snapshot_ids[2],
    )
    assert lf.select(pl.len()).collect().item() == 4
    assert lf.head(3).select(pl.len()).collect().item() == 3


@pytest.mark.ci_only
def test_convert_predicate_nanosecond_literal() -> None:
    # Truncating to microseconds would narrow the predicate.
    assert (
        try_convert_pyarrow_predicate(
            "(pa.compute.field('t') < to_py_datetime(1500,'ns'))"
        )
        is None
    )
    assert try_convert_pyarrow_predicate(
        "(pa.compute.field('t') < to_py_datetime(2000,'ns'))"
    ) == LessThan("t", "1970-01-01T00:00:00.000002")


@pytest.mark.write_disk
@pytest.mark.parametrize("iceberg_type", [FloatType(), DoubleType()])
def test_sink_iceberg_identity_partition_nan_and_signed_zero(
    tmp_path: Path, iceberg_type: Any
) -> None:
    # Parquet statistics have no bounds for NaN columns, and do not tell signed zeros
    # apart; Iceberg keeps -0.0 and 0.0 in separate partitions.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "f", iceberg_type), NestedField(2, "i", LongType())
        ),
        partition_spec=PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "f")),
    )
    dtype = pl.Float32 if isinstance(iceberg_type, FloatType) else pl.Float64
    df = pl.DataFrame(
        {
            "f": pl.Series(
                [float("nan"), float("nan"), 1.5, None, -0.0, 0.0], dtype=dtype
            ),
            "i": range(6),
        }
    )
    df.lazy().sink_iceberg(tbl, mode="append")

    def partition_key(v: float | None) -> bytes | str | None:
        # Tells signed zeros apart.
        return None if v is None else "nan" if math.isnan(v) else struct.pack(">d", v)

    partitions = {
        (partition_key(task.file.partition[0]), task.file.record_count)
        for task in tbl.scan().plan_files()
    }
    assert partitions == {
        ("nan", 2),
        (partition_key(1.5), 1),
        (None, 1),
        (partition_key(-0.0), 1),
        (partition_key(0.0), 1),
    }

    for predicate in [
        pl.col("f").is_not_null(),
        pl.col("f").is_nan(),
        pl.col("f").is_null(),
        pl.col("f") == 0.0,
        pl.col("f") > 1.0,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect().sort("i"),
            df.filter(predicate),
        )


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("iceberg_type", "dtype", "bits"),
    [(IntegerType(), pl.Int32, 32), (LongType(), pl.Int64, 64)],
)
def test_sink_iceberg_truncate_partition_near_minimum(
    tmp_path: Path, iceberg_type: Any, dtype: pl.DataType, bits: int
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "x", iceberg_type), NestedField(2, "i", LongType())
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, TruncateTransform(10), "x_trunc")
        ),
    )
    minimum = -(1 << (bits - 1))
    values = [minimum, minimum + 5, -1, 0]
    df = pl.DataFrame({"x": pl.Series(values, dtype=dtype), "i": range(len(values))})
    df.lazy().sink_iceberg(tbl, mode="append")

    def wrapped_truncate(v: int) -> int:
        # As Iceberg Java (and PyIceberg's append) write it: a wrapping subtraction.
        t = v - v % 10
        return (t - minimum) % (1 << bits) + minimum

    assert {task.file.partition[0] for task in tbl.scan().plan_files()} == {
        wrapped_truncate(v) for v in values
    }
    for predicate in [pl.col("x") < 0, pl.col("x") == minimum, pl.col("x") >= -1]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect().sort("i"),
            df.filter(predicate),
        )


@pytest.mark.write_disk
def test_scan_iceberg_truncate_partition_near_int_minimum_after_promotion(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", IntegerType())),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, TruncateTransform(100), "x_trunc")
        ),
    )
    # One manifest whose partition summary bounds (written while `x` is an int) hide
    # `truncate[100]` of `-2**31`, wrapped to 2147483596, below the upper bound
    # 2147483600.
    df = pl.DataFrame({"x": pl.Series([-(2**31), 5, 2**31 - 1], dtype=pl.Int32)})
    tbl.append(df.to_arrow())
    with tbl.update_schema() as update:
        update.update_column("x", LongType())

    df = df.cast(pl.Int64)
    for predicate in [
        pl.col("x") <= -5,
        pl.col("x") == -(2**31),
        pl.col("x") < 0,
        pl.col("x") >= 2**31 - 1,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect().sort("x"),
            df.filter(predicate).sort("x"),
        )


@pytest.mark.write_disk
def test_scan_iceberg_truncate_partition_near_int_minimum_merged_manifest(
    tmp_path: Path,
) -> None:
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", IntegerType())),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, TruncateTransform(100), "x_trunc")
        ),
        properties={
            "commit.manifest-merge.enabled": "true",
            "commit.manifest.min-count-to-merge": "2",
        },
    )
    df = pl.DataFrame({"x": pl.Series([-(2**31), 2147483400], dtype=pl.Int32)})
    tbl.append(df.to_arrow())
    with tbl.update_schema() as update:
        update.update_column("x", LongType())
    tbl = catalog.load_table(tbl.name())

    # Merged into one manifest with the int-era files: its upper bound lies beyond
    # the int range, and `truncate[100]` of `-2**31`, wrapped to 2147483596, between
    # its bounds.
    appended = pl.DataFrame({"x": [3_000_000_000]})
    tbl.append(appended.to_arrow())
    assert len(tbl.current_snapshot().manifests(tbl.io)) == 1  # type: ignore[union-attr]

    df = pl.concat([df.cast(pl.Int64), appended])
    for predicate in [
        pl.col("x") < 0,
        pl.col("x") == -(2**31),
        pl.col("x") >= 3_000_000_000,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect().sort("x"),
            df.filter(predicate).sort("x"),
        )


@pytest.mark.write_disk
def test_scan_iceberg_truncate_partition_near_int_minimum_int_schema_removed(
    tmp_path: Path,
) -> None:
    from pyiceberg.table.update import RemoveSchemasUpdate

    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", IntegerType())),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, TruncateTransform(100), "x_trunc")
        ),
    )
    # The manifest's partition summary bounds hide `truncate[100]` of `-2**31`,
    # wrapped to 2147483596, below the upper bound 2147483600.
    df = pl.DataFrame({"x": pl.Series([-(2**31), 5, 2**31 - 1], dtype=pl.Int32)})
    tbl.append(df.to_arrow())
    first_snapshot_id = tbl.current_snapshot().snapshot_id  # type: ignore[union-attr]
    with tbl.update_schema() as update:
        update.update_column("x", LongType())
    tbl = catalog.load_table(tbl.name())
    appended = pl.DataFrame({"x": [7]})
    tbl.append(appended.to_arrow())

    # Metadata cleanup: once no snapshot uses the int schema, it can be removed. The
    # manifest written while `x` was an int is unchanged.
    tbl.maintenance.expire_snapshots().by_id(first_snapshot_id).commit()
    with tbl.transaction() as tx:
        tx._apply((RemoveSchemasUpdate.model_validate({"schema-ids": [0]}),))
    tbl = catalog.load_table(tbl.name())
    assert [s.schema_id for s in tbl.metadata.schemas] == [1]

    df = pl.concat([df.cast(pl.Int64), appended])
    for predicate in [
        pl.col("x") < 0,
        pl.col("x") == -(2**31),
        pl.col("x") <= -5,
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect().sort("x"),
            df.filter(predicate).sort("x"),
        )


@pytest.mark.write_disk
def test_sink_iceberg_partition_long_values(tmp_path: Path) -> None:
    # Parquet statistics of long values are truncated.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "s", StringType()),
            NestedField(2, "b", BinaryType()),
            NestedField(3, "t", StringType()),
            NestedField(4, "i", LongType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "s"),
            PartitionField(2, 1001, IdentityTransform(), "b"),
            PartitionField(3, 1002, TruncateTransform(70), "t_trunc"),
        ),
    )
    long = "é" * 50 + "x" * 50
    df = pl.DataFrame(
        {
            "s": [long, long, "a"],
            "b": [long.encode(), long.encode(), b"a"],
            "t": [long, long + "y", "a"],
            "i": [0, 1, 2],
        }
    )
    df.lazy().sink_iceberg(tbl, mode="append")

    partitions = [task.file.partition for task in tbl.scan().plan_files()]
    assert {(p[0], p[1], p[2]) for p in partitions} == {
        (long, long.encode(), long[:70]),
        ("a", b"a", "a"),
    }
    assert_frame_equal(
        pl.scan_iceberg(tbl).filter(pl.col("s") == long).collect().sort("i"),
        df.head(2),
    )


@pytest.mark.write_disk
def test_scan_iceberg_missing_struct_with_child_initial_default_is_null(
    tmp_path: Path,
) -> None:
    # A struct added after data was written is null in the older files, even if its
    # fields have defaults.
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl.append(pa.Table.from_pylist([{"a": 1}], schema=tbl.schema().as_arrow()))

    md_path = Path(
        tbl.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())
    md_object["format-version"] = 3
    md_object["schemas"][-1]["fields"].append(
        {
            "id": 2,
            "name": "s",
            "required": False,
            "type": {
                "type": "struct",
                "fields": [
                    {
                        "id": 3,
                        "name": "b",
                        "required": False,
                        "type": "long",
                        "initial-default": 5,
                    }
                ],
            },
        }
    )
    md_object["last-column-id"] = 3
    md_path.write_text(json.dumps(md_object))
    tbl = catalog.load_table(tbl.name())

    expect = pl.DataFrame(
        {"a": [1], "s": [None]},
        schema={"a": pl.Int64, "s": pl.Struct({"b": pl.Int64})},
    )
    for reader_override in [None, "pyiceberg"]:
        lf = pl.scan_iceberg(tbl, reader_override=reader_override)  # type: ignore[arg-type]
        assert_frame_equal(lf.collect(), expect)
        assert_frame_equal(lf.filter(pl.col("s").is_null()).collect(), expect)
        assert lf.filter(pl.col("s").struct.field("b") == 5).collect().height == 0


@pytest.mark.write_disk
def test_scan_iceberg_missing_struct_child_default_with_identity_partition(
    tmp_path: Path,
) -> None:
    # The identity-partition values of files whose spec lacks the identity field hold
    # the field's `initial-default`, which is not partition data: a struct missing from
    # such a file stays null. A file of a spec with the identity field gets the struct
    # from its partition value.
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl.append(pa.Table.from_pylist([{"a": 1}], schema=tbl.schema().as_arrow()))

    with tbl.update_schema() as update:
        update.add_column("s", StructType(NestedField(3, "b", LongType())))
    tbl = catalog.load_table(tbl.name())
    b_id = tbl.schema().find_field("s.b").field_id
    with tbl.update_spec() as update:
        update.add_identity("s.b")
    tbl = catalog.load_table(tbl.name())
    tbl.append(
        pa.Table.from_pylist([{"a": 2, "s": {"b": 7}}], schema=tbl.schema().as_arrow())
    )

    # Leave only `a` in the second file, so that `s` comes from its partition value.
    for task in tbl.scan().plan_files():
        if task.file.spec_id != 0:
            pq.write_table(
                pa.Table.from_pylist(
                    [{"a": 2}],
                    schema=schema_to_pyarrow(tbl.schema().select("a")),
                ),
                task.file.file_path.removeprefix("file://"),
            )

    md_path = Path(
        tbl.metadata_location.removeprefix("file:")
        # Windows //C:/... -> C:/...
        .removeprefix("//")
    )
    md_object = json.loads(md_path.read_text())
    md_object["format-version"] = 3
    for schema in md_object["schemas"]:
        for field in schema["fields"]:
            if field["name"] == "s":
                field["type"]["fields"][0]["initial-default"] = 5
                assert field["type"]["fields"][0]["id"] == b_id
    md_path.write_text(json.dumps(md_object))
    tbl = catalog.load_table(tbl.name())

    dtype = pl.Struct({"b": pl.Int64})
    expect = pl.DataFrame(
        {"a": [1, 2], "s": [None, {"b": 7}]}, schema={"a": pl.Int64, "s": dtype}
    )
    engines: list[EngineType] = ["in-memory", "streaming"]
    for engine in engines:
        lf = pl.scan_iceberg(tbl, reader_override="native")
        assert_frame_equal(
            lf.collect(engine=engine),
            expect,
            check_row_order=False,
        )
        assert_frame_equal(
            lf.filter(pl.col("s").is_null()).collect(engine=engine),
            expect.head(1),
        )
        assert (
            lf.filter(pl.col("s").struct.field("b") == 5).collect(engine=engine).height
            == 0
        )
        # Files pruned by `a` shift the source indices of the presence masks.
        for a, i in [(1, 0), (2, 1)]:
            assert_frame_equal(
                lf.filter(pl.col("a") == a).collect(engine=engine),
                expect.slice(i, 1),
            )


@pytest.mark.write_disk
def test_scan_iceberg_required_column_added_without_default(tmp_path: Path) -> None:
    # PyIceberg and Java allow it (as an incompatible change); older files hold
    # nulls. (PyIceberg's reader refuses these files.)
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    tbl.append(
        pa.Table.from_pylist([{"a": 1}, {"a": 2}], schema=tbl.schema().as_arrow())
    )
    with tbl.update_schema(allow_incompatible_changes=True) as update:
        update.add_column("r", LongType(), required=True)
    tbl = catalog.load_table(tbl.name())

    lf = pl.scan_iceberg(tbl)
    assert lf.filter(pl.col("r").is_null()).collect().height == 2
    assert lf.filter(~pl.col("r").is_not_null()).collect().height == 2


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", [None, "pyiceberg"])
def test_scan_iceberg_or_predicate_with_null_partition_values(
    tmp_path: Path, reader_override: str | None
) -> None:
    # PyIceberg's residual evaluator fails on `i > 0` for a null partition value.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "i", IntegerType()), NestedField(2, "s", StringType())
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, IdentityTransform(), "i_p")
        ),
    )
    arrow_schema = pa.schema([pa.field("i", pa.int32()), pa.field("s", pa.string())])
    df = pl.DataFrame(
        {"i": [1, None, -5, None], "s": ["a", "b", "c", "a"]},
        schema={"i": pl.Int32, "s": pl.String},
    )
    tbl.append(df.head(2).to_arrow().cast(arrow_schema))
    tbl.append(df.tail(2).to_arrow().cast(arrow_schema))

    lf = pl.scan_iceberg(tbl, reader_override=reader_override)  # type: ignore[arg-type]
    for predicate in [
        (pl.col("i") > 0) | (pl.col("s") == "a"),
        ~((pl.col("i") > 0) & (pl.col("s") == "b")),
        (pl.col("i") > 0) ^ (pl.col("s") == "a"),
    ]:
        assert_frame_equal(
            lf.filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.write_disk
def test_scan_iceberg_files_without_field_ids_use_name_mapping(
    tmp_path: Path,
) -> None:
    # E.g. files imported with `add_files`: their columns are found by the table's
    # name mapping, including after renames.
    tbl, catalog = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "s", StructType(NestedField(3, "x", StringType()))),
            NestedField(4, "l", ListType(5, LongType(), element_required=False)),
        ),
    )
    path = tmp_path / "no-field-ids.parquet"
    pq.write_table(
        pa.table(
            {
                "a": pa.array([1, 2], pa.int64()),
                "s": pa.array([{"x": "p"}, {"x": "q"}]),
                "l": pa.array([[1], [2, 3]], pa.list_(pa.int64())),
            }
        ),
        path,
    )
    tbl.add_files([format_file_uri_iceberg(path)])
    with tbl.update_schema() as update:
        update.rename_column("a", "b")
    tbl = catalog.load_table(tbl.name())

    expect = pl.DataFrame(
        {"b": [1, 2], "s": [{"x": "p"}, {"x": "q"}], "l": [[1], [2, 3]]}
    )
    lf = pl.scan_iceberg(tbl)
    assert_frame_equal(lf.collect(), expect)
    assert_frame_equal(lf.filter(pl.col("b") == 2).collect(), expect.tail(1))


@pytest.mark.write_disk
def test_scan_iceberg_name_mapping_projections(tmp_path: Path) -> None:
    # Projections that leave out (nested) columns of files without field IDs, or
    # project no column of the file at all.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "s", StructType(NestedField(3, "x", StringType()))),
            NestedField(
                4,
                "m",
                MapType(
                    5,
                    StringType(),
                    6,
                    StructType(NestedField(7, "y", LongType())),
                    value_required=False,
                ),
            ),
            NestedField(
                8,
                "l",
                ListType(
                    9,
                    StructType(NestedField(10, "z", LongType())),
                    element_required=False,
                ),
            ),
        ),
    )
    path = tmp_path / "no-field-ids.parquet"
    pq.write_table(
        pa.table(
            {
                "a": pa.array([1, 2], pa.int64()),
                "s": pa.array([{"x": "p"}, {"x": "q"}]),
                "m": pa.array(
                    [[("k", {"y": 1})], []],
                    pa.map_(pa.string(), pa.struct([("y", pa.int64())])),
                ),
                "l": pa.array(
                    [[{"z": 1}], None], pa.list_(pa.struct([("z", pa.int64())]))
                ),
            }
        ),
        path,
    )
    tbl.add_files([format_file_uri_iceberg(path)])

    lf = pl.scan_iceberg(tbl, use_metadata_statistics=False)
    df = lf.collect()
    assert df.height == 2

    assert_frame_equal(lf.select("a").collect(), df.select("a"))
    assert_frame_equal(lf.drop("s").collect(), df.drop("s"))
    assert_frame_equal(lf.select("m").collect(), df.select("m"))
    assert_frame_equal(lf.select("l").collect(), df.select("l"))
    assert_frame_equal(lf.drop("l", "m").collect(), df.drop("l", "m"))
    assert lf.select(pl.len()).collect().item() == 2
    assert lf.with_row_index().select("index").collect().to_series().to_list() == [
        0,
        1,
    ]


@pytest.mark.write_disk
def test_scan_iceberg_name_mapping_of_dropped_columns_only(tmp_path: Path) -> None:
    # The name mapping only names a column that has been dropped, so it maps no
    # current column. It still exists: the file's unmapped columns are not read by
    # name, and the current column is missing from the file.
    tbl, catalog = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    path = tmp_path / "no-field-ids.parquet"
    pq.write_table(pa.table({"a": pa.array([1, 2], pa.int64())}), path)
    tbl.add_files([format_file_uri_iceberg(path)])
    with tbl.update_schema() as update:
        update.add_column("b", LongType())
        update.delete_column("a")
    with tbl.transaction() as tx:
        tx.set_properties(
            {
                "schema.name-mapping.default": json.dumps(
                    [{"field-id": 1, "names": ["a"]}]
                )
            }
        )
    tbl = catalog.load_table(tbl.name())

    expect = pl.DataFrame({"b": pl.Series([None, None], dtype=pl.Int64)})
    lf = pl.scan_iceberg(tbl)
    assert_frame_equal(lf.collect(), expect)
    assert_frame_equal(lf.filter(pl.col("b").is_null()).collect(), expect)


@pytest.mark.write_disk
def test_scan_iceberg_name_mapping_change_rescans(tmp_path: Path) -> None:
    # The name mapping can change without a new snapshot; collecting the same
    # LazyFrame again uses the new one.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", LongType())
        ),
    )
    path = tmp_path / "no-field-ids.parquet"
    pq.write_table(
        pa.table({"a": pa.array([1], pa.int64()), "b": pa.array([2], pa.int64())}),
        path,
    )
    tbl.add_files([format_file_uri_iceberg(path)])

    lf = pl.scan_iceberg(tbl)
    assert lf.collect().rows() == [(1, 2)]

    swapped = [{"field-id": 1, "names": ["b"]}, {"field-id": 2, "names": ["a"]}]
    with tbl.transaction() as tx:
        tx.set_properties({"schema.name-mapping.default": json.dumps(swapped)})
    tbl.refresh()

    assert lf.collect().rows() == [(2, 1)]


@pytest.mark.write_disk
def test_scan_iceberg_partial_field_ids(tmp_path: Path) -> None:
    # As in Iceberg Java, a file with field IDs is read by them; its fields without
    # one are missing.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", StringType()),
            NestedField(3, "s", StructType(NestedField(4, "x", LongType()))),
        ),
    )

    def field_id(field: pa.Field, id: int) -> pa.Field:
        return field.with_metadata({b"PARQUET:field_id": str(id).encode()})

    schema = pa.schema(
        [
            field_id(pa.field("a", pa.int64()), 1),
            pa.field("b", pa.string()),
            field_id(pa.field("s", pa.struct([pa.field("x", pa.int64())])), 3),
        ]
    )
    path = tmp_path / "partial-field-ids.parquet"
    pq.write_table(
        pa.table(
            {"a": [1, 2], "b": ["x", "y"], "s": [{"x": 1}, {"x": 2}]}, schema=schema
        ),
        path,
    )
    tbl.add_files([format_file_uri_iceberg(path)])

    assert_frame_equal(
        pl.scan_iceberg(tbl).collect(),
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": pl.Series([None, None], dtype=pl.String),
                "s": pl.Series([{"x": None}, {"x": None}]).cast(
                    pl.Struct({"x": pl.Int64})
                ),
            }
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_timestamp_millis_files(tmp_path: Path) -> None:
    # Iceberg Java reads TIMESTAMP_MILLIS columns, e.g. of files added with
    # `add_files`.
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "ts", TimestampType()))
    )
    path = tmp_path / "millis.parquet"
    pq.write_table(
        pa.table(
            {
                "ts": pa.array(
                    [datetime(2024, 1, 1, 12, 0, 0, 500000)], pa.timestamp("ms")
                )
            }
        ),
        path,
    )
    tbl.add_files([format_file_uri_iceberg(path)])

    assert pl.scan_iceberg(tbl).collect().to_series().to_list() == [
        datetime(2024, 1, 1, 12, 0, 0, 500000)
    ]


def _append_hand_built_data_file(
    tbl: pyiceberg.table.Table, table: pa.Table, **data_file_args: Any
) -> pyiceberg.table.Table:
    """Append `table` as a data file with metadata from `data_file_args`."""
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    local_path = Path(
        tbl.location().removeprefix("file://"), "data", f"{uuid.uuid4()}.parquet"
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, local_path)

    data_file = DataFile.from_args(
        **{
            "content": DataFileContent.DATA,
            "file_path": format_file_uri_iceberg(local_path),
            "file_format": FileFormat.PARQUET,
            "partition": Record(),
            "record_count": table.num_rows,
            "file_size_in_bytes": local_path.stat().st_size,
            **data_file_args,
        }
    )
    data_file.spec_id = tbl.spec().spec_id

    with tbl.transaction() as tx, tx.update_snapshot().fast_append() as append:
        append.append_data_file(data_file)

    return tbl.catalog.load_table(tbl.name())


@pytest.mark.write_disk
def test_scan_iceberg_unknown_record_count(tmp_path: Path) -> None:
    # Some format v1 writers write -1 for an unknown record count.
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "x", LongType())),
        properties={"format-version": "1"},
    )
    schema = pa.schema(
        [pa.field("x", pa.int64(), metadata={b"PARQUET:field_id": b"1"})]
    )
    tbl = _append_hand_built_data_file(
        tbl, pa.table({"x": [1, 2, 3]}, schema=schema), record_count=-1
    )

    lf = pl.scan_iceberg(tbl)
    assert lf.select(pl.len()).collect().item() == 3
    assert lf.collect().height == 3

    tbl = _append_hand_built_data_file(tbl, pa.table({"x": [4, 5]}, schema=schema))

    lf = pl.scan_iceberg(tbl)
    assert lf.select(pl.len()).collect().item() == 5
    assert lf.collect().height == 5


@pytest.mark.write_disk
def test_scan_iceberg_null_count_beyond_u32(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "x", LongType()), NestedField(2, "y", LongType())
        ),
    )
    schema = pa.schema(
        [
            pa.field("x", pa.int64(), metadata={b"PARQUET:field_id": b"1"}),
            pa.field("y", pa.int64(), metadata={b"PARQUET:field_id": b"2"}),
        ]
    )
    # Metadata as for a file with more rows than fit in a u32, of which all `x` are
    # null.
    n = 2**32 + 10
    tbl = _append_hand_built_data_file(
        tbl,
        pa.table({"x": [None, None], "y": [1, 2]}, schema=schema),
        record_count=n,
        null_value_counts={1: n, 2: 0},
    )

    lf = pl.scan_iceberg(tbl)
    assert lf.filter(pl.col("x").is_null()).select(
        "y"
    ).collect().to_series().to_list() == [1, 2]
    assert lf.filter(pl.col("y") > 0).select("y").collect().to_series().to_list() == [
        1,
        2,
    ]


@pytest.mark.write_disk
def test_scan_iceberg_metadata_file_removed_after_load(tmp_path: Path) -> None:
    # E.g. by `write.metadata.previous-versions-max` after newer commits.
    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    df = pl.DataFrame({"a": [1, 2]})
    tbl.append(df.to_arrow())
    Path(tbl.metadata_location.removeprefix("file:").removeprefix("//")).unlink()

    assert_frame_equal(pl.scan_iceberg(tbl).collect(), df)


@pytest.mark.write_disk
def test_scan_iceberg_reader_pyiceberg_negated_predicates_keep_nulls(
    tmp_path: Path,
) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "i", LongType()), NestedField(2, "s", StringType())
        ),
    )
    df = pl.DataFrame({"i": [1, 2, None], "s": ["ab", "cd", None]})
    tbl.append(df.to_arrow())

    lf = pl.scan_iceberg(tbl, reader_override="pyiceberg")
    for predicate in [
        ~pl.col("i").is_in([1, 3], nulls_equal=True),
        pl.col("i").ne_missing(1),
        ~pl.col("s").str.starts_with("a"),
    ]:
        assert_frame_equal(
            lf.filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.parametrize(
    ("dtype", "iceberg_type", "values"),
    [
        (pl.Int8, IntegerType(), [0, 1, -1, 127, -128]),
        (pl.Int32, IntegerType(), [0, 34, -1, 2**31 - 1, -(2**31)]),
        (pl.Int64, LongType(), [0, 34, -1, 2**63 - 1, -(2**63)]),
        (pl.UInt32, LongType(), [0, 34, 2**32 - 1]),
        (pl.UInt64, LongType(), [0, 34, 2**63 - 1]),
        (pl.Date, DateType(), [date(2017, 11, 16), date(1969, 12, 31), date(1, 1, 1)]),
        (pl.Time, TimeType(), [time(22, 31, 8), time(0), time(23, 59, 59, 999999)]),
        *[
            (
                dtype,
                TimestampType(),
                [datetime(2017, 11, 16, 22, 31, 8), datetime(1969, 12, 31, 23, 59)],
            )
            for dtype in [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
        ],
        (
            pl.Datetime("us", "UTC"),
            TimestamptzType(),
            [datetime(2017, 11, 16, 22, 31, 8, tzinfo=zoneinfo.ZoneInfo("UTC"))],
        ),
        (
            pl.Decimal(38, 2),
            DecimalType(38, 2),
            [
                D("14.20"),
                D("0"),
                D("-0.01"),
                D("1.27"),
                D("1.28"),
                D("-1.28"),
                D("-1.29"),
                D("999999999999999999999999999999999999.99"),
                D("-999999999999999999999999999999999999.99"),
            ],
        ),
        (pl.String, StringType(), ["iceberg", "", "a", "😁", "abcd", "abcde"]),
        (pl.Binary, BinaryType(), [b"\x00\x01\x02\x03", b"", b"\xff", b"abcdefg"]),
    ],
)
def test_mm3_hash_matches_iceberg_bucket_hash(
    dtype: pl.DataType, iceberg_type: Any, values: list[Any]
) -> None:
    hash_func = BucketTransform(16).transform(iceberg_type, bucket=False)

    s = pl.Series("a", [*values, None], dtype=dtype)
    expected = pl.Series("a", [*map(hash_func, values), None], dtype=pl.Int32)

    assert_series_equal(
        s.to_frame().select(pl.col("a")._mm3_hash()).to_series(), expected
    )
    assert_series_equal(
        s.to_frame().lazy().select(pl.col("a")._mm3_hash()).collect().to_series(),
        expected,
    )


def test_mm3_hash_timestamp_ns_truncates_to_micros() -> None:
    hash_func = BucketTransform(16).transform(TimestampNanoType(), bucket=False)
    nanos = [0, 1, 999, 1000, -1, -999, -1000, -1001, 1510871468123456789]

    s = pl.Series("a", nanos, dtype=pl.Int64).cast(pl.Datetime("ns"))
    expected = pl.Series("a", [hash_func(v) for v in nanos], dtype=pl.Int32)

    assert_series_equal(
        s.to_frame().select(pl.col("a")._mm3_hash()).to_series(), expected
    )


def test_mm3_hash_null_and_unsupported() -> None:
    assert_series_equal(
        pl.select(pl.lit(None)._mm3_hash()).to_series(),
        pl.Series("literal", [None], dtype=pl.Int32),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.select(pl.lit(1.0)._mm3_hash())

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.LazyFrame({"a": [1.0]}).select(pl.col("a")._mm3_hash()).collect_schema()

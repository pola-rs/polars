# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
from polars.io.iceberg._utils import _convert_predicate, _to_ast
from polars.testing import assert_frame_equal


@pytest.fixture
def iceberg_path(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return f"file://{iceberg_path.resolve()}"


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only
class TestIcebergScanIO:
    """Test coverage for `iceberg` scan ops."""

    def test_scan_iceberg_plain(self, iceberg_path: str) -> None:
        df = pl.scan_iceberg(iceberg_path)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id(self, iceberg_path: str) -> None:
        df = pl.scan_iceberg(iceberg_path, snapshot_id=7051579356916758811)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
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


@pytest.mark.ci_only
class TestIcebergExpressions:
    """Test coverage for `iceberg` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        from pyiceberg.expressions import IsNull

        expr = _to_ast("(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == IsNull("id")

    def test_is_not_null_expression(self) -> None:
        from pyiceberg.expressions import IsNull, Not

        expr = _to_ast("~(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == Not(IsNull("id"))

    def test_isin_expression(self) -> None:
        from pyiceberg.expressions import In, literal

        expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
        assert _convert_predicate(expr) == In(
            "id", {literal(1), literal(2), literal(3)}
        )

    def test_parse_combined_expression(self) -> None:
        from pyiceberg.expressions import (
            And,
            EqualTo,
            GreaterThan,
            In,
            Or,
            Reference,
            literal,
        )

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
        from pyiceberg.expressions import GreaterThan

        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        from pyiceberg.expressions import GreaterThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_lt(self) -> None:
        from pyiceberg.expressions import LessThan

        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        from pyiceberg.expressions import LessThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

    def test_compare_boolean(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(True))")
        assert _convert_predicate(expr) == EqualTo("ts", True)

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(False))")
        assert _convert_predicate(expr) == EqualTo("ts", False)


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings("ignore:Delete operation did not match any records")
@pytest.mark.filterwarnings(
    "ignore:Iceberg does not have a dictionary type. <class 'pyarrow.lib.DictionaryType'> will be inferred as large_string on read."
)
def test_write_iceberg(df: pl.DataFrame, tmp_path: Path) -> None:
    from pyiceberg.catalog.sql import SqlCatalog

    # time64[ns] type is currently not supported in pyiceberg.
    # https://github.com/apache/iceberg-python/issues/1169
    df = df.drop("time", "cat", "enum")

    # in-memory catalog
    catalog = SqlCatalog(
        "default", uri="sqlite:///:memory:", warehouse=f"file://{tmp_path}"
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


@pytest.mark.slow
@pytest.mark.write_disk
def test_sink_iceberg_no_partition(tmp_path: Path) -> None:
    import datetime

    from pyiceberg.catalog import load_catalog
    from pyiceberg.io.pyarrow import pyarrow_to_schema
    from pyiceberg.schema import NestedField
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table.name_mapping import MappedField, NameMapping
    from pyiceberg.types import StringType, TimestampType

    tmp_path.mkdir(parents=True, exist_ok=True)

    warehouse_path = str(tmp_path)
    catalog = load_catalog(
        "default",
        type="sql",
        uri="sqlite:///:memory:",
        warehouse=f"file://{warehouse_path}",
    )
    catalog.create_namespace("default")

    TABLE_NAME = "default.simple"

    df = pl.DataFrame(
        {
            "a": [
                datetime.datetime(2025, 1, 25),
                datetime.datetime(2025, 1, 27),
                datetime.datetime(2025, 1, 28),
            ],
            "b": ["x", "y", "z"],
        }
    )
    schema = df.to_arrow().schema

    array = []
    for fieldId, name in enumerate(schema.names):
        array.append(MappedField(field_id=fieldId + 1, names=[name]))

    name_mapping = NameMapping(array)
    icebergSchema = pyarrow_to_schema(
        schema, name_mapping, downcast_ns_timestamp_to_us=True
    )

    icebergSchema = IcebergSchema(
        NestedField(field_id=1, name="a", field_type=TimestampType(), required=False),
        NestedField(field_id=2, name="b", field_type=StringType(), required=False),
    )

    catalog.create_table(
        TABLE_NAME,
        schema=icebergSchema,
    )
    table = catalog.load_table(TABLE_NAME)

    assert_frame_equal(pl.scan_iceberg(table).collect(), df.clear())
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), pl.concat([df] * 2))
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), pl.concat([df] * 3))
    df.lazy().sink_iceberg(table, mode="overwrite")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)
    df.lazy().sink_iceberg(table, mode="overwrite")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)


@pytest.mark.slow
@pytest.mark.write_disk
def test_sink_iceberg_partition(tmp_path: Path) -> None:
    import datetime

    import pyiceberg.transforms as ts
    from pyiceberg.catalog import load_catalog
    from pyiceberg.io.pyarrow import pyarrow_to_schema
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import NestedField
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table.name_mapping import MappedField, NameMapping
    from pyiceberg.types import StringType, TimestampType

    tmp_path.mkdir(parents=True, exist_ok=True)

    warehouse_path = str(tmp_path)
    catalog = load_catalog(
        "default",
        type="sql",
        uri="sqlite:///:memory:",
        warehouse=f"file://{warehouse_path}",
    )
    catalog.create_namespace("default")

    TABLE_NAME = "default.simple"

    df = pl.DataFrame(
        {
            "a": [
                datetime.datetime(2025, 1, 25),
                datetime.datetime(2025, 1, 27),
                datetime.datetime(2025, 1, 28),
            ],
            "b": ["x", "y", "z"],
        }
    )
    schema = df.to_arrow().schema

    array = []
    for fieldId, name in enumerate(schema.names):
        array.append(MappedField(field_id=fieldId + 1, names=[name]))

    name_mapping = NameMapping(array)
    icebergSchema = pyarrow_to_schema(
        schema, name_mapping, downcast_ns_timestamp_to_us=True
    )

    icebergSchema = IcebergSchema(
        NestedField(field_id=1, name="a", field_type=TimestampType(), required=False),
        NestedField(field_id=2, name="b", field_type=StringType(), required=False),
    )

    partition_spec = PartitionSpec(
        PartitionField(
            source_id=1, field_id=1000, transform=ts.DayTransform(), name="a_day"
        )
    )
    catalog.create_table(
        TABLE_NAME,
        schema=icebergSchema,
        partition_spec=partition_spec,
    )
    table = catalog.load_table(TABLE_NAME)

    assert_frame_equal(pl.scan_iceberg(table).collect(), df.clear())
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), pl.concat([df] * 2))
    df.lazy().sink_iceberg(table, mode="append")
    assert_frame_equal(pl.scan_iceberg(table).collect(), pl.concat([df] * 3))
    df.lazy().sink_iceberg(table, mode="overwrite")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)
    df.lazy().sink_iceberg(table, mode="overwrite")
    assert_frame_equal(pl.scan_iceberg(table).collect(), df)

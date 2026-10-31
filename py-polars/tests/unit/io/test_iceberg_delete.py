from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    LongType,
    NestedField,
    StringType,
)

import polars as pl
from polars.exceptions import ColumnNotFoundError, InvalidOperationError
from polars.testing import assert_frame_equal
from tests.unit.io.test_iceberg import (  # noqa: F401
    new_iceberg_table,
    sqlcatalog_uses_null_pool_connections,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pyiceberg.table
    from pyiceberg.types import PrimitiveType

SCHEMA = IcebergSchema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "x", LongType()),
    NestedField(3, "y", LongType()),
    NestedField(4, "s", StringType()),
    NestedField(5, "d", DateType()),
    NestedField(6, "b", BooleanType()),
    NestedField(7, "f", DoubleType()),
)

DATA = pl.DataFrame(
    {
        "id": [0, 1, 2, 3, 4, 5],
        "x": [1, 2, None, 4, None, 1],
        "y": [2, 2, 2, None, 3, None],
        "s": ["apple", "banana", None, "avocado", "cherry", "apricot"],
        "d": [
            date(2020, 1, 1),
            date(2021, 1, 1),
            None,
            date(2022, 1, 1),
            date(2023, 1, 1),
            date(2024, 1, 1),
        ],
        "b": [True, False, None, True, False, None],
        "f": [1.5, float("nan"), None, -2.0, 3.0, float("nan")],
    },
    schema={
        "id": pl.Int64,
        "x": pl.Int64,
        "y": pl.Int64,
        "s": pl.String,
        "d": pl.Date,
        "b": pl.Boolean,
        "f": pl.Float64,
    },
)


def new_table(
    tmp_path: Path, partition_spec: PartitionSpec | None = None
) -> pyiceberg.table.Table:
    table, _ = new_iceberg_table(tmp_path, schema=SCHEMA, partition_spec=partition_spec)
    table.append(DATA.to_arrow().cast(schema_to_pyarrow(SCHEMA)))
    return table


def read(table: pyiceberg.table.Table) -> pl.DataFrame:
    table.refresh()
    return pl.scan_iceberg(table).sort("id").collect()


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("id").is_in([1, 2, 3]),
        pl.col("id") == 4,
        pl.col("x") != 1,
        ~(pl.col("x") == 1),
        ~pl.col("x").is_in([1, 2]),
        pl.col("x").is_in([1, None], nulls_equal=False),
        pl.col("x").is_null(),
        pl.col("x").is_not_null(),
        pl.col("x").eq_missing(1),
        pl.col("x").ne_missing(1),
        pl.col("x").is_between(2, 4),
        (pl.col("x") > 1) | (pl.col("x").is_null() & (pl.col("y") == 2)),
        (pl.col("x") > 1) | (pl.col("x").is_null() & (pl.col("y") == 3)),
        ~((pl.col("x") > 1) | pl.col("y").is_null()),
        (pl.col("x") == 1) ^ (pl.col("y") == 2),
        pl.col("s").str.starts_with("ap"),
        ~pl.col("s").str.starts_with("ap"),
        pl.col("d") >= date(2022, 1, 1),
        pl.col("b"),
        ~pl.col("b"),
        pl.col("f") < 2.0,
        pl.col("f") == 3.0,
        pl.col("f").is_nan(),
        ~pl.col("f").is_nan(),
        pl.col("f").is_in([1.5, -2.0]),
        pl.col("f").is_null(),
        pl.col("f").is_not_null(),
        ~pl.col("f").is_not_null(),
        (pl.col("f") < 2.0) | (pl.col("x") == 1),
        # Top-level conjunctions are split into multiple filters.
        (pl.col("x") == 1) & (pl.col("s") == "apple"),
        (pl.col("x") > 1) & (pl.col("y") == 2) & pl.col("d").is_not_null(),
        pl.lit(True),
        pytest.param(
            pl.lit(False),
            marks=pytest.mark.filterwarnings(
                "ignore:Delete operation did not match any records"
            ),
        ),
    ],
)
def test_delete_iceberg(tmp_path: Path, predicate: pl.Expr) -> None:
    table = new_table(tmp_path)
    pl.delete_iceberg(table, predicate)
    assert_frame_equal(read(table), DATA.remove(predicate))


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("x") + 1 > 2,
        pl.col("x") > pl.col("y"),
        pl.col("s").str.contains("an"),
        pl.col("s").str.len_bytes() > 3,
        pl.col("f") > 1.0,
        pl.col("f") != 1.0,
        ~(pl.col("f") < 1.0),
        pl.col("f") == 0.0,
        pl.col("f").is_in([0.0]),
        pl.col("x").is_in([1, None], nulls_equal=True),
        ~pl.col("x").is_in([1, 2], nulls_equal=True),
        ~pl.col("x").is_in(pl.lit([], dtype=pl.List(pl.Int64))),
        pl.col("x") == 1.5,
    ],
)
def test_delete_iceberg_inexact_predicate(tmp_path: Path, predicate: pl.Expr) -> None:
    table = new_table(tmp_path)
    snapshot_id = table.metadata.current_snapshot_id

    with pytest.raises(InvalidOperationError, match="cannot convert predicate"):
        pl.delete_iceberg(table, predicate)

    table.refresh()
    assert table.metadata.current_snapshot_id == snapshot_id
    assert_frame_equal(read(table), DATA)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("field_type", "dtype", "values", "predicate", "exact"),
    [
        # Polars compares as `Float64`; PyIceberg rounds the literal to 1.2.
        (DecimalType(10, 1), pl.Decimal(10, 1), [1.2, 1.3], pl.col("v") == 1.24, False),
        # Polars casts the column to `Float64`.
        (
            FloatType(),
            pl.Float32,
            [0.1, 1.0],
            pl.col("v") <= pl.lit(0.1, dtype=pl.Float64),
            False,
        ),
        (FloatType(), pl.Float32, [0.1, 1.0], pl.col("v") <= 0.1, True),
    ],
)
def test_delete_iceberg_type_coercion(
    tmp_path: Path,
    field_type: PrimitiveType,
    dtype: pl.DataType,
    values: list[float],
    predicate: pl.Expr,
    exact: bool,
) -> None:
    schema = IcebergSchema(NestedField(1, "v", field_type))
    df = pl.DataFrame({"v": values}, schema={"v": dtype})
    table, _ = new_iceberg_table(tmp_path, schema=schema)
    table.append(df.to_arrow().cast(schema_to_pyarrow(schema)))

    if exact:
        pl.delete_iceberg(table, predicate)
        expected = df.remove(predicate)
    else:
        with pytest.raises(InvalidOperationError, match="cannot convert predicate"):
            pl.delete_iceberg(table, predicate)
        expected = df

    table.refresh()
    assert_frame_equal(pl.scan_iceberg(table).collect(), expected)


@pytest.mark.write_disk
def test_delete_iceberg_invalid_predicate(tmp_path: Path) -> None:
    table = new_table(tmp_path)

    with pytest.raises(ColumnNotFoundError):
        pl.delete_iceberg(table, pl.col("missing") == 1)

    with pytest.raises(InvalidOperationError, match="boolean"):
        pl.delete_iceberg(table, pl.col("x"))

    with pytest.raises(TypeError, match="Expr"):
        pl.delete_iceberg(table, "x = 1")  # type: ignore[arg-type]


@pytest.mark.write_disk
def test_delete_iceberg_whole_partition(tmp_path: Path) -> None:
    spec = PartitionSpec(
        PartitionField(
            source_id=6, field_id=1000, transform=IdentityTransform(), name="b"
        )
    )
    table = new_table(tmp_path, spec)

    pl.delete_iceberg(table, pl.col("b"))

    table.refresh()
    snapshot = table.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary is not None
    # Metadata-only: no data file is rewritten.
    assert snapshot.summary.operation.value == "delete"
    assert_frame_equal(read(table), DATA.remove(pl.col("b")))

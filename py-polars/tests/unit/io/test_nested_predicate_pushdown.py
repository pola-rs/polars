from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def check(query: pl.LazyFrame) -> None:
    expected = query.collect(
        optimizations=pl.QueryOptFlags(
            predicate_pushdown=False, projection_pushdown=False
        )
    )
    for engine in ("streaming", "in-memory"):
        assert_frame_equal(query.collect(engine=engine), expected)


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_nested_leaf_definition_levels(
    tmp_path: Path, version: str, dictionary: bool, required: bool, depth: int
) -> None:
    # Required physical leaves beneath nullable ancestors need a validity bitmap
    # even though their own parquet repetition is REQUIRED.
    dtype = pa.struct(
        [
            pa.field("value", pa.int64(), nullable=not required),
            pa.field("text", pa.string()),
            pa.field("list", pa.list_(pa.int64())),
        ]
    )
    values = [
        {
            "value": i if required or i % 7 else None,
            "text": f"v{i % 11}",
            "list": [i, i + 1],
        }
        for i in range(513)
    ]
    for level in range(depth):
        values = [
            None if i % (13 + level) == 0 else {"a": value}
            for i, value in enumerate(values)
        ]
        dtype = pa.struct([pa.field("a", dtype)])
    table = pa.table({"s": pa.array(values, type=dtype), "other": pa.array(range(513))})
    path = tmp_path / "nested.parquet"
    pq.write_table(
        table,
        path,
        data_page_version=version,
        use_dictionary=dictionary,
        row_group_size=129,
        data_page_size=128,
        write_batch_size=32,
    )
    leaf = pl.col("s")
    for _ in range(depth):
        leaf = leaf.struct.field("a")
    value = leaf.struct.field("value")
    text = leaf.struct.field("text")
    lf = pl.scan_parquet(path, use_statistics=False, parallel="prefiltered")
    predicates = [
        value > 250,
        value.is_null(),
        value.is_not_null(),
        value == 42,
        value.is_between(100, 201),
        value.is_in([4, 42, 150]),
        text.str.starts_with("v1"),
        text.str.contains("[23]"),
        (value + value) > 500,
        (value + text.str.len_bytes()) > 500,
        (value > 10) & (text == "v2"),
        (value > 10) & (value < 250),
        (value > 10) & pl.col("s").is_not_null(),
        (value > 10) & ((value + pl.col("other")) > 700),
    ]
    for predicate in predicates:
        for selection in [pl.col("s"), pl.col("other"), text]:
            check(lf.filter(predicate).select(selection))
    check(lf.filter(value > 250).head(8))
    check(lf.with_row_index().filter(value > 250).select("index", "s"))


def test_nested_leaf_missing_and_cast_fallback(tmp_path: Path) -> None:
    dtype = pl.Struct({"a": pl.Struct({"v": pl.Int32, "text": pl.String})})
    pl.DataFrame(
        {"s": [{"a": {"v": 1, "text": "x"}}, None, {"a": None}]}, schema={"s": dtype}
    ).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame(
        {"s": [{"a": {"text": "y"}}, None, {"a": None}]},
        schema={"s": pl.Struct({"a": pl.Struct({"text": pl.String})})},
    ).write_parquet(tmp_path / "b.parquet")
    pl.DataFrame({"other": [0, 1]}).write_parquet(tmp_path / "c.parquet")
    schema = {"s": pl.Struct({"a": pl.Struct({"v": pl.Int64, "text": pl.String})})}
    lf = pl.scan_parquet(
        tmp_path / "*.parquet",
        schema=schema,
        cast_options=pl.ScanCastOptions(
            integer_cast="upcast", missing_struct_fields="insert"
        ),
        missing_columns="insert",
        extra_columns="ignore",
        use_statistics=False,
        parallel="prefiltered",
    )
    v = pl.col("s").struct.field("a").struct.field("v")
    for pred in [v.is_null(), v.is_not_null(), v == 1]:
        check(lf.filter(pred))


def test_nested_leaf_all_null_and_no_null_groups(tmp_path: Path) -> None:
    dtype = pl.Struct({"a": pl.Struct({"v": pl.Int64, "sibling": pl.Int64})})
    values = [None] * 64 + [{"a": None}] * 64
    values += [{"a": {"v": i, "sibling": i * 2}} for i in range(128)]
    path = tmp_path / "null-groups.parquet"
    pl.DataFrame({"s": values}, schema={"s": dtype}).write_parquet(
        path, row_group_size=64
    )
    lf = pl.scan_parquet(path, use_statistics=False, parallel="prefiltered")
    v = pl.col("s").struct.field("a").struct.field("v")
    for predicate in [v.is_null(), v.is_not_null(), v == 42, v > 10]:
        check(lf.filter(predicate))


@pytest.mark.parametrize("parallel", ["auto", "prefiltered"])
@pytest.mark.parametrize("width", [2, 10])
@pytest.mark.parametrize("lists", [False, True])
def test_nested_leaf_root_fallback_coalescing(
    tmp_path: Path, parallel: str, width: int, lists: bool
) -> None:
    i = pl.int_range(100)
    fields = [(i + k).alias(f"v{k}") for k in range(width)]
    if lists:
        fields.append(pl.concat_list(i, i + 1).alias("list"))
    path = tmp_path / "coalesce.parquet"
    pl.select(
        s=pl.when(i % 7 == 0).then(None).otherwise(pl.struct(a=pl.struct(fields))),
        other=i,
    ).write_parquet(path)
    lf = pl.scan_parquet(path, parallel=parallel, use_statistics=False)
    a = pl.col("s").struct.field("a")
    # Multiple fallback paths, an original whole-root predicate, and a rest
    # conjunct must each survive without decoding the root more than once.
    check(
        lf.filter(
            (a.struct.field("v0") > 10)
            & (a.struct.field("v1") < 60)
            & pl.col("s").is_not_null()
        )
    )
    check(
        lf.filter(
            (a.struct.field("v0") > 10)
            & ((a.struct.field("v1") + pl.col("other")) > 50)
        )
    )


def test_nested_leaf_rejects_rows_before_decoding_sibling(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.parquet"
    pl.DataFrame(
        {"s": [{"a": {"v": i, "garbage": f"data{i}"}} for i in range(100)]}
    ).write_parquet(path)
    md = pq.ParquetFile(path).metadata.row_group(0)
    col = next(
        md.column(i)
        for i in range(md.num_columns)
        if md.column(i).path_in_schema == "s.a.garbage"
    )
    start = min(
        col.dictionary_page_offset or col.data_page_offset, col.data_page_offset
    )
    with path.open("r+b") as file:
        file.seek(start)
        file.write(b"X" * col.total_compressed_size)
    # Projecting the sibling requires its bytes; a leaf predicate rejecting the
    # entire row group must avoid deserializing its corrupt pages.
    q = (
        pl.scan_parquet(path, use_statistics=False, parallel="prefiltered")
        .filter(pl.col("s").struct.field("a").struct.field("v") > 1000)
        .select(pl.col("s").struct.field("a").struct.field("garbage"))
    )
    assert q.collect(engine="streaming").height == 0


@pytest.mark.parametrize("kind", ["bool", "float", "binary", "decimal", "date", "enum"])
def test_nested_leaf_physical_types(tmp_path: Path, kind: str) -> None:
    from datetime import date
    from decimal import Decimal

    series = {
        "bool": pl.Series("v", [True, False, None, True]),
        "float": pl.Series("v", [1.5, float("nan"), None, -2.0]),
        "binary": pl.Series("v", [b"a", b"b", None, b"a"]),
        "decimal": pl.Series(
            "v",
            [Decimal("1.25"), Decimal("2.50"), None, Decimal("1.25")],
            dtype=pl.Decimal(12, 2),
        ),
        "date": pl.Series(
            "v", [date(2020, 1, 1), date(2021, 1, 1), None, date(2020, 1, 1)]
        ),
        "enum": pl.Series("v", ["a", "b", None, "a"], dtype=pl.Enum(["a", "b", "c"])),
    }[kind]
    df = (
        series.to_frame()
        .with_row_index("i")
        .select(
            s=pl.when(pl.col("i") == 3)
            .then(None)
            .otherwise(pl.struct(a=pl.struct("v", sibling=pl.col("i")))),
            i=pl.col("i"),
        )
    )
    path = tmp_path / f"{kind}.parquet"
    df.write_parquet(path, row_group_size=2)
    lf = pl.scan_parquet(path, parallel="prefiltered", use_statistics=False)
    v = pl.col("s").struct.field("a").struct.field("v")
    for predicate in [
        v == pl.lit(series[0], dtype=series.dtype),
        v.is_null(),
        v.is_not_null(),
    ]:
        check(lf.filter(predicate))
        check(lf.filter(predicate).select("i"))

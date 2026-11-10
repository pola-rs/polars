from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import JoinStrategy
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


def scan(
    tmp_path: Path, name: str, df: pl.DataFrame, row_group_size: int = 100
) -> pl.LazyFrame:
    path = tmp_path / f"{name}.parquet"
    df.write_parquet(path, row_group_size=row_group_size, statistics="full")
    return pl.scan_parquet(path)


def keys(n: int, offset: int = 0, *, b_offset: int = 0) -> pl.DataFrame:
    i = pl.int_range(offset, offset + n, dtype=pl.Int64)
    return pl.select(
        a=pl.when(i % 11 != 0).then(i % 1000),
        b=pl.when(i % 13 != 0).then(i % 37 - 18 + b_offset),
        v=i,
    )


def packed_layouts(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, list[str]]:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    return out, re.findall(r"packed keys: (.*)", err)


def assert_engines_equal(out: pl.DataFrame, q: pl.LazyFrame) -> None:
    assert_frame_equal(out, q.collect(engine="in-memory"), check_row_order=False)


def test_group_by_packs_key_ranges(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    lf = scan(tmp_path, "keys", keys(2000))
    q = lf.group_by("a", "b").agg(pl.len(), pl.col("v").sum())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    # 1000 values and 37 values, each plus the null code.
    assert layouts == ["u32 [0..10) range of 1000 [10..16) range of 37"]
    assert_engines_equal(out, q)


def test_group_by_duplicate_keys_pack_key_ranges(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    lf = scan(tmp_path, "keys", keys(2000))
    q = lf.group_by(pl.col("a").alias("x"), pl.col("a").alias("y"), "b").agg(pl.len())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert layouts == ["u32 [0..10) range of 1000 [10..16) range of 37"]
    assert_engines_equal(out, q)


def test_group_by_hive_column_has_no_file_range(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    part = tmp_path / "a=1000"
    part.mkdir()
    pl.select(a=pl.int_range(2000) % 3, b=pl.int_range(2000) % 7).write_parquet(
        part / "x.parquet", statistics="full", row_group_size=100
    )
    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    q = lf.group_by("a", "b").agg(pl.len())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    # The hive value replaces `a`, so the file statistics of `a` don't hold.
    assert layouts == ["u128 [0..64) [65..68) range of 7"]
    assert_engines_equal(out, q)


def test_unique_packs_key_ranges(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    lf = scan(tmp_path, "keys", keys(2000))
    q = lf.unique(["a", "b"], keep="first")
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert any("range" in layout for layout in layouts)
    assert_engines_equal(out, q)


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_join_packs_key_ranges(
    how: JoinStrategy,
    nulls_equal: bool,
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The sides have different ranges, both are packed with their union.
    left = scan(tmp_path, "left", keys(600))
    right = scan(tmp_path, "right", keys(400, offset=500, b_offset=5))
    q = left.join(right, on=["a", "b"], how=how, nulls_equal=nulls_equal)
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert len(layouts) == 1
    # The left keys are 1..=599, the right ones 500..=899.
    assert layouts == ["u32 [0..10) range of 899 [10..16) range of 42"]
    assert_engines_equal(out, q)


def test_join_key_without_range_is_full_width(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    left = scan(tmp_path, "left", keys(600))
    right = keys(400, offset=500).lazy()
    q = left.join(right, on=["a", "b"])
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert layouts == ["u128 [0..64) [64..128)"]
    assert_engines_equal(out, q)


def test_full_join_coalesced_key_has_both_ranges(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    left = scan(tmp_path, "left", pl.DataFrame({"a": [0, 5, 10], "x": [1, 2, 3]}))
    right = scan(tmp_path, "right", pl.DataFrame({"a": [5, 100, 110], "y": [4, 5, 6]}))
    joined = left.join(right, on="a", how="full", coalesce=True)
    q = joined.group_by("a", "x").agg(pl.col("y").sum())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert "u32 [0..7) range of 111 [7..9) range of 3" in layouts
    assert_engines_equal(out, q)


def test_computed_key_has_no_range(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    lf = scan(tmp_path, "keys", keys(2000))
    q = lf.with_columns(pl.col("a") * 1000).group_by("a", "b").agg(pl.col("v").sum())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    # The null bit of `a` follows its 64 bits.
    assert layouts == ["u128 [0..64) [65..71) range of 37"]
    assert_engines_equal(out, q)


def test_sampled_range_is_not_used(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # More row groups than the planner reads statistics of, so the ranges are
    # estimates.
    lf = scan(tmp_path, "keys", keys(3000), row_group_size=1)
    q = lf.group_by("a", "b").agg(pl.col("v").sum())
    out, layouts = packed_layouts(q, plmonkeypatch, capfd)
    assert layouts == []
    assert_engines_equal(out, q)


@pytest.mark.parametrize(
    "dtypes",
    [
        [pl.Boolean] * 8,
        [pl.Int32, pl.Int16],
        [pl.Int8, pl.UInt16, pl.Date],
        [pl.Float32, pl.Boolean],
    ],
)
@pytest.mark.parametrize("nulls", [False, True])
def test_full_width_packed_keys(dtypes: list[pl.DataType], nulls: bool) -> None:
    n = 500
    i = pl.int_range(n)
    cols: dict[str, pl.Expr] = {}
    for c, dtype in enumerate(dtypes):
        value = (i * (c + 3) % 7 - 3).cast(pl.Int64)
        if dtype == pl.Boolean:
            col = (i // (c + 1)) % 2 == 0
        elif dtype == pl.Float32:
            col = pl.when(value == 0).then(-0.0).otherwise(value).cast(pl.Float32)
        elif dtype in (pl.UInt16, pl.Date):
            col = (value + 3).cast(dtype)
        else:
            col = value.cast(dtype)
        if nulls:
            col = pl.when(i % (5 + c) != 0).then(col)
        cols[f"c{c}"] = col
    lf = pl.select(*(c.alias(name) for name, c in cols.items()), i.alias("v")).lazy()
    names = list(cols)
    for q in [
        lf.group_by(names).agg(pl.len(), pl.col("v").sum()),
        lf.unique(names),
        lf.join(lf.head(50), on=names, how="inner"),
        lf.join(lf.head(50), on=names, how="anti", nulls_equal=True),
    ]:
        assert_engines_equal(q.collect(engine="streaming"), q)

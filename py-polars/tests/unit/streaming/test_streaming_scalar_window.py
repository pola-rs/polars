from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.fixture(autouse=True)
def small_morsels(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "100")


def _frame(n: int = 3_000) -> pl.DataFrame:
    i = pl.col("id")
    return pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        g=pl.when(i % 101 == 0).then(None).otherwise(i % 97),
        h=(i % 13).cast(pl.String),
        k=i % 5,
        x=pl.when(i % 7 == 0).then(None).otherwise((i * 104729) % 1_000).cast(pl.Int64),
        y=((i * 7919) % 500).cast(pl.Float64),
        z=pl.when(i % 3 == 0).then(None).otherwise(i % 11).cast(pl.Int8),
        n=pl.lit(None, dtype=pl.Int32),
    )


def _scan(df: pl.DataFrame, path: Path, parts: int = 6) -> pl.LazyFrame:
    size = df.height // parts + 1
    for i in range(parts):
        df.slice(i * size, size).write_parquet(path / f"{i:02}.parquet")
    return pl.scan_parquet(path / "*.parquet")


def _physical_windows(q: pl.LazyFrame) -> list[str]:
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    return [line for line in dot.splitlines() if "window[" in line]


def _is_scalar(q: pl.LazyFrame) -> bool:
    windows = _physical_windows(q)
    assert len(windows) == 1
    return "scalar-window[" in windows[0]


def _assert_same(q: pl.LazyFrame, *, check_row_order: bool = True) -> None:
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=check_row_order,
        check_exact=False,
    )


@pytest.mark.parametrize(
    "exprs",
    [
        [pl.col("x").sum().over("g").alias("w")],
        [pl.col("x").mean().over("g").alias("w")],
        [pl.col("y").min().over("h").alias("w")],
        [pl.col("y").max().over("h").alias("w")],
        [pl.col("x").count().over("g").alias("w")],
        [pl.len().over("g").alias("w")],
        [pl.col("z").sum().over("k").alias("w")],
        [pl.col("n").sum().over("g").alias("w")],
        [pl.col("n").mean().over("g").alias("w")],
        [pl.col("x").sum().over("g", "h").alias("w")],
        [pl.col("x").sum().over("g", "h", "k").alias("w")],
        [
            pl.col("x").sum().over("g").alias("a"),
            pl.col("y").mean().over("g").alias("b"),
            pl.col("z").min().over("g").alias("c"),
            pl.col("x").max().over("g").alias("d"),
            pl.col("x").count().over("g").alias("e"),
            pl.len().over("g").alias("f"),
        ],
        [(pl.col("y") - pl.col("y").mean().over("k")).alias("w")],
        [(pl.col("x").sum().over("g") / pl.col("x").count().over("g")).alias("w")],
    ],
)
def test_scalar_window_matches_in_memory(tmp_path: Path, exprs: list[pl.Expr]) -> None:
    q = _scan(_frame(), tmp_path).with_columns(exprs)
    assert _is_scalar(q)
    _assert_same(q)


def test_scalar_window_from_in_memory_source() -> None:
    q = _frame().lazy().with_columns(w=pl.col("x").sum().over("g"))
    assert _is_scalar(q)
    _assert_same(q)


def test_scalar_window_two_specs(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(
        a=pl.col("x").sum().over("g"),
        b=pl.col("x").sum().over("h"),
        c=pl.col("y").mean().over("g"),
    )
    windows = _physical_windows(q)
    assert len(windows) == 2
    assert all("scalar-window[" in w for w in windows)
    _assert_same(q)


def test_scalar_window_keeps_all_columns_and_order(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=pl.col("y").sum().over("k"))
    out = q.collect(engine="streaming")
    assert out.columns == [*_frame().columns, "w"]
    assert out["id"].to_list() == list(range(3_000))
    _assert_same(q)


@pytest.mark.parametrize(
    "consumer",
    [
        lambda lf: lf.select(pl.col("w").sum(), pl.col("id").sum()),
        lambda lf: lf.group_by("k").agg(pl.col("w").first(), pl.col("id").sum()),
        lambda lf: lf.sort("id"),
    ],
)
def test_scalar_window_unordered_consumer(
    tmp_path: Path, consumer: Callable[[pl.LazyFrame], pl.LazyFrame]
) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=pl.col("x").sum().over("k"))
    q = consumer(q)
    assert _is_scalar(q)
    _assert_same(q, check_row_order=False)


def test_scalar_window_single_group(tmp_path: Path) -> None:
    q = (
        _scan(_frame(), tmp_path)
        .with_columns(o=pl.lit(1))
        .with_columns(w=pl.col("y").sum().over("o"))
    )
    assert _is_scalar(q)
    _assert_same(q)


def test_scalar_window_empty() -> None:
    q = (
        _frame()
        .lazy()
        .filter(pl.col("id") < 0)
        .with_columns(w=pl.col("x").sum().over("g"), v=pl.col("y").mean().over("g"))
    )
    out = q.collect(engine="streaming")
    assert out.height == 0
    assert_frame_equal(out, q.collect(engine="in-memory"))


def test_scalar_window_stops_early(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=pl.col("x").sum().over("k")).head(5)
    _assert_same(q)


def test_scalar_window_all_null_group() -> None:
    df = pl.DataFrame(
        {"g": [1, 1, 2, 2, 3, None], "x": [None, None, 1, None, 5, 7]},
        schema={"g": pl.Int64, "x": pl.Int64},
    )
    q = df.lazy().with_columns(
        s=pl.col("x").sum().over("g"),
        m=pl.col("x").mean().over("g"),
        c=pl.col("x").count().over("g"),
        n=pl.len().over("g"),
    )
    assert _is_scalar(q)
    _assert_same(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").median().over("g"),
        pl.col("x").n_unique().over("g"),
        pl.col("x").first().over("g"),
        pl.col("x").last().over("g"),
        pl.col("x").sum().over("g", order_by="id"),
        pl.col("x").filter(pl.col("y") > 100).sum().over("g"),
        (pl.col("x") - pl.col("x").mean()).over("g"),
        pl.col("x").cum_sum().over("g"),
        pl.col("x").std().over("g"),
        pl.col("x").sum().over("g", mapping_strategy="join"),
        pl.col("h").first().over("g"),
        pl.col("h").count().over("g"),
        (pl.col("x") + 1).sum().over("g"),
    ],
)
def test_scalar_window_fallbacks(tmp_path: Path, expr: pl.Expr) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=expr)
    windows = _physical_windows(q)
    assert not any("scalar-window[" in w for w in windows)
    _assert_same(q)


def test_scalar_window_mixed_spec_falls_back(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(
        a=pl.col("x").sum().over("g"), b=pl.col("x").cum_sum().over("g")
    )
    windows = _physical_windows(q)
    assert len(windows) == 1
    assert "scalar-window[" not in windows[0]
    _assert_same(q)


def test_scalar_window_first_last_across_morsels(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(
        f=pl.col("y").first().over("k"), l=pl.col("y").last().over("k")
    )
    _assert_same(q)


@pytest.mark.parametrize("groups", [10, 70_000, 100_000])
def test_scalar_window_many_groups(tmp_path: Path, groups: int) -> None:
    n = 100_000
    df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        k=(pl.col("id") * 7919) % groups,
        j=pl.col("id") % 3,
        x=pl.when(pl.col("id") % 5 == 0).then(None).otherwise(pl.col("id") % 1_000),
    )
    q = _scan(df, tmp_path).with_columns(
        a=pl.col("x").sum().over("k"),
        b=pl.col("x").mean().over("k"),
        c=pl.len().over("k"),
    )
    assert _is_scalar(q)
    _assert_same(q)
    q = _scan(df, tmp_path).with_columns(a=pl.col("x").max().over("k", "j"))
    _assert_same(q)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ],
)
def test_scalar_window_value_dtypes(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"k": [1, 2, 1, 2, 3, 1], "x": [1, None, 3, 4, None, 6]},
        schema={"k": pl.Int64, "x": dtype},
    )
    q = df.lazy().with_columns(
        s=pl.col("x").sum().over("k"),
        m=pl.col("x").mean().over("k"),
        lo=pl.col("x").min().over("k"),
        hi=pl.col("x").max().over("k"),
        c=pl.col("x").count().over("k"),
    )
    assert _is_scalar(q)
    _assert_same(q)


@pytest.mark.parametrize("groups", [7, 70_000])
@pytest.mark.parametrize(
    "key",
    [
        pl.col("i").cast(pl.String),
        pl.col("i").cast(pl.String).cast(pl.Categorical),
        pl.col("i").cast(pl.Decimal(10, 2)),
        pl.col("i").cast(pl.Float64) / 2 - 1,
        pl.col("i").cast(pl.Date),
        pl.col("i") % 2 == 0,
        pl.struct(a=pl.col("i"), b=pl.col("i").cast(pl.String)),
        pl.concat_list(pl.col("i"), pl.col("i") % 3),
    ],
)
def test_scalar_window_key_dtypes(tmp_path: Path, key: pl.Expr, groups: int) -> None:
    n = 100_000
    df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        i=pl.when(pl.col("id") % 11 == 0).then(None).otherwise(pl.col("id") % groups),
        x=pl.col("id") % 1_000,
    )
    q = (
        _scan(df, tmp_path)
        .with_columns(k=key)
        .with_columns(s=pl.col("x").sum().over("k"), c=pl.len().over("k"))
    )
    assert _is_scalar(q)
    _assert_same(q)

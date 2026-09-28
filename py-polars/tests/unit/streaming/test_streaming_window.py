from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.xdist_group("streaming")

ENGINES = ["in-memory", "streaming"]


def _frame(n: int = 20_000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.int_range(n, eager=True),
            "k": pl.int_range(n, eager=True) % 5_003,
            "g": pl.int_range(n, eager=True) % 97,
            "h": pl.int_range(n, eager=True) % 13,
            "x": (pl.int_range(n, eager=True) * 7919) % 1_000,
        }
    )


def _write_parts(df: pl.DataFrame, path: Path, parts: int = 8) -> pl.LazyFrame:
    size = df.height // parts + 1
    for i in range(parts):
        df.slice(i * size, size).write_parquet(path / f"{i:02}.parquet")
    return pl.scan_parquet(path / "*.parquet")


@pytest.mark.parametrize("engine", ENGINES)
def test_stacked_windows_over_group_by_evaluate_in_same_order(engine: str) -> None:
    q = (
        _frame()
        .lazy()
        .group_by("k")
        .agg(pl.col("g").first(), pl.col("h").first(), pl.col("x").sum())
        .with_columns(w1=pl.col("x").cum_sum().over("g"))
        .with_columns(w2=pl.col("w1").cum_sum().over("h"))
        .with_columns(w3=pl.col("x").cum_sum().over("g") + pl.col("w2") * 0)
    )
    out = q.collect(engine=engine)  # type: ignore[call-overload]
    assert_series_equal(out["w1"], out["w3"], check_names=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_window_in_cached_branch_stays_aligned(engine: str) -> None:
    base = (
        _frame()
        .lazy()
        .group_by("k")
        .agg(pl.col("g").first(), pl.col("x").sum())
        .cache()
    )
    left = base.select("k", c=pl.col("x").cum_sum().over("g"))
    right = base.select(k2="k", g2="g", x2="x")
    out = pl.concat([left, right], how="horizontal").collect(engine=engine)  # type: ignore[call-overload]

    assert_series_equal(out["k"], out["k2"], check_names=False)
    expected = out.select(pl.col("x2").cum_sum().over("g2"))
    assert_series_equal(out["c"], expected.to_series(), check_names=False)


def test_window_with_order_dependent_key(tmp_path: Path) -> None:
    lf = _write_parts(_frame(), tmp_path)
    key = (pl.col("h") == 0).cum_sum()

    q = lf.with_columns(s=pl.col("x").sum().over(key))
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))

    q = q.group_by("s").agg(pl.col("id").sum()).sort("s")
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("reverse", [False, True])
def test_window_seeded_random_rank(reverse: bool) -> None:
    df = _frame().with_columns(pl.col("x") // 100)
    if reverse:
        df = df.reverse()

    q = df.lazy().with_columns(r=pl.col("x").rank("random", seed=1).over("g"))
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_seeded_random_rank_unordered_consumer(tmp_path: Path) -> None:
    lf = _write_parts(_frame().with_columns(pl.col("x") // 100), tmp_path)

    q = (
        lf.with_columns(r=pl.col("x").rank("random", seed=1).over("g"))
        .group_by("g", "r")
        .agg(pl.col("id").sum())
        .sort("g", "r")
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))

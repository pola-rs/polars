from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def lopsided(tmp_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """A large and a small scan whose row counts the parquet footer guarantees."""
    tmp_path.mkdir(exist_ok=True)
    big = pl.DataFrame({"k": range(100_000), "v": [1] * 100_000})
    small = pl.DataFrame({"k": range(100), "w": [2] * 100})
    big.write_parquet(tmp_path / "big.parquet")
    small.write_parquet(tmp_path / "small.parquet")
    return (
        pl.scan_parquet(tmp_path / "big.parquet"),
        pl.scan_parquet(tmp_path / "small.parquet"),
    )


def test_build_side_is_the_smaller_scan(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    big, small = lopsided
    assert "BUILD SIDE: PreferRight" in big.join(small, on="k").explain()
    assert "BUILD SIDE: PreferLeft" in small.join(big, on="k").explain()

    assert big.join(small, on="k").collect().height == 100
    assert small.join(big, on="k").collect().height == 100


def test_build_side_survives_a_filter_on_the_large_side(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    # The filter's selectivity is unknown, but it can only remove rows, so the
    # small side is still bounded well below the large one.
    big, small = lopsided
    q = big.filter(pl.col("v") > 0).join(small, on="k")
    assert "BUILD SIDE: PreferRight" in q.explain()
    assert q.collect().height == 100


def test_no_build_side_for_similar_sizes(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    for name in ("a", "b"):
        pl.DataFrame({"k": range(1_000)}).write_parquet(tmp_path / f"{name}.parquet")
    a = pl.scan_parquet(tmp_path / "a.parquet")
    b = pl.scan_parquet(tmp_path / "b.parquet")
    assert "BUILD SIDE" not in a.join(b, on="k").explain()


def test_no_build_side_when_a_side_is_unbounded(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    big, small = lopsided
    unbounded = big.map_batches(lambda df: df, schema={"k": pl.Int64, "v": pl.Int64})
    assert "BUILD SIDE" not in unbounded.join(small, on="k").explain()


def test_maintain_order_keeps_its_own_build_side(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    big, small = lopsided
    q = big.join(small, on="k", maintain_order="left")
    assert "BUILD SIDE" not in q.explain()


def test_explicit_build_side_is_not_overridden(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    big, small = lopsided
    q = big.join(small, on="k", build_side="force_left")
    assert "BUILD SIDE: ForceLeft" in q.explain()

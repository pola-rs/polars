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


def cross_join_build_side(plan: str) -> str | None:
    """The build side of the one cross join in `plan`, read off the line below it."""
    lines = plan.splitlines()
    heads = [i for i, line in enumerate(lines) if line.strip() == "CROSS JOIN:"]
    assert len(heads) == 1, f"expected one cross join, found {len(heads)}"
    below = lines[heads[0] + 1].strip()
    return below.removeprefix("BUILD SIDE: ") if below.startswith("BUILD SIDE:") else None


def test_cross_join_builds_the_one_row_side_over_an_unbounded_side(
    lopsided: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    # An equi join has no row bound, so only the estimates can be compared here.
    big, small = lopsided
    joined = big.join(small, on="k")
    scalar = big.select(pl.col("v").sum())

    assert cross_join_build_side(joined.join(scalar, how="cross").explain()) == (
        "PreferRight"
    )
    assert cross_join_build_side(scalar.join(joined, how="cross").explain()) == (
        "PreferLeft"
    )

    forwards = joined.join(scalar, how="cross").collect(engine="streaming")
    backwards = scalar.join(joined, how="cross").collect(engine="streaming")
    assert forwards.height == 100
    assert backwards.height == 100


def test_cross_join_prefers_the_row_bound_over_the_estimate(tmp_path: Path) -> None:
    # Every predicate passes, but each is credited a flat selectivity, so the
    # estimate puts the right side two orders of magnitude below the left one.
    tmp_path.mkdir(exist_ok=True)
    pl.DataFrame({"a": range(10)}).write_parquet(tmp_path / "small.parquet")
    pl.DataFrame({"b": range(1_000), "c": range(1_000)}).write_parquet(
        tmp_path / "big.parquet"
    )
    small = pl.scan_parquet(tmp_path / "small.parquet")
    big = pl.scan_parquet(tmp_path / "big.parquet").filter(
        pl.col("b") >= 0,
        pl.col("b") < 1_000,
        pl.col("c") >= 0,
        pl.col("c") < 1_000,
        pl.col("b") != -1,
    )

    q = small.join(big.select("b"), how="cross")
    assert cross_join_build_side(q.explain()) == "PreferLeft"
    assert q.collect(engine="streaming").height == 10_000


def test_no_cross_join_build_side_for_similar_sizes(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    for name in ("a", "b"):
        pl.DataFrame({name: range(100)}).write_parquet(tmp_path / f"{name}.parquet")
    a = pl.scan_parquet(tmp_path / "a.parquet")
    b = pl.scan_parquet(tmp_path / "b.parquet")
    assert cross_join_build_side(a.join(b, how="cross").explain()) is None

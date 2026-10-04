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
    return (
        below.removeprefix("BUILD SIDE: ") if below.startswith("BUILD SIDE:") else None
    )


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
        (pl.col("b") - pl.col("c")) == 0,
        (pl.col("b") + pl.col("c")) >= 0,
        pl.col("b").abs() < 1_000,
        pl.col("c").abs() < 1_000,
        (pl.col("b") * 2) >= 0,
    )

    q = small.join(big.select("b"), how="cross")
    assert cross_join_build_side(q.explain()) == "PreferLeft"
    assert q.collect(engine="streaming").height == 10_000


def test_cross_join_estimate_uses_the_range_a_filter_keeps(tmp_path: Path) -> None:
    # The filter keeps 80% of the left side. The column's min/max tells the planner
    # so, and the joined right side is estimated far below it.
    tmp_path.mkdir(exist_ok=True)
    pl.DataFrame({"k": range(4_000)}).write_parquet(tmp_path / "big.parquet")
    pl.DataFrame({"k": range(200), "w": range(200)}).write_parquet(
        tmp_path / "small.parquet"
    )
    big = pl.scan_parquet(tmp_path / "big.parquet").filter(
        pl.col("k") >= 400, pl.col("k") < 3_600
    )
    small = pl.scan_parquet(tmp_path / "small.parquet")
    joined = small.join(small, on="k", suffix="_r").select("w")

    q = big.join(joined, how="cross")
    assert cross_join_build_side(q.explain()) == "PreferRight"
    assert q.select(pl.len()).collect(engine="streaming").item() == 640_000


@pytest.mark.parametrize("keep", ["head", "tail"])
def test_cross_join_estimate_uses_the_range_of_sampled_row_groups(
    tmp_path: Path, keep: str
) -> None:
    # More row groups than the planner reads statistics from, so it only sees a
    # sample of their ranges. The filter keeps 0.1% of the sorted left side, at
    # either end of the range.
    tmp_path.mkdir(exist_ok=True)
    pl.DataFrame({"k": range(20_000)}).write_parquet(
        tmp_path / "big.parquet", row_group_size=2
    )
    pl.DataFrame({"k": range(2_000), "w": range(2_000)}).write_parquet(
        tmp_path / "small.parquet"
    )
    kept = pl.col("k") < 20 if keep == "head" else pl.col("k") >= 19_980
    big = pl.scan_parquet(tmp_path / "big.parquet").filter(kept)
    small = pl.scan_parquet(tmp_path / "small.parquet")
    joined = small.join(small, on="k", suffix="_r").select("w")

    q = big.join(joined, how="cross")
    assert cross_join_build_side(q.explain()) == "PreferLeft"
    assert q.select(pl.len()).collect(engine="streaming").item() == 40_000


@pytest.mark.parametrize(
    ("n_columns", "n_row_groups", "expected"),
    [
        # The planner reads row groups 0, 4 and 8; the filter keeps only the last.
        (2_049, 12, "PreferRight"),
        # It reads only row group 0, which says nothing about the others.
        (4_097, 2, None),
    ],
)
def test_filter_past_the_sampled_row_groups_is_not_estimated_empty(
    tmp_path: Path, n_columns: int, n_row_groups: int, expected: str | None
) -> None:
    tmp_path.mkdir(exist_ok=True)
    # `k` is the index of its row group; the many columns make the planner sample.
    n = 100 * n_row_groups
    pl.DataFrame({"k": [i // 100 for i in range(n)], "v": range(n)}).with_columns(
        pl.lit(0).alias(f"c{i}") for i in range(n_columns - 2)
    ).write_parquet(tmp_path / "wide.parquet", row_group_size=100)
    pl.DataFrame({"k": range(10), "w": range(10)}).write_parquet(
        tmp_path / "small.parquet"
    )
    last = pl.col("k") >= n_row_groups - 1
    wide = pl.scan_parquet(tmp_path / "wide.parquet").filter(last).select("v")
    small = pl.scan_parquet(tmp_path / "small.parquet")
    joined = small.join(small, on="k", suffix="_r").select("w")

    q = wide.join(joined, how="cross")
    assert cross_join_build_side(q.explain()) == expected
    assert q.select(pl.len()).collect(engine="streaming").item() == 1_000


def test_no_cross_join_build_side_for_similar_sizes(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    for name in ("a", "b"):
        pl.DataFrame({name: range(100)}).write_parquet(tmp_path / f"{name}.parquet")
    a = pl.scan_parquet(tmp_path / "a.parquet")
    b = pl.scan_parquet(tmp_path / "b.parquet")
    assert cross_join_build_side(a.join(b, how="cross").explain()) is None

"""Tests for column drops fused into joins on the streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

NO_OPT = pl.QueryOptFlags.none()


def physical_plan(q: pl.LazyFrame) -> str:
    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert plan is not None
    return plan


def check(q: pl.LazyFrame, expected: pl.DataFrame | None = None) -> None:
    if expected is None:
        expected = q.collect(optimizations=NO_OPT, engine="in-memory")
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)


@pytest.mark.parametrize("coalesce", [True, False])
@pytest.mark.parametrize("suffix", ["_right", "_R"])
def test_fused_predicate_both_sides_and_collisions(coalesce: bool, suffix: str) -> None:
    left = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3, 4], "v": [10, 20, 30, 40, 50, 60], "keep": list("abcdef")}
    )
    right = pl.LazyFrame(
        {"k": [1, 2, 2, 3, 5], "v": [15, 25, 45, 5, 99], "r": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    predicate = pl.col(f"v{suffix}") < pl.col("v")
    q = (
        left.join(right, on="k", suffix=suffix, coalesce=coalesce)
        .filter(predicate)
        .select("keep", "r")
    )
    plan = physical_plan(q)
    assert "fused predicate" in plan
    assert "project 2 / " in plan
    assert "select" not in plan

    expected = (
        left.join(right, on="k", suffix=suffix, coalesce=coalesce)
        .collect()
        .filter(predicate)
        .select("keep", "r")
    )
    check(q, expected)


def test_fused_predicate_keeps_key_and_predicate_columns() -> None:
    left = pl.LazyFrame({"k": [1, 1, 2, 3], "a": [5, 25, 35, 1], "x": [0, 1, 2, 3]})
    right = pl.LazyFrame({"k": [1, 2, 2, 3], "b": [1, 2, 30, 4], "y": [7, 8, 9, 10]})
    predicate = pl.col("b") * 2 < pl.col("a") + pl.col("k")
    full = left.join(right, on="k").collect().filter(predicate)
    for cols in (["k", "x"], ["y"], ["a", "b"], ["x", "y"], ["k", "a", "y"]):
        q = left.join(right, on="k").filter(predicate).select(cols)
        check(q, full.select(cols))


def test_fused_predicate_reordered_select_correct() -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [5, 25, 35], "x": [0, 1, 2]})
    right = pl.LazyFrame({"k": [1, 2, 2], "b": [1, 2, 30]})
    q = left.join(right, on="k").filter(pl.col("b") < pl.col("a")).select("x", "k")
    check(q)


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("coalesce", [True, False])
def test_plain_join_drop(how: str, coalesce: bool) -> None:
    left = pl.LazyFrame(
        {"k": [1, 1, 2, 3, None], "a": [1, 2, 3, 4, 5], "x": list("abcde")}
    )
    right = pl.LazyFrame(
        {"k": [1, 2, 2, 7, None], "a": [9, 8, 7, 6, 5], "y": [0.5] * 5}
    )
    joined = left.join(right, on="k", how=how, coalesce=coalesce)  # type: ignore[arg-type]
    full = joined.collect()
    column_sets = [["x"], ["k", "x"], ["a"]]
    if how not in ("semi", "anti"):
        column_sets.append(["x", "y"])
    for cols in column_sets:
        check(joined.select(cols), full.select(cols))


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_semi_anti_drop_fused(how: str) -> None:
    left = pl.LazyFrame({"k": [1, 2, 3, 4], "a": [1, 2, 3, 4], "x": list("abcd")})
    right = pl.LazyFrame({"k": [2, 4, 6]})
    q = left.join(right, on="k", how=how).select("x")  # type: ignore[arg-type]
    # Only the input of the join is pruned, the drop is part of the join.
    assert 'label="select\\nx"' not in physical_plan(q)
    check(q)


@pytest.mark.slow
def test_fused_predicate_drop_many_morsels() -> None:
    n = 300_000
    left = pl.LazyFrame(
        {"k": [i % 1000 for i in range(n)], "a": list(range(n)), "x": list(range(n))}
    )
    right = pl.LazyFrame(
        {"k": [i % 1000 for i in range(5000)], "b": [i * 7 for i in range(5000)]}
    )
    q = left.join(right, on="k").filter(pl.col("b") < pl.col("a")).select("k", "x")
    assert "project 2 / " in physical_plan(q)
    check(q)


def test_range_join_expression_keys_dropped() -> None:
    left = pl.LazyFrame({"id": [1, 2, 3, 4], "t": [10, 20, 30, 40], "w": list("abcd")})
    right = pl.LazyFrame({"lo": [5, 15, 25], "hi": [18, 28, 50], "z": [1.0, 2.0, 3.0]})
    q = left.join_where(
        right,
        pl.col("t") * 2 > pl.col("lo"),
        pl.col("t") * 2 <= pl.col("hi") + 3,
    )
    full = q.collect(optimizations=NO_OPT, engine="in-memory")
    assert "range-join" in physical_plan(q)
    check(q, full)
    for cols in (["w"], ["id", "z"]):
        check(q.select(cols), full.select(cols))


@pytest.mark.parametrize("flip", [False, True])
def test_range_join_single_expression_key(flip: bool) -> None:
    a = pl.LazyFrame({"p": [1, 5, 9, 12], "pv": list("wxyz")})
    b = pl.LazyFrame({"lo": [0, 6], "bv": [1, 2]})
    if flip:
        q = b.join_where(a, pl.col("lo") + 1 < pl.col("p") * 2)
    else:
        q = a.join_where(b, pl.col("p") * 2 > pl.col("lo") + 1)
    check(q)
    check(q.select("pv", "bv"))
    check(q.select("bv"))


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("use_by", [False, True])
def test_asof_expression_keys(strategy: str, use_by: bool) -> None:
    left = pl.LazyFrame(
        {"t": [1, 3, 5, 7, 9, 11], "g": [1, 1, 2, 2, 1, 2], "lv": list("abcdef")}
    )
    right = pl.LazyFrame(
        {"t": [2, 4, 6, 8, 10], "g": [1, 2, 1, 2, 1], "rv": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    if use_by:
        left, right = left.sort("g", "t"), right.sort("g", "t")
    else:
        left, right = left.sort("t"), right.sort("t")
    q = left.join_asof(
        right,
        left_on=pl.col("t") * 2,
        right_on=pl.col("t") * 2,
        by="g" if use_by else None,
        strategy=strategy,  # type: ignore[arg-type]
        coalesce=False,
    )
    check(q)
    check(q.select("lv", "rv"))
    check(q.select("lv"))


@pytest.mark.parametrize("how", ["inner", "left", "full"])
def test_sorted_join_on_expression_and_multiple_keys(how: str) -> None:
    left = pl.LazyFrame(
        {"a": [1, 1, 2, 2, 3], "b": [1, 2, 1, 2, 1], "x": list("abcde")}
    ).sort("a", "b")
    right = pl.LazyFrame(
        {"a": [1, 2, 2, 3, 4], "b": [2, 1, 2, 1, 1], "y": [1, 2, 3, 4, 5]}
    ).sort("a", "b")
    for q in (
        left.join(right, on=["a", "b"], how=how, coalesce=False),  # type: ignore[arg-type]
        left.join(
            right,
            left_on=pl.col("a") + 0,
            right_on=pl.col("a") + 0,
            how=how,  # type: ignore[arg-type]
            coalesce=False,
        ),
    ):
        check(q)
        check(q.select("x", "y"))


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("morsel_size", [2, 100_000])
@pytest.mark.parametrize("columns", [["v"], ["k", "v_right"], ["v", "v_right"], ["k"]])
def test_outer_join_where_predicate_columns_dropped(
    how: str, morsel_size: int, columns: list[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", str(morsel_size))
    left = pl.LazyFrame({"k": [1, 1, 1, 2, 2, 3, None], "v": [1, 5, 4, 2, None, 9, 1]})
    right = pl.LazyFrame({"k": [1, 1, 2, 3, 4, None], "v": [3, 0, None, 1, 0, 2]})
    q = left.join_where(
        right,
        pl.col("k") == pl.col("k_right"),
        pl.col("v") <= pl.col("v_right"),
        how=how,  # type: ignore[arg-type]
    ).select(columns)
    check(q)


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize("morsel_size", [2, 100_000])
@pytest.mark.parametrize(
    "columns", [["v"], ["k"], ["k", "v"], ["k_r"], ["w"], ["k", "k_r", "w"]]
)
def test_coalesced_join_key_dropped_and_kept(
    how: str, morsel_size: int, columns: list[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", str(morsel_size))
    left = pl.LazyFrame({"k": [1, 1, 2, 3, 6, None], "v": [1, 5, 4, 2, 9, 1]})
    right = pl.LazyFrame({"k_r": [1, 2, 2, 4, 5, None], "w": [3, 0, 7, 1, 0, 2]})
    q = left.join(
        right,
        left_on="k",
        right_on="k_r",
        how=how,  # type: ignore[arg-type]
        coalesce=True,
    )
    # A coalesced right join outputs the right key only.
    names = q.collect_schema().names()
    q = q.select([c for c in columns if c in names] or ["v"])
    check(q)
    q = left.join(
        right,
        left_on="k",
        right_on="k_r",
        how=how,  # type: ignore[arg-type]
        coalesce=False,
    ).select(columns)
    check(q)


@pytest.mark.parametrize("how", ["left", "right", "full"])
@pytest.mark.parametrize("columns", [["v"], ["k", "w"], ["w"], ["v", "w"]])
def test_fused_predicate_many_unmatched_build_rows(
    how: str, columns: list[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "16")
    n = 1000
    left = pl.LazyFrame(
        {"k": [i % 500 for i in range(n)], "v": list(range(n)), "p": list(range(n))}
    )
    right = pl.LazyFrame(
        {
            "k": [i % 500 + (i % 3) * 400 for i in range(n)],
            "w": list(range(n)),
            "q": [i * 3 % 7 for i in range(n)],
        }
    )
    q = left.join_where(
        right,
        pl.col("k") == pl.col("k_right"),
        pl.col("p") < pl.col("q") * 100,
        how=how,  # type: ignore[arg-type]
    ).select(columns)
    check(q)

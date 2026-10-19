"""A top-k over a left/right/cross join copied onto the join's preserved side."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import JoinStrategy, MaintainOrderJoin

ON = pl.QueryOptFlags()
OFF = pl.QueryOptFlags.none()

N = 1000
FACT = pl.LazyFrame(
    {
        "f_id": list(range(N)),
        # 40..49 have no match in `DIM`.
        "f_dim": [i % 50 for i in range(N)],
        # Unique where not null.
        "f_val": [float(i * 37 % N) if i % 7 else None for i in range(N)],
        "f_grp": [i % 10 for i in range(N)],
    }
)
# Key 3 matches twice.
DIM = pl.LazyFrame(
    {
        "d_key": [*range(40), 3],
        "d_name": [f"d{i}" for i in range(40)] + ["dup"],
    }
)
SMALL = pl.LazyFrame({"s_key": [3, 1, 2], "s_w": [0.5, 0.25, 1.0]})


def top_k_copies(plan: str) -> list[list[tuple[str, str]]]:
    """Get, for each top-k below a join, the joins and sides above it, top down."""
    copies = []
    joins: dict[int, str] = {}
    path: dict[int, tuple[str, str]] = {}
    for line in plan.splitlines():
        indent = len(line) - len(line.lstrip())
        text = line.strip()
        path = {i: p for i, p in path.items() if i < indent}
        if text.endswith(" JOIN:"):
            joins[indent] = text.removesuffix(" JOIN:")
        elif " PLAN ON: [" in text:
            path[indent] = (joins[indent], text.split()[0].lower())
        elif text.startswith("SORT BY [slice: (0, ") and path:
            copies.append(list(path.values()))
    return copies


def assert_correct_top_k(lf: pl.LazyFrame, q: pl.LazyFrame, by: list[str]) -> None:
    """Compare the keys of `q` with the unoptimized run, and check every row is in `lf`.

    Rows that tie at the k-th key may legally differ between runs.
    """
    expected = q.collect(optimizations=OFF)
    out = q.collect(optimizations=ON)
    assert out.schema == expected.schema
    assert_frame_equal(out.select(by), expected.select(by))
    rows = lf.collect(optimizations=OFF)
    assert out.join(rows, on=out.columns, how="anti", nulls_equal=True).is_empty()


def left_join(**kwargs: Any) -> pl.LazyFrame:
    return FACT.join(DIM, left_on="f_dim", right_on="d_key", how="left", **kwargs)


def assert_no_copy(lf: pl.LazyFrame) -> None:
    assert top_k_copies(lf.explain(optimizations=ON)) == []


def test_copy_below_right_join() -> None:
    by = ["f_dim", "f_id"]
    # `f_dim` is the coalesced key.
    lf = DIM.join(FACT, left_on="d_key", right_on="f_dim", how="right")
    q = lf.sort(by, descending=True).head(10)
    assert top_k_copies(q.explain(optimizations=ON)) == [[("RIGHT", "right")]]
    assert_correct_top_k(lf, q, by)


@pytest.mark.parametrize(("by", "side"), [(["f_val"], "left"), (["s_w"], "right")])
def test_copy_below_cross_join(by: list[str], side: str) -> None:
    lf = FACT.join(SMALL, how="cross")
    q = lf.sort(by).head(5)
    assert top_k_copies(q.explain(optimizations=ON)) == [[("CROSS", side)]]
    assert_correct_top_k(lf, q, by)


def test_single_copy_below_chain_of_left_joins() -> None:
    lf = left_join().join(SMALL, left_on="f_grp", right_on="s_key", how="left")
    q = lf.sort("f_val", descending=True, nulls_last=True).head(10)
    assert top_k_copies(q.explain(optimizations=ON)) == [
        [("LEFT", "left"), ("LEFT", "left")]
    ]
    assert_correct_top_k(lf, q, ["f_val"])


@pytest.mark.parametrize(
    "project",
    [
        lambda lf: lf.with_columns(g=pl.col("f_val") * 2, d_name=pl.lit("x")),
        lambda lf: lf.select("d_name", "f_val", (pl.col("f_id") + 1).alias("g")),
        lambda lf: lf.select("f_val", "d_name"),
    ],
)
def test_copy_through_projections(
    project: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> None:
    lf = project(left_join())
    q = lf.sort("f_val").head(10)
    assert top_k_copies(q.explain(optimizations=ON)) == [[("LEFT", "left")]]
    assert_correct_top_k(lf, q, ["f_val"])


@pytest.mark.parametrize(
    ("join_order", "sort_maintain_order"),
    [("none", True), ("right", False)],
)
def test_copy_with_compatible_order(
    join_order: MaintainOrderJoin, sort_maintain_order: bool
) -> None:
    lf = left_join(maintain_order=join_order)
    q = lf.sort("f_grp", maintain_order=sort_maintain_order).head(10)
    assert top_k_copies(q.explain(optimizations=ON)) == [[("LEFT", "left")]]
    assert_correct_top_k(lf, q, ["f_grp"])


@pytest.mark.parametrize("how", ["inner", "full"])
def test_no_copy_below_non_preserving_join(how: JoinStrategy) -> None:
    lf = FACT.join(DIM, left_on="f_dim", right_on="d_key", how=how)
    assert_no_copy(lf.sort("f_val").head(10))


def test_no_copy_below_asof_join() -> None:
    lf = FACT.set_sorted("f_id").join_asof(
        DIM.sort("d_key"), left_on="f_id", right_on="d_key"
    )
    assert_no_copy(lf.sort("f_val").head(10))


def test_no_copy_below_join_where() -> None:
    lf = FACT.join_where(
        DIM, pl.col("f_dim") < pl.col("d_key"), pl.col("f_id") > pl.col("d_key")
    )
    assert_no_copy(lf.sort("f_val").head(10))


@pytest.mark.parametrize(
    "by",
    [
        [pl.col("f_val"), pl.col("d_name")],
        [pl.col("f_val") - pl.col("f_val").mean()],
    ],
)
def test_no_copy_for_unpushable_keys(by: list[pl.Expr]) -> None:
    assert_no_copy(left_join().sort(by).head(10))


def test_no_copy_for_renamed_key() -> None:
    lf = left_join().select(pl.col("f_val").alias("v"), "d_name")
    assert_no_copy(lf.sort("v").head(10))


def test_no_copy_for_suffixed_key() -> None:
    other = FACT.select(pl.col("f_id"), pl.col("f_val") * 2)
    # The right `f_val` comes out as `f_val_right`.
    lf = FACT.join(other, on="f_id", how="right", coalesce=False)
    assert_no_copy(lf.sort("f_val_right").head(10))


def test_no_copy_for_non_elementwise_projection() -> None:
    lf = left_join().with_columns(m=pl.col("f_id").max())
    assert_no_copy(lf.sort("f_val").head(10))


def test_no_copy_through_filter() -> None:
    lf = left_join().filter(pl.col("f_val").fill_null(0).cum_sum() > 100)
    assert_no_copy(lf.sort("f_val").head(10))


def test_no_copy_through_shared_cache() -> None:
    lf = left_join()
    q = pl.concat([lf.sort("f_val").head(10), lf.sort("f_val").head(20)])
    plan = q.explain(optimizations=ON)
    assert "CACHE" in plan
    assert top_k_copies(plan) == []


def test_no_copy_for_negative_offset() -> None:
    assert_no_copy(left_join().sort("f_val").slice(-10, 5))


def test_no_copy_without_length() -> None:
    assert_no_copy(left_join().sort("f_val").slice(5))


@pytest.mark.parametrize("join_order", ["left", "right_left"])
def test_no_copy_when_join_keeps_preserved_order(join_order: MaintainOrderJoin) -> None:
    assert_no_copy(left_join(maintain_order=join_order).sort("f_val").head(10))


def test_no_copy_when_stable_sort_follows_other_side() -> None:
    lf = left_join(maintain_order="right")
    assert_no_copy(lf.sort("f_val", maintain_order=True).head(10))


def test_no_copy_for_validated_join() -> None:
    unique = DIM.filter(pl.col("d_name") != "dup")
    lf = FACT.join(
        unique, left_on="f_dim", right_on="d_key", how="left", validate="m:1"
    )
    assert_no_copy(lf.sort("f_val").head(10))


@pytest.mark.parametrize(
    ("by", "options", "offset", "length"),
    [
        (["f_val"], {"descending": True, "nulls_last": True}, 5, 10),
        (["f_grp", "f_val"], {"descending": [True, False]}, 0, 10),
        (["f_dim", "f_id"], {}, 0, 10),
        (["f_grp"], {}, 0, 10),
        (["f_grp"], {"maintain_order": True}, 0, 10),
        (["f_val"], {}, 990, 100),
    ],
)
def test_copy_below_left_join(
    by: list[str], options: dict[str, Any], offset: int, length: int
) -> None:
    lf = left_join()
    q = lf.sort(by, **options).slice(offset, length)
    assert top_k_copies(q.explain(optimizations=ON)) == [[("LEFT", "left")]]
    assert_correct_top_k(lf, q, by)

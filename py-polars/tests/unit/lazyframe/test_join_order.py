"""Tests for the join-reordering optimization (`QueryOptFlags(join_order=...)`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from polars._typing import PolarsDataType

ON = pl.QueryOptFlags(join_order=True)
OFF = pl.QueryOptFlags(join_order=False)


def star_frames(tmp_path: Path) -> dict[str, pl.LazyFrame]:
    """A miniature star schema written to parquet.

    The pass reads row counts from scan metadata, so the relations must be real files.
    """
    fact = pl.DataFrame(
        {
            "f_dim_a": [i % 50 for i in range(1000)],
            "f_dim_b": [i % 20 for i in range(1000)],
            "f_val": list(range(1000)),
        }
    )
    # Selective: a filter leaves very few rows.
    dim_a = pl.DataFrame(
        {"a_key": list(range(50)), "a_flag": [i == 7 for i in range(50)]}
    )
    # Unselective: joined on its key it reproduces the fact table.
    dim_b = pl.DataFrame(
        {"b_key": list(range(20)), "b_name": [f"b{i}" for i in range(20)]}
    )

    return write_scans(tmp_path, fact=fact, dim_a=dim_a, dim_b=dim_b)


def write_scans(tmp_path: Path, **frames: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    scans = {}
    for name, df in frames.items():
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path)
        scans[name] = pl.scan_parquet(path)
    return scans


def shared_key_star(*, stray_key_in_dim_b: bool = False) -> dict[str, pl.DataFrame]:
    """A star schema whose keys are named the same on both sides.

    Coalescing folds such a pair into one column whichever side ends up left, so
    these joins may be reordered.
    """
    dim_b = pl.DataFrame(
        {"k_b": list(range(20)), "b_name": [f"b{i}" for i in range(20)]}
    )
    if stray_key_in_dim_b:
        # Holds `k_a` without being joined on it, so `k_a` would survive twice.
        dim_b = dim_b.with_columns(k_a=pl.col("k_b"))

    return {
        "fact": pl.DataFrame(
            {
                "k_a": [i % 50 for i in range(1000)],
                "k_b": [i % 20 for i in range(1000)],
                "f_val": list(range(1000)),
            }
        ),
        "dim_a": pl.DataFrame(
            {"k_a": list(range(50)), "a_flag": [i == 7 for i in range(50)]}
        ),
        "dim_b": dim_b,
    }


def shared_key_frames(
    tmp_path: Path, *, stray_key_in_dim_b: bool = False
) -> dict[str, pl.LazyFrame]:
    return write_scans(
        tmp_path, **shared_key_star(stray_key_in_dim_b=stray_key_in_dim_b)
    )


def shared_key_query(frames: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    return (
        frames["fact"]
        .join(frames["dim_b"], on="k_b")
        .join(frames["dim_a"].filter(pl.col("a_flag")), on="k_a")
    )


def star_query(frames: dict[str, pl.LazyFrame], **join_kwargs: Any) -> pl.LazyFrame:
    """Fact joined to both dimensions; `join_kwargs` is forwarded to both joins."""
    kwargs: dict[str, Any] = {"coalesce": False, **join_kwargs}
    return (
        frames["fact"]
        .join(frames["dim_b"], left_on="f_dim_b", right_on="b_key", **kwargs)
        .join(
            frames["dim_a"].filter(pl.col("a_flag")),
            left_on="f_dim_a",
            right_on="a_key",
            **kwargs,
        )
    )


def scan_order(plan: str) -> list[str]:
    return [
        line.split("[")[1].split("]")[0].split("/")[-1].removesuffix(".parquet")
        for line in plan.splitlines()
        if "Parquet SCAN" in line
    ]


def test_selective_dimension_is_joined_first(tmp_path: Path) -> None:
    lf = star_query(star_frames(tmp_path))

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    # dim_a filters down to one row, so it is folded in before dim_b.
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]


def test_reordering_preserves_results_and_schema(tmp_path: Path) -> None:
    lf = star_query(star_frames(tmp_path))

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)

    # Reordered joins emit columns in a different order and are projected back, so
    # names and order must both survive.
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


@pytest.mark.parametrize(
    "join_kwargs",
    [
        # Coalescing keeps the left key's name, so swapping inputs would rename the
        # column when the two sides are named differently.
        pytest.param({"coalesce": True}, id="coalesce_renaming_key"),
        # Only inner joins are safe to commute.
        pytest.param({"how": "left"}, id="outer_join"),
        # Validation checks a named side, which reordering would change.
        pytest.param({"validate": "m:1"}, id="validation"),
    ],
)
def test_unsafe_clusters_are_left_alone(
    tmp_path: Path, join_kwargs: dict[str, object]
) -> None:
    lf = star_query(star_frames(tmp_path), **join_kwargs)
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)


def in_memory_star() -> pl.LazyFrame:
    """The `shared_key_frames` star schema, held in memory rather than on disk."""
    return shared_key_query({n: df.lazy() for n, df in shared_key_star().items()})


def test_in_memory_frames_are_reordered() -> None:
    # An in-memory frame knows its own height, so it needs no scan metadata.
    lf = in_memory_star()

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)

    assert lf.explain(optimizations=ON) != lf.explain(optimizations=OFF)
    assert off.height > 0
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


def test_group_by_leaf_is_estimated(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    # A group-by emits at most one row per input row, so `dim_a` is still the
    # smaller side and is folded in first.
    rollup = frames["fact"].group_by("f_dim_a", "f_dim_b").agg(pl.col("f_val").sum())
    lf = rollup.join(
        frames["dim_b"], left_on="f_dim_b", right_on="b_key", coalesce=False
    ).join(
        frames["dim_a"].filter(pl.col("a_flag")),
        left_on="f_dim_a",
        right_on="a_key",
        coalesce=False,
    )

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)
    assert off.height > 0
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


def test_group_by_with_a_user_function_is_left_alone(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    # `map_groups` can emit any number of rows per group, so the count is unknown.
    rollup = (
        frames["fact"]
        .group_by("f_dim_a", "f_dim_b")
        .map_groups(lambda df: df.head(1), schema=None)
    )
    lf = rollup.join(
        frames["dim_b"], left_on="f_dim_b", right_on="b_key", coalesce=False
    ).join(
        frames["dim_a"].filter(pl.col("a_flag")),
        left_on="f_dim_a",
        right_on="a_key",
        coalesce=False,
    )
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)


def test_sql_star_join_is_reordered(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    ctx = pl.SQLContext(frames)
    q = """
        SELECT f_val
        FROM fact, dim_b, dim_a
        WHERE f_dim_b = b_key AND f_dim_a = a_key AND a_flag
    """
    lf = ctx.execute(q, eager=False)

    assert scan_order(lf.explain()) == ["fact", "dim_a", "dim_b"]

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


def test_coalescing_star_join_is_reordered(tmp_path: Path) -> None:
    lf = shared_key_query(shared_key_frames(tmp_path))

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]


def test_coalescing_reordering_preserves_results_and_schema(tmp_path: Path) -> None:
    lf = shared_key_query(shared_key_frames(tmp_path))

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)

    assert off.height > 0
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


def test_coalesced_key_held_by_an_unjoined_leaf_is_left_alone(tmp_path: Path) -> None:
    lf = shared_key_query(shared_key_frames(tmp_path, stray_key_in_dim_b=True))
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)


def implied_edge_frames(tmp_path: Path) -> dict[str, pl.LazyFrame]:
    """Three relations sharing key `k`, where `a` and `c` are only joined on `m`.

    `k` is written as `a`-`b` and `b`-`c`, so placing `c` next to `a` keeps both
    copies of `k` unless the implied `a`-`c` equality is added.
    """
    a = pl.DataFrame(
        {
            "k": [i % 50 for i in range(1000)],
            "m": [(i // 50) % 10 for i in range(1000)],
            "va": list(range(1000)),
        }
    )
    b = pl.DataFrame({"k": list(range(20)), "vb": [f"b{i}" for i in range(20)]})
    c = pl.DataFrame(
        {
            "k": list(range(50)),
            "m": [i % 7 for i in range(50)],
            "c_flag": [i == 7 for i in range(50)],
            "vc": list(range(50)),
        }
    )
    return write_scans(tmp_path, a=a, b=b, c=c)


def test_implied_key_equality_keeps_coalescing_sound(tmp_path: Path) -> None:
    frames = implied_edge_frames(tmp_path)
    lf = frames["a"].join(
        frames["b"].join(frames["c"].filter(pl.col("c_flag")), on="k"), on=["k", "m"]
    )

    off = lf.collect(optimizations=OFF)
    on = lf.collect(optimizations=ON)

    assert lf.explain(optimizations=ON) != lf.explain(optimizations=OFF)
    assert off.height > 0
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))


def test_select_on_a_leaf_is_estimated(tmp_path: Path) -> None:
    # `fast_projection` rewrites a plain `select` into a `SimpleProjection`, but it
    # runs after this pass, so the estimate has to see through the `select` itself.
    frames = {k: v.select(pl.all()) for k, v in star_frames(tmp_path).items()}
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_with_columns_on_a_leaf_is_estimated(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    frames["fact"] = frames["fact"].with_columns(f_double=pl.col("f_val") * 2)
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def between_joins(
    frames: dict[str, pl.LazyFrame],
    step: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> pl.LazyFrame:
    """The star query with `step` applied between the two joins."""
    inner = frames["fact"].join(
        frames["dim_b"], left_on="f_dim_b", right_on="b_key", coalesce=False
    )
    return step(inner).join(
        frames["dim_a"].filter(pl.col("a_flag")),
        left_on="f_dim_a",
        right_on="a_key",
        coalesce=False,
    )


def joins_around(
    frames: dict[str, pl.LazyFrame], middle: pl.Expr | str
) -> pl.LazyFrame:
    """The star query with a projection sitting between the two joins."""
    return between_joins(frames, lambda lf: lf.select("f_dim_a", "f_val", middle))


def test_projection_between_joins_keeps_one_cluster(tmp_path: Path) -> None:
    # Without looking past it the outer join sees two leaves and no cluster forms.
    lf = joins_around(star_frames(tmp_path), "b_name")

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_computed_projection_between_joins_is_not_dropped(tmp_path: Path) -> None:
    # The restoring projection can only pick columns out of what the joins produce,
    # so a projection that computes one cannot be looked past.
    lf = joins_around(star_frames(tmp_path), (pl.col("b_name") + "!").alias("shout"))

    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)
    result = lf.collect(optimizations=ON)
    assert "shout" in result.columns
    assert_frame_equal(
        lf.collect(optimizations=OFF).sort(pl.all()), result.sort(pl.all())
    )


def assert_reordered(
    lf: pl.LazyFrame,
    before: list[str],
    after: list[str],
    *,
    on_flags: pl.QueryOptFlags = ON,
    off_flags: pl.QueryOptFlags = OFF,
) -> pl.DataFrame:
    """Check `lf` reorders from `before` to `after` and returns the same rows."""
    assert scan_order(lf.explain(optimizations=off_flags)) == before
    assert scan_order(lf.explain(optimizations=on_flags)) == after

    off = lf.collect(optimizations=off_flags)
    on = lf.collect(optimizations=on_flags)
    assert off.height > 0
    assert on.columns == off.columns
    assert_frame_equal(off.sort(pl.all()), on.sort(pl.all()))
    return on


def test_rename_between_joins_keeps_one_cluster(tmp_path: Path) -> None:
    # A rename reaches the IR as a `select` naming its output differently. The
    # rebuilt joins read the leaves directly, so the rename has to travel down to
    # the leaf holding the column.
    lf = between_joins(star_frames(tmp_path), lambda lf: lf.rename({"f_val": "val"}))

    reordered = assert_reordered(
        lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"]
    )
    assert "val" in reordered.columns


def test_rename_of_a_join_key_between_joins(tmp_path: Path) -> None:
    # The renamed column is the key the outer join is written on, so the key
    # expression has to be rewritten along with the leaf.
    frames = star_frames(tmp_path)
    lf = (
        frames["fact"]
        .join(frames["dim_b"], left_on="f_dim_b", right_on="b_key", coalesce=False)
        .rename({"f_dim_a": "a_ref"})
        .join(
            frames["dim_a"].filter(pl.col("a_flag")),
            left_on="a_ref",
            right_on="a_key",
            coalesce=False,
        )
    )

    reordered = assert_reordered(
        lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"]
    )
    assert "a_ref" in reordered.columns


def test_projection_reading_one_column_twice_is_left_alone(tmp_path: Path) -> None:
    # Two output columns read `f_val`, which no rename of the leaf can produce.
    lf = between_joins(
        star_frames(tmp_path),
        lambda lf: lf.select(pl.all(), pl.col("f_val").alias("copy")),
    )
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)


def test_rename_onto_a_dropped_column_is_left_alone(tmp_path: Path) -> None:
    # The projection drops `f_dim_b` and renames `f_val` onto it. `fact` still holds
    # both, so pushing the rename down would give it two columns of that name.
    lf = between_joins(
        star_frames(tmp_path),
        lambda lf: lf.select("f_dim_a", pl.col("f_val").alias("f_dim_b")),
    )
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)


def test_unique_leaf_is_estimated(tmp_path: Path) -> None:
    # `unique` emits one row per distinct subset, which is a group-by over it.
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].unique(subset=["a_key"])
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_unique_over_every_column_is_estimated(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].unique()
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_scan_slice_is_estimated(tmp_path: Path) -> None:
    # Slice pushdown folds a `head` into the scan, so it never reaches the `Slice`
    # node; the scan's own `pre_slice` is what narrows the estimate.
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].head(10)
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_slice_node_is_estimated(tmp_path: Path) -> None:
    # Without slice pushdown the slice stays a node of its own.
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].head(10)
    lf = star_query(frames)

    assert_reordered(
        lf,
        ["fact", "dim_b", "dim_a"],
        ["fact", "dim_a", "dim_b"],
        on_flags=pl.QueryOptFlags(join_order=True, slice_pushdown=False),
        off_flags=pl.QueryOptFlags(join_order=False, slice_pushdown=False),
    )


def test_sort_leaf_is_estimated(tmp_path: Path) -> None:
    # A sort keeps every row, and a `head` on it becomes the sort's own top-k slice.
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].sort("a_key").head(10)
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def test_union_leaf_is_estimated(tmp_path: Path) -> None:
    # A union holds every row of every input, so it is the sum of them. The two
    # inputs must be different files, or common-subplan elimination caches them
    # into one scan.
    frames = star_frames(tmp_path)
    more = write_scans(
        tmp_path,
        dim_a_more=pl.DataFrame({"a_key": list(range(50, 60)), "a_flag": [False] * 10}),
    )["dim_a_more"]
    lf = star_query({**frames, "dim_a": pl.concat([frames["dim_a"], more])})

    assert_reordered(
        lf,
        ["fact", "dim_b", "dim_a", "dim_a_more"],
        ["fact", "dim_a", "dim_a_more", "dim_b"],
    )


def test_gather_leaf_is_estimated(tmp_path: Path) -> None:
    # The indices are a frame of their own, so the gather's height is that frame's.
    frames = star_frames(tmp_path)
    frames["dim_a"] = frames["dim_a"].gather(pl.LazyFrame({"i": [0, 7, 9]}))
    lf = star_query(frames)

    assert_reordered(lf, ["fact", "dim_b", "dim_a"], ["fact", "dim_a", "dim_b"])


def null_count_frames(tmp_path: Path, *, nulls: int) -> dict[str, pl.LazyFrame]:
    """A star schema whose `dim_a` filter is an `is_not_null`.

    `dim_a` is the larger dimension, so it only sorts ahead of `dim_b` when the
    null count says the filter is more selective than the flat fallback.
    """
    fact = pl.DataFrame(
        {
            "f_dim_a": [i % 1000 for i in range(20_000)],
            "f_dim_b": [i % 100 for i in range(20_000)],
        }
    )
    dim_a = pl.DataFrame(
        {
            "a_key": list(range(1000)),
            "a_flag": [None if i < nulls else i for i in range(1000)],
        }
    )
    dim_b = pl.DataFrame({"b_key": list(range(100)), "b_val": list(range(100))})
    return write_scans(tmp_path, fact=fact, dim_a=dim_a, dim_b=dim_b)


def null_count_query(frames: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    return (
        frames["fact"]
        .join(
            frames["dim_a"].filter(pl.col("a_flag").is_not_null()),
            left_on="f_dim_a",
            right_on="a_key",
            coalesce=False,
        )
        .join(
            frames["dim_b"].filter(pl.col("b_val") > 5),
            left_on="f_dim_b",
            right_on="b_key",
            coalesce=False,
        )
    )


def test_null_count_drives_filter_selectivity(tmp_path: Path) -> None:
    # 99% of `a_flag` is null, so `is_not_null` leaves ~10 of 1000 rows. That beats
    # `dim_b`, whose opaque predicate falls back to the flat selectivity.
    lf = null_count_query(null_count_frames(tmp_path, nulls=990))

    assert_reordered(lf, ["fact", "dim_a", "dim_b"], ["fact", "dim_a", "dim_b"])


def test_no_nulls_leaves_the_larger_dimension_last(tmp_path: Path) -> None:
    # The same query over a column with no nulls at all: `is_not_null` keeps every
    # row, so the smaller `dim_b` is joined first instead.
    lf = null_count_query(null_count_frames(tmp_path, nulls=0))

    assert_reordered(lf, ["fact", "dim_a", "dim_b"], ["fact", "dim_b", "dim_a"])


def self_join_frames(tmp_path: Path) -> dict[str, pl.LazyFrame]:
    """A fact table joined to one dimension twice, plus a small second dimension."""
    fact = pl.DataFrame(
        {
            "f_a": [i % 400 for i in range(4000)],
            "f_b": [i % 400 for i in range(4000)],
            "f_c": [i % 4 for i in range(4000)],
        }
    )
    # Held by two leaves at once, so `d_key` and `d_val` are ambiguous names.
    dim = pl.DataFrame(
        {"d_key": list(range(400)), "d_val": [f"v{i}" for i in range(400)]}
    )
    # Selective: a filter leaves a single row.
    other = pl.DataFrame(
        {"o_key": list(range(4)), "o_flag": [i == 2 for i in range(4)]}
    )
    return write_scans(tmp_path, fact=fact, dim=dim, other=other)


def self_join_query(frames: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """`fact` joined to `dim` twice, with the selective dimension written last."""
    dim = frames["dim"]
    return (
        frames["fact"]
        .join(dim, left_on="f_a", right_on="d_key", coalesce=False)
        .join(dim, left_on="f_b", right_on="d_key", coalesce=False)
        .join(
            frames["other"].filter(pl.col("o_flag")),
            left_on="f_c",
            right_on="o_key",
            coalesce=False,
        )
    )


def test_self_join_on_shared_column_names_is_reordered(tmp_path: Path) -> None:
    # `d_key` and `d_val` are each held by two leaves. Once those are renamed apart
    # the cluster can be ordered, and the selective `other` sorts ahead of both
    # copies of `dim`.
    lf = self_join_query(self_join_frames(tmp_path))

    # Both copies of `dim` scan one file, which common-subplan elimination would
    # otherwise fold into a single cached scan.
    assert_reordered(
        lf,
        ["fact", "dim", "dim", "other"],
        ["fact", "other", "dim", "dim"],
        on_flags=pl.QueryOptFlags(join_order=True, comm_subplan_elim=False),
        off_flags=pl.QueryOptFlags(join_order=False, comm_subplan_elim=False),
    )


def test_self_join_keeps_each_copy_of_a_shared_column(tmp_path: Path) -> None:
    # Which physical column a suffixed name refers to depends on the join order, so
    # the rebuilt plan aliases the renamed columns back onto the original ones.
    lf = self_join_query(self_join_frames(tmp_path)).select(
        "f_a", "f_b", "d_val", "d_val_right"
    )

    on = lf.collect(optimizations=ON)
    assert on.height > 0
    assert_frame_equal(lf.collect(optimizations=OFF).sort(pl.all()), on.sort(pl.all()))
    # `d_val` is `dim` looked up by `f_a`, `d_val_right` by `f_b`.
    assert on.get_column("d_val").to_list() == [
        f"v{a}" for a in on.get_column("f_a").to_list()
    ]
    assert on.get_column("d_val_right").to_list() == [
        f"v{b}" for b in on.get_column("f_b").to_list()
    ]


def correlated_key_frames(tmp_path: Path) -> dict[str, pl.LazyFrame]:
    """A fact table, its returns joined on three correlated keys, and a dimension."""
    rows = 2000
    fact = pl.DataFrame(
        {
            "f_customer": [i % 200 for i in range(rows)],
            "f_item": [i % 100 for i in range(rows)],
            "f_ticket": list(range(rows)),
            "f_day": [i % 50 for i in range(rows)],
        }
    )
    # The three keys together are a key of `fact`, so they do not vary independently.
    ret = fact.select(
        r_customer="f_customer", r_item="f_item", r_ticket="f_ticket"
    ).head(rows // 2)
    # Selective: a filter leaves a single day.
    day = pl.DataFrame(
        {"d_day": list(range(50)), "d_flag": [i == 3 for i in range(50)]}
    )
    return write_scans(tmp_path, fact=fact, ret=ret, day=day)


def test_correlated_multi_key_join_does_not_look_free(tmp_path: Path) -> None:
    # One domain per key multiplied puts the three-key join at the cardinality floor,
    # which no dimension can beat, so it would always be joined first.
    frames = correlated_key_frames(tmp_path)
    lf = (
        frames["fact"]
        .join(
            frames["ret"],
            left_on=["f_customer", "f_item", "f_ticket"],
            right_on=["r_customer", "r_item", "r_ticket"],
            coalesce=False,
        )
        .join(
            frames["day"].filter(pl.col("d_flag")),
            left_on="f_day",
            right_on="d_day",
            coalesce=False,
        )
    )

    assert_reordered(lf, ["fact", "ret", "day"], ["fact", "day", "ret"])


def fanout_frames(
    tmp_path: Path, dtype: PolarsDataType, spread: int
) -> dict[str, pl.LazyFrame]:
    """Two relations sharing 100 key values, and a dimension hanging off one of them.

    `spread` scales the keys apart without changing how many there are, so a domain
    read from the key's value range only bounds the join for a small `spread`.
    """
    sales = pl.DataFrame(
        {"s_item": pl.Series([(i % 100) * spread for i in range(2000)], dtype=dtype)}
    )
    inv = pl.DataFrame(
        {
            "i_item": pl.Series([(i % 100) * spread for i in range(3000)], dtype=dtype),
            "i_wh": [i % 30 for i in range(3000)],
        }
    )
    wh = pl.DataFrame(
        {"w_key": list(range(30)), "w_name": [f"w{i}" for i in range(30)]}
    )
    return write_scans(tmp_path, sales=sales, inv=inv, wh=wh)


def fanout_query(frames: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    return (
        frames["sales"]
        .join(frames["inv"], left_on="s_item", right_on="i_item", coalesce=False)
        .join(frames["wh"], left_on="i_wh", right_on="w_key", coalesce=False)
    )


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
def test_key_value_range_defers_a_fan_out_join(
    tmp_path: Path, dtype: PolarsDataType
) -> None:
    lf = fanout_query(fanout_frames(tmp_path, dtype, spread=1))

    # 100 values shared by both sides fan out, so the dimension is folded in first.
    assert_reordered(lf, ["sales", "inv", "wh"], ["inv", "wh", "sales"])


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.UInt64, pl.Float64])
def test_key_value_range_wider_than_the_relation_does_not_bound_it(
    tmp_path: Path, dtype: PolarsDataType
) -> None:
    lf = fanout_query(fanout_frames(tmp_path, dtype, spread=10_000_000))

    assert scan_order(lf.explain(optimizations=ON)) == ["sales", "inv", "wh"]


@pytest.mark.parametrize(
    ("spread", "expected"),
    [(1, ["inv", "wh", "sales"]), (10_000_000, ["sales", "inv", "wh"])],
)
def test_key_value_range_survives_a_partial_footer_read(
    tmp_path: Path, spread: int, expected: list[str]
) -> None:
    # More files than one footer wave, so only some of their footers are read.
    parts = tmp_path / "sales"
    parts.mkdir()
    n_files = 40
    for k in range(n_files):
        pl.DataFrame(
            {
                "s_item": pl.Series(
                    [(i % 100) * spread for i in range(k, 2000, n_files)],
                    dtype=pl.Int32,
                )
            }
        ).write_parquet(parts / f"p{k:03}.parquet")

    frames = fanout_frames(tmp_path, pl.Int32, spread=spread)
    frames["sales"] = pl.scan_parquet(parts / "*.parquet")

    order = scan_order(fanout_query(frames).explain(optimizations=ON))
    assert ["sales" if n.startswith("p000") else n for n in order] == expected


def test_non_elementwise_filter_between_joins_is_left_alone(tmp_path: Path) -> None:
    # `b` matches every `a` row twice, so a window over the join sees groups of two.
    frames = write_scans(
        tmp_path,
        a=pl.DataFrame({"a_id": [1, 2, 3, 4], "av": [10, 20, 30, 40]}),
        b=pl.DataFrame({"b_id": [1, 1, 2, 2, 3, 3, 4, 4], "bv": list(range(8))}),
        c=pl.DataFrame({"c_id": [1, 2, 3, 4], "cv": [100, 200, 300, 400]}),
    )
    lf = (
        frames["a"]
        .join(frames["b"], left_on="a_id", right_on="b_id", coalesce=False)
        .filter(pl.len().over("a_id") == 1)
        .join(frames["c"], left_on="a_id", right_on="c_id", coalesce=False)
    )

    assert_frame_equal(lf.collect(optimizations=OFF), lf.collect(optimizations=ON))


def test_computed_join_key_does_not_borrow_a_column_range(tmp_path: Path) -> None:
    # A computed key is named after its left-most column, here a constant whose value
    # range would otherwise be read as the key's own.
    frames = fanout_frames(tmp_path, pl.Int64, spread=10_000_000)
    frames["sales"] = frames["sales"].with_columns(z1=pl.lit(0, dtype=pl.Int64))
    frames["inv"] = frames["inv"].with_columns(z2=pl.lit(0, dtype=pl.Int64))

    lf = (
        frames["sales"]
        .join(
            frames["inv"],
            left_on=pl.col("z1") + pl.col("s_item"),
            right_on=pl.col("z2") + pl.col("i_item"),
            coalesce=False,
        )
        .join(frames["wh"], left_on="i_wh", right_on="w_key", coalesce=False)
    )

    assert scan_order(lf.explain(optimizations=ON)) == ["sales", "inv", "wh"]

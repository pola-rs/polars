"""Tests for the join-reordering optimization (`QueryOptFlags(join_order=...)`)."""

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal

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


def star_query(frames: dict[str, pl.LazyFrame], **join_kwargs: object) -> pl.LazyFrame:
    """Fact joined to both dimensions; `join_kwargs` is forwarded to both joins."""
    kwargs: dict[str, object] = {"coalesce": False, **join_kwargs}
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


def test_join_order_is_off_by_default(tmp_path: Path) -> None:
    lf = star_query(star_frames(tmp_path))
    assert lf.explain() == lf.explain(optimizations=OFF)


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
    ctx = pl.SQLContext(**frames)
    q = """
        SELECT f_val
        FROM fact, dim_b, dim_a
        WHERE f_dim_b = b_key AND f_dim_a = a_key AND a_flag
    """
    lf = ctx.execute(q, eager=False)

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

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]
    assert_frame_equal(
        lf.collect(optimizations=OFF).sort(pl.all()),
        lf.collect(optimizations=ON).sort(pl.all()),
    )


def test_with_columns_on_a_leaf_is_estimated(tmp_path: Path) -> None:
    frames = star_frames(tmp_path)
    frames["fact"] = frames["fact"].with_columns(f_double=pl.col("f_val") * 2)
    lf = star_query(frames)

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]
    assert_frame_equal(
        lf.collect(optimizations=OFF).sort(pl.all()),
        lf.collect(optimizations=ON).sort(pl.all()),
    )


def joins_around(
    frames: dict[str, pl.LazyFrame], middle: pl.Expr | str
) -> pl.LazyFrame:
    """The star query with a projection sitting between the two joins."""
    return (
        frames["fact"]
        .join(frames["dim_b"], left_on="f_dim_b", right_on="b_key", coalesce=False)
        .select("f_dim_a", "f_val", middle)
        .join(
            frames["dim_a"].filter(pl.col("a_flag")),
            left_on="f_dim_a",
            right_on="a_key",
            coalesce=False,
        )
    )


def test_projection_between_joins_keeps_one_cluster(tmp_path: Path) -> None:
    # Without looking past it the outer join sees two leaves and no cluster forms.
    lf = joins_around(star_frames(tmp_path), "b_name")

    assert scan_order(lf.explain(optimizations=OFF)) == ["fact", "dim_b", "dim_a"]
    assert scan_order(lf.explain(optimizations=ON)) == ["fact", "dim_a", "dim_b"]
    assert_frame_equal(
        lf.collect(optimizations=OFF).sort(pl.all()),
        lf.collect(optimizations=ON).sort(pl.all()),
    )


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

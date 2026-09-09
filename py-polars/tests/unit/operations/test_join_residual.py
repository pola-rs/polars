"""Tests for predicates fused into an inner join as its residual match condition."""

from __future__ import annotations

import re

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

ENGINES = ["in-memory", "streaming"]


def assert_fused(lf: pl.LazyFrame) -> None:
    assert "RESIDUAL" in lf.explain()


def assert_not_fused(lf: pl.LazyFrame) -> None:
    assert "RESIDUAL" not in lf.explain()


def assert_native_streaming(lf: pl.LazyFrame) -> None:
    """The residual reached the streaming hash join rather than a fallback filter."""
    plan = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert plan is not None
    assert "equi-join" in plan
    assert "residual:" in plan
    assert "filter" not in plan


def reference(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    predicate: pl.Expr,
    **join_kwargs: object,
) -> pl.DataFrame:
    """Join and filter as two separate materialized steps, so nothing can fuse them."""
    return left.join(right, **join_kwargs).collect().filter(predicate)  # type: ignore[arg-type]


@pytest.fixture
def frames() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2, 3, 4],
            "a": [10, 20, 30, 40, 50, 60],
        }
    )
    right = pl.LazyFrame(
        {
            "k": [1, 2, 2, 3, 5],
            "b": [15, 25, 45, 5, 99],
        }
    )
    return left, right


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_matches_unfused_reference(
    frames: tuple[pl.LazyFrame, pl.LazyFrame], engine: str
) -> None:
    left, right = frames
    q = left.join(right, on="k").filter(pl.col("b") < pl.col("a"))

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, pl.col("b") < pl.col("a"), on="k"),
        check_row_order=False,
    )


def test_residual_native_streaming_path(
    frames: tuple[pl.LazyFrame, pl.LazyFrame],
) -> None:
    left, right = frames
    q = left.join(right, on="k").filter(pl.col("b") < pl.col("a"))
    assert_native_streaming(q)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("b") < pl.col("a"),
        pl.col("b") == pl.col("a"),
        pl.col("b") + pl.col("a") > 40,
        (pl.col("b") < pl.col("a")) & (pl.col("a") > 20),
        (pl.col("b") < pl.col("a")) | (pl.col("a") > 50),
        pl.col("a").cast(pl.Float64) > pl.col("b").cast(pl.Float64),
    ],
)
def test_residual_expressions(
    frames: tuple[pl.LazyFrame, pl.LazyFrame], predicate: pl.Expr, engine: str
) -> None:
    left, right = frames
    q = left.join(right, on="k").filter(predicate)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_composite_and_duplicate_keys(engine: str) -> None:
    left = pl.LazyFrame(
        {
            "k1": [1, 1, 1, 2, 2],
            "k2": ["a", "a", "b", "a", "a"],
            "v": [1, 2, 3, 4, 5],
        }
    )
    right = pl.LazyFrame(
        {
            "k1": [1, 1, 2, 2],
            "k2": ["a", "a", "a", "a"],
            "w": [0, 5, 1, 9],
        }
    )
    q = left.join(right, on=["k1", "k2"]).filter(pl.col("w") < pl.col("v"))

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, pl.col("w") < pl.col("v"), on=["k1", "k2"]),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_string_keys(engine: str) -> None:
    left = pl.LazyFrame({"k": ["x", "x", "y", "z"], "a": [1, 2, 3, 4]})
    right = pl.LazyFrame({"k": ["x", "y", "y", "w"], "b": [2, 1, 9, 0]})
    q = left.join(right, on="k").filter(pl.col("b") < pl.col("a"))

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, pl.col("b") < pl.col("a"), on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("nulls_equal", [True, False])
def test_residual_null_keys_and_values(engine: str, nulls_equal: bool) -> None:
    left = pl.LazyFrame({"k": [1, None, 2, None], "a": [10, 20, None, 40]})
    right = pl.LazyFrame({"k": [1, None, 2], "b": [1, 5, None]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k", nulls_equal=nulls_equal).filter(predicate)

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", nulls_equal=nulls_equal),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("left_k", "right_k", "predicate_true"),
    [
        ([1, 2], [3, 4], True),  # no equi matches
        ([1, 2], [1, 2], False),  # every candidate rejected
        ([], [1, 2], True),  # empty left
        ([1, 2], [], True),  # empty right
        ([], [], True),  # both empty
    ],
)
def test_residual_degenerate_inputs(
    engine: str, left_k: list[int], right_k: list[int], predicate_true: bool
) -> None:
    left = pl.LazyFrame(
        {"k": left_k, "a": [1] * len(left_k)}, schema={"k": pl.Int64, "a": pl.Int64}
    )
    right = pl.LazyFrame(
        {"k": right_k, "b": [1] * len(right_k)}, schema={"k": pl.Int64, "b": pl.Int64}
    )
    predicate = (
        pl.col("b") <= pl.col("a") if predicate_true else pl.col("b") < pl.col("a")
    )
    q = left.join(right, on="k").filter(predicate)

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("suffix", ["_right", "_R"])
@pytest.mark.parametrize("coalesce", [True, False])
def test_residual_suffix_and_coalesce(engine: str, suffix: str, coalesce: bool) -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "v": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2, 2], "v": [15, 5, 99]})
    predicate = pl.col(f"v{suffix}") < pl.col("v")
    q = left.join(right, on="k", suffix=suffix, coalesce=coalesce).filter(predicate)

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", suffix=suffix, coalesce=coalesce),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_only_columns_dropped_by_select(engine: str) -> None:
    """`v`/`v_right` are read only by the residual, so they must survive to the join."""
    left = pl.LazyFrame({"k": [1, 1, 2], "v": [10, 20, 30], "keep": ["a", "b", "c"]})
    right = pl.LazyFrame({"k": [1, 2, 2], "v": [15, 5, 99]})
    predicate = pl.col("v_right") < pl.col("v")
    q = left.join(right, on="k").filter(predicate).select("k", "keep")

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k").select("k", "keep"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_suffix_dropped_when_collision_disappears(engine: str) -> None:
    """Pruning the left `v` un-suffixes the right one; the residual must follow."""
    left = pl.LazyFrame({"k": [1, 1, 2], "v": [99, 99, 99], "x": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2, 2], "v": [15, 5, 99]})
    predicate = pl.col("v_right") < pl.col("x")
    q = left.join(right, on="k").filter(predicate).select("k", "x")

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k").select("k", "x"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_reads_right_key(engine: str) -> None:
    """The right key must survive even though nothing else projects it.

    Predicate pushdown may rewrite `k_right` to `k` and push it to one side, so this
    does not necessarily fuse; either way the answer must hold.
    """
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [5, 25, 35]})
    right = pl.LazyFrame({"k": [1, 2, 2], "b": [1, 2, 3]})
    predicate = pl.col("k_right") + pl.col("b") < pl.col("a")
    q = left.join(right, on="k", coalesce=False).filter(predicate).select("a", "b")

    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", coalesce=False).select("a", "b"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "maintain_order", ["none", "left", "right", "left_right", "right_left"]
)
def test_residual_maintain_order(engine: str, maintain_order: str) -> None:
    left = pl.LazyFrame({"k": [1, 2, 1, 3, 2], "a": [10, 20, 30, 40, 50]})
    right = pl.LazyFrame({"k": [1, 2, 2, 3], "b": [5, 15, 45, 5]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k", maintain_order=maintain_order).filter(predicate)  # type: ignore[arg-type]

    expected = (
        left.join(right, on="k", maintain_order=maintain_order)  # type: ignore[arg-type]
        .collect()
        .filter(predicate)
    )
    result = q.collect(engine=engine)

    # `left`/`right` only pin the order of that side, so compare the side that is
    # actually specified; the two-sided modes pin the whole row order.
    if maintain_order in ("left_right", "right_left"):
        assert_frame_equal(result, expected)
    else:
        assert_frame_equal(result, expected, check_row_order=False)
        if maintain_order == "left":
            assert result["a"].to_list() == expected["a"].to_list()
        elif maintain_order == "right":
            assert result["b"].to_list() == expected["b"].to_list()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("build_side", ["force_left", "force_right"])
def test_residual_forced_build_side(engine: str, build_side: str) -> None:
    left = pl.LazyFrame({"k": [1, 1, 2, 3], "a": [10, 20, 30, 40]})
    right = pl.LazyFrame({"k": [1, 2, 2, 4], "b": [5, 15, 45, 0]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k", build_side=build_side).filter(predicate)  # type: ignore[arg-type]

    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_many_morsels_and_skew(engine: str) -> None:
    """More rows than one morsel, plus a key whose duplicate list exceeds the limit."""
    n = 60_000
    left = pl.LazyFrame({"k": [0] * n + list(range(n)), "a": list(range(2 * n))})
    right = pl.LazyFrame({"k": [0] * 32 + list(range(n)), "b": list(range(32 + n))})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k").filter(predicate)

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_rejected_batch_followed_by_accepted(engine: str) -> None:
    """Leading candidates are all rejected; later survivors must still be emitted."""
    n = 20_000
    left = pl.LazyFrame({"k": [1] * n + [2] * n, "a": [0] * n + [100] * n})
    right = pl.LazyFrame({"k": [1, 2], "b": [50, 50]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k").filter(predicate)

    result = q.collect(engine=engine)
    assert result.height == n
    assert_frame_equal(
        result, reference(left, right, predicate, on="k"), check_row_order=False
    )


@pytest.mark.parametrize("how", ["left", "right", "full"])
def test_residual_not_fused_while_join_stays_outer(how: str) -> None:
    """Keeping null-extended rows leaves the join outer, so it cannot fuse."""
    left = pl.LazyFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2, 4], "b": [5, 25, 0]})
    predicate = pl.col("b").is_null() | (pl.col("b") < pl.col("a"))

    q = left.join(right, on="k", how=how).filter(predicate)  # type: ignore[arg-type]
    assert_not_fused(q)
    assert_frame_equal(
        q.collect(),
        reference(left, right, predicate, on="k", how=how),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("how", ["left", "right", "full"])
def test_residual_outer_join_reduced_to_inner(engine: str, how: str) -> None:
    """Rejecting null-extended rows turns the join inner, which may then fuse."""
    left = pl.LazyFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2, 4], "b": [5, 25, 0]})
    predicate = pl.col("b") < pl.col("a")

    q = left.join(right, on="k", how=how).filter(predicate)  # type: ignore[arg-type]
    assert "INNER JOIN" in q.explain()
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", how=how),
        check_row_order=False,
    )


def test_residual_not_fused_for_non_elementwise_predicates() -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2, 2], "b": [5, 15, 45]})

    # An aggregate is not decidable per candidate pair.
    assert_not_fused(left.join(right, on="k").filter(pl.col("b") < pl.col("a").max()))
    # Neither is a window function.
    assert_not_fused(
        left.join(right, on="k").filter(pl.col("b") < pl.col("a").sum().over("k"))
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_slice_ordering(engine: str) -> None:
    left = pl.LazyFrame({"k": [1, 1, 2, 2], "a": [10, 20, 30, 40]})
    right = pl.LazyFrame({"k": [1, 2], "b": [15, 5]})
    predicate = pl.col("b") < pl.col("a")

    # filter-then-slice: the slice must stay above the join.
    q = left.join(right, on="k", maintain_order="left_right").filter(predicate).head(1)
    assert q.collect(engine=engine).height == 1

    # slice-then-filter: the slice must not be absorbed past the filter.
    q = left.join(right, on="k", maintain_order="left_right").head(2).filter(predicate)
    expected = (
        left.join(right, on="k", maintain_order="left_right")
        .collect()
        .head(2)
        .filter(predicate)
    )
    assert_frame_equal(q.collect(engine=engine), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_validation_still_errors(engine: str) -> None:
    """Validation runs on the keys; a residual rejecting the pair does not excuse it."""
    left = pl.LazyFrame({"k": [1, 2], "a": [10, 20]})
    right = pl.LazyFrame({"k": [1, 1, 2], "b": [99, 99, 99]})
    q = left.join(right, on="k", validate="m:1").filter(pl.col("b") < pl.col("a"))

    with pytest.raises(ComputeError):
        q.collect(engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("join_order", [True, False])
def test_residual_with_join_order(engine: str, join_order: bool) -> None:
    a = pl.LazyFrame({"k": [1, 2, 3], "x": [1, 2, 3]})
    b = pl.LazyFrame({"k": [1, 2, 3], "y": [3, 2, 1]})
    c = pl.LazyFrame({"k": [1, 2, 3], "z": [1, 1, 1]})
    predicate = pl.col("y") > pl.col("x")

    q = a.join(b, on="k").join(c, on="k").filter(predicate)
    result = q.collect(
        engine=engine, optimizations=pl.QueryOptFlags(join_order=join_order)
    )
    expected = a.join(b, on="k").join(c, on="k").collect().filter(predicate)
    assert_frame_equal(result, expected, check_row_order=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_residual_shared_join_with_distinct_predicates(engine: str) -> None:
    """One join feeding two filters must not have either fused into the other."""
    left = pl.LazyFrame({"k": [1, 1, 2, 2], "a": [10, 20, 30, 40]})
    right = pl.LazyFrame({"k": [1, 2], "b": [15, 35]})
    joined = left.join(right, on="k")

    lo = joined.filter(pl.col("b") < pl.col("a"))
    hi = joined.filter(pl.col("b") > pl.col("a"))
    q = pl.concat([lo, hi])

    expected = pl.concat(
        [
            reference(left, right, pl.col("b") < pl.col("a"), on="k"),
            reference(left, right, pl.col("b") > pl.col("a"), on="k"),
        ]
    )
    assert_frame_equal(q.collect(engine=engine), expected, check_row_order=False)


def test_residual_optimization_is_idempotent() -> None:
    """Optimizing the same plan repeatedly must not lose or duplicate the residual."""
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2], "b": [15, 5]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k").filter(predicate)

    assert q.explain().count("RESIDUAL") == 1
    assert q.explain() == q.explain()

    roundtripped = pl.LazyFrame.deserialize(q.serialize())
    assert roundtripped.explain().count("RESIDUAL") == 1
    assert_frame_equal(
        roundtripped.collect(),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


def test_residual_unfused_when_predicate_pushdown_disabled() -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 2], "b": [15, 5]})
    predicate = pl.col("b") < pl.col("a")
    q = left.join(right, on="k").filter(predicate)

    unfused = q.explain(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    assert "RESIDUAL" not in unfused
    assert "FILTER" in unfused

    assert_frame_equal(
        q.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False)),
        q.collect(),
        check_row_order=False,
    )


def join_keys(lf: pl.LazyFrame) -> tuple[str, str]:
    plan = lf.explain()
    left = re.search(r"LEFT PLAN ON: \[(.*)\]", plan)
    right = re.search(r"RIGHT PLAN ON: \[(.*)\]", plan)
    assert left is not None
    assert right is not None
    return left.group(1), right.group(1)


@pytest.fixture
def eq_frames() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left = pl.LazyFrame(
        {
            "k": [1, 1, 2, 2, 3],
            "a": [10, 20, 30, 40, 50],
        }
    )
    right = pl.LazyFrame(
        {
            "k": [1, 1, 2, 3, 3],
            "b": [10, 99, 40, 50, 60],
        }
    )
    return left, right


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_becomes_join_key(
    eq_frames: tuple[pl.LazyFrame, pl.LazyFrame], engine: str
) -> None:
    left, right = eq_frames
    predicate = pl.col("a") == pl.col("b")
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    left_keys, right_keys = join_keys(q)
    assert 'col("a")' in left_keys
    assert 'col("b")' in right_keys

    assert q.collect_schema().names() == ["k", "a", "b"]
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_key_rejects_nulls(engine: str) -> None:
    left = pl.LazyFrame(
        {"k": [1, 1, 2], "a": [None, 5, None]}, schema_overrides={"a": pl.Int64}
    )
    right = pl.LazyFrame(
        {"k": [1, 2, 2], "b": [None, None, 5]}, schema_overrides={"b": pl.Int64}
    )
    predicate = pl.col("a") == pl.col("b")
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_not_promoted_when_join_matches_nulls(
    eq_frames: tuple[pl.LazyFrame, pl.LazyFrame], engine: str
) -> None:
    """`==` rejects a null pair, so it cannot become a key that matches them."""
    left, right = eq_frames
    predicate = pl.col("a") == pl.col("b")
    q = left.join(right, on="k", nulls_equal=True).filter(predicate)

    assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", nulls_equal=True),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_eq_missing_matches_the_join(engine: str, nulls_equal: bool) -> None:
    left = pl.LazyFrame(
        {"k": [1, 1, 2], "a": [None, 5, None]}, schema_overrides={"a": pl.Int64}
    )
    right = pl.LazyFrame(
        {"k": [1, 2, 2], "b": [None, None, 5]}, schema_overrides={"b": pl.Int64}
    )
    predicate = pl.col("a").eq_missing(pl.col("b"))
    q = left.join(right, on="k", nulls_equal=nulls_equal).filter(predicate)

    # `eq_missing` accepts a null pair, which only a null-matching join key reproduces.
    if nulls_equal:
        assert_not_fused(q)
    else:
        assert_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", nulls_equal=nulls_equal),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_float_equality_becomes_join_key(engine: str) -> None:
    """`==` and a key pair agree on NaN, so floats promote like any other dtype."""
    nan = float("nan")
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [1.5, nan, nan]})
    right = pl.LazyFrame({"k": [1, 2, 2], "b": [1.5, nan, 2.5]})
    predicate = pl.col("a") == pl.col("b")
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_promoted_and_rest_stays_residual(
    eq_frames: tuple[pl.LazyFrame, pl.LazyFrame], engine: str
) -> None:
    left, right = eq_frames
    predicate = (pl.col("a") == pl.col("b")) & (pl.col("a") + pl.col("b") > 40)
    q = left.join(right, on="k").filter(predicate)

    assert_fused(q)
    left_keys, _ = join_keys(q)
    assert 'col("a")' in left_keys
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_promoted_with_suffixed_column(engine: str) -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 99, 30]})
    predicate = pl.col("a") == pl.col("a_right")
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    assert q.collect_schema().names() == ["k", "a", "a_right"]
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_several_equalities_sharing_a_left_column(engine: str) -> None:
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [10, 20, 30]})
    right = pl.LazyFrame({"k": [1, 1, 2], "b": [10, 20, 30], "c": [10, 99, 30]})
    predicate = (pl.col("a") == pl.col("b")) & (pl.col("a") == pl.col("c"))
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    assert q.collect_schema().names() == ["k", "a", "b", "c"]
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_promoted_without_coalesce(
    eq_frames: tuple[pl.LazyFrame, pl.LazyFrame], engine: str
) -> None:
    left, right = eq_frames
    predicate = pl.col("a") == pl.col("b")
    q = left.join(right, on="k", coalesce=False).filter(predicate)

    assert_not_fused(q)
    assert q.collect_schema().names() == ["k", "a", "k_right", "b"]
    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", coalesce=False),
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_equality_resolves_suffixed_output_column(engine: str) -> None:
    """An output name can belong to a different column than it shares a name with."""
    left = pl.LazyFrame({"k": [1, 1], "v": [10, 20]})
    right = pl.LazyFrame({"v_right": [1, 1], "v": [10, 20]})
    predicate = pl.col("v") == pl.col("v_right")
    q = left.join(right, left_on="k", right_on="v_right").filter(predicate)

    assert_not_fused(q)
    assert q.collect_schema().names() == ["k", "v", "v_right"]

    # This shape trips a projection pushdown bug that has nothing to do with the join
    # condition, so the reference is taken without that pass.
    no_pushdown = pl.QueryOptFlags(projection_pushdown=False)
    expected = (
        left.join(right, left_on="k", right_on="v_right")
        .collect(optimizations=no_pushdown)
        .filter(predicate)
    )
    assert_frame_equal(
        q.collect(engine=engine, optimizations=no_pushdown),
        expected,
        check_row_order=False,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_promoted_key_keeps_right_payload_in_merge_join(engine: str) -> None:
    """Sorted inputs take the merge join, which coalesces by name."""
    left = pl.LazyFrame({"k": [1, 1, 2], "v": [10, 20, 30]}).sort("k", "v")
    right = pl.LazyFrame({"k": [1, 1, 2], "v": [10, 99, 30]}).sort("k", "v")
    predicate = pl.col("v") == pl.col("v_right")
    q = left.join(right, on="k", maintain_order="left_right").filter(predicate)

    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k", maintain_order="left_right"),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_fallible_inside_list_eval_is_not_promoted(engine: str) -> None:
    """A key is evaluated on every row, so a nested fallible cast must block it."""
    left = pl.LazyFrame({"k": [1, 2], "a": [["1"], ["bad"]]})
    right = pl.LazyFrame({"k": [1], "b": [[1]]})
    predicate = pl.col("a").list.eval(pl.element().cast(pl.Int64)) == pl.col("b")
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    assert_frame_equal(
        q.collect(engine=engine), reference(left, right, predicate, on="k")
    )


@pytest.mark.parametrize("maintain_order", ["left", "right", "left_right"])
def test_residual_not_native_when_order_is_preserved(maintain_order: str) -> None:
    """The ordered probe would evaluate the residual a group at a time."""
    left = pl.LazyFrame({"k": [1, 1, 2], "a": [1, 5, 9]})
    right = pl.LazyFrame({"k": [1, 2, 2], "b": [3, 4, 7]})
    predicate = pl.col("a") < pl.col("b")
    q = left.join(right, on="k", maintain_order=maintain_order).filter(predicate)  # type: ignore[arg-type]

    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert plan is not None
    assert "residual:" not in plan
    assert "filter" in plan

    assert_frame_equal(
        q.collect(engine="streaming"),
        reference(left, right, predicate, on="k", maintain_order=maintain_order),
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_nondeterministic_equality_is_not_promoted(engine: str) -> None:
    """A key is drawn once per input row; the filter draws once per candidate pair."""
    left = pl.LazyFrame({"k": [0, 1], "a": [list(range(10)), list(range(10, 20))]})
    right = pl.LazyFrame({"k": [0] * 10 + [1] * 10, "b": range(20)})
    predicate = pl.col("a").list.sample(n=1).list.first() == pl.col("b")
    q = left.join(right, on="k", maintain_order="left_right").filter(predicate)

    assert_fused(q)
    left_keys, _ = join_keys(q)
    assert "sample" not in left_keys

    # Promotion pins the height at one row per key; per-pair sampling does not.
    assert len({q.collect(engine=engine).height for _ in range(30)}) > 1


@pytest.mark.parametrize("engine", ENGINES)
def test_strict_regex_equality_is_not_promoted(engine: str) -> None:
    """An invalid pattern on an unmatched row must not be evaluated."""
    left = pl.LazyFrame({"k": [1, 2], "pat": ["x", "["]})
    right = pl.LazyFrame({"k": [1], "b": [True]})
    predicate = pl.lit("x").str.contains(pl.col("pat")) == pl.col("b")
    q = left.join(right, on="k").filter(predicate)

    assert_frame_equal(
        q.collect(engine=engine), reference(left, right, predicate, on="k")
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_non_strict_cast_equality_is_promoted(engine: str) -> None:
    left = pl.LazyFrame({"k": [1, 1], "a": [10, 20]})
    right = pl.LazyFrame({"k": [1, 1], "b": [10, 99]})
    predicate = pl.col("a").cast(pl.Int32, strict=False) == pl.col("b").cast(
        pl.Int32, strict=False
    )
    q = left.join(right, on="k").filter(predicate)

    assert_not_fused(q)
    left_keys, right_keys = join_keys(q)
    assert "cast" in left_keys
    assert "cast" in right_keys

    assert_frame_equal(
        q.collect(engine=engine),
        reference(left, right, predicate, on="k"),
        check_row_order=False,
    )

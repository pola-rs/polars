"""Inner joins and left/semi/anti joins reordered, `LEFT JOIN … IS NULL` -> anti."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import JoinStrategy, JoinValidation
    from tests.conftest import PlMonkeyPatch

ON = pl.QueryOptFlags(join_order=True)
OFF = pl.QueryOptFlags(join_order=False)


def frames() -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    n = 1000
    fact = pl.LazyFrame(
        {
            "f_id": list(range(n)),
            "f_ret": [i if i % 3 else None for i in range(n)],
            "f_dim": [i % 50 for i in range(n)],
            "f_val": [float(i) for i in range(n)],
        }
    )
    # Matches half of the non-null fact keys, some of them twice.
    matched = [i for i in range(n) if i % 3 == 1]
    returns = pl.LazyFrame(
        {
            "r_ret": matched + [1, 4, None],
            "r_amt": [1.0] * len(matched) + [2.0, 3.0, 4.0],
        }
    )
    dim = pl.LazyFrame(
        {"d_key": list(range(50)), "d_flag": [i % 10 == 0 for i in range(50)]}
    )
    return fact, returns, dim


def all_joins(plan: str) -> list[str]:
    return [line.strip() for line in plan.splitlines() if "JOIN:" in line]


def first_join(plan: str) -> str:
    return all_joins(plan)[0]


def assert_same_result(
    lf: pl.LazyFrame, on: pl.QueryOptFlags = ON, off: pl.QueryOptFlags = OFF
) -> None:
    expected = lf.collect(optimizations=off)
    for engine in ("in-memory", "streaming"):
        out = lf.collect(engine=engine, optimizations=on)
        assert out.schema == lf.collect_schema()
        assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("how", ["left", "semi", "anti"])
def test_inner_join_moves_below_preserving_join(how: JoinStrategy) -> None:
    fact, returns, dim = frames()
    # Filtering the right side is what lets the estimator price a semi/anti join.
    returns = returns.filter(pl.col("r_amt") < 1.5)
    lf = fact.join(returns, left_on="f_ret", right_on="r_ret", how=how).join(
        dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key"
    )
    assert first_join(lf.explain(optimizations=OFF)) == "INNER JOIN:"
    assert first_join(lf.explain(optimizations=ON)) == f"{how.upper()} JOIN:"
    assert_same_result(lf)


def test_unpriceable_anti_join_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="anti").join(
            dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key"
        )
    )


def test_left_join_is_null_becomes_anti_join() -> None:
    fact, returns, dim = frames()
    lf = (
        fact.join(
            returns, left_on="f_ret", right_on="r_ret", how="left", coalesce=False
        )
        .filter(pl.col("r_ret").is_null())
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
        .select("f_id", "f_val", "d_flag")
    )
    plan = lf.explain(optimizations=ON)
    assert first_join(plan) == "ANTI JOIN:"
    assert "is_null" not in plan
    assert_same_result(lf)


def test_anti_rewrite_keeps_null_extended_columns() -> None:
    fact, returns, _ = frames()
    lf = fact.join(
        returns, left_on="f_ret", right_on="r_ret", how="left", coalesce=False
    ).filter(pl.col("r_ret").is_null())
    plan = lf.explain(optimizations=ON)
    assert "ANTI JOIN:" in plan
    assert_same_result(lf)
    out = lf.collect(optimizations=ON)
    assert out["r_amt"].null_count() == out.height


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_anti_rewrite_multi_key(nulls_equal: bool) -> None:
    left = pl.LazyFrame(
        {"k1": [1, 1, 2, None, 3], "k2": [1, 2, 1, 1, None], "v": [1, 2, 3, 4, 5]}
    )
    right = pl.LazyFrame(
        {"k1": [1, 1, None, 3], "k2": [1, 1, 1, None], "w": [7, 8, 9, 10]}
    )
    for keys in (["k1_right"], ["k1_right", "k2_right"]):
        lf = left.join(
            right, on=["k1", "k2"], how="left", coalesce=False, nulls_equal=nulls_equal
        ).filter(pl.all_horizontal(pl.col(k).is_null() for k in keys))
        assert ("ANTI JOIN:" in lf.explain(optimizations=ON)) is not nulls_equal
        assert_same_result(lf)


def test_anti_rewrite_needs_a_plain_key_column() -> None:
    left = pl.LazyFrame({"k": [0, 1, 2], "v": [1, 2, 3]})
    right = pl.LazyFrame({"k": [None, 1], "w": [7, 8]})
    lf = left.join(
        right,
        left_on="k",
        right_on=pl.col("k").fill_null(0),
        how="left",
        coalesce=False,
    ).filter(pl.col("k_right").is_null())
    assert "ANTI JOIN:" not in lf.explain(optimizations=ON)
    assert_same_result(lf)
    # The null right key matched k == 0, so the row survives the filter.
    assert lf.collect(optimizations=ON)["v"].sort().to_list() == [1, 3]


def test_cross_side_filter_above_inner_join_stays_fusable() -> None:
    fact, returns, dim = frames()
    returns = returns.filter(pl.col("r_amt") < 1.5)
    lf = (
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="anti")
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
        .filter((pl.col("f_val") > 500) | pl.col("d_flag").not_())
    )
    plan = lf.explain(optimizations=ON)
    assert first_join(plan) == "ANTI JOIN:"
    graph = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "fused predicate" in graph
    assert_same_result(lf)


def test_coalescing_inner_join_on_the_outer_key_moves() -> None:
    a = pl.LazyFrame({"k": list(range(100)), "v": list(range(100))})
    b = pl.LazyFrame({"k": [i * 2 for i in range(30)], "w": list(range(30))})
    c = pl.LazyFrame({"k": [5, 6], "x": ["p", "q"]})
    lf = a.join(b, on="k", how="left").join(c, on="k")
    assert first_join(lf.explain(optimizations=ON)) == "LEFT JOIN:"
    assert_same_result(lf)


def test_stacked_outer_joins_are_all_crossed() -> None:
    fact, returns, dim = frames()
    other = pl.LazyFrame({"o_id": [1, 2, 3], "o_v": [1, 2, 3]})
    lf = (
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left")
        .join(other, left_on="f_id", right_on="o_id", how="left")
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
    )
    plan = lf.explain(optimizations=ON)
    assert all_joins(plan) == ["LEFT JOIN:", "LEFT JOIN:", "INNER JOIN:"]
    assert_same_result(lf)


def not_rewritten(lf: pl.LazyFrame) -> None:
    assert lf.explain(optimizations=ON) == lf.explain(optimizations=OFF)
    assert_same_result(lf)


def test_validated_outer_join_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        fact.join(
            returns.unique("r_ret"),
            left_on="f_id",
            right_on="r_ret",
            how="left",
            validate="1:1",
        ).join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
    )


def test_outer_join_on_the_right_input_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        dim.filter(pl.col("d_flag")).join(
            fact.join(returns, left_on="f_ret", right_on="r_ret", how="left"),
            left_on="d_key",
            right_on="f_dim",
        )
    )


def test_projection_between_the_joins_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left")
        .select("f_dim", "r_amt")
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
    )


def test_key_from_the_null_extended_side_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left").join(
            dim.filter(pl.col("d_flag")),
            left_on=pl.col("r_amt").cast(pl.Int64),
            right_on="d_key",
        )
    )


def test_name_collision_with_the_null_extended_side_is_left_alone() -> None:
    fact, returns, dim = frames()
    not_rewritten(
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left").join(
            dim.filter(pl.col("d_flag")).rename({"d_flag": "r_amt"}),
            left_on="f_dim",
            right_on="d_key",
        )
    )


@pytest.mark.slow
def test_expanding_inner_join_is_left_alone() -> None:
    fact, returns, _ = frames()
    wide = pl.LazyFrame({"w_key": [i % 50 for i in range(5000)], "w_v": range(5000)})
    not_rewritten(
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left").join(
            wide, left_on="f_dim", right_on="w_key"
        )
    )


def test_sql_left_join_is_null_idiom() -> None:
    from tests.unit.sql.asserts import assert_sql_matches

    fact, returns, dim = frames()
    frames_ = {"fact": fact, "returns": returns, "dim": dim}
    query = """
        SELECT f_id, f_val, d_key
        FROM fact
        LEFT JOIN returns ON r_ret = f_ret
        JOIN dim ON d_key = f_dim
        WHERE r_ret IS NULL AND d_flag
        ORDER BY f_id
    """
    plan = pl.SQLContext(frames=frames_).execute(query).explain(optimizations=ON)
    assert first_join(plan) == "ANTI JOIN:"
    assert_sql_matches(frames_, query=query, compare_with="duckdb")


def test_every_inner_join_above_the_outer_join_moves() -> None:
    fact, returns, dim = frames()
    other = pl.LazyFrame({"o_key": [0, 1, 2, 3], "o_v": [1, 2, 3, 4]})
    lf = (
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left")
        .join(other, left_on="f_dim", right_on="o_key")
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
    )
    plan = lf.explain(optimizations=ON)
    assert all_joins(plan) == ["LEFT JOIN:", "INNER JOIN:", "INNER JOIN:"]
    assert_same_result(lf)


@pytest.mark.parametrize("validate", ["m:1", "1:m", "1:1"])
def test_anti_rewrite_keeps_validation(validate: JoinValidation) -> None:
    left = pl.LazyFrame({"a": [1, 1, 2]})
    right = pl.LazyFrame({"b": [1, 1]})
    lf = left.join(
        right, left_on="a", right_on="b", how="left", coalesce=False, validate=validate
    ).filter(pl.col("b").is_null())
    assert "ANTI JOIN:" not in lf.explain(optimizations=ON)
    for engine in ("in-memory", "streaming"):
        with pytest.raises(pl.exceptions.ComputeError, match="validation"):
            lf.collect(engine=engine, optimizations=ON)


def test_suffix_collision_in_the_candidate_is_left_alone() -> None:
    a = pl.LazyFrame({"k": [1, 2, 3], "x": [1, 2, 3]})
    b = pl.LazyFrame({"bk": [1, 2], "x": [10, 20]})
    c = pl.LazyFrame({"ck": [1], "x_right": [100]})
    lf = a.join(b, left_on="k", right_on="bk", how="left").join(
        c, left_on="k", right_on="ck"
    )
    assert lf.collect_schema().names() == ["k", "x", "x_right", "x_right_right"]
    not_rewritten(lf)


def test_fallible_filter_does_not_move(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")
    left = pl.LazyFrame({"k": [1, 2, 3], "s": ["1", "2", "bad"]})
    right = pl.LazyFrame({"k": [3], "w": [0]})
    dim = pl.LazyFrame({"d": [1, 2, 3]})
    # The anti join removes the row whose cast raises.
    lf = (
        left.join(right.filter(pl.col("w") == 0), on="k", how="anti")
        .join(dim.filter(pl.col("d") < 3), left_on="k", right_on="d")
        .filter(pl.col("s").cast(pl.Int64) > 0)
    )
    assert first_join(lf.explain(optimizations=ON)) == "ANTI JOIN:"
    assert_same_result(lf)
    # The same cast between the joins, and as the outer join's key: those would see
    # fewer rows after the rewrite, so the plan is left alone.
    left = left.with_columns(pl.col("k").cast(pl.String).alias("s"))
    lf = (
        left.join(right, on="k", how="left", coalesce=False)
        .filter(pl.col("s").cast(pl.Int64) > 0)
        .join(dim.filter(pl.col("d") < 3), left_on="k", right_on="d")
    )
    not_rewritten(lf)
    lf = left.join(
        right.with_columns(pl.col("k").cast(pl.String)),
        left_on=pl.col("s").cast(pl.Int64).cast(pl.String),
        right_on="k",
        how="left",
        coalesce=False,
    ).join(dim.filter(pl.col("d") < 3), left_on="k", right_on="d")
    not_rewritten(lf)


def test_filter_reaches_the_innermost_inner_join_through_stacked_outer_joins() -> None:
    fact, returns, dim = frames()
    other = pl.LazyFrame({"o_id": [1, 2, 3], "o_v": [1, 2, 3]})
    lf = (
        fact.join(returns, left_on="f_ret", right_on="r_ret", how="left")
        .join(other, left_on="f_id", right_on="o_id", how="left")
        .join(dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key")
        .filter((pl.col("f_val") > 500) | pl.col("d_flag").not_())
    )
    graph = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "fused predicate" in graph
    assert_same_result(lf)


def test_fixture_has_matches_and_misses() -> None:
    fact, returns, _ = frames()
    joined = fact.join(
        returns, left_on="f_ret", right_on="r_ret", how="left", coalesce=False
    )
    counts = joined.select(
        matched=pl.col("r_ret").is_not_null().sum(), total=pl.len()
    ).collect()
    assert 0 < counts["matched"][0] < counts["total"][0]


def test_user_function_does_not_move(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")
    left = pl.LazyFrame({"k": [1, 2, 3], "s": ["1", "2", "bad"]})
    right = pl.LazyFrame({"k": [3], "w": [0]})
    dim = pl.LazyFrame({"d": [1, 2, 3]})
    lf = (
        left.join(right.filter(pl.col("w") == 0), on="k", how="anti")
        .join(dim.filter(pl.col("d") <= 3), left_on="k", right_on="d")
        .filter(
            pl.col("s").map_batches(
                lambda s: s.cast(pl.Int64), return_dtype=pl.Int64, is_elementwise=True
            )
            > 0
        )
    )
    # Predicate pushdown moves the function on its own; keep it out of the picture.
    on = pl.QueryOptFlags(join_order=True, predicate_pushdown=False)
    off = pl.QueryOptFlags(join_order=False, predicate_pushdown=False)
    assert first_join(lf.explain(optimizations=on)) == "ANTI JOIN:"
    assert_same_result(lf, on=on, off=off)


def test_filter_stays_on_the_join_where_it_fuses() -> None:
    a = pl.LazyFrame({"k": list(range(100)), "av": list(range(100))})
    b = pl.LazyFrame({"bk": list(range(0, 100, 2)), "bv": list(range(50))})
    e = pl.LazyFrame({"ek": list(range(0, 100, 5)), "ev": list(range(20))})
    c = pl.LazyFrame({"ck": [1, 2, 3, 4, 5], "cv": [1, 2, 3, 4, 5]})
    lf = (
        a.join(b, left_on="k", right_on="bk", how="left")
        .join(e, left_on="k", right_on="ek", how="left")
        .join(c, left_on="k", right_on="ck")
        .filter((pl.col("bv") > 1) | (pl.col("cv") > 1))
    )
    graph = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "fused predicate" in graph
    # C moved below E but stays above B, where the predicate can be fused.
    assert all_joins(lf.explain(optimizations=ON)) == [
        "LEFT JOIN:",
        "INNER JOIN:",
        "LEFT JOIN:",
    ]
    assert_same_result(lf)


def test_first_push_keeps_the_filter_fused() -> None:
    a = pl.LazyFrame({"k": list(range(100)), "av": list(range(100))})
    b = pl.LazyFrame({"bk": list(range(0, 100, 2)), "bv": list(range(50))})
    c = pl.LazyFrame({"ck": [1, 2, 3, 4, 5], "cv": [1, 2, 3, 4, 5]})
    lf = (
        a.join(b, left_on="k", right_on="bk", how="left")
        .join(c, left_on="k", right_on="ck")
        .filter((pl.col("bv") > 1) | (pl.col("cv") > 1))
    )
    graph = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "fused predicate" in graph
    not_rewritten(lf)


def scanned_frames(tmp_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Parquet scans, so the estimator knows the key ranges."""
    fact, _, dim = frames()
    paths = {"fact": fact, "dim": dim}
    for name, lf in paths.items():
        lf.collect().write_parquet(tmp_path / f"{name}.parquet")
    return (
        pl.scan_parquet(tmp_path / "fact.parquet"),
        pl.scan_parquet(tmp_path / "dim.parquet"),
    )


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_selective_semi_anti_join_moves_below_inner_join(
    tmp_path: Path, how: JoinStrategy
) -> None:
    fact, dim = scanned_frames(tmp_path)
    wanted = pl.LazyFrame({"w": [3, 7]})
    # The inner join keeps every fact row; the semi join keeps two of them.
    lf = fact.join(dim, left_on="f_dim", right_on="d_key").join(
        wanted, left_on="f_id", right_on="w", how=how
    )
    assert all_joins(lf.explain(optimizations=OFF)) == [
        f"{how.upper()} JOIN:",
        "INNER JOIN:",
    ]
    assert all_joins(lf.explain(optimizations=ON)) == [
        "INNER JOIN:",
        f"{how.upper()} JOIN:",
    ]
    assert_same_result(lf)


def test_semi_join_moves_into_the_right_side_and_through_a_filter(
    tmp_path: Path,
) -> None:
    fact, dim = scanned_frames(tmp_path)
    wanted = pl.LazyFrame({"w": [10]})
    # A filter reading both sides stays between the joins.
    lf = (
        fact.join(dim, left_on="f_dim", right_on="d_key", coalesce=False)
        .filter(pl.col("f_val") > pl.col("d_flag").cast(pl.Float64) * 100)
        .join(wanted, left_on="d_key", right_on="w", how="semi")
    )
    plan = lf.explain(optimizations=ON)
    assert all_joins(plan) == ["INNER JOIN:", "SEMI JOIN:"]
    # The semi join sits on the dimension, the filter stays on the inner join.
    assert plan.index("SEMI JOIN:") > plan.index("FILTER")
    assert_same_result(lf)


def test_semi_join_that_keeps_more_than_the_inner_join_stays(
    tmp_path: Path,
) -> None:
    fact, dim = scanned_frames(tmp_path)
    # Half the fact keys, against an inner join that keeps a fifth of the rows.
    wanted = pl.LazyFrame({"w": list(range(0, 1000, 2))})
    lf = fact.join(
        dim.filter(pl.col("d_flag")), left_on="f_dim", right_on="d_key"
    ).join(wanted, left_on="f_id", right_on="w", how="semi")
    assert all_joins(lf.explain(optimizations=ON)) == ["SEMI JOIN:", "INNER JOIN:"]
    assert_same_result(lf)


def qualifying_keys(fact: pl.LazyFrame, at_least: float) -> pl.LazyFrame:
    """A `HAVING` shape: the estimator only knows its rows are at most the fact's."""
    return (
        fact.group_by("f_dim")
        .agg(pl.col("f_val").sum().alias("total"))
        .filter(pl.col("total") >= at_least)
        .select("f_dim")
    )


@pytest.mark.parametrize("at_least", [10_000.0, 0.0])
def test_semi_join_probing_no_more_rows_moves_below_inner_join(
    tmp_path: Path, at_least: float
) -> None:
    fact, dim = scanned_frames(tmp_path)
    # The inner join keeps every fact row, so the semi join probes the same rows on
    # the fact alone, however many groups qualify.
    lf = fact.join(dim, left_on="f_dim", right_on="d_key").join(
        qualifying_keys(fact, at_least), on="f_dim", how="semi"
    )
    assert all_joins(lf.explain(optimizations=OFF))[:2] == [
        "SEMI JOIN:",
        "INNER JOIN:",
    ]
    assert all_joins(lf.explain(optimizations=ON))[:2] == [
        "INNER JOIN:",
        "SEMI JOIN:",
    ]
    assert_same_result(lf)


def test_semi_join_above_a_selective_filter_stays(tmp_path: Path) -> None:
    fact, dim = scanned_frames(tmp_path)
    # The filter reads both sides, so it stays on the inner join. Pushed, the semi
    # join would probe more rows than reach it now.
    lf = (
        fact.join(dim, left_on="f_dim", right_on="d_key")
        .filter(pl.col("f_val") + pl.col("d_flag").cast(pl.Float64) < 100)
        .join(qualifying_keys(fact, 10_000.0), on="f_dim", how="semi")
    )
    assert all_joins(lf.explain(optimizations=ON))[:2] == [
        "SEMI JOIN:",
        "INNER JOIN:",
    ]
    assert_same_result(lf)


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_pushed_down_join_keys_a_suffixed_column_by_its_own_name(
    how: JoinStrategy,
) -> None:
    a = pl.LazyFrame({"k": [0, 1, 0, 1], "x": [10, 11, 12, 13]})
    # `x_right` is coalesced away, and `x` comes out as `x_right`.
    b = pl.LazyFrame({"x_right": [0, 1], "x": [100, 101]})
    wanted = pl.LazyFrame({"w": [101]})
    lf = a.join(b, left_on="k", right_on="x_right").join(
        wanted, left_on="x_right", right_on="w", how=how
    )
    # Projection pushdown cannot resolve `x_right` in this shape on its own.
    on = pl.QueryOptFlags(join_order=True, projection_pushdown=False)
    off = pl.QueryOptFlags(join_order=False, projection_pushdown=False)
    plan = lf.explain(optimizations=on)
    assert all_joins(plan) == ["INNER JOIN:", f"{how.upper()} JOIN:"]
    assert 'LEFT PLAN ON: [col("x")]' in plan
    assert_same_result(lf, on, off)

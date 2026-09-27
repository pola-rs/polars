from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from polars._typing import EngineType


def _frames() -> dict[str, pl.DataFrame]:
    return {
        "t1": pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}),
        "t2": pl.DataFrame({"g": [10, 10, 20], "w": [1, 2, 3]}),
    }


def test_correlated_count_inequality() -> None:
    # COUNT over no matches is 0 (never NULL).
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, (SELECT COUNT(*) FROM t1 AS x WHERE x.b < t1.b) AS cnt "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "cnt": [0, 1, 2]},
    )


def test_correlated_sum_empty_match_is_null() -> None:
    # SUM over no matches is NULL.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, (SELECT SUM(x.b) FROM t1 AS x WHERE x.b < t1.b) AS s "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "s": [None, 10, 30]},
    )


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("MIN(x.b)", [None, 10, 10]),
        ("MAX(x.b)", [None, 10, 20]),
        ("AVG(x.b)", [None, 10.0, 15.0]),
    ],
)
def test_correlated_min_max_avg(agg: str, expected: list[float | None]) -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            f"SELECT a, (SELECT {agg} FROM t1 AS x WHERE x.b < t1.b) AS v "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "v": expected},
    )


def test_correlated_equality_across_tables() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, (SELECT COUNT(*) FROM t2 WHERE t2.g = t1.b) AS c "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "c": [2, 1, 0]},
    )


def test_correlated_count_equality_self() -> None:
    # Self-correlated equality: count rows of the same table sharing `b`,
    # excluding the row itself via an inequality on `a`.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, (SELECT COUNT(*) FROM t1 AS x "
            "WHERE x.b = t1.b AND x.a <> t1.a) AS c FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "c": [0, 0, 0]},
    )


def test_correlated_with_inner_only_filter() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, (SELECT COUNT(*) FROM t2 WHERE t2.g = t1.b AND t2.w > 1) AS c "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "c": [1, 1, 0]},
    )


def test_correlated_subquery_in_where() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 "
            "WHERE (SELECT COUNT(*) FROM t1 AS x WHERE x.b < t1.b) > 0 "
            "ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_multiple_correlated_subqueries() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, "
            "(SELECT COUNT(*) FROM t1 AS x WHERE x.b < t1.b) AS c, "
            "(SELECT SUM(x.b) FROM t1 AS x WHERE x.b < t1.b) AS s "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "c": [0, 1, 2], "s": [None, 10, 30]},
    )


def test_uncorrelated_scalar_subquery_still_works() -> None:
    # An uncorrelated scalar subquery must stay on the generic scalar path.
    assert_sql_matches(
        frames=_frames(),
        query="SELECT a, (SELECT MAX(b) FROM t1) AS mx FROM t1 ORDER BY a",
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "mx": [30, 30, 30]},
    )


def _having_frames() -> dict[str, pl.DataFrame]:
    # Chosen so the correlated and uncorrelated readings of the same HAVING
    # subquery select different groups.
    return {
        "t": pl.DataFrame({"g": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]}),
        "r": pl.DataFrame({"k": [1, 2, 3], "c": [25, 100, 10]}),
    }


def test_correlated_subquery_in_having() -> None:
    # Per-group thresholds are 25/100/10, so groups 1 and 3 pass. Resolving the
    # correlation against the inner relation instead would compare every group
    # against SUM(c) = 135 and select nothing.
    assert_sql_matches(
        frames=_having_frames(),
        query=(
            "SELECT g, SUM(v) AS s FROM t "
            "GROUP BY g "
            "HAVING SUM(v) > (SELECT SUM(c) FROM r WHERE r.k = t.g) "
            "ORDER BY g"
        ),
        compare_with="duckdb",
        expected={"g": [1, 3], "s": [30, 50]},
    )


def test_correlated_exists_in_having() -> None:
    assert_sql_matches(
        frames=_having_frames(),
        query=(
            "SELECT g, SUM(v) AS s FROM t "
            "GROUP BY g "
            "HAVING EXISTS (SELECT 1 FROM r WHERE r.k = t.g AND r.c > 50) "
            "ORDER BY g"
        ),
        compare_with="duckdb",
        expected={"g": [2], "s": [70]},
    )


def test_uncorrelated_subquery_in_having_still_works() -> None:
    # AVG(c) = 45, so groups 2 and 3 pass.
    assert_sql_matches(
        frames=_having_frames(),
        query=(
            "SELECT g, SUM(v) AS s FROM t "
            "GROUP BY g "
            "HAVING SUM(v) > (SELECT AVG(c) FROM r) "
            "ORDER BY g"
        ),
        compare_with="duckdb",
        expected={"g": [2, 3], "s": [70, 50]},
    )


def _decorrelation_count(ctx: pl.SQLContext[pl.LazyFrame], query: str) -> int:
    """Count decorrelation pipelines by their unique `__POLARS_CORR_*` id.

    Equality correlation lowers to a `GROUP BY` + join with no `_idx` column, so the
    pipeline id itself (not a strategy-specific column suffix) is what's counted.
    """
    plan = ctx.execute(query).explain()
    return len(set(re.findall(r"__POLARS_CORR.*?(POLARS_TMP_\d+)_", plan)))


def test_repeated_correlated_subquery_is_decorrelated_once() -> None:
    # Decorrelation is expensive, so the same subquery appearing in several places
    # must be lowered once rather than once per occurrence.
    frames = {
        "t1": pl.DataFrame({"k": [1, 2, 3]}),
        "t2": pl.DataFrame({"k": [1, 1, 2], "w": [5, 7, 9]}),
    }
    sub = "(SELECT SUM(w) FROM t2 WHERE t2.k = t1.k)"

    with pl.SQLContext(frames=frames) as ctx:
        # twice in the SELECT list
        assert (
            _decorrelation_count(ctx, f"SELECT {sub} AS a, {sub} + 1 AS b FROM t1") == 1
        )
        # in WHERE and in the SELECT list
        assert (
            _decorrelation_count(ctx, f"SELECT {sub} AS a FROM t1 WHERE {sub} > 9") == 1
        )
        # in the SELECT list and in HAVING
        assert (
            _decorrelation_count(
                ctx, f"SELECT k, {sub} AS a FROM t1 GROUP BY k HAVING {sub} > 9"
            )
            == 1
        )
        # genuinely different subqueries still get one each
        other = "(SELECT MAX(w) FROM t2 WHERE t2.k = t1.k)"
        assert (
            _decorrelation_count(ctx, f"SELECT {sub} AS a, {other} AS b FROM t1") == 2
        )
        # EXISTS and a scalar subquery over the same inner query must not share:
        # one yields a value column, the other a boolean flag
        assert (
            _decorrelation_count(
                ctx, f"SELECT k, EXISTS {sub} AS e, {sub} AS s FROM t1"
            )
            == 2
        )

    # ...and the values are still right
    assert_sql_matches(
        frames=frames,
        query=f"SELECT k, {sub} AS a, {sub} + 1 AS b FROM t1 ORDER BY k",
        compare_with="duckdb",
        expected={"k": [1, 2, 3], "a": [12, 9, None], "b": [13, 10, None]},
    )


def _sales_frames() -> dict[str, pl.DataFrame]:
    return {
        "cs": pl.DataFrame(
            {
                "cs_item_sk": [1, 1, 2, 2, 3, 3],
                "amt": [10.0, 20.0, 5.0, 100.0, 7.0, 8.0],
                "dsk": [1, 2, 1, 2, 1, 2],
            }
        ),
        "item": pl.DataFrame({"i_item_sk": [1, 2, 3], "i_manufact_id": [977, 977, 42]}),
        "dd": pl.DataFrame({"d_date_sk": [1, 2], "d_year": [2000, 2001]}),
    }


@pytest.mark.parametrize(
    "correlation",
    [
        # the outer relation named by the correlation may be qualified or not,
        # and need not be the first relation of the outer FROM
        "c2.cs_item_sk = item.i_item_sk",
        "c2.cs_item_sk = i_item_sk",
    ],
)
def test_correlated_scalar_subquery_multi_table_outer(correlation: str) -> None:
    # a correlated subquery inside a comparison, over a multi-relation outer FROM
    assert_sql_matches(
        frames=_sales_frames(),
        query=f"""
            SELECT sum(amt) AS s FROM cs, item
            WHERE i_item_sk = cs_item_sk
              AND amt > (SELECT avg(c2.amt) FROM cs c2 WHERE {correlation})
        """,
        compare_with="duckdb",
    )


def test_correlated_subquery_name_in_both_scopes() -> None:
    # `cs_item_sk` exists in the inner and the outer relation; an unqualified
    # name binds to the innermost scope that holds it
    assert_sql_matches(
        frames=_sales_frames(),
        query="""
            SELECT sum(amt) AS s FROM cs, item
            WHERE i_item_sk = cs_item_sk
              AND amt > (SELECT 1.3 * avg(amt) FROM cs WHERE cs_item_sk = i_item_sk)
        """,
        compare_with="duckdb",
    )


def test_correlated_subquery_multi_relation_inner_from() -> None:
    # the subquery's own FROM comma-joins two relations
    assert_sql_matches(
        frames=_sales_frames(),
        query="""
            SELECT sum(amt) AS s FROM cs, item, dd
            WHERE i_item_sk = cs_item_sk AND d_date_sk = dsk
              AND amt > (
                SELECT avg(amt) FROM cs, dd
                WHERE cs_item_sk = i_item_sk AND d_date_sk = dsk AND d_year = 2001
              )
        """,
        compare_with="duckdb",
    )


def test_correlated_subquery_predicate_shared_across_or_branches() -> None:
    # the correlation sits inside both branches of an OR rather than at the top
    # level of the subquery's WHERE
    assert_sql_matches(
        frames=_sales_frames(),
        query="""
            SELECT DISTINCT i_item_sk FROM item i1
            WHERE (
                SELECT count(*) FROM item
                WHERE (i_manufact_id = i1.i_manufact_id AND i_item_sk < 3)
                   OR (i_manufact_id = i1.i_manufact_id AND i_item_sk > 2)
            ) > 0
            ORDER BY i_item_sk
        """,
        compare_with="duckdb",
    )


# --- equality correlation restricted to the outer keys ---------------------------

ENGINES: list[EngineType] = ["in-memory", "streaming"]


def _key_frames() -> dict[str, pl.DataFrame]:
    # Outer keys repeat with different payloads, one is unmatched (2), one is
    # null. Inner rows repeat within a key (1), hold keys the outer frame lacks
    # (4, 9), a null key and an all-null value group (5). The inner frame is
    # over twice the outer one, so the aggregate is restricted to the outer keys.
    return {
        "o": pl.DataFrame(
            {
                "k": [1, 1, 2, 3, 5, None],
                "j": ["x", "y", "x", "x", "x", "x"],
                "p": ["a", "b", "c", "d", "e", "f"],
                "s": [5, 15, 0, 100, 1, 1],
            }
        ),
        "i": pl.DataFrame(
            {
                "k": [1, 1, 1, 3, 4, 5, None, 4, 9, 9, 4, 9],
                "j": ["x", "x", None, "x", "x", "x", "x", "x", "x", "x", "x", "x"],
                "v": [10, None, 7, 30, 40, None, 50, 41, 90, 91, 42, 92],
            },
            schema={"k": pl.Int32, "j": pl.String, "v": pl.Int64},
        ),
    }


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("COUNT(*)", [3, 3, 0, 1, 1, 0]),
        ("COUNT(v)", [2, 2, 0, 1, 0, 0]),
        ("SUM(v)", [17, 17, None, 30, None, None]),
        ("MIN(v)", [7, 7, None, 30, None, None]),
        ("MAX(v)", [10, 10, None, 30, None, None]),
        ("AVG(v)", [8.5, 8.5, None, 30.0, None, None]),
        ("0.5 * SUM(v)", [8.5, 8.5, None, 15.0, None, None]),
        ("SUM(v) FILTER (WHERE v > 8)", [10, 10, None, 30, None, None]),
        ("MAX(CAST(v AS VARCHAR))", ["7", "7", None, "30", None, None]),
    ],
)
def test_correlated_aggregate_single_key(agg: str, expected: list[Any]) -> None:
    assert_sql_matches(
        frames=_key_frames(),
        query=f"SELECT p, (SELECT {agg} FROM i WHERE i.k = o.k) AS r FROM o ORDER BY p",
        compare_with="duckdb",
        expected={"p": list("abcdef"), "r": expected},
        engines=ENGINES,
    )


def test_correlated_aggregate_result_dtypes() -> None:
    with pl.SQLContext(frames=_key_frames()) as ctx:
        schema = ctx.execute(
            "SELECT p, "
            "(SELECT COUNT(*) FROM i WHERE i.k = o.k) AS c, "
            "(SELECT SUM(v) FROM i WHERE i.k = o.k) AS s, "
            "(SELECT AVG(v) FROM i WHERE i.k = o.k) AS a FROM o"
        ).collect_schema()
    assert schema == pl.Schema(
        {"p": pl.String, "c": pl.Int64, "s": pl.Int64, "a": pl.Float64}
    )


@pytest.mark.parametrize(
    "correlation",
    [
        "i.k = o.k AND i.j = o.j",
        "o.k = i.k AND o.j = i.j",
        "i.j = o.j AND i.k = o.k",
        # the same key pair spelled twice
        "i.k = o.k AND i.j = o.j AND o.k = i.k",
    ],
)
def test_correlated_aggregate_composite_key(correlation: str) -> None:
    # a partially null composite key never matches
    assert_sql_matches(
        frames=_key_frames(),
        query=(
            f"SELECT p, (SELECT SUM(v) FROM i WHERE {correlation}) AS r "
            "FROM o ORDER BY p"
        ),
        compare_with="duckdb",
        expected={"p": list("abcdef"), "r": [10, None, None, 30, None, None]},
        engines=ENGINES,
    )


@pytest.mark.parametrize(
    ("outer", "inner", "expected"),
    [
        # empty outer
        ([], [1, 2], []),
        # empty inner
        ([1, 2], [], [0, 0]),
        # disjoint keys
        ([1, 2], [3, 4], [0, 0]),
        # both empty
        ([], [], []),
    ],
)
def test_correlated_aggregate_empty_or_disjoint_inputs(
    outer: list[int], inner: list[int], expected: list[int]
) -> None:
    frames = {
        "o": pl.DataFrame({"k": outer}, schema={"k": pl.Int64}),
        "i": pl.DataFrame({"k": inner}, schema={"k": pl.Int64}),
    }
    assert_sql_matches(
        frames=frames,
        query="SELECT k, (SELECT COUNT(*) FROM i WHERE i.k = o.k) AS c FROM o ORDER BY k",
        compare_with="duckdb",
        expected={"k": outer, "c": expected},
        engines=ENGINES,
    )


@pytest.mark.parametrize(
    "agg",
    [
        "COUNT(*) + 1",
        "COALESCE(SUM(v), 0)",
        "COUNT(DISTINCT v)",
    ],
)
@pytest.mark.xfail(
    reason="only a bare COUNT is recognised as non-null over an unmatched outer row",
    strict=True,
)
def test_correlated_aggregate_wrapped_count_over_no_match(agg: str) -> None:
    assert_sql_matches(
        frames=_key_frames(),
        query=f"SELECT p, (SELECT {agg} FROM i WHERE i.k = o.k) AS r FROM o ORDER BY p",
        compare_with="duckdb",
        engines=ENGINES,
    )


def test_correlated_aggregate_strict_cast_only_over_matched_rows() -> None:
    # The inner group no outer row asks for holds a value the aggregate can't
    # cast; the subquery is never evaluated for that group.
    frames = {
        "o": pl.DataFrame({"k": [1, 2]}),
        "i": pl.DataFrame({"k": [1, 1, 3, 3], "v": ["10", "20", "x", "30"]}),
    }
    assert_sql_matches(
        frames=frames,
        query=(
            "SELECT k, (SELECT SUM(CAST(v AS INT)) FROM i WHERE i.k = o.k) AS r "
            "FROM o ORDER BY k"
        ),
        compare_with=None,
        expected={"k": [1, 2], "r": [30, None]},
        engines=ENGINES,
    )


def test_correlated_aggregate_row_dependent_inner_source() -> None:
    # The inner relation is a lazy plan whose values depend on the rows around
    # them; restricting it to the outer keys must not change what it holds.
    inner = pl.LazyFrame({"k": [3, 1, 3, 1, 2], "v": [1, 1, 1, 1, 1]}).with_columns(
        pl.col("v").cum_sum().alias("v")
    )
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {
        "o": pl.DataFrame({"k": [1, 3]}),
        "i": inner,
    }
    assert_sql_matches(
        frames=frames,
        query=(
            "SELECT k, (SELECT SUM(v) FROM i WHERE i.k = o.k) AS r FROM o ORDER BY k"
        ),
        compare_with="duckdb",
        expected={"k": [1, 3], "r": [6, 4]},
        engines=ENGINES,
    )


def test_correlated_aggregate_reused_across_clauses() -> None:
    # One subquery in WHERE, SELECT and HAVING is lowered once, over one cached
    # outer input.
    frames = _key_frames()
    sub = "(SELECT SUM(v) FROM i WHERE i.k = o.k)"
    query = (
        f"SELECT k, {sub} AS a, MAX(s) AS m FROM o WHERE {sub} > 10 "
        f"GROUP BY k HAVING {sub} < 100 ORDER BY k"
    )
    with pl.SQLContext(frames=frames) as ctx:
        plan = ctx.execute(query).explain(optimized=False)
    assert _decorrelation_count(ctx, query) == 1
    assert len(set(re.findall(r"CACHE\[id: ([^\]]+)\]", plan))) == 1
    assert_sql_matches(
        frames=frames,
        query=query,
        compare_with="duckdb",
        expected={"k": [1, 3], "a": [17, 30], "m": [15, 100]},
        engines=ENGINES,
    )


def test_correlated_aggregates_several_over_one_outer() -> None:
    assert_sql_matches(
        frames=_key_frames(),
        query="""
            SELECT p,
              (SELECT SUM(v) FROM i WHERE i.k = o.k) AS a,
              (SELECT MAX(v) FROM i WHERE i.k = o.k) AS b,
              (SELECT COUNT(*) FROM i WHERE i.j = o.j) AS c,
              (SELECT COUNT(*) FROM o AS x WHERE x.k = o.k) AS d
            FROM o ORDER BY p
        """,
        compare_with="duckdb",
        expected={
            "p": list("abcdef"),
            "a": [17, 17, None, 30, None, None],
            "b": [10, 10, None, 30, None, None],
            "c": [11, 0, 11, 11, 11, 11],
            "d": [2, 2, 1, 1, 1, 0],
        },
        engines=ENGINES,
    )
    with pl.SQLContext(frames=_key_frames()) as ctx:
        out = ctx.execute("SELECT (SELECT SUM(v) FROM i WHERE i.k = o.k) AS a FROM o")
    assert out.collect_schema().names() == ["a"]


@pytest.mark.parametrize(
    "where",
    [
        # the selective conjuncts in either order around the scalar comparison
        "j = 'x' AND s > (SELECT SUM(v) FROM i WHERE i.k = o.k) AND k IN (SELECT k FROM i WHERE v > 20)",
        "k IN (SELECT k FROM i WHERE v > 20) AND s > (SELECT SUM(v) FROM i WHERE i.k = o.k) AND j = 'x'",
        "(j = 'x' AND (k IN (SELECT k FROM i WHERE v > 20))) AND s > (SELECT SUM(v) FROM i WHERE i.k = o.k)",
        # a disjunction and a negation aren't split
        "j = 'y' OR s > (SELECT SUM(v) FROM i WHERE i.k = o.k)",
        "NOT (j = 'x' AND s > (SELECT SUM(v) FROM i WHERE i.k = o.k))",
        # a conjunct reading the scalar result stays after it
        "s > (SELECT SUM(v) FROM i WHERE i.k = o.k) AND (SELECT SUM(v) FROM i WHERE i.k = o.k) < 100",
        "EXISTS (SELECT 1 FROM i WHERE i.k = o.k AND v > 8) AND s > (SELECT MIN(v) FROM i WHERE i.k = o.k)",
        "k NOT IN (SELECT k FROM i WHERE v > 35) AND s > (SELECT MIN(v) FROM i WHERE i.k = o.k)",
        "s = (SELECT MAX(s) FROM o) AND s > (SELECT MIN(v) FROM i WHERE i.k = o.k)",
        "1 = 1 AND s > (SELECT MIN(v) FROM i WHERE i.k = o.k)",
    ],
)
def test_correlated_aggregate_where_conjunct_scheduling(where: str) -> None:
    assert_sql_matches(
        frames=_key_frames(),
        query=f"SELECT p FROM o WHERE {where} ORDER BY p",
        compare_with="duckdb",
        engines=ENGINES,
    )


def _assert_restricted_to_outer_keys(plan: str) -> None:
    (result_col,) = set(re.findall(r"__POLARS_CORR__POLARS_TMP_\d+_res", plan))
    lines = plan.splitlines()
    agg_input = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("AGGREGATE") and result_col in lines[idx + 1]:
            from_line = lines[idx + 2]
            assert from_line.strip() == "FROM"
            indent = len(from_line) - len(from_line.lstrip())
            body = []
            for below in lines[idx + 3 :]:
                if below.strip() and len(below) - len(below.lstrip()) < indent:
                    break
                body.append(below)
            agg_input = "\n".join(body)
            break
    assert agg_input is not None, plan
    # the aggregate reads a semi join against the cached outer input...
    assert agg_input.lstrip().startswith("SEMI JOIN:")
    cache_ids = re.findall(r"CACHE\[id: ([^\]]+)\]", agg_input)
    assert len(cache_ids) == 1
    # ...and that cache is the one the aggregate result is joined back onto
    assert plan.count(f"CACHE[id: {cache_ids[0]}]") == 2


def _tpch_q20_frames() -> dict[str, pl.LazyFrame]:
    return {
        "part": pl.LazyFrame(
            {"p_partkey": [1, 2, 3], "p_name": ["forest a", "b", "forest c"]}
        ),
        "partsupp": pl.LazyFrame(
            {
                "ps_partkey": [1, 1, 2, 3],
                "ps_suppkey": [10, 20, 10, 30],
                "ps_availqty": [100, 1, 100, 100],
            }
        ),
        "lineitem": pl.LazyFrame(
            {
                "l_partkey": [1, 1, 2, 3, 3, 2, 2, 3, 1],
                "l_suppkey": [10, 20, 10, 30, 30, 10, 10, 30, 20],
                "l_quantity": [10, 10, 1000, 300, 10, 5, 5, 1, 1],
                "l_shipdate": pl.Series(
                    [
                        "1994-06-01",
                        "1994-06-01",
                        "1994-06-01",
                        "1994-06-01",
                        "1995-06-01",
                        "1994-07-01",
                        "1994-08-01",
                        "1993-01-01",
                        "1996-01-01",
                    ]
                ).str.to_date(),
            }
        ),
        "supplier": pl.LazyFrame(
            {
                "s_suppkey": [10, 20, 30],
                "s_name": ["s10", "s20", "s30"],
                "s_address": ["a10", "a20", "a30"],
                "s_nationkey": [1, 1, 2],
            }
        ),
        "nation": pl.LazyFrame({"n_nationkey": [1, 2], "n_name": ["CANADA", "PERU"]}),
    }


TPCH_Q20 = """
    SELECT s_name, s_address FROM supplier, nation
    WHERE s_suppkey IN (
        SELECT ps_suppkey FROM partsupp
        WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
          AND ps_availqty > (
            SELECT 0.5 * SUM(l_quantity) FROM lineitem
            WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
              AND l_shipdate >= DATE '1994-01-01'
              AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
          )
    )
    AND s_nationkey = n_nationkey AND n_name = 'CANADA'
    ORDER BY s_name
"""


def test_correlated_aggregate_restricted_by_filtered_outer_plan() -> None:
    # The lineitem aggregate reads only the part/supplier keys of the forest
    # parts, without the query being rewritten.
    with pl.SQLContext(frames=_tpch_q20_frames()) as ctx:
        lf = ctx.execute(TPCH_Q20)
    for plan in (
        lf.explain(optimized=False),
        lf.explain(),
        lf.explain(optimizations=pl.QueryOptFlags(join_order=False)),
    ):
        _assert_restricted_to_outer_keys(plan)
    # the forest-parts restriction sits below the cache the aggregate reads
    plan = lf.explain(optimized=False)
    cache_line = plan.index("CACHE[id:")
    assert "p_name" in plan[cache_line:]
    assert_sql_matches(
        frames=_tpch_q20_frames(),
        query=TPCH_Q20,
        compare_with="duckdb",
        expected={"s_name": ["s10"], "s_address": ["a10"]},
        engines=ENGINES,
    )


def test_correlated_aggregate_not_restricted_by_larger_outer() -> None:
    # The semi join only pays off when the inner frame is well over twice the
    # outer one; here the outer frame is the larger one.
    frames = {
        "o": pl.DataFrame({"k": list(range(10)), "s": list(range(10))}),
        "i": pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]}),
    }
    query = "SELECT k, (SELECT SUM(v) FROM i WHERE i.k = o.k) AS r FROM o ORDER BY k"
    with pl.SQLContext(frames=frames) as ctx:
        plan = ctx.execute(query).explain(optimized=False)
    assert "SEMI JOIN" not in plan
    assert "CACHE" not in plan
    assert_sql_matches(
        frames=frames,
        query=query,
        compare_with="duckdb",
        expected={"k": list(range(10)), "r": [None, 10, 20, 30] + [None] * 6},
        engines=ENGINES,
    )


def test_correlated_aggregate_inequality_keeps_row_index_path() -> None:
    with pl.SQLContext(frames=_key_frames()) as ctx:
        plan = ctx.execute(
            "SELECT p, (SELECT SUM(v) FROM i WHERE i.k < o.k) AS r FROM o"
        ).explain(optimized=False)
    assert "SEMI JOIN" not in plan
    assert "_idx" in plan


def _counting_source(
    df: pl.DataFrame, on_batch: Callable[[pl.DataFrame], None]
) -> pl.LazyFrame:
    from polars.io.plugins import register_io_source

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        _n_rows: int | None,
        _batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        for offset in range(0, df.height, 10_000):
            batch = df.slice(offset, 10_000)
            if predicate is not None:
                batch = batch.filter(predicate)
            if with_columns is not None:
                batch = batch.select(with_columns)
            on_batch(batch)
            yield batch

    return register_io_source(source, schema=df.schema, is_pure=True)


def test_correlated_aggregate_unknown_outer_size_not_restricted() -> None:
    # A source without a row count could be arbitrarily large, so the aggregate
    # is not restricted to its keys: caching it would read it whole before a
    # LIMIT could stop it.
    outer = pl.DataFrame({"k": range(2_000_000), "s": range(2_000_000)})
    yielded = 0

    def count(batch: pl.DataFrame) -> None:
        nonlocal yielded
        yielded += batch.height

    frames = {
        "o": _counting_source(outer, count),
        "i": pl.DataFrame({"k": [0, 1, 2, 3], "v": [10, 20, 30, 40]}),
    }
    query = "SELECT k, (SELECT SUM(v) FROM i WHERE i.k = o.k) AS r FROM o LIMIT 1"
    with pl.SQLContext(frames=frames) as ctx:
        lf = ctx.execute(query)
    assert "CACHE" not in lf.explain(optimized=False)
    yielded = 0
    assert lf.collect(engine="streaming").height == 1
    assert 0 < yielded < outer.height


def test_correlated_aggregate_outer_read_once(tmp_path: Path) -> None:
    # Both consumers of the cached outer frame read one evaluation of it: the
    # outer scan is planned once and every result row is produced once.
    path = tmp_path / "o.parquet"
    pl.DataFrame({"k": [1, 2, 3], "s": [5, 50, 500]}).write_parquet(path)
    frames = {
        "o": pl.scan_parquet(path),
        "i": pl.DataFrame({"k": [1, 1, 3, 4, 5, 6, 7], "v": [10, 20, 30, 40, 1, 1, 1]}),
    }
    query = (
        "SELECT k, s FROM o WHERE s > (SELECT SUM(v) FROM i WHERE i.k = o.k) ORDER BY k"
    )
    with pl.SQLContext(frames=frames) as ctx:
        lf = ctx.execute(query)
    plan = lf.explain()
    assert plan.count("Parquet SCAN") == 1
    assert len(set(re.findall(r"CACHE\[id: ([^\]]+)\]", plan))) == 1
    for engine in ENGINES:
        # the unmatched key compares against NULL and is dropped
        assert lf.collect(engine=engine).to_dict(as_series=False) == {
            "k": [3],
            "s": [500],
        }


@pytest.mark.parametrize("reverse", [False, True])
def test_correlated_aggregate_fallible_predicate_stays_after_scalar_filter(
    reverse: bool,
) -> None:
    # The strict cast fails on the row the scalar comparison removes, so it is
    # not moved ahead of the scalar lowering.
    frames = {
        "o": pl.DataFrame({"k": [1, 2], "s": ["10", "bad"]}),
        "i": pl.DataFrame({"k": [1, 1, 2, 2, 3, 3], "v": [10, 10, -10, -10, 1, 1]}),
    }
    predicates = ["CAST(s AS INT) > 0", "(SELECT SUM(v) FROM i WHERE i.k = o.k) > 0"]
    if reverse:
        predicates.reverse()
    assert_sql_matches(
        frames=frames,
        query="SELECT k FROM o WHERE " + " AND ".join(predicates) + " ORDER BY k",
        compare_with=None,
        expected={"k": [1]},
        engines=ENGINES,
    )


def test_correlated_aggregate_fallible_membership_key_stays_residual() -> None:
    frames = {
        "o": pl.DataFrame({"k": [1, 2], "s": ["10", "bad"]}),
        "i": pl.DataFrame({"k": [1, 1, 2, 2, 3, 3], "v": [10, 10, -10, -10, 1, 1]}),
    }
    query = (
        "SELECT k FROM o WHERE CAST(s AS INT) IN (SELECT v FROM i) "
        "AND (SELECT SUM(v) FROM i WHERE i.k = o.k) > 0 ORDER BY k"
    )
    with pl.SQLContext(frames=frames) as ctx:
        plan = ctx.execute(query).explain(optimized=False)
    # the membership join sits above the scalar lowering's cache
    assert plan.index("SEMI JOIN") < plan.index("CACHE")


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("{sum} = {max}", [2, 3]),
        ("{sum} IN (SELECT v FROM i)", [2, 3]),
        ("k + {sum} = (SELECT MAX(v) FROM i)", []),
        ("CASE WHEN k > 1 THEN {sum} ELSE 0 END = (SELECT MAX(v) FROM i)", [3]),
    ],
)
def test_correlated_aggregate_inside_membership_key(
    condition: str, expected: list[int]
) -> None:
    # A membership key holding a correlated scalar subquery is lowered first.
    condition = condition.format(
        sum="(SELECT SUM(v) FROM i WHERE i.k = o.k)",
        max="(SELECT MAX(v) FROM i WHERE i.k = o.k)",
    )
    assert_sql_matches(
        frames={
            "o": pl.DataFrame({"k": [1, 2, 3]}),
            "i": pl.DataFrame({"k": [1, 1, 2, 2, 3, 3], "v": [1, 2, 4, 0, 5, 0]}),
        },
        query=f"SELECT k FROM o WHERE {condition} ORDER BY k",
        compare_with="duckdb",
        expected={"k": expected},
        engines=ENGINES,
    )


@pytest.mark.parametrize(
    "membership",
    [
        "k IN (SELECT k FROM i WHERE v < s)",
        "k NOT IN (SELECT k FROM i WHERE v < s)",
        "s = (SELECT MAX(v) FROM i WHERE v < s)",
    ],
)
@pytest.mark.parametrize("with_scalar", [False, True])
def test_membership_subquery_correlated_by_unqualified_outer_column(
    membership: str, with_scalar: bool
) -> None:
    # `s` is an outer column; only the decorrelation, not the uncorrelated
    # rewrite, can resolve it.
    where = membership
    if with_scalar:
        where += " AND (SELECT COUNT(*) FROM i WHERE i.k = o.k) > 0"
    assert_sql_matches(
        frames={
            "o": pl.DataFrame({"k": [1, 2, 3], "s": [15, 5, 100]}),
            "i": pl.DataFrame({"k": [1, 1, 2, 3, 3, 3], "v": [10, 20, 30, 40, 1, 2]}),
        },
        query=f"SELECT k FROM o WHERE {where} ORDER BY k",
        compare_with="duckdb",
        engines=ENGINES,
    )


def test_correlated_aggregate_one_outer_column_in_two_key_pairs() -> None:
    frames = {
        "o": pl.DataFrame({"k": [1, 2]}),
        "i": pl.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 9],
                "b": [1, 2, 1, 2, 1, 9],
                "v": [10, 20, 30, 40, 50, 60],
            }
        ),
    }
    assert_sql_matches(
        frames=frames,
        query=(
            "SELECT k, (SELECT SUM(v) FROM i WHERE i.a = o.k AND i.b = o.k) AS s "
            "FROM o ORDER BY k"
        ),
        compare_with="duckdb",
        expected={"k": [1, 2], "s": [40, 40]},
        engines=ENGINES,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_correlated_aggregate_limit_and_empty_results(engine: EngineType) -> None:
    # A shared outer input must not stall a query that stops early or yields nothing.
    frames = {
        "o": pl.DataFrame({"k": list(range(300)), "s": list(range(300))}),
        "i": pl.DataFrame({"k": list(range(0, 2000, 2)), "v": [1] * 1000}),
    }
    with pl.SQLContext(frames=frames) as ctx:
        limited = ctx.execute(
            "SELECT k, (SELECT SUM(v) FROM i WHERE i.k = o.k) AS r FROM o ORDER BY k LIMIT 2"
        ).collect(engine=engine)
        assert limited.to_dict(as_series=False) == {"k": [0, 1], "r": [1, None]}
        empty = ctx.execute(
            "SELECT k FROM o WHERE s < 0 AND s > (SELECT SUM(v) FROM i WHERE i.k = o.k)"
        ).collect(engine=engine)
        assert empty.height == 0
        none_match = ctx.execute(
            "SELECT k, (SELECT SUM(v) FROM i WHERE i.k = o.k) AS r FROM o WHERE k < 0"
        ).collect(engine=engine)
        assert none_match.height == 0

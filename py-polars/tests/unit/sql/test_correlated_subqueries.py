import re

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


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
        "item": pl.DataFrame(
            {"i_item_sk": [1, 2, 3], "i_manufact_id": [977, 977, 42]}
        ),
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
    # the subquery names `i_item_sk`, which made the enclosing comparison look
    # like a join predicate between the two outer relations
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
    # level of the subquery's WHERE, so it must be factored out to be seen
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

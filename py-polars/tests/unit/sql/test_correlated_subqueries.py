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

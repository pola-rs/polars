import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


def _frames() -> dict[str, pl.DataFrame]:
    return {
        "t1": pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}),
        "t2": pl.DataFrame({"g": [10, 10, 20], "w": [1, 2, 3]}),
    }


def test_equality_correlated_exists_still_works() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )
    # an equality correlation must still be lowered to a semi join, not routed
    # through the general (decorrelated flag column) EXISTS path
    with pl.SQLContext(frames=_frames()) as ctx:
        plan = ctx.execute(
            "SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)"
        ).explain()
        assert "SEMI JOIN" in plan, plan


def test_equality_correlated_not_exists_still_works() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE NOT EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [3]},
    )
    # an equality correlation must still be lowered to an anti join, not
    # routed through the general (decorrelated flag column) EXISTS path
    with pl.SQLContext(frames=_frames()) as ctx:
        plan = ctx.execute(
            "SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)"
        ).explain()
        assert "ANTI JOIN" in plan, plan


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("<", [2, 3]),
        ("<=", [1, 2, 3]),
        (">", [1, 2]),
        (">=", [1, 2, 3]),
        ("<>", [1, 2, 3]),
    ],
)
def test_exists_inequality_correlation(op: str, expected: list[int]) -> None:
    # Correlation through a self-referencing alias (`t1 AS x` vs outer `t1`).
    assert_sql_matches(
        frames=_frames(),
        query=(
            f"SELECT a FROM t1 WHERE EXISTS "
            f"(SELECT 1 FROM t1 AS x WHERE x.b {op} t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": expected},
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("<", [1]),
        ("<=", []),
        (">", [3]),
        (">=", []),
        ("<>", []),
    ],
)
def test_not_exists_inequality_correlation(op: str, expected: list[int]) -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            f"SELECT a FROM t1 WHERE NOT EXISTS "
            f"(SELECT 1 FROM t1 AS x WHERE x.b {op} t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": expected},
    )


def test_exists_empty_inner_result() -> None:
    # No `x.b` value is ever both less than and greater than the same
    # `t1.b`, so EXISTS is false for every outer row.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": []},
    )


def test_not_exists_all_rows_match() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE NOT EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3]},
    )


def test_exists_inequality_combined_with_other_where_conjunct() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a > 1 AND EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_exists_inequality_with_inner_local_filter() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g < t1.b AND t2.w > 1) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_not_exists_inequality_null_edge_case() -> None:
    # A NULL correlation column makes every comparison NULL (never true), so
    # the subquery matches zero rows and NOT EXISTS is true.
    frames = {"t3": pl.DataFrame({"a": [1, 2, 3], "b": [10, None, 30]})}
    assert_sql_matches(
        frames=frames,
        query=(
            "SELECT a FROM t3 WHERE NOT EXISTS "
            "(SELECT 1 FROM t3 AS x WHERE x.b < t3.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )


# --- EXISTS in general expression position (OR / CASE / SELECT list) -------
#
# None of these shapes are a whole WHERE filter or a top-level AND-conjunct
# of it, so they can't be lowered to a semi/anti join or count-filter; they
# exercise the decorrelated boolean flag column path instead.


def test_exists_or_predicate() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_not_exists_or_predicate() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1]},
    )


def test_equality_correlated_exists_in_or_position() -> None:
    # An equality correlation, but not a top-level AND-conjunct, so this must
    # go through the general decorrelation path rather than the semi-join
    # fast path.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )


def test_exists_in_case_expression() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, CASE WHEN EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b) THEN 1 ELSE 0 END AS flag "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "flag": [0, 1, 1]},
    )


def test_exists_in_select_list() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e "
            "FROM t1 ORDER BY a"
        ),
        compare_with=None,
        expected={"a": [1, 2, 3], "e": [False, True, True]},
    )


def test_not_exists_in_select_list() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e "
            "FROM t1 ORDER BY a"
        ),
        compare_with=None,
        expected={"a": [1, 2, 3], "e": [True, False, False]},
    )


def test_uncorrelated_exists_in_expression_position() -> None:
    # Uncorrelated EXISTS is a constant boolean, broadcast onto every row.
    assert_sql_matches(
        frames=_frames(),
        query="SELECT a FROM t1 WHERE a = 1 OR EXISTS (SELECT 1 FROM t2) ORDER BY a",
        compare_with="duckdb",
        expected={"a": [1, 2, 3]},
    )
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 1 "
            "OR EXISTS (SELECT 1 FROM t2 WHERE g > 1000) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1]},
    )

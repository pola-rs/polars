from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "category": ["A", "A", "A", "B", "B", "B"],
            "value": [100, 200, 150, 300, 250, 400],
        }
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category)",
            id="above_avg",
        ),
        pytest.param(
            "value = MAX(value) OVER (PARTITION BY category)",
            id="equals_max",
        ),
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category) AND value < 500",
            id="compound_expr",
        ),
    ],
)
def test_qualify_constraints(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY category, value
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
        },
    )


def test_qualify_distinct() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [100, 100, 200, 200, 300, 300],
        }
    )
    assert_sql_matches(
        {"df": df},
        query="""
            SELECT DISTINCT category, value
            FROM df
            QUALIFY value = MAX(value) OVER (PARTITION BY category)
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "category": ["A", "B", "C"],
            "value": [100, 200, 300],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "400 < SUM(value) OVER (PARTITION BY category)",
            id="sum_window",
        ),
        pytest.param(
            "COUNT(*) OVER (PARTITION BY category) = 3",
            id="count_window",
        ),
    ],
)
def test_qualify_matches_all_rows(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY id DESC
        """,
        compare_with="duckdb",
        expected={
            "id": [6, 5, 4, 3, 2, 1],
            "category": ["B", "B", "B", "A", "A", "A"],
            "value": [400, 250, 300, 150, 200, 100],
        },
    )


def test_qualify_multiple_clauses(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df
            QUALIFY
              value >= 300
              AND SUM(value) OVER (PARTITION BY category) > 500
            ORDER BY value
        """,
        compare_with="duckdb",
        expected={
            "id": [4, 6],
            "category": ["B", "B"],
            "value": [300, 400],
        },
    )
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df
            QUALIFY
              value = MAX(value) OVER (PARTITION BY category)
              OR value = MIN(value) OVER (PARTITION BY category)
            ORDER BY id
        """,
        compare_with="duckdb",
        expected={
            "id": [1, 2, 5, 6],
            "category": ["A", "A", "B", "B"],
            "value": [100, 200, 250, 400],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > MAX(value) OVER (PARTITION BY category)",
            id="greater_than_max",
        ),
        pytest.param(
            "value < MIN(value) OVER (PARTITION BY category)",
            id="less_than_min",
        ),
    ],
)
def test_qualify_returns_no_rows(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df QUALIFY {qualify_clause}
        """,
        compare_with="duckdb",
        expected={"id": [], "category": [], "value": []},
    )


def test_qualify_using_select_alias(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT
              id,
              category,
              value,
              MAX(value) OVER (PARTITION BY category) as max_value
            FROM df
            QUALIFY value = max_value
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
            "max_value": [200, 400],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > avg_value AND COUNT(*) OVER (PARTITION BY category) = 3",
            id="mixed_alias_and_explicit",
        ),
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category)",
            id="window_in_select",
        ),
    ],
)
def test_qualify_miscellaneous(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT
              id,
              category,
              value,
              AVG(value) OVER (PARTITION BY category) as avg_value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
            "avg_value": [150.0, 316.6666666666667],
        },
    )


def test_qualify_with_internal_cumulative_sum() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 3, 4, 2, 5],
            "value": [10, 30, 40, 20, 50],
        }
    )
    assert_sql_matches(
        {"df": df},
        query="""
            SELECT id, value
            FROM df
            QUALIFY SUM(value) OVER (ORDER BY id) <= 60
            ORDER BY id
        """,
        compare_with="duckdb",
        expected={
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        },
    )


def test_qualify_with_alias_and_comparison(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, SUM(value) OVER (PARTITION BY category) as total
            FROM df QUALIFY total > 500
            ORDER BY id DESC
        """,
        compare_with="duckdb",
        expected={
            "id": [6, 5, 4],
            "total": [950, 950, 950],
        },
    )


def test_qualify_with_where_clause(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df WHERE value > 200
            QUALIFY value != MAX(value) OVER (PARTITION BY category)
            ORDER BY value
        """,
        compare_with="duckdb",
        expected={
            "id": [5, 4],
            "category": ["B", "B"],
            "value": [250, 300],
        },
    )


def test_qualify_expected_errors(df_test: pl.DataFrame) -> None:
    ctx = pl.SQLContext(df=df_test, eager=True)
    with pytest.raises(
        SQLSyntaxError,
        match="QUALIFY clause must reference window functions",
    ):
        ctx.execute("SELECT id, category, value FROM df QUALIFY value > 200")

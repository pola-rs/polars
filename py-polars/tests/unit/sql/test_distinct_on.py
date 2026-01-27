from __future__ import annotations

import polars as pl
from tests.unit.sql import assert_sql_matches

# Note: SQLite does not support "DISTINCT ON", so only compare results against DuckDB


def test_distinct_on_single_column(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT ON with single column - keeps first row per distinct value."""
    assert_sql_matches(
        df_distinct,
        query="""
              SELECT DISTINCT ON (category)
                  category, subcategory, value, status, score
              FROM self
              ORDER BY category NULLS FIRST, value
              """,
        compare_with="duckdb",
        expected={
            "category": [None, "A", "B", "C"],
            "subcategory": ["x", "x", "y", "x"],
            "value": [600, 100, 200, 400],
            "status": ["active", "active", "active", "inactive"],
            "score": [70, 10, 30, 50],
        },
    )


def test_distinct_on_multiple_columns(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT ON with multiple columns."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT ON (category, subcategory)
                category, subcategory, value, status, score
            FROM self
            ORDER BY category NULLS FIRST, subcategory, score
        """,
        compare_with="duckdb",
        expected={
            "category": [None, None, "A", "B", "B", "C", "C"],
            "subcategory": ["x", "y", "x", "y", "z", "x", "y"],
            "value": [600, 700, 100, 200, 300, 400, 500],
            "status": [
                "active",
                "inactive",
                "active",
                "active",
                "active",
                "inactive",
                "active",
            ],
            "score": [70, 80, 10, 30, 40, 50, 60],
        },
    )


def test_distinct_on_after_join(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT ON applied after a JOIN operation."""
    df_supplements = pl.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C"],
            "supplement_id": [1, 2, 3, 4, 5],
            "supplement_score": [95, 85, 75, 90, 80],
        }
    )
    assert_sql_matches(
        frames={"data": df_distinct, "supplements": df_supplements},
        query="""
              SELECT DISTINCT ON (d.category)
                  d.category,
                  d.value,
                  s.supplement_id,
                  s.supplement_score
              FROM data d
                  INNER JOIN supplements s ON d.category = s.category
              ORDER BY
                  d.category,
                  s.supplement_score DESC,
                  d.value ASC
              """,
        compare_with="duckdb",
        expected={
            "category": ["A", "B", "C"],
            "value": [100, 200, 400],
            "supplement_id": [1, 4, 5],
            "supplement_score": [95, 90, 80],
        },
    )


def test_distinct_on_with_ordering(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT ON where ORDER BY determines which row is kept."""
    # ascending 'value' order
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT ON (status)
                status, value
            FROM self
            WHERE category IS NOT NULL
            ORDER BY status, value ASC
        """,
        compare_with="duckdb",
        expected={
            "status": ["active", "inactive"],
            "value": [100, 200],
        },
    )

    # descending 'value' order
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT ON (status)
                status, value
            FROM self
            WHERE category IS NOT NULL
            ORDER BY status, value DESC
        """,
        compare_with="duckdb",
        expected={
            "status": ["active", "inactive"],
            "value": [500, 400],
        },
    )

    # ascending category with 'nulls first', descending score
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT ON (category)
                category, subcategory, value, score
            FROM self
            ORDER BY category NULLS FIRST, score DESC
        """,
        compare_with="duckdb",
        expected={
            "category": [None, "A", "B", "C"],
            "subcategory": ["y", "x", "z", "y"],
            "value": [700, 100, 300, 500],
            "score": [80, 20, 40, 60],
        },
    )

    # mixed ordering, multiple 'distinct on' cols
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT ON (category, status)
                category, status, subcategory, value, score
            FROM self
            ORDER BY category NULLS FIRST, status, value DESC, score ASC
        """,
        compare_with="duckdb",
        expected={
            "category": [None, None, "A", "B", "B", "C", "C"],
            "status": [
                "active",
                "inactive",
                "active",
                "active",
                "inactive",
                "active",
                "inactive",
            ],
            "subcategory": ["x", "y", "x", "z", "y", "y", "x"],
            "value": [600, 700, 100, 300, 200, 500, 400],
            "score": [70, 80, 10, 40, 30, 60, 50],
        },
    )


def test_distinct_on_with_distinct_aggregation_in_join(
    df_distinct: pl.DataFrame,
) -> None:
    """Test DISTINCT COUNT in aggregations used within JOINs."""
    df_targets = pl.DataFrame(
        {
            "category": ["A", "B", "C", "A"],
            "target_subcats": [1, 2, 3, 4],
        }
    )
    assert_sql_matches(
        frames={"dist": df_distinct, "targets": df_targets},
        query="""
              SELECT DISTINCT ON (a.category)
                  a.category,
                  a.unique_subcats,
                  t.target_subcats,
                  CASE
                  WHEN a.unique_subcats >= t.target_subcats THEN 'yes'
                  ELSE 'no'
              END AS met_target
            FROM (
                SELECT
                    category,
                    COUNT(DISTINCT subcategory) AS unique_subcats
                FROM dist
                WHERE category IS NOT NULL
                GROUP BY category
            ) a
            INNER JOIN targets t ON a.category = t.category
            ORDER BY a.category, target_subcats DESC
              """,
        compare_with=["duckdb"],
        expected={
            "category": ["A", "B", "C"],
            "unique_subcats": [1, 2, 2],
            "target_subcats": [4, 2, 3],
            "met_target": ["no", "yes", "no"],
        },
    )

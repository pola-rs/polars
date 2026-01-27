from __future__ import annotations

import polars as pl
from tests.unit.sql import assert_sql_matches


def test_distinct_basic_single_column(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT on a single column."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT category
            FROM self ORDER BY category NULLS FIRST
        """,
        compare_with=["sqlite"],
        expected={"category": [None, "A", "B", "C"]},
    )


def test_distinct_basic_multiple_columns(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT across multiple columns."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT category, subcategory
            FROM self ORDER BY category NULLS FIRST, subcategory
        """,
        compare_with=["sqlite"],
        expected={
            "category": [None, None, "A", "B", "B", "C", "C"],
            "subcategory": ["x", "y", "x", "y", "z", "x", "y"],
        },
    )


def test_distinct_basic_all_columns(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT across all columns."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT * FROM self
            ORDER BY category NULLS FIRST, subcategory, value, status, score
        """,
        compare_with=["sqlite"],
        expected={
            "category": [None, None, "A", "A", "B", "B", "B", "C", "C"],
            "subcategory": ["x", "y", "x", "x", "y", "y", "z", "x", "y"],
            "value": [600, 700, 100, 100, 200, 200, 300, 400, 500],
            "status": [
                "active",
                "inactive",
                "active",
                "active",
                "active",
                "inactive",
                "active",
                "inactive",
                "active",
            ],
            "score": [70, 80, 10, 20, 30, 30, 40, 50, 60],
        },
    )


def test_distinct_with_expressions(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT with column expressions and aggregations."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT
                category,
                value * 2 AS doubled_value
            FROM self
            WHERE category IS NOT NULL
            ORDER BY category, doubled_value
        """,
        compare_with=["sqlite"],
        expected={
            "category": ["A", "B", "B", "C", "C"],
            "doubled_value": [200, 400, 600, 800, 1000],
        },
    )


def test_distinct_with_full_outer_join(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT with FULL OUTER JOIN producing NULLs on both sides."""
    df_extended = pl.DataFrame(
        {
            "category": ["A", "D", "E"],
            "extra_info": ["info_a", "info_d", "info_e"],
        }
    )
    assert_sql_matches(
        frames={"data": df_distinct, "extended": df_extended},
        query="""
            SELECT DISTINCT
                COALESCE(d.category, e.category) AS category,
                e.extra_info
            FROM (SELECT DISTINCT category FROM data WHERE category IS NOT NULL) d
            FULL JOIN extended e USING (category)
            ORDER BY category
        """,
        compare_with=["sqlite"],
        expected={
            "category": ["A", "B", "C", "D", "E"],
            "extra_info": ["info_a", None, None, "info_d", "info_e"],
        },
    )


def test_distinct_with_group_by(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT in combination with GROUP BY."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT
                category,
                COUNT(DISTINCT subcategory) AS distinct_subcats,
                COUNT(DISTINCT status) AS distinct_statuses,
                SUM(value) AS total_value
            FROM self
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY category
        """,
        compare_with=["sqlite"],
        expected={
            "category": ["A", "B", "C"],
            "distinct_subcats": [1, 2, 2],
            "distinct_statuses": [1, 2, 2],
            "total_value": [300, 700, 900],
        },
    )
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT *
            FROM (
                SELECT
                    subcategory,
                    COUNT(DISTINCT category) AS distinct_categories
                FROM self
                WHERE category IS NOT NULL
                GROUP BY subcategory
            ) AS agg
            WHERE distinct_categories > 1
            ORDER BY subcategory
        """,
        compare_with=["sqlite"],
        expected={
            "subcategory": ["x", "y"],
            "distinct_categories": [2, 2],
        },
    )


def test_distinct_with_join(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT with multiway JOINs."""
    df_categories = pl.DataFrame(
        {
            "category": ["A", "B", "C", "D"],
            "category_name": ["Alpha", "Beta", "Gamma", "Delta"],
        }
    )
    df_status_info = pl.DataFrame(
        {
            "status": ["active", "inactive", "pending"],
            "priority": [1, 2, 3],
        }
    )
    assert_sql_matches(
        {
            "data": df_distinct,
            "categories": df_categories,
            "status_info": df_status_info,
        },
        query="""
            SELECT DISTINCT
                d.category,
                c.category_name,
                d.status,
                s.priority
            FROM data d
            INNER JOIN categories c ON d.category = c.category
            INNER JOIN status_info s ON d.status = s.status
            ORDER BY d.category, d.status
        """,
        compare_with=["sqlite"],
        expected={
            "category": ["A", "B", "B", "C", "C"],
            "category_name": ["Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            "status": ["active", "active", "inactive", "active", "inactive"],
            "priority": [1, 1, 2, 1, 2],
        },
    )


def test_distinct_with_left_join_nulls(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT behaviour with NULL values introduced by LEFT JOIN."""
    df_lookup = pl.DataFrame(
        {
            "category": ["A", "B"],
            "region": ["North", "South"],
        }
    )
    assert_sql_matches(
        frames={"data": df_distinct, "lookup": df_lookup},
        query="""
            SELECT DISTINCT
                d.category,
                l.region
            FROM data d
            LEFT JOIN lookup l ON d.category = l.category
            ORDER BY d.category NULLS FIRST, l.region
        """,
        compare_with=["sqlite"],
        expected={
            "category": [None, "A", "B", "C"],
            "region": [None, "North", "South", None],
        },
    )


def test_distinct_with_nulls_handling(df_distinct: pl.DataFrame) -> None:
    """Test that DISTINCT treats NULL values as different from each other."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT category, status
            FROM self ORDER BY category NULLS FIRST, status
        """,
        compare_with=["sqlite"],
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
        },
    )


def test_distinct_with_where_filter(df_distinct: pl.DataFrame) -> None:
    """Test DISTINCT with various WHERE clause filters."""
    assert_sql_matches(
        df_distinct,
        query="""
            SELECT DISTINCT category, status
            FROM self
            WHERE value >= 200 AND category IS NOT NULL
            ORDER BY category, status
          """,
        compare_with=["sqlite"],
        expected={
            "category": ["B", "B", "C", "C"],
            "status": ["active", "inactive", "active", "inactive"],
        },
    )

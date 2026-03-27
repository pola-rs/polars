from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "category": ["A", "A", "A", "B", "B", "B", "C"],
            "value": [20, 10, 30, 15, 40, 25, 35],
        }
    )


def test_over_with_order_by(df_test: pl.DataFrame) -> None:
    query = """
        SELECT
            id,
            value,
            SUM(value) OVER (ORDER BY value) AS sum_by_value
        FROM self
        ORDER BY id
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4, 5, 6, 7],
            "value": [20, 10, 30, 15, 40, 25, 35],
            "sum_by_value": [45, 10, 100, 25, 175, 70, 135],
        },
    )


def test_over_with_partition_by(df_test: pl.DataFrame) -> None:
    df = df_test.remove(pl.col("id") == 6)
    query = """
        SELECT
            category,
            value,
            ROW_NUMBER() OVER (PARTITION BY category ORDER BY value) AS row_num,
            COUNT(*) OVER w0 AS cat_count,
            SUM(value) OVER w0 AS cat_sum
        FROM self
        WINDOW w0 AS (PARTITION BY category)
        ORDER BY category, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "A", "B", "B", "C"],
            "value": [10, 20, 30, 15, 40, 35],
            "row_num": [1, 2, 3, 1, 2, 1],
            "cat_count": [3, 3, 3, 2, 2, 1],
            "cat_sum": [60, 60, 60, 55, 55, 35],
        },
    )


def test_over_with_cumulative_window_funcs(df_test: pl.DataFrame) -> None:
    query = """
        SELECT
            category,
            value,
            AVG(value) OVER (PARTITION BY category ORDER BY value) AS cum_avg,
            MIN(value) OVER (PARTITION BY category ORDER BY value) AS cum_min,
            MAX(value) OVER (PARTITION BY category ORDER BY value) AS cum_max,
            SUM(value) OVER (PARTITION BY category ORDER BY value) AS cum_sum
        FROM self
        ORDER BY category, value
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "A", "B", "B", "B", "C"],
            "value": [10, 20, 30, 15, 25, 40, 35],
            "cum_avg": [10.0, 15.0, 20.0, 15.0, 20.0, 26.666666666, 35.0],
            "cum_min": [10, 10, 10, 15, 15, 15, 35],
            "cum_max": [10, 20, 30, 15, 25, 40, 35],
            "cum_sum": [10, 30, 60, 15, 40, 80, 35],
        },
    )


def test_window_function_over_empty(df_test: pl.DataFrame) -> None:
    query = """
        SELECT
            id,
            COUNT(*) OVER () AS total_count,
            SUM(value) OVER () AS total_sum
        FROM self
        ORDER BY id
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4, 5, 6, 7],
            "total_count": [7, 7, 7, 7, 7, 7, 7],
            "total_sum": [175, 175, 175, 175, 175, 175, 175],
        },
    )


def test_window_function_order_by_asc_desc(df_test: pl.DataFrame) -> None:
    query = """
        SELECT
            id,
            value,
            SUM(value) OVER (ORDER BY value ASC) AS sum_asc,
            SUM(value) OVER (ORDER BY value DESC) AS sum_desc,
            ROW_NUMBER() OVER (ORDER BY value DESC) AS row_num_desc
        FROM self
        ORDER BY id
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4, 5, 6, 7],
            "value": [20, 10, 30, 15, 40, 25, 35],
            "sum_asc": [45, 10, 100, 25, 175, 70, 135],
            "sum_desc": [150, 175, 105, 165, 40, 130, 75],
            "row_num_desc": [5, 7, 3, 6, 1, 4, 2],
        },
    )


def test_window_function_misc_aggregations(df_test: pl.DataFrame) -> None:
    df = df_test.filter(pl.col("id").is_in([1, 3, 4, 5, 7]))
    query = """
        SELECT
            category,
            value,
            COUNT(*) OVER (PARTITION BY category) AS cat_count,
            SUM(value) OVER (PARTITION BY category) AS cat_sum,
            AVG(value) OVER (PARTITION BY category) AS cat_avg,
            COUNT(*) OVER () AS total_count
        FROM self
        ORDER BY category, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "B", "B", "C"],
            "value": [20, 30, 15, 40, 35],
            "cat_count": [2, 2, 2, 2, 1],
            "cat_sum": [50, 50, 55, 55, 35],
            "cat_avg": [25.0, 25.0, 27.5, 27.5, 35.0],
            "total_count": [5, 5, 5, 5, 5],
        },
    )


def test_window_function_partition_by_multi() -> None:
    df = pl.DataFrame(
        {
            "region": ["North", "North", "North", "South", "South", "South"],
            "category": ["A", "A", "B", "A", "B", "B"],
            "value": [10, 20, 15, 30, 25, 35],
        }
    )
    query = """
        SELECT
            region,
            category,
            value,
            COUNT(*) OVER (PARTITION BY region, category) AS group_count,
            SUM(value) OVER (PARTITION BY region, category) AS group_sum
        FROM self
        ORDER BY region, category, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "region": ["North", "North", "North", "South", "South", "South"],
            "category": ["A", "A", "B", "A", "B", "B"],
            "value": [10, 20, 15, 30, 25, 35],
            "group_count": [2, 2, 1, 1, 2, 2],
            "group_sum": [30, 30, 15, 30, 60, 60],
        },
    )


def test_window_function_order_by_multi() -> None:
    df = pl.DataFrame(
        {
            "category": ["A", "A", "A", "B", "B"],
            "subcategory": ["X", "Y", "X", "Y", "X"],
            "value": [10, 20, 15, 30, 25],
        }
    )
    # Note: Polars uses ROWS semantics, not RANGE semantics; we make that explicit in
    # the query below so we can compare the result with SQLite (relational databases
    # usually default to RANGE semantics if not given an explicit frame spec)
    #
    # RANGE >> gives peer groups the same value: (A,X) → [25, 25, ...]
    # ROWS >> gives each row its own cumulative: (A,X) → [10, 25, ...]
    query = """
        SELECT
            category,
            subcategory,
            value,
            SUM(value) OVER (
                ORDER BY category ASC, subcategory ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS sum_asc
        FROM self
        ORDER BY category, subcategory, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "A", "B", "B"],
            "subcategory": ["X", "X", "Y", "X", "Y"],
            "value": [10, 15, 20, 25, 30],
            "sum_asc": [10, 25, 45, 70, 100],
        },
    )

    query = """
        SELECT
            category,
            subcategory,
            value,
            SUM(value) OVER (
                ORDER BY category DESC, subcategory DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS sum_desc
        FROM self
        ORDER BY category DESC, subcategory DESC, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["B", "B", "A", "A", "A"],
            "subcategory": ["Y", "X", "Y", "X", "X"],
            "value": [30, 25, 20, 10, 15],
            "sum_desc": [30, 55, 75, 85, 100],
        },
    )


def test_window_function_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "category": ["A", "A", None, "B", "B"],
            "value": [10, None, 15, 30, 25],
        }
    )
    # COUNT with PARTITION BY (where NULL is in the partition)
    query = """
        SELECT
            category,
            value,
            COUNT(*) OVER (PARTITION BY category) AS cat_count,
            COUNT(value) OVER (PARTITION BY category) AS value_count,
            COUNT(category) OVER () AS cat_count_global
        FROM self
        ORDER BY category NULLS LAST, value NULLS FIRST
    """
    assert_sql_matches(
        df,
        query=query,
        check_dtypes=False,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "B", "B", None],
            "value": [None, 10, 25, 30, 15],
            "cat_count": [2, 2, 2, 2, 1],
            "value_count": [1, 1, 2, 2, 1],
            "cat_count_global": [4, 4, 4, 4, 4],
        },
    )


def test_window_cumulative_agg_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "idx": [3, 1, 5, 2, 2, 4, 1, 3],
            "grp": ["yy", "xx", "xx", "yy", "xx", "xx", "yy", "xx"],
            "val": [None, 10.0, 50.0, 40.0, 20.0, None, None, 30.0],
        }
    )
    query = """
        SELECT
            *,
            SUM(val) OVER w AS cum_sum,
            MIN(val) OVER w AS cum_min,
            MAX(val) OVER w AS cum_max
        FROM self
        WINDOW w AS (
            PARTITION BY grp
            ORDER BY idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        ORDER BY grp, idx
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "grp": ["xx", "xx", "xx", "xx", "xx", "yy", "yy", "yy"],
            "idx": [1, 2, 3, 4, 5, 1, 2, 3],
            "val": [10.0, 20.0, 30.0, None, 50.0, None, 40.0, None],
            "cum_sum": [10.0, 30.0, 60.0, 60.0, 110.0, None, 40.0, 40.0],
            "cum_min": [10.0, 10.0, 10.0, 10.0, 10.0, None, 40.0, 40.0],
            "cum_max": [10.0, 20.0, 30.0, 30.0, 50.0, None, 40.0, 40.0],
        },
    )


def test_window_function_min_max(df_test: pl.DataFrame) -> None:
    df = df_test.filter(pl.col("id").is_in([1, 3, 4, 5, 7]))
    query = """
        SELECT
            category,
            value,
            MIN(value) OVER (PARTITION BY category) AS cat_min,
            MAX(value) OVER (PARTITION BY category) AS cat_max,
            MIN(value) OVER () AS global_min,
            MAX(value) OVER () AS global_max
        FROM self
        ORDER BY category, value
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "B", "B", "C"],
            "value": [20, 30, 15, 40, 35],
            "cat_min": [20, 20, 15, 15, 35],
            "cat_max": [30, 30, 40, 40, 35],
            "global_min": [15, 15, 15, 15, 15],
            "global_max": [40, 40, 40, 40, 40],
        },
    )


def test_window_function_first_last() -> None:
    df = pl.DataFrame(
        {
            "idx": [6, 5, 4, 3, 2, 1, 0],
            "category": ["A", "A", "A", "A", "B", "B", "C"],
            "value": [10, 20, 15, 30, None, 25, 5],
        }
    )
    for first, last, expected_first_last in (
        (
            "FIRST_VALUE(value) OVER (PARTITION BY category ORDER BY idx ASC) AS first_val",
            "LAST_VALUE(value) OVER (PARTITION BY category ORDER BY idx DESC) AS last_val",
            {
                "first_val": [30, 30, 30, 30, 25, 25, 5],
                "last_val": [10, 15, 20, 30, 25, None, 5],
            },
        ),
        (
            "FIRST_VALUE(value) OVER (PARTITION BY category ORDER BY idx DESC) AS first_val",
            "LAST_VALUE(value) OVER (PARTITION BY category ORDER BY idx ASC) AS last_val",
            {
                "first_val": [10, 10, 10, 10, None, None, 5],
                "last_val": [10, 15, 20, 30, 25, None, 5],
            },
        ),
    ):
        query = f"""
            SELECT category, value, {first}, {last},
            FROM self ORDER BY category, value
        """
        expected = pl.DataFrame(
            {
                "category": ["A", "A", "A", "A", "B", "B", "C"],
                "value": [10, 15, 20, 30, 25, None, 5],
                **expected_first_last,
            }
        )
        assert_frame_equal(df.sql(query), expected)
        assert_sql_matches(df, query=query, compare_with="duckdb", expected=expected)


def test_window_function_over_clause_misc() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        }
    )

    # OVER with empty spec
    query = "SELECT id, COUNT(*) OVER () AS cnt FROM self ORDER BY id"
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={"id": [1, 2, 3, 4], "cnt": [4, 4, 4, 4]},
    )

    # OVER with only PARTITION BY
    query = """
        SELECT id, category, COUNT(*) OVER (PARTITION BY category) AS count
        FROM self ORDER BY id
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "count": [2, 2, 2, 2],
        },
    )

    # OVER with only ORDER BY
    query = """
        SELECT id, value, SUM(value) OVER (ORDER BY value) AS sum_val
        FROM self ORDER BY id
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4],
            "value": [10, 20, 30, 40],
            "sum_val": [10, 30, 60, 100],
        },
    )

    # OVER with both PARTITION BY and ORDER BY
    query = """
        SELECT
            id,
            category,
            value,
            COUNT(*) OVER (PARTITION BY category ORDER BY value) AS cnt
        FROM self ORDER BY id
    """
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "id": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "cnt": [1, 2, 1, 2],
        },
    )


def test_window_named_window(df_test: pl.DataFrame) -> None:
    # One named window, applied multiple times
    query = """
        SELECT
            category,
            value,
            SUM(value) OVER w AS cumsum,
            MIN(value) OVER w AS cummin,
            MAX(value) OVER w AS cummax
        FROM self
        WINDOW w AS (PARTITION BY category ORDER BY value)
        ORDER BY category, value
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected=pl.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B", "C"],
                "value": [10, 20, 30, 15, 25, 40, 35],
                "cumsum": [10, 30, 60, 15, 40, 80, 35],
                "cummin": [10, 10, 10, 15, 15, 15, 35],
                "cummax": [10, 20, 30, 15, 25, 40, 35],
            }
        ),
    )


def test_window_multiple_named_windows(df_test: pl.DataFrame) -> None:
    # Multiple named windows with different properties
    query = """
        SELECT
            category,
            value,
            AVG(value) OVER w1 AS category_avg,
            SUM(value) OVER w2 AS running_sum,
            COUNT(*) OVER w3 AS total_count
        FROM self
        WINDOW
            w1 AS (PARTITION BY category),
            w2 AS (ORDER BY value),
            w3 AS ()
        ORDER BY category, value
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected=pl.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B", "C"],
                "value": [10, 20, 30, 15, 25, 40, 35],
                "category_avg": [
                    20.0,
                    20.0,
                    20.0,
                    26.666667,
                    26.666667,
                    26.666667,
                    35.0,
                ],
                "running_sum": [10, 45, 100, 25, 70, 175, 135],
                "total_count": [7, 7, 7, 7, 7, 7, 7],
            }
        ),
    )


def test_window_frame_validation() -> None:
    df = pl.DataFrame({"lbl": ["aa", "cc", "bb"], "value": [50, 75, -100]})

    # Omitted window frame => implicit ROWS semantics
    # (for Polars; for databases it usually implies RANGE semantics)
    for query in (
        """
        SELECT lbl, SUM(value) OVER (ORDER BY lbl) AS sum_value
        FROM self ORDER BY lbl ASC
        """,
        """
        SELECT lbl, SUM(value) OVER (
            ORDER BY lbl
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS sum_value
        FROM self ORDER BY lbl ASC
        """,
    ):
        assert df.sql(query).rows() == [("aa", 50), ("bb", -50), ("cc", 25)]
        assert_sql_matches(df, query=query, compare_with="sqlite")

    # Rejected: RANGE frame (peer group semantics not supported)
    query = """
        SELECT lbl, SUM(value) OVER (
            ORDER BY lbl
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS sum_value
        FROM self
    """
    with pytest.raises(
        SQLInterfaceError,
        match="RANGE-based window frames are not supported",
    ):
        df.sql(query)

    # Rejected: GROUPS frame
    query = """
        SELECT lbl, SUM(value) OVER (
            ORDER BY lbl
            GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS sum_value
        FROM self
    """
    with pytest.raises(
        SQLInterfaceError,
        match="GROUPS-based window frames are not supported",
    ):
        df.sql(query)

    # Rejected: ROWS with incompatible bounds
    query = """
        SELECT lbl, SUM(value) OVER (
            ORDER BY lbl
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS sum_value
        FROM self
    """
    with pytest.raises(
        SQLInterfaceError,
        match=(
            "only 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW' is currently "
            "supported; found 'ROWS BETWEEN 1 PRECEDING AND CURRENT ROW'"
        ),
    ):
        df.sql(query)

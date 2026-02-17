from __future__ import annotations

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "category": ["A", "A", "A", "B", "B", "B", "C"],
            "value": [20, 10, 25, 10, 40, 25, 35],
        }
    )


def test_rank_funcs_comparison(df_test: pl.DataFrame) -> None:
    # Compare ROW_NUMBER, RANK, and DENSE_RANK; can see the
    # differences between them when there are tied values
    query = """
        SELECT
            value,
            ROW_NUMBER() OVER (ORDER BY value) AS row_num,
            RANK() OVER (ORDER BY value) AS rank,
            DENSE_RANK() OVER (ORDER BY value) AS dense_rank
        FROM self
        ORDER BY value, id
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "value": [10, 10, 20, 25, 25, 35, 40],
            "row_num": [1, 2, 3, 4, 5, 6, 7],
            "rank": [1, 1, 3, 4, 4, 6, 7],
            "dense_rank": [1, 1, 2, 3, 3, 4, 5],
        },
    )


def test_rank_funcs_with_partition(df_test: pl.DataFrame) -> None:
    # All three ranking functions should return identical
    # results if there are no ties within the partitions
    query = """
        SELECT
            category,
            value,
            ROW_NUMBER() OVER (PARTITION BY category ORDER BY value) AS row_num,
            RANK() OVER (PARTITION BY category ORDER BY value) AS rank,
            DENSE_RANK() OVER (PARTITION BY category ORDER BY value) AS dense
        FROM self
        ORDER BY category, value
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "A", "B", "B", "B", "C"],
            "value": [10, 20, 25, 10, 25, 40, 35],
            # No ties within each partition, so identical results
            "row_num": [1, 2, 3, 1, 2, 3, 1],
            "rank": [1, 2, 3, 1, 2, 3, 1],
            "dense": [1, 2, 3, 1, 2, 3, 1],
        },
    )

    # We expect to see differences (in the same query) when there *are* ties
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "category": ["A", "A", "A", "B", "B", "B", "B", "C"],
            "value": [10, 10, 20, 20, 30, 30, 30, 30],
        }
    )
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        expected={
            "category": ["A", "A", "A", "B", "B", "B", "B", "C"],
            "value": [10, 10, 20, 20, 30, 30, 30, 30],
            # ROW_NUMBER: always unique
            "row_num": [1, 2, 3, 1, 2, 3, 4, 1],
            # RANK: ties get same rank, next rank skips
            "rank": [1, 1, 3, 1, 2, 2, 2, 1],
            # DENSE_RANK: ties get same rank, next rank is consecutive
            "dense": [1, 1, 2, 1, 2, 2, 2, 1],
        },
    )


def test_rank_funcs_desc(df_test: pl.DataFrame) -> None:
    query = """
        SELECT
            value,
            ROW_NUMBER() OVER (ORDER BY value DESC, category DESC) AS row_num,
            RANK() OVER (ORDER BY value DESC, category DESC) AS rank,
            DENSE_RANK() OVER (ORDER BY value DESC, category DESC) AS dense_rank
        FROM self
        ORDER BY value, id DESC
    """
    assert_sql_matches(
        df_test,
        query=query,
        compare_with="sqlite",
        expected={
            "value": [10, 10, 20, 25, 25, 35, 40],
            "row_num": [6, 7, 5, 3, 4, 2, 1],
            "rank": [6, 7, 5, 3, 4, 2, 1],
            "dense_rank": [6, 7, 5, 3, 4, 2, 1],
        },
    )


def test_rank_funcs_require_order_by(df_test: pl.DataFrame) -> None:
    # ROW_NUMBER without ORDER BY is fine (uses arbitrary order)
    query_row_num = "SELECT id, ROW_NUMBER() OVER () FROM self"
    result = df_test.sql(query_row_num)
    assert result.height == 7  # Just verify it runs

    # RANK without ORDER BY should error
    query_rank = "SELECT id, RANK() OVER (PARTITION BY category) FROM self"
    with pytest.raises(
        pl.exceptions.SQLSyntaxError,
        match="RANK requires an OVER clause with ORDER BY",
    ):
        df_test.sql(query_rank)

    # DENSE_RANK without ORDER BY should error
    query_dense = "SELECT id, DENSE_RANK() OVER () FROM self"
    with pytest.raises(
        pl.exceptions.SQLSyntaxError,
        match="DENSE_RANK requires an OVER clause with ORDER BY",
    ):
        df_test.sql(query_dense)

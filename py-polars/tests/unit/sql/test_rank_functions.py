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


def test_rank_funcs_without_order_by(df_test: pl.DataFrame) -> None:
    # ROW_NUMBER without ORDER BY is fine (uses arbitrary order)
    query_row_num = "SELECT id, ROW_NUMBER() OVER () FROM self"
    result = df_test.sql(query_row_num)
    assert result.height == 7  # Just verify it runs

    # without ORDER BY, all rows of the partition are peers
    query = """
        SELECT
            id,
            RANK() OVER (PARTITION BY category) AS rank,
            DENSE_RANK() OVER () AS dense_rank,
            PERCENT_RANK() OVER (PARTITION BY category) AS percent_rank,
            CUME_DIST() OVER () AS cume_dist
        FROM self
        ORDER BY id
    """
    assert_sql_matches(df_test, query=query, compare_with="duckdb", check_dtypes=True)


@pytest.fixture
def df_ties() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "g": [1, 1, 1, 1, 2, 2, None, 3],
            "x": [10, None, 20, 10, None, None, 3, 1],
            "y": [1, 2, 2, 1, None, 3, 3, None],
        }
    )


@pytest.mark.parametrize(
    "order_by",
    [
        "x",
        "x DESC",
        "x NULLS FIRST",
        "x DESC NULLS LAST",
        "x, y DESC",
        "x DESC NULLS LAST, y NULLS FIRST",
        # constant keys don't change the order
        "x, 1",
        "LOWER('A'), x DESC",
    ],
)
@pytest.mark.parametrize("partition_by", ["", "PARTITION BY g"])
def test_rank_funcs_nulls_and_ties(
    df_ties: pl.DataFrame, partition_by: str, order_by: str
) -> None:
    query = f"""
        SELECT
            id,
            RANK() OVER w AS rank,
            DENSE_RANK() OVER w AS dense_rank,
            PERCENT_RANK() OVER w AS percent_rank,
            CUME_DIST() OVER w AS cume_dist
        FROM self
        WINDOW w AS ({partition_by} ORDER BY {order_by})
        ORDER BY id
    """
    assert_sql_matches(
        df_ties,
        query=query,
        compare_with="duckdb",
        check_dtypes=True,
        engines=["in-memory", "streaming"],
    )


def test_rank_funcs_empty_input(df_ties: pl.DataFrame) -> None:
    query = """
        SELECT
            RANK() OVER () AS rank,
            PERCENT_RANK() OVER () AS percent_rank,
            CUME_DIST() OVER (ORDER BY x) AS cume_dist,
            NTILE(2) OVER (PARTITION BY g ORDER BY id) AS ntile
        FROM self
    """
    assert_sql_matches(
        df_ties.clear(),
        query=query,
        compare_with="duckdb",
        check_dtypes=True,
        engines=["in-memory", "streaming"],
    )


@pytest.mark.parametrize("buckets", [1, 2, 3, 5, 9])
def test_ntile(df_ties: pl.DataFrame, buckets: int) -> None:
    query = f"""
        SELECT
            id,
            NTILE({buckets}) OVER (ORDER BY id) AS a,
            NTILE({buckets}) OVER (PARTITION BY g ORDER BY x DESC, id) AS b
        FROM self
        ORDER BY id
    """
    assert_sql_matches(
        df_ties,
        query=query,
        compare_with="duckdb",
        check_dtypes=True,
        engines=["in-memory", "streaming"],
    )


@pytest.mark.parametrize(
    ("window_fn", "error"),
    [
        ("NTILE(0) OVER (ORDER BY id)", "NTILE expects a positive integer"),
        ("NTILE(x) OVER (ORDER BY id)", "NTILE expects a positive integer"),
        ("NTILE() OVER (ORDER BY id)", r"NTILE expects 1 argument \(found 0\)"),
        ("NTILE(2)", "NTILE requires an OVER clause"),
        ("CUME_DIST()", "CUME_DIST requires an OVER clause"),
        ("RANK(x) OVER (ORDER BY id)", r"RANK expects 0 arguments \(found 1\)"),
    ],
)
def test_rank_funcs_errors(df_ties: pl.DataFrame, window_fn: str, error: str) -> None:
    with pytest.raises(pl.exceptions.SQLSyntaxError, match=error):
        df_ties.sql(f"SELECT {window_fn} AS a FROM self")

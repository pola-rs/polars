from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "label": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "value": [100, 200, 150, 250, 120, 220, 180, 280, 130, 230],
        }
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (  # Basic FETCH FIRST / NEXT variants
            "SELECT * FROM self ORDER BY id FETCH FIRST 3 ROWS ONLY",
            {"id": [1, 2, 3], "label": ["A", "B", "A"], "value": [100, 200, 150]},
        ),
        (
            "SELECT * FROM self ORDER BY id FETCH NEXT 3 ROWS ONLY",
            {"id": [1, 2, 3], "label": ["A", "B", "A"], "value": [100, 200, 150]},
        ),
        (
            "SELECT * FROM self ORDER BY id FETCH FIRST 1 ROW ONLY",
            {"id": [1], "label": ["A"], "value": [100]},
        ),
        (  # OFFSET with FETCH
            "SELECT * FROM self ORDER BY id OFFSET 8 ROWS FETCH NEXT 5 ROWS ONLY",
            {"id": [9, 10], "label": ["A", "B"], "value": [130, 230]},
        ),
        (  # FETCH with WHERE
            "SELECT * FROM self WHERE label = 'A' ORDER BY value FETCH FIRST 3 ROWS ONLY",
            {"id": [1, 5, 9], "label": ["A", "A", "A"], "value": [100, 120, 130]},
        ),
        (  # FETCH with GROUP BY
            "SELECT label, SUM(value) AS total FROM self GROUP BY label ORDER BY total DESC FETCH FIRST 1 ROW ONLY",
            {"label": ["B"], "total": [1180]},
        ),
        (  # FETCH with DISTINCT
            "SELECT DISTINCT label FROM self ORDER BY label FETCH FIRST 1 ROW ONLY",
            {"label": ["A"]},
        ),
        (  # FETCH in subquery
            "SELECT * FROM (SELECT * FROM self ORDER BY value DESC FETCH FIRST 3 ROWS ONLY) AS top3 ORDER BY id",
            {"id": [4, 8, 10], "label": ["B", "B", "B"], "value": [250, 280, 230]},
        ),
        (  # Queries that should return no rows
            "SELECT * FROM self FETCH FIRST 0 ROWS ONLY",
            {"id": [], "label": [], "value": []},
        ),
        (
            "SELECT * FROM self ORDER BY id OFFSET 100 ROWS FETCH FIRST 5 ROWS ONLY",
            {"id": [], "label": [], "value": []},
        ),
        (  # FETCH in CTE
            """
            WITH top5 AS (
                SELECT * FROM self ORDER BY value DESC FETCH FIRST 5 ROWS ONLY
            )
            SELECT label, COUNT(*) AS cnt
            FROM top5
            GROUP BY label
            ORDER BY label
            """,
            {"label": ["B"], "cnt": [5]},
        ),
    ],
)
def test_fetch_clause(
    df: pl.DataFrame, query: str, expected: dict[str, Sequence[Any]]
) -> None:
    assert_sql_matches(
        df,
        query=query,
        compare_with="duckdb",
        expected=expected,
    )


def test_fetch_with_join(df: pl.DataFrame) -> None:
    categories = pl.DataFrame({"label": ["A", "B"], "description": ["Alpha", "Beta"]})
    assert_sql_matches(
        frames={
            "test": df,
            "categories": categories,
        },
        query="""
            SELECT test.id, test.value, categories.description
            FROM test
            JOIN categories ON test.label = categories.label
            ORDER BY test.value DESC
            FETCH FIRST 3 ROWS ONLY
        """,
        compare_with="duckdb",
        expected={
            "id": [8, 4, 10],
            "value": [280, 250, 230],
            "description": ["Beta", "Beta", "Beta"],
        },
    )


def test_fetch_with_union(df: pl.DataFrame) -> None:
    assert_sql_matches(
        frames={"tbl": df},
        query="""
          (
            SELECT id, value FROM tbl WHERE label = 'A'
            UNION ALL
            SELECT id, value FROM tbl WHERE label = 'B'
          )
          ORDER BY value
          FETCH FIRST 5 ROWS ONLY
        """,
        expected={"id": [1, 5, 9, 3, 7], "value": [100, 120, 130, 150, 180]},
        compare_with="duckdb",
    )


@pytest.mark.parametrize(
    ("query", "error_type", "match"),
    [
        (
            "SELECT * FROM self FETCH FIRST 50 PERCENT ROWS ONLY",
            SQLInterfaceError,
            r"`FETCH` with `PERCENT` is not supported",
        ),
        (
            "SELECT * FROM self ORDER BY value FETCH FIRST 5 ROWS WITH TIES",
            SQLInterfaceError,
            r"`FETCH` with `WITH TIES` is not supported",
        ),
        (
            "SELECT * FROM self LIMIT 5 FETCH FIRST 3 ROWS ONLY",
            SQLSyntaxError,
            r"cannot use both `LIMIT` and `FETCH`",
        ),
    ],
)
def test_fetch_errors(
    df: pl.DataFrame, query: str, error_type: type[Exception], match: str
) -> None:
    """Test error conditions for unsupported FETCH features."""
    with pytest.raises(error_type, match=match):
        df.sql(query)

from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import ColumnNotFoundError, SQLInterfaceError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def sales() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "region": ["n", "n", "s", "s", "s"],
            "quarter": ["q1", "q2", "q1", "q1", None],
            "amount": [10, 20, 30, None, 5],
            "rep": ["a", "a", "b", "b", "c"],
        }
    )


@pytest.fixture
def wide() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2], "q1": [10, 30], "q2": [None, 40]})


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('q1', 'q2', NULL))",
            {
                "region": ["n", "s", "s"],
                "rep": ["a", "b", "c"],
                "q1": [10, 30, None],
                "q2": [20, None, None],
                "NULL": [None, None, 5],
            },
        ),
        (
            """
            SELECT * FROM (SELECT rep, quarter, amount FROM sales)
            PIVOT (SUM(amount) AS total, COUNT(*) AS n FOR quarter IN ('q1' AS first, 'q3'))
            """,
            {
                "rep": ["a", "b", "c"],
                "first_total": [10, 30, None],
                "first_n": [1, 2, 0],
                "q3_total": [None, None, None],
                "q3_n": [0, 0, 0],
            },
        ),
        (
            # no column is left to group by: one row
            """
            SELECT * FROM (SELECT quarter, amount FROM sales)
            PIVOT (SUM(amount * 2) FOR UPPER(quarter) IN ('Q1', 'Q2'))
            """,
            {"Q1": [80], "Q2": [40]},
        ),
        (
            "SELECT * FROM sales PIVOT (MAX(rep) FOR amount IN (10, 30))",
            {
                "region": ["n", "n", "s", "s"],
                "quarter": ["q1", "q2", "q1", None],
                "10": ["a", None, None, None],
                "30": [None, None, "b", None],
            },
        ),
        (
            "SELECT p.rep, p.q1 FROM sales PIVOT (SUM(amount) FOR quarter IN ('q1')) AS p",
            {"rep": ["a", "b", "c"], "q1": [10, 30, None]},
        ),
        (
            # the columns a subquery reads are not read by the PIVOT
            """
            SELECT * FROM (SELECT rep, amount FROM sales)
            PIVOT (SUM(amount) FOR COALESCE((SELECT rep), '') IN ('a', 'b'))
            """,
            {"rep": ["a", "b", "c"], "a": [30, None, None], "b": [None, 30, None]},
        ),
    ],
)
def test_pivot(sales: pl.DataFrame, query: str, expected: dict[str, Any]) -> None:
    assert_sql_matches(
        {"sales": sales},
        query=query,
        compare_with="duckdb",
        check_row_order=False,
        expected=expected,
        engines=["in-memory", "streaming"],
    )


def test_pivot_names(sales: pl.DataFrame) -> None:
    df = sales.with_columns(d=pl.date(2024, 1, pl.int_range(1, 6)))
    res = pl.sql(
        """
        SELECT * FROM (SELECT d, amount FROM df)
        PIVOT (SUM(amount), MAX(amount) FOR d IN (DATE '2024-01-01', '2024-01-03'))
        """,
        eager=True,
    )
    assert res.columns == [
        "2024-01-01_SUM(amount)",
        "2024-01-01_MAX(amount)",
        "2024-01-03_SUM(amount)",
        "2024-01-03_MAX(amount)",
    ]
    assert res.rows() == [(10, 10, 30, 30)]

    res = pl.sql(
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('q1')) AS p(a, b, c)",
        eager=True,
    )
    assert res.columns == ["a", "b", "c"]


def test_pivot_aggregate_filter() -> None:
    df = pl.DataFrame({"k": ["x", "x", "y"], "v": [1, 2, 3]})
    for condition, expected in [
        ("1", [(3, 3)]),
        ("v > 1", [(2, 3)]),
        ("0", [(None, None)]),
    ]:
        query = f"SELECT * FROM df PIVOT (SUM(v) FILTER (WHERE {condition}) FOR k IN ('x', 'y'))"
        assert pl.sql(query, eager=True).rows() == expected


def test_pivot_column_references() -> None:
    df = pl.DataFrame({"t": ["a", "b"], "k": ["x", "x"], "v": [1, 2]})
    expected = [("a", 1), ("b", 2)]
    with pl.SQLContext({"df": df, "ns.t": df}) as ctx:
        for query in [
            # the qualifier `t` is not the column `t`
            "SELECT * FROM df AS t PIVOT (SUM(t.v) FOR t.k IN ('x'))",
            'SELECT * FROM df AS "a.b" PIVOT (SUM("a.b".v) FOR k IN (\'x\'))',
            "SELECT * FROM ns.t PIVOT (SUM(ns.t.v) FOR k IN ('x'))",
            # a subquery's own columns are not read from the input
            "SELECT * FROM df PIVOT (SUM(v) FOR COALESCE(k, (SELECT MIN(t) FROM df)) IN ('x'))",
            # a correlated subquery can name the input
            """
            SELECT * FROM df AS t
            PIVOT (SUM(v) FOR COALESCE((SELECT s.k FROM df AS s WHERE s.v = t.v), k) IN ('x'))
            """,
        ]:
            assert sorted(ctx.execute(query, eager=True).rows()) == expected


@pytest.mark.parametrize(
    ("query", "error"),
    [
        ("PIVOT (SUM(amount) FOR quarter IN (ANY))", "requires a list of values"),
        ("PIVOT (SUM(amount) FOR (quarter, rep) IN (('q1', 'a')))", "a single column"),
        ("PIVOT (ABS(amount) FOR quarter IN ('q1'))", "expects aggregate functions"),
        ("PIVOT (SUM(amount) FOR MAX(rep) IN ('a'))", "an aggregate after FOR"),
        ("PIVOT (SUM(amount) FOR quarter IN (q1))", "values must be constants"),
        ("PIVOT (SUM(amount) FOR quarter IN ((SELECT 'q1')))", "must be constants"),
        (
            "PIVOT (SUM(amount) FOR quarter IN (COLUMNS('quarter')))",
            "must be constants",
        ),
        ("PIVOT (SUM(amount) FOR quarter IN (RANDOM()))", "must be constants"),
        (
            "PIVOT (SUM(amount) FOR quarter IN (INTERVAL (SUM(amount)) DAY))",
            "must be constants",
        ),
        ("PIVOT (SUM(SUM(amount)) OVER () FOR quarter IN ('q1'))", "window functions"),
        ("PIVOT (SUM((SELECT 1)) FOR quarter IN ('q1'))", "cannot contain subqueries"),
        (
            "PIVOT (SUM(amount) FOR quarter IN ('q1') DEFAULT ON NULL (0))",
            "DEFAULT ON NULL",
        ),
    ],
)
def test_pivot_errors(sales: pl.DataFrame, query: str, error: str) -> None:
    with pytest.raises(SQLInterfaceError, match=error):
        pl.sql(f"SELECT * FROM sales {query}", eager=True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "SELECT * FROM wide UNPIVOT (v FOR k IN (q1, q2))",
            {"id": [1, 2, 2], "k": ["q1", "q1", "q2"], "v": [10, 30, 40]},
        ),
        (
            "SELECT * FROM wide UNPIVOT INCLUDE NULLS (v FOR k IN (q1, q2))",
            {
                "id": [1, 1, 2, 2],
                "k": ["q1", "q2", "q1", "q2"],
                "v": [10, None, 30, 40],
            },
        ),
        (
            "SELECT * FROM wide AS w UNPIVOT (v FOR k IN (w.q1 AS 'first', q2))",
            {"id": [1, 2, 2], "k": ["first", "first", "q2"], "v": [10, 30, 40]},
        ),
        (
            "SELECT u.k, u.v FROM wide UNPIVOT (v FOR k IN (q2)) AS u",
            {"k": ["q2"], "v": [40]},
        ),
        (
            # a column listed twice, and labels that name other columns
            "SELECT * FROM wide UNPIVOT (v FOR k IN (q1 AS id, q1 AS q2, q1))",
            {
                "id": [1, 1, 1, 2, 2, 2],
                "q2": [None, None, None, 40, 40, 40],
                "k": ["id", "q2", "q1", "id", "q2", "q1"],
                "v": [10, 10, 10, 30, 30, 30],
            },
        ),
    ],
)
def test_unpivot(wide: pl.DataFrame, query: str, expected: dict[str, Any]) -> None:
    assert_sql_matches(
        {"wide": wide},
        query=query,
        compare_with="duckdb",
        check_row_order=False,
        expected=expected,
        engines=["in-memory", "streaming"],
    )


def test_unpivot_errors(wide: pl.DataFrame) -> None:
    with pytest.raises(SQLInterfaceError, match="a single value column"):
        pl.sql("SELECT * FROM wide UNPIVOT ((a, b) FOR k IN ((q1, q2)))", eager=True)
    with pytest.raises(SQLInterfaceError, match="expects column names"):
        pl.sql("SELECT * FROM wide UNPIVOT (v FOR k IN (q1 + 1))", eager=True)
    with pytest.raises(ColumnNotFoundError, match="'nosuch' was not found"):
        pl.sql("SELECT * FROM wide UNPIVOT (v FOR k IN (nosuch))", eager=True)
    with pytest.raises(ColumnNotFoundError, match=r"'nosuch\.q1' was not found"):
        pl.sql("SELECT * FROM wide UNPIVOT (v FOR k IN (nosuch.q1))", eager=True)


def test_pivot_unpivot_round_trip(wide: pl.DataFrame) -> None:
    res = pl.sql(
        """
        SELECT * FROM wide
        UNPIVOT INCLUDE NULLS (v FOR k IN (q1, q2))
        PIVOT (MAX(v) FOR k IN ('q1', 'q2'))
        """,
        eager=True,
    )
    assert_frame_equal(res, wide, check_row_order=False)

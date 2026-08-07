from __future__ import annotations

import re
from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_sql_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["xyz", "abcde", None]})
    sql_exprs = pl.sql_expr(
        [
            "MIN(a)",
            "POWER(a,a) AS aa",
            "SUBSTR(b,2,2) AS b2",
        ]
    )
    result = df.select(*sql_exprs)
    expected = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "aa": [1, 4, 27],
            "b2": ["yz", "bc", None],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "clause"),
    [
        ("1 + 2 ORDER BY a", "ORDER"),
        ("EXCEPT x", "EXCEPT"),
        ("EXPLAIN SELECT 1", "EXPLAIN"),
        ("FROM tbl", "FROM"),
        ("GROUP BY a", "GROUP"),
        ("HAVING count(*) > 1", "HAVING"),
        ("INTERSECT y", "INTERSECT"),
        ("INTO outfile", "INTO"),
        ("LIMIT 10", "LIMIT"),
        ("MAX(a) UNION SELECT b", "UNION"),
        ("ORDER BY a", "ORDER"),
        ("SELECT xyz", "SELECT"),
        ("UNION ALL", "UNION"),
        ("WHERE abcd = 1", "WHERE"),
        ("WITH cte AS (SELECT 1)", "WITH"),
        ("a = 3 WHERE x = 0", "WHERE"),
        ("a SELECT b", "SELECT"),
        ("x + 1 LIMIT 10", "LIMIT"),
    ],
)
def test_sql_expr_rejects_clauses(expr: str, clause: str) -> None:
    with pytest.raises(
        SQLInterfaceError,
        match=rf"expected an expression \(found '{clause}' clause\)",
    ):
        pl.sql_expr(expr)


@pytest.mark.parametrize(
    ("expr", "token"),
    [("a, b", ","), ("x AS y %", "%"), ("a; DROP TABLE t", ";")],
)
def test_sql_expr_rejects_invalid_expressions(expr: str, token: str) -> None:
    with pytest.raises(
        SQLInterfaceError,
        match=rf"invalid expression \(found unexpected token '{re.escape(token)}'\)",
    ):
        pl.sql_expr(expr)


@pytest.mark.parametrize(
    "expr",
    ["@#$$% = 100", "||| AS abcd", "xyz.*"],
)
def test_sql_expr_invalid_colnames(expr: str) -> None:
    with pytest.raises(
        SQLInterfaceError,
        match=rf"unable to parse '{re.escape(expr)}' as Expr",
    ):
        pl.sql_expr(expr)

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": [1, 1, 2, 2],
            "v": [1, 2, 3, 4],
            "l": [[1, 2], [3, None], None, []],
            "s": [[{"a": 1}], [{"a": 2}, {"a": 3}], None, []],
            "n": [[[1, 2], [3]], [[4]], None, []],
        }
    )


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        ("list_transform(l, lambda x: x + 1)", [[2, 3], [4, None], None, []]),
        ("array_transform(l, lambda x:x*2)", [[2, 4], [6, None], None, []]),
        ("apply(l, lambda x: x::VARCHAR)", [["1", "2"], ["3", None], None, []]),
        ("list_apply(l, lambda x: 1)", [[1, 1], [1, 1], None, []]),
        ("array_apply(s, lambda x: x.a)", [[1], [2, 3], None, []]),
        ("list_filter(l, lambda x: x > 1)", [[2], [3], None, []]),
        ("array_filter(l, lambda x: x % 2)", [[1], [3], None, []]),
        ("filter(l, lambda x: x IS NULL)", [[], [None], None, []]),
        (
            "list_transform(n, lambda x: list_filter(x, lambda x: x > 1))",
            [[[2], [3]], [[4]], None, []],
        ),
        (
            "list_transform(n, lambda x: list_transform(x, lambda y: y * 10))",
            [[[10, 20], [30]], [[40]], None, []],
        ),
        ("list_transform(n, lambda x: x[1])", [[1, 3], [4], None, []]),
        (
            "list_transform(list_filter(l, lambda x: x > 1), lambda x: x * x)",
            [[4], [9], None, []],
        ),
    ],
)
def test_lambda_functions(df: pl.DataFrame, func: str, expected: list[Any]) -> None:
    res = df.sql(f"SELECT {func} AS r FROM self")
    assert res["r"].to_list() == expected


def test_lambda_in_clauses(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT g, list_transform(ARRAY_AGG(v ORDER BY v), lambda x: x * 2) AS r
        FROM self
        WHERE ARRAY_LENGTH(list_filter(l, lambda x: x > 1)) > 0
        GROUP BY g
        ORDER BY g
        """
    )
    assert res.rows() == [(1, [2, 4])]

    res = df.sql(
        """
        SELECT v, list_transform(l, lambda v: v + 1) AS l2,
          list_transform(l2, lambda x: x * 2) AS r
        FROM self
        QUALIFY ROW_NUMBER() OVER (ORDER BY v) <= 2
        ORDER BY v
        """
    )
    assert res.rows() == [(1, [2, 3], [4, 6]), (2, [4, None], [8, None])]

    ctx = pl.SQLContext(t=df)
    ctx.execute("CREATE VIEW w AS SELECT list_filter(l, lambda x: x < 3) AS r FROM t")
    assert ctx.execute("SELECT r FROM w", eager=True)["r"].to_list() == [
        [1, 2],
        [],
        None,
        [],
    ]


def test_lambda_input_types() -> None:
    res = pl.sql(
        """
        SELECT
          list_transform([1, 2], lambda x: x * 2) AS a,
          list_transform(NULL, lambda x: x + 1) AS b,
          list_transform(['a', 'bb'], lambda x: upper(x) || x) AS c
        """,
        eager=True,
    )
    assert res.row(0) == ([2, 4], None, ["Aa", "BBbb"])

    df = pl.DataFrame({"a": [[1, 2], [3, 4]]}, schema={"a": pl.Array(pl.Int64, 2)})
    res = df.sql(
        """
        SELECT list_transform(a, lambda x: x * 2) AS t,
          list_filter(a, lambda x: x > 1) AS f
        FROM self
        """
    )
    expected = pl.DataFrame({"t": [[2, 4], [6, 8]], "f": [[2], [3, 4]]})
    assert_frame_equal(res, expected)


def test_lambda_element_type() -> None:
    res = pl.sql(
        """
        SELECT
          list_filter([0, 1, 2, -1, NULL], lambda x: x IS TRUE) AS a,
          list_filter([0, 1, 2, -1, NULL], lambda x: x IS NOT TRUE) AS b,
          list_filter(
            [TIMESTAMP '2020-01-01 00:00:00', NULL],
            lambda x: coalesce(x, '2020-01-03 00:00:00') = '2020-01-01 00:00:00'
          ) AS c
        """,
        eager=True,
    )
    assert res.row(0) == ([1, 2, -1], [0, None], [datetime(2020, 1, 1)])


def test_lambda_constant_list_rows() -> None:
    res = pl.sql(
        """
        SELECT list_filter([1, 2, 3], lambda x: x > 1) AS r,
          list_transform([[1, 2], [3]], lambda x: list_filter(x, lambda y: y > 1)) AS n
        FROM (VALUES (1), (2)) AS t(i)
        """,
        eager=True,
    )
    assert res.rows() == [([2, 3], [[2], [3]])] * 2


def test_lambda_typed_inputs() -> None:
    res = pl.sql(
        """
        SELECT
          list_filter((SELECT [0, 1, 2, -1, NULL]), lambda x: x IS TRUE) AS a,
          list_filter(ARRAY_AGG(i) OVER (), lambda x: x IS TRUE) AS b
        FROM (VALUES (0), (2)) AS t(i)
        """,
        eager=True,
    )
    assert res.rows() == [([1, 2, -1], [2])] * 2
    with pytest.raises(SQLSyntaxError, match="LIST_FILTER expects a list"):
        pl.sql("SELECT list_filter(1, lambda x: x)", eager=True)


def test_lambda_sql_expr() -> None:
    df = pl.DataFrame({"l": [[1, 2], [3]]})
    res = df.select(pl.sql_expr("list_transform(l, lambda x: x - 1) AS r"))
    assert res["r"].to_list() == [[0, 1], [2]]


def test_lambda_columns(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT
          list_transform(l, lambda x: x + v) AS a,
          list_filter(l, lambda x: x >= self.v) AS b,
          list_transform(n, lambda x: list_filter(x, lambda y: y > g)) AS c
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "a": [[2, 3], [5, None], None, []],
        "b": [[1, 2], [3], None, []],
        "c": [[[2], [3]], [[4]], None, []],
    }

    other = pl.DataFrame({"g": [1, 2], "t": [1, 3]})
    res = pl.SQLContext(t=df, o=other).execute(
        """
        SELECT t.v, list_filter(t.l, lambda x: x > o.t) AS r
        FROM t JOIN o ON t.g = o.g
        ORDER BY t.v
        """,
        eager=True,
    )
    assert res.rows() == [(1, [2]), (2, [3]), (3, None), (4, [])]


def test_lambda_named_column() -> None:
    df = pl.DataFrame({"lambda": [1, 2]})
    assert df.sql("SELECT lambda FROM self")["lambda"].to_list() == [1, 2]


@pytest.mark.parametrize(
    ("func", "error", "match"),
    [
        (
            "list_transform(n, lambda x: list_transform(x, lambda y: y + x))",
            SQLInterfaceError,
            "a nested lambda cannot use the parameter of an outer lambda",
        ),
        (
            "list_transform(l, lambda x, i: x + i)",
            SQLInterfaceError,
            r"takes one parameter \(found 2\)",
        ),
        (
            "list_transform(l, lambda x: (SELECT 1))",
            SQLInterfaceError,
            "subqueries are not supported in a lambda",
        ),
        (
            "list_transform(l, lambda x: SUM(x))",
            SQLInterfaceError,
            "aggregate functions are not supported in a lambda",
        ),
        (
            "list_transform(l, lambda x: ROW_NUMBER() OVER ())",
            SQLInterfaceError,
            "window functions are not supported in a lambda",
        ),
        (
            "list_transform(l, lambda x: x + RANDOM())",
            SQLInterfaceError,
            "RANDOM",
        ),
        (
            "array_reverse(l, lambda x: x)",
            SQLSyntaxError,
            "a lambda is only supported in LIST_TRANSFORM and LIST_FILTER",
        ),
        (
            "list_transform(l, 1)",
            SQLSyntaxError,
            "the second argument of LIST_TRANSFORM must be a lambda",
        ),
    ],
)
def test_lambda_errors(
    df: pl.DataFrame, func: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        df.sql(f"SELECT {func} AS r FROM self")

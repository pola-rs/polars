from __future__ import annotations

from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import (
    InvalidOperationError,
    SQLInterfaceError,
    SQLSyntaxError,
)
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT * FROM generate_series(3)", [0, 1, 2, 3]),
        ("SELECT * FROM generate_series(1, 3)", [1, 2, 3]),
        ("SELECT * FROM generate_series(1, 6, 2)", [1, 3, 5]),
        ("SELECT * FROM generate_series(1, 7, 2)", [1, 3, 5, 7]),
        ("SELECT * FROM generate_series(3, 1, -1)", [3, 2, 1]),
        ("SELECT * FROM generate_series(3, 1)", []),
        ("SELECT * FROM range(3)", [0, 1, 2]),
        ("SELECT * FROM range(1, 3)", [1, 2]),
        ("SELECT * FROM range(1, 7, 2)", [1, 3, 5]),
        ("SELECT * FROM range(3, 0, -1)", [3, 2, 1]),
        ("SELECT * FROM range(3, 1)", []),
    ],
)
def test_series_table_function_int(query: str, expected: list[int]) -> None:
    res = pl.sql(query, eager=True)
    name = "generate_series" if "generate_series" in query else "range"
    assert_frame_equal(res, pl.DataFrame({name: expected}, schema={name: pl.Int64}))


def test_series_table_function_temporal() -> None:
    res = pl.sql(
        """
        SELECT * FROM generate_series(
          DATE '2026-01-30', DATE '2026-04-30', INTERVAL '1 month'
        ) AS cal(d)
        """,
        eager=True,
    )
    expected = [datetime(2026, 1, 30), datetime(2026, 2, 28), datetime(2026, 3, 30)]
    expected.append(datetime(2026, 4, 30))
    assert_frame_equal(res, pl.DataFrame({"d": expected}))

    res = pl.sql(
        """
        SELECT * FROM range(
          TIMESTAMP '2026-01-01 00:00', TIMESTAMP '2026-01-01 01:00', INTERVAL 20 MINUTE
        )
        """,
        eager=True,
    )
    assert res["range"].to_list() == [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 20),
        datetime(2026, 1, 1, 0, 40),
    ]


def test_series_table_function_in_query() -> None:
    df = pl.DataFrame({"n": [2, 4]})
    res = pl.sql(
        """
        SELECT g.i, df.n
        FROM generate_series(1, 4) AS g(i)
        JOIN df ON g.i = df.n
        WHERE g.i > (SELECT MIN(n) FROM df) - 1
        ORDER BY g.i
        """,
        eager=True,
    )
    assert_frame_equal(res, pl.DataFrame({"i": [2, 4], "n": [2, 4]}))

    res = pl.sql(
        "SELECT * FROM generate_series(1, (SELECT MAX(n) FROM df))", eager=True
    )
    assert res["generate_series"].to_list() == [1, 2, 3, 4]


def test_series_scalar_function() -> None:
    df = pl.DataFrame({"a": [1, 5, 3], "b": [3, 6, 1], "s": [1, 1, -1]})
    res = pl.sql(
        """
        SELECT
          generate_series(a, b, s) AS gs,
          range(a, b, s) AS r,
          generate_series(b) AS gs_one
        FROM df
        """,
        eager=True,
    )
    expected = pl.DataFrame(
        {
            "gs": [[1, 2, 3], [5, 6], [3, 2, 1]],
            "r": [[1, 2], [5], [3, 2]],
            "gs_one": [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], [0, 1]],
        },
        schema={k: pl.List(pl.Int64) for k in ("gs", "r", "gs_one")},
    )
    assert_frame_equal(res, expected)


def test_series_unnest_calendar() -> None:
    res = pl.sql(
        """
        SELECT unnest(
          generate_series(DATE '2026-01-01', DATE '2026-01-03', INTERVAL 1 DAY)
        ) AS d
        """,
        eager=True,
    )
    expected = [datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3)]
    assert_frame_equal(res, pl.DataFrame({"d": expected}))

    df = pl.DataFrame({"start": [date(2026, 1, 1)], "stop": [date(2026, 1, 2)]})
    res = pl.sql(
        "SELECT range(start, stop, INTERVAL '12 hours') AS d FROM df", eager=True
    )
    assert res["d"].to_list() == [[datetime(2026, 1, 1), datetime(2026, 1, 1, 12)]]


def test_series_errors() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(SQLInterfaceError, match="takes only constant arguments"):
        pl.sql("SELECT * FROM df, generate_series(df.a, 3)", eager=True)
    with pytest.raises(SQLSyntaxError, match="expects 1 to 3 arguments"):
        pl.sql("SELECT * FROM range(1, 2, 3, 4)", eager=True)
    with pytest.raises(SQLSyntaxError, match="expects 1 to 3 arguments"):
        pl.sql("SELECT generate_series()", eager=True)
    with pytest.raises(InvalidOperationError, match="step must not be zero"):
        pl.sql("SELECT * FROM generate_series(1, 3, 0)", eager=True)
    with pytest.raises(SQLInterfaceError, match="negative INTERVAL steps"):
        pl.sql(
            "SELECT * FROM range(DATE '2026-01-02', DATE '2026-01-01', -INTERVAL 1 DAY)",
            eager=True,
        )

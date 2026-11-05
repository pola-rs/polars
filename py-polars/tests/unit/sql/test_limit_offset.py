from __future__ import annotations

from typing import Literal

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def frames() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.DataFrame({"x": list(range(10))}),
        "c": pl.DataFrame({"y": [1, 2, 3]}),
    }


@pytest.mark.parametrize(
    "query",
    [
        "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM c)",
        "SELECT x FROM t ORDER BY x LIMIT (SELECT COUNT(*) FROM c)",
        "SELECT x FROM t ORDER BY x LIMIT 2 OFFSET (SELECT COUNT(*) FROM c) / 2",
        "SELECT x FROM t ORDER BY x LIMIT (SELECT COUNT(*) FROM c) OFFSET 7 / 2",
        "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM c) * 0.5",
        "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM t) / 4",
        "SELECT x FROM t ORDER BY x LIMIT CAST(2.5 AS DOUBLE) OFFSET CAST(2.5 AS DECIMAL(3, 1))",
        "SELECT x FROM t ORDER BY x LIMIT CAST(0.5 AS DOUBLE)",
        "SELECT x FROM t ORDER BY x LIMIT (SELECT COUNT(*) FROM t) / 4",
        "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM t) * 2",
        "SELECT x FROM t ORDER BY x LIMIT (SELECT NULL) OFFSET (SELECT NULL)",
        (
            "SELECT x FROM t ORDER BY x OFFSET CASE WHEN (SELECT COUNT(*) FROM c) < 5 "
            "THEN 1 ELSE 0 END ROWS FETCH NEXT 2 ROWS ONLY"
        ),
        "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM c) ROWS FETCH NEXT ROW ONLY",
        "SELECT x FROM t ORDER BY x OFFSET 2 ROWS FETCH NEXT ROW ONLY",
        "SELECT x FROM t ORDER BY x LIMIT 1.5",
        "SELECT x FROM t ORDER BY x LIMIT -0.1",
        "SELECT x FROM t ORDER BY x LIMIT '2'",
        "SELECT x FROM t ORDER BY x LIMIT '9223372036854775807'",
        (
            "SELECT x FROM t ORDER BY x "
            "LIMIT CAST('1.49999999999999999999' AS DECIMAL(38, 20)) "
            "OFFSET CAST('0.50000000000000000000' AS DECIMAL(38, 20))"
        ),
        "SELECT x FROM t ORDER BY x LIMIT CAST('0.49999999999999999999' AS DECIMAL(38, 20))",
        (
            "WITH w AS (SELECT * FROM c) "
            "SELECT x FROM t ORDER BY x OFFSET (SELECT COUNT(*) FROM w) * 2"
        ),
        (
            "SELECT (SELECT x FROM t ORDER BY x LIMIT 1 "
            "OFFSET (SELECT COUNT(*) FROM t) / 2) AS x"
        ),
        "SELECT x FROM t WHERE x < 0 ORDER BY x LIMIT (SELECT 1)",
    ],
)
def test_limit_offset_expressions(query: str, frames: dict[str, pl.DataFrame]) -> None:
    assert_sql_matches(frames=frames, query=query, compare_with="duckdb")
    lf = pl.SQLContext(frames).execute(query)
    assert_frame_equal(lf.collect(engine="streaming"), lf.collect(engine="in-memory"))


@pytest.mark.parametrize(
    ("clause", "err"),
    [
        ("LIMIT (SELECT -1)", "conversion from"),
        ("OFFSET (SELECT -1)", "conversion from"),
        ("LIMIT 9223372036854775808", "conversion from"),
        ("LIMIT '1.5'", "conversion from"),
        ("LIMIT x", "must be one value for the query"),
        ("LIMIT COUNT(*)", "must be one value for the query"),
        ("LIMIT ROW_NUMBER() OVER ()", "must be one value for the query"),
        ("LIMIT UNNEST([1, 2])", "must be one value for the query"),
        ("LIMIT (SELECT y FROM c)", "single value"),
        ("LIMIT CAST('2020-01-01' AS DATE)", "must be a number"),
        ("LIMIT TRUE", "must be a number"),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_limit_offset_expression_errors(
    clause: str,
    err: str,
    engine: Literal["in-memory", "streaming"],
    frames: dict[str, pl.DataFrame],
) -> None:
    with pytest.raises(Exception, match=err):
        pl.SQLContext(frames).execute(f"SELECT x FROM t {clause}").collect(
            engine=engine
        )


@pytest.mark.parametrize("clause", ["LIMIT (SELECT 1)", "OFFSET (SELECT 2)"])
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_limit_offset_expression_keeps_column_names(
    clause: str,
    engine: Literal["in-memory", "streaming"],
    frames: dict[str, pl.DataFrame],
) -> None:
    res = (
        pl.SQLContext(frames)
        .execute(f'SELECT x AS "^foo$", x AS "*", x + 10 AS y FROM t {clause}')
        .collect(engine=engine)
    )
    assert res.columns == ["^foo$", "*", "y"]


def test_limit_offset_literals_keep_slice(frames: dict[str, pl.DataFrame]) -> None:
    plan = (
        pl.SQLContext(frames)
        .execute("SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 2")
        .explain()
    )
    assert "slice: (2, 3" in plan
    assert ".slice(" not in plan

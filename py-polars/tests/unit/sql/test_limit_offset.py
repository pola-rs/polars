from __future__ import annotations

from typing import Literal

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
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
        ("LIMIT ROW_NUMBER()", "LIMIT must be a single value"),
        ("OFFSET ROW_NUMBER()", "OFFSET must be a single value"),
        # The count doesn't depend on the input, so it is negative also on no rows.
        ("WHERE x < 0 LIMIT FLOOR(RANDOM()) - 1", "conversion from"),
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


@pytest.mark.parametrize(
    "query",
    [
        "SELECT x FROM t ORDER BY x LIMIT FLOOR(RANDOM()) + 3",
        "SELECT x FROM t ORDER BY x OFFSET FLOOR(RANDOM()) + 3",
        (
            "SELECT x FROM t ORDER BY x "
            "LIMIT FLOOR(RANDOM()) + 3 OFFSET FLOOR(RANDOM()) + 3"
        ),
        (
            "SELECT x FROM t ORDER BY x "
            "OFFSET (FLOOR(RANDOM()) + 3) ROWS FETCH NEXT 2 ROWS ONLY"
        ),
    ],
)
def test_limit_offset_random(query: str, frames: dict[str, pl.DataFrame]) -> None:
    assert_sql_matches(frames=frames, query=query, compare_with="duckdb")
    expected = pl.SQLContext(frames).execute(query).collect()
    assert_frame_equal(frames["t"].sql(query.replace("FROM t", "FROM self")), expected)


def test_limit_random_is_one_draw_per_query() -> None:
    # A seed whose draw is neither 0 nor 1000 rows.
    seed = 1
    ctx = pl.SQLContext(
        one=pl.DataFrame({"x": [0]}), t=pl.DataFrame({"x": range(1000)})
    )
    pl.set_random_seed(seed)
    n = ctx.execute("SELECT FLOOR(RANDOM() * 1000) AS n FROM one").collect().item()
    assert 0 < n < 1000

    pl.set_random_seed(seed)
    res = ctx.execute("SELECT x FROM t LIMIT FLOOR(RANDOM() * 1000)").collect()
    assert res.height == n


def test_limit_random_keeps_prefix(frames: dict[str, pl.DataFrame]) -> None:
    query = "SELECT x FROM t ORDER BY x LIMIT RANDOM() * 5 + 2"
    res = pl.SQLContext(frames).execute(query).collect()
    assert 2 <= res.height <= 7
    assert res["x"].to_list() == list(range(res.height))


@pytest.mark.parametrize(
    ("query", "max_height"),
    [
        ("SELECT y FROM one LIMIT RANDOM()", 1),
        ("SELECT y FROM one OFFSET RANDOM()", 1),
        ("SELECT x FROM t WHERE x < 0 LIMIT RANDOM()", 0),
        ("SELECT x FROM t WHERE x < 0 OFFSET RANDOM()", 0),
    ],
)
def test_limit_offset_random_small_input(
    query: str, max_height: int, frames: dict[str, pl.DataFrame]
) -> None:
    ctx = pl.SQLContext(frames, one=pl.DataFrame({"y": [1]}))
    assert ctx.execute(query).collect().height <= max_height


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * EXCLUDE x FROM t LIMIT FLOOR(RANDOM()) - 1",
        "SELECT * EXCLUDE x FROM t OFFSET FLOOR(RANDOM()) - 1",
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_limit_offset_no_columns_raises(
    query: str,
    engine: Literal["in-memory", "streaming"],
    frames: dict[str, pl.DataFrame],
) -> None:
    with pytest.raises(InvalidOperationError, match="conversion from"):
        pl.SQLContext(frames).execute(query).collect(engine=engine)

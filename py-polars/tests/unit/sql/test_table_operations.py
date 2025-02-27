from __future__ import annotations

import re
from datetime import date

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal


@pytest.fixture
def test_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": [1, 2, 3],
            "y": ["aaa", "bbb", "ccc"],
            "z": [date(2000, 12, 31), date(1978, 11, 15), date(2077, 10, 20)],
        },
        schema_overrides={"x": pl.UInt8},
    )


@pytest.mark.parametrize(
    ("delete_constraint", "expected_ids"),
    [
        # basic constraints
        ("WHERE id = 200", {100, 300}),
        ("WHERE id = 200 OR id = 300", {100}),
        ("WHERE id IN (200, 300, 400)", {100}),
        ("WHERE id NOT IN (200, 300, 400)", {200, 300}),
        # more involved constraints
        ("WHERE EXTRACT(year FROM dt) >= 2000", {200}),
        # null-handling (in the data)
        ("WHERE v1 < 0", {100, 300}),
        ("WHERE v1 > 0", {200, 300}),
        # null handling (in the constraint)
        ("WHERE v1 IS NULL", {100, 200}),
        ("WHERE v1 IS NOT NULL", {300}),
        # boolean handling (delete all/none)
        ("WHERE FALSE", {100, 200, 300}),
        ("WHERE TRUE", set()),
        # no constraint; equivalent to TRUNCATE (drop all rows)
        ("", set()),
    ],
)
def test_delete_clause(delete_constraint: str, expected_ids: set[int]) -> None:
    df = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "dt": [date(2020, 10, 10), date(1999, 1, 2), date(2001, 7, 5)],
            "v1": [3.5, -4.0, None],
            "v2": [10.0, 2.5, -1.5],
        }
    )
    res = df.sql(f"DELETE FROM self {delete_constraint}")
    assert set(res["id"]) == expected_ids


def test_drop_table(test_frame: pl.LazyFrame) -> None:
    # 'drop' completely removes the table from sql context
    expected = pl.DataFrame()

    with pl.SQLContext(frame=test_frame, eager=True) as ctx:
        res = ctx.execute("DROP TABLE frame")
        assert_frame_equal(res, expected)

        with pytest.raises(SQLInterfaceError, match="'frame' was not found"):
            ctx.execute("SELECT * FROM frame")


def test_explain_query(test_frame: pl.LazyFrame) -> None:
    # 'explain' returns the query plan for the given sql
    with pl.SQLContext(frame=test_frame) as ctx:
        plan = (
            ctx.execute("EXPLAIN SELECT * FROM frame")
            .select(pl.col("Logical Plan").str.join())
            .collect()
            .item()
        )
        assert (
            re.search(
                pattern=r"PROJECT.+?COLUMNS",
                string=plan,
                flags=re.IGNORECASE,
            )
            is not None
        )


def test_show_tables(test_frame: pl.LazyFrame) -> None:
    # 'show tables' lists all tables registered with the sql context in sorted order
    with pl.SQLContext(
        tbl3=test_frame,
        tbl2=test_frame,
        tbl1=test_frame,
    ) as ctx:
        res = ctx.execute("SHOW TABLES").collect()
        assert_frame_equal(res, pl.DataFrame({"name": ["tbl1", "tbl2", "tbl3"]}))


@pytest.mark.parametrize(
    "truncate_sql",
    [
        "TRUNCATE TABLE frame",
        "TRUNCATE frame",
    ],
)
def test_truncate_table(truncate_sql: str, test_frame: pl.LazyFrame) -> None:
    # 'truncate' preserves the table, but optimally drops all rows within it
    expected = pl.DataFrame(schema=test_frame.collect_schema())

    with pl.SQLContext(frame=test_frame, eager=True) as ctx:
        res = ctx.execute(truncate_sql)
        assert_frame_equal(res, expected)

        res = ctx.execute("SELECT * FROM frame")
        assert_frame_equal(res, expected)

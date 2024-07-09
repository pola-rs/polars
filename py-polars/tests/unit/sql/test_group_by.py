from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_group_by(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(eager=True)
    ctx.register("foods", lf)

    out = ctx.execute(
        """
        SELECT
            count(category) as n,
            category,
            max(calories) as max_cal,
            median(calories) as median_cal,
            min(fats_g) as min_fats
        FROM foods
        GROUP BY category
        HAVING n > 5
        ORDER BY n, category DESC
        """
    )
    assert out.to_dict(as_series=False) == {
        "n": [7, 7, 8],
        "category": ["vegetables", "fruit", "seafood"],
        "max_cal": [45, 130, 200],
        "median_cal": [25.0, 50.0, 145.0],
        "min_fats": [0.0, 0.0, 1.5],
    }

    lf = pl.LazyFrame(
        {
            "grp": ["a", "b", "c", "c", "b"],
            "att": ["x", "y", "x", "y", "y"],
        }
    )
    assert ctx.tables() == ["foods"]

    ctx.register("test", lf)
    assert ctx.tables() == ["foods", "test"]

    out = ctx.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(as_series=False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_group_by_all() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx", "yy", "xx", "zz"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [99, 99, 66, 66, 66, 66],
        }
    )

    # basic group/agg
    res = df.sql(
        """
        SELECT
            a,
            SUM(b),
            SUM(c),
            COUNT(*) AS n
        FROM self
        GROUP BY ALL
        ORDER BY ALL
        """
    )
    expected = pl.DataFrame(
        {
            "a": ["xx", "yy", "zz"],
            "b": [9, 6, 6],
            "c": [231, 165, 66],
            "n": [3, 2, 1],
        }
    )
    assert_frame_equal(expected, res, check_dtypes=False)

    # more involved determination of agg/group columns
    res = df.sql(
        """
        SELECT
            SUM(b) AS sum_b,
            SUM(c) AS sum_c,
            (SUM(b) + SUM(c)) / 2.0 AS sum_bc_over_2,  -- nested agg
            a as grp, --aliased group key
        FROM self
        GROUP BY ALL
        ORDER BY grp
        """
    )
    expected = pl.DataFrame(
        {
            "sum_b": [9, 6, 6],
            "sum_c": [231, 165, 66],
            "sum_bc_over_2": [120.0, 85.5, 36.0],
            "grp": ["xx", "yy", "zz"],
        }
    )
    assert_frame_equal(expected, res.sort(by="grp"))


def test_group_by_all_multi() -> None:
    dt1 = date(1999, 12, 31)
    dt2 = date(2028, 7, 5)

    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "xx", "yy", "xx", "xx"],
            "dt": [dt1, dt1, dt1, dt2, dt2, dt2],
            "value": [10.5, -5.5, 20.5, 8.0, -3.0, 5.0],
        }
    )
    expected = pl.DataFrame(
        {
            "dt": [dt1, dt1, dt2, dt2],
            "key": ["xx", "yy", "xx", "yy"],
            "sum_value": [31.0, -5.5, 2.0, 8.0],
            "ninety_nine": [99, 99, 99, 99],
        },
        schema_overrides={"ninety_nine": pl.Int16},
    )

    # the following groupings should all be equivalent
    for group in (
        "ALL",
        "1, 2",
        "dt, key",
    ):
        res = df.sql(
            f"""
            SELECT dt, key, sum_value, ninety_nine::int2 FROM
            (
                SELECT
                  dt,
                  key,
                  SUM(value) AS sum_value,
                  99 AS ninety_nine
                FROM self
                GROUP BY {group}
                ORDER BY dt, key
            ) AS grp
            """
        )
        assert_frame_equal(expected, res)


def test_group_by_ordinal_position() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx", "yy", "xx", "zz"],
            "b": [1, None, 3, 4, 5, 6],
            "c": [99, 99, 66, 66, 66, 66],
        }
    )
    expected = pl.LazyFrame(
        {
            "c": [66, 99],
            "total_b": [18, 1],
            "count_b": [4, 1],
            "count_star": [4, 2],
        }
    )

    with pl.SQLContext(frame=df) as ctx:
        res1 = ctx.execute(
            """
            SELECT
              c,
              SUM(b) AS total_b,
              COUNT(b) AS count_b,
              COUNT(*) AS count_star
            FROM frame
            GROUP BY 1
            ORDER BY c
            """
        )
        assert_frame_equal(res1, expected, check_dtypes=False)

        res2 = ctx.execute(
            """
            WITH "grp" AS (
              SELECT NULL::date as dt, c, SUM(b) AS total_b
              FROM frame
              GROUP BY 2, 1
            )
            SELECT c, total_b FROM grp ORDER BY c"""
        )
        assert_frame_equal(res2, expected.select(pl.nth(0, 1)))


def test_group_by_errors() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx"],
            "b": [10, 20, 30],
            "c": [99, 99, 66],
        }
    )

    with pytest.raises(
        SQLSyntaxError,
        match=r"negative ordinal values are invalid for GROUP BY; found -99",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY -99, a")

    with pytest.raises(
        SQLSyntaxError,
        match=r"GROUP BY requires a valid expression or positive ordinal; found '!!!'",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY a, '!!!'")

    with pytest.raises(
        SQLSyntaxError,
        match=r"'a' should participate in the GROUP BY clause or an aggregate function",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY b")

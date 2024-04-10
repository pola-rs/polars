from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_group_by(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(eager_execution=True)
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


def test_group_by_ordinal_position() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx", "yy", "xx", "zz"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [99, 99, 66, 66, 66, 66],
        }
    )
    expected = pl.LazyFrame({"c": [66, 99], "total_b": [18, 3]})

    with pl.SQLContext(frame=df) as ctx:
        res1 = ctx.execute(
            """
            SELECT c, SUM(b) AS total_b
            FROM frame
            GROUP BY 1
            ORDER BY c
            """
        )
        assert_frame_equal(res1, expected)

        res2 = ctx.execute(
            """
            WITH "grp" AS (
              SELECT NULL::date as dt, c, SUM(b) AS total_b
              FROM frame
              GROUP BY 2, 1
            )
            SELECT c, total_b FROM grp ORDER BY c"""
        )
        assert_frame_equal(res2, expected)

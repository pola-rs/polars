from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_any_all() -> None:
    df = pl.DataFrame(
        {
            "x": [-1, 0, 1, 2, 3, 4],
            "y": [1, 0, 0, 1, 2, 3],
        }
    )
    res = pl.SQLContext(df=df).execute(
        """
        SELECT
        x >= ALL(df.y) as 'All Geq',
        x > ALL(df.y) as 'All G',
        x < ALL(df.y) as 'All L',
        x <= ALL(df.y) as 'All Leq',
        x >= ANY(df.y) as 'Any Geq',
        x > ANY(df.y) as 'Any G',
        x < ANY(df.y) as 'Any L',
        x <= ANY(df.y) as 'Any Leq',
        x == ANY(df.y) as 'Any eq',
        x != ANY(df.y) as 'Any Neq',
        FROM df
        """,
        eager=True,
    )

    assert res.to_dict(as_series=False) == {
        "All Geq": [0, 0, 0, 0, 1, 1],
        "All G": [0, 0, 0, 0, 0, 1],
        "All L": [1, 0, 0, 0, 0, 0],
        "All Leq": [1, 1, 0, 0, 0, 0],
        "Any Geq": [0, 1, 1, 1, 1, 1],
        "Any G": [0, 0, 1, 1, 1, 1],
        "Any L": [1, 1, 1, 1, 0, 0],
        "Any Leq": [1, 1, 1, 1, 1, 0],
        "Any eq": [0, 1, 1, 1, 1, 0],
        "Any Neq": [1, 0, 0, 0, 0, 1],
    }


def test_distinct() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    ctx = pl.SQLContext(register_globals=True, eager_execution=True)
    res1 = ctx.execute("SELECT DISTINCT a FROM df ORDER BY a DESC")
    assert_frame_equal(
        left=df.select("a").unique().sort(by="a", descending=True),
        right=res1,
    )

    res2 = ctx.execute(
        """
        SELECT DISTINCT
          a * 2 AS two_a,
          b / 2 AS half_b
        FROM df
        ORDER BY two_a ASC, half_b DESC
        """,
    )
    assert res2.to_dict(as_series=False) == {
        "two_a": [2, 2, 4, 6],
        "half_b": [1, 0, 2, 3],
    }

    # test unregistration
    ctx.unregister("df")
    with pytest.raises(ComputeError, match=".*'df'.*not found"):
        ctx.execute("SELECT * FROM df")


def test_in_no_ops_11946() -> None:
    df = pl.LazyFrame(
        [
            {"i1": 1},
            {"i1": 2},
            {"i1": 3},
        ]
    )
    ctx = pl.SQLContext(frame_data=df, eager_execution=False)
    out = ctx.execute(
        "SELECT * FROM frame_data WHERE i1 in (1, 3)", eager=False
    ).collect()
    assert out.to_dict(as_series=False) == {"i1": [1, 3]}


def test_limit_offset() -> None:
    n_values = 11
    lf = pl.LazyFrame({"a": range(n_values), "b": reversed(range(n_values))})
    ctx = pl.SQLContext(tbl=lf)

    assert ctx.execute("SELECT * FROM tbl LIMIT 3 OFFSET 4", eager=True).rows() == [
        (4, 6),
        (5, 5),
        (6, 4),
    ]
    for offset, limit in [(0, 3), (1, n_values), (2, 3), (5, 3), (8, 5), (n_values, 1)]:
        out = ctx.execute(
            f"SELECT * FROM tbl LIMIT {limit} OFFSET {offset}", eager=True
        )
        assert_frame_equal(out, lf.slice(offset, limit).collect())
        assert len(out) == min(limit, n_values - offset)


def test_order_by(foods_ipc_path: Path) -> None:
    foods = pl.scan_ipc(foods_ipc_path)
    nums = pl.LazyFrame({"x": [1, 2, 3], "y": [4, 3, 2]})

    order_by_distinct_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT DISTINCT category
        FROM foods1
        ORDER BY category DESC
        """,
        eager=True,
    )
    assert order_by_distinct_res.to_dict(as_series=False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"]
    }

    order_by_group_by_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT category
        FROM foods1
        GROUP BY category
        ORDER BY category DESC
        """,
        eager=True,
    )
    assert order_by_group_by_res.to_dict(as_series=False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"]
    }

    order_by_constructed_group_by_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT category, SUM(calories) as summed_calories
        FROM foods1
        GROUP BY category
        ORDER BY summed_calories DESC
        """,
        eager=True,
    )
    assert order_by_constructed_group_by_res.to_dict(as_series=False) == {
        "category": ["seafood", "meat", "fruit", "vegetables"],
        "summed_calories": [1250, 540, 410, 192],
    }

    order_by_unselected_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT SUM(calories) as summed_calories
        FROM foods1
        GROUP BY category
        ORDER BY summed_calories DESC
        """,
        eager=True,
    )
    assert order_by_unselected_res.to_dict(as_series=False) == {
        "summed_calories": [1250, 540, 410, 192],
    }

    order_by_unselected_nums_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.x,
        df.y as y_alias
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_unselected_nums_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y_alias": [2, 3, 4],
    }

    order_by_wildcard_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        *,
        df.y as y_alias
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_wildcard_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y": [2, 3, 4],
        "y_alias": [2, 3, 4],
    }

    order_by_qualified_wildcard_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.*
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_qualified_wildcard_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y": [2, 3, 4],
    }

    order_by_exclude_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        * EXCLUDE y
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_exclude_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
    }

    order_by_qualified_exclude_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.* EXCLUDE y
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_qualified_exclude_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
    }

    order_by_expression_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        x % y as modded
        FROM df
        ORDER BY x % y
        """,
        eager=True,
    )
    assert order_by_expression_res.to_dict(as_series=False) == {
        "modded": [1, 1, 2],
    }


def test_register_context() -> None:
    # use as context manager unregisters tables created within each scope
    # on exit from that scope; arbitrary levels of nesting are supported.
    with pl.SQLContext() as ctx:
        _lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": ["m", "n", "o"]})
        _lf2 = pl.LazyFrame({"a": [2, 3, 4], "c": ["p", "q", "r"]})
        ctx.register_globals()
        assert ctx.tables() == ["_lf1", "_lf2"]

        with ctx:
            _lf3 = pl.LazyFrame({"a": [3, 4, 5], "b": ["s", "t", "u"]})
            _lf4 = pl.LazyFrame({"a": [4, 5, 6], "c": ["v", "w", "x"]})
            ctx.register_globals(n=2)
            assert ctx.tables() == ["_lf1", "_lf2", "_lf3", "_lf4"]

        assert ctx.tables() == ["_lf1", "_lf2"]

    assert ctx.tables() == []

import os
import warnings
from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


# TODO: Do not rely on I/O for these tests
@pytest.fixture()
def foods_ipc_path() -> str:
    return str(Path(os.path.dirname(__file__)) / "io" / "files" / "foods1.ipc")


def test_sql_distinct() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    c = pl.SQLContext(register_globals=True, eager_execution=True)
    res1 = c.execute("SELECT DISTINCT a FROM df ORDER BY a DESC")
    assert_frame_equal(
        left=df.select("a").unique().sort(by="a", descending=True),
        right=res1,
    )

    res2 = c.execute(
        """
        SELECT DISTINCT
          a*2 AS two_a,
          b/2 AS half_b
        FROM df
        ORDER BY two_a ASC, half_b DESC
        """,
    )
    assert res2.to_dict(False) == {
        "two_a": [2, 2, 4, 6],
        "half_b": [1, 0, 2, 3],
    }

    # test unregistration
    c.unregister("df")
    with pytest.raises(pl.ComputeError, match=".*'df'.*not found"):
        c.execute("SELECT * FROM df")


def test_sql_groupby(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext(eager_execution=True)
    c.register("foods", lf)

    out = c.execute(
        """
        SELECT
            category,
            count(category) as n,
            max(calories),
            min(fats_g)
        FROM foods
        GROUP BY category
        HAVING n > 5
        ORDER BY n, category DESC
        """
    )
    assert out.to_dict(False) == {
        "category": ["vegetables", "fruit", "seafood"],
        "n": [7, 7, 8],
        "calories": [45, 130, 200],
        "fats_g": [0.0, 0.0, 1.5],
    }

    lf = pl.LazyFrame(
        {
            "grp": ["a", "b", "c", "c", "b"],
            "att": ["x", "y", "x", "y", "y"],
        }
    )
    assert c.tables() == ["foods"]

    c.register("test", lf)
    assert c.tables() == ["foods", "test"]

    out = c.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_sql_limit_offset() -> None:
    n_values = 11
    lf = pl.LazyFrame({"a": range(n_values), "b": reversed(range(n_values))})
    c = pl.SQLContext(tbl=lf)

    assert c.execute("SELECT * FROM tbl LIMIT 3 OFFSET 4", eager=True).rows() == [
        (4, 6),
        (5, 5),
        (6, 4),
    ]
    for offset, limit in [(0, 3), (1, n_values), (2, 3), (5, 3), (8, 5), (n_values, 1)]:
        out = c.execute(f"SELECT * FROM tbl LIMIT {limit} OFFSET {offset}", eager=True)
        assert_frame_equal(out, lf.slice(offset, limit).collect())
        assert len(out) == min(limit, n_values - offset)


def test_sql_join_inner(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register_many(foods1=lf, foods2=lf)

    for join_clause in (
        "ON foods1.category = foods2.category",
        "USING (category)",
    ):
        out = c.execute(
            f"""
            SELECT *
            FROM foods1
            INNER JOIN foods2 {join_clause}
            LIMIT 2
            """
        )
        assert out.collect().to_dict(False) == {
            "category": ["vegetables", "vegetables"],
            "calories": [45, 20],
            "fats_g": [0.5, 0.0],
            "sugars_g": [2, 2],
            "calories_right": [45, 45],
            "fats_g_right": [0.5, 0.5],
            "sugars_g_right": [2, 2],
        }


def test_sql_join_left() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    c = pl.SQLContext(frames)
    out = c.execute(
        """
        SELECT a, b, c, d
        FROM tbl_a
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        ORDER BY c DESC
        """
    )
    assert out.collect().rows() == [
        (1, 4, "z", 25.5),
        (2, None, "y", -50.0),
        (3, 6, "x", None),
    ]
    assert c.tables() == ["tbl_a", "tbl_b", "tbl_c"]


def test_sql_is_between(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext(foods1=lf, eager_execution=True)
    out = c.execute(
        """
        SELECT *
        FROM foods1
        WHERE foods1.calories BETWEEN 22 AND 30
        ORDER BY "calories" DESC, "sugars_g" DESC
    """
    )
    assert out.rows() == [
        ("fruit", 30, 0.0, 5),
        ("vegetables", 30, 0.0, 5),
        ("fruit", 30, 0.0, 3),
        ("vegetables", 25, 0.0, 4),
        ("vegetables", 25, 0.0, 3),
        ("vegetables", 25, 0.0, 2),
        ("vegetables", 22, 0.0, 3),
    ]

    out = c.execute(
        """
        SELECT *
        FROM foods1
        WHERE calories NOT BETWEEN 22 AND 30
        ORDER BY "calories" ASC
        """
    )
    assert not any((22 <= cal <= 30) for cal in out["calories"])


def test_sql_trim(foods_ipc_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        out = pl.SQLContext(foods1=pl.scan_ipc(foods_ipc_path)).query(  # type: ignore[attr-defined]
            """
            SELECT DISTINCT TRIM(LEADING 'vmf' FROM category)
            FROM foods1
            ORDER BY category DESC
            """
        )
        assert out.to_dict(False) == {
            "category": ["seafood", "ruit", "egetables", "eat"]
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


def test_sql_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})
    sql_expr = pl.sql_expr("MIN(a)")
    expected = pl.DataFrame({"a": [1]})
    assert df.select(sql_expr).frame_equal(expected)

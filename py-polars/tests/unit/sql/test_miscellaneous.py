from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_any_all() -> None:
    df = pl.DataFrame(  # noqa: F841
        {
            "x": [-1, 0, 1, 2, 3, 4],
            "y": [1, 0, 0, 1, 2, 3],
        }
    )
    res = pl.sql(
        """
        SELECT
          x >= ALL(df.y) AS "All Geq",
          x  > ALL(df.y) AS "All G",
          x  < ALL(df.y) AS "All L",
          x <= ALL(df.y) AS "All Leq",
          x >= ANY(df.y) AS "Any Geq",
          x  > ANY(df.y) AS "Any G",
          x  < ANY(df.y) AS "Any L",
          x <= ANY(df.y) AS "Any Leq",
          x == ANY(df.y) AS "Any eq",
          x != ANY(df.y) AS "Any Neq",
        FROM df
        """,
    ).collect()

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
    ctx = pl.SQLContext(register_globals=True, eager=True)
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
    with pytest.raises(SQLInterfaceError, match="relation 'df' was not found"):
        ctx.execute("SELECT * FROM df")


def test_frame_sql_globals_error() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [2, 3, 4], "b": [7, 6, 5]})  # noqa: F841

    query = """
        SELECT df1.a, df2.b
        FROM df2 JOIN df1 ON df1.a = df2.a
        ORDER BY b DESC
    """
    with pytest.raises(SQLInterfaceError, match="relation.*not found.*"):
        df1.sql(query=query)

    res = pl.sql(query=query, eager=True)
    assert res.to_dict(as_series=False) == {"a": [2, 3], "b": [7, 6]}


def test_in_no_ops_11946() -> None:
    lf = pl.LazyFrame(
        [
            {"i1": 1},
            {"i1": 2},
            {"i1": 3},
        ]
    )
    out = lf.sql(
        query="SELECT * FROM frame_data WHERE i1 in (1, 3)",
        table_name="frame_data",
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


def test_sql_on_compatible_frame_types() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # create various different frame types
    dfp = df.to_pandas()
    dfa = df.to_arrow()
    dfb = dfa.to_batches()[0]  # noqa: F841

    # run polars sql query against all frame types
    for dfs in (  # noqa: B007
        (df["a"] * 2).rename("c"),  # polars series
        (dfp["a"] * 2).rename("c"),  # pandas series
    ):
        res = pl.sql("""
            SELECT a, b, SUM(c) AS cc FROM (
              SELECT * FROM df               -- polars frame
                UNION ALL SELECT * FROM dfp  -- pandas frame
                UNION ALL SELECT * FROM dfa  -- pyarrow table
                UNION ALL SELECT * FROM dfb  -- pyarrow record batch
            ) tbl
            INNER JOIN dfs ON dfs.c == tbl.b -- join on pandas/polars series
            GROUP BY "a", "b"
            ORDER BY "a", "b"
        """).collect()

        expected = pl.DataFrame({"a": [1, 3], "b": [4, 6], "cc": [16, 24]})
        assert_frame_equal(left=expected, right=res)

    # register and operate on non-polars frames
    for obj in (dfa, dfp):
        with pl.SQLContext(obj=obj) as ctx:
            res = ctx.execute("SELECT * FROM obj", eager=True)
            assert_frame_equal(df, res)

    # don't register all compatible objects
    with pytest.raises(SQLInterfaceError, match="relation 'dfp' was not found"):
        pl.SQLContext(
            register_globals=True,
            all_compatible=False,
        ).execute("SELECT * FROM dfp")

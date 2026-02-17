import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("cols", "join_type", "constraint"),
    [
        ("x", "INNER", ""),
        ("y", "INNER", ""),
        ("x", "LEFT", "WHERE y IN (0,1,2,3,4,5)"),
        ("y", "LEFT", "WHERE y >= 0"),
        ("df1.*", "FULL", "WHERE y >= 0"),
        ("df2.*", "FULL", "WHERE x >= 0"),
        ("* EXCLUDE y", "LEFT", "WHERE y >= 0"),
        ("* EXCLUDE x", "LEFT", "WHERE x >= 0"),
    ],
)
def test_from_subquery(cols: str, join_type: str, constraint: str) -> None:
    df1 = pl.DataFrame({"x": [-1, 0, 3, 1, 2, -1]})
    df2 = pl.DataFrame({"y": [0, 1, 2, 3]})

    sql = pl.SQLContext(df1=df1, df2=df2)
    res = sql.execute(
        query=f"""
          SELECT {cols} FROM (SELECT * FROM df1) AS df1
          {join_type} JOIN (SELECT * FROM df2) AS df2
          ON df1.x = df2.y {constraint}
        """,
        eager=True,
    )
    assert sorted(res.to_series()) == [0, 1, 2, 3]


@pytest.mark.may_fail_cloud  # reason: with_context
def test_in_subquery() -> None:
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 3, 4, 5, 6, 7],
        }
    )
    df_other = pl.DataFrame(
        {
            "w": [1, 2, 3, 4, 5, 6],
            "z": [2, 3, 4, 5, 6, 7],
        }
    )
    df_chars = pl.DataFrame(
        {
            "one": ["a", "b", "c", "d", "e", "f"],
            "two": ["b", "c", "d", "e", "f", "g"],
        }
    )

    ctx = pl.SQLContext(df=df, df_other=df_other, df_chars=df_chars)
    res_same = ctx.execute(
        query="""
          SELECT df.x as x
          FROM df
          WHERE x IN (SELECT y FROM df)
        """,
        eager=True,
    )
    df_expected_same = pl.DataFrame({"x": [2, 3, 4, 5, 6]})
    assert_frame_equal(
        left=df_expected_same,
        right=res_same,
    )

    res_double = ctx.execute(
        query="""
          SELECT df.x as x
          FROM df
          WHERE x IN (SELECT y FROM df)
            AND y IN (SELECT w FROM df_other)
        """,
        eager=True,
    )
    df_expected_double = pl.DataFrame({"x": [2, 3, 4, 5]})
    assert_frame_equal(
        left=df_expected_double,
        right=res_double,
    )

    res_expressions = ctx.execute(
        query="""
          SELECT
          df.x as x
          FROM df
          WHERE x+1 IN (SELECT y FROM df)
            AND y IN (SELECT w-1 FROM df_other)
        """,
        eager=True,
    )
    df_expected_expressions = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_frame_equal(
        left=df_expected_expressions,
        right=res_expressions,
    )

    res_not_in = ctx.execute(
        query="""
          SELECT
          df.x as x
          FROM df
          WHERE x NOT IN (SELECT y-5 FROM df)
            AND y NOT IN (SELECT w+5 FROM df_other)
        """,
        eager=True,
    )
    df_not_in = pl.DataFrame({"x": [3, 4]})
    assert_frame_equal(
        left=df_not_in,
        right=res_not_in,
    )

    res_chars = ctx.execute(
        query="""
          SELECT
          df_chars.one
          FROM df_chars
          WHERE one IN (SELECT two FROM df_chars)
        """,
        eager=True,
    )
    df_expected_chars = pl.DataFrame({"one": ["b", "c", "d", "e", "f"]})
    assert_frame_equal(
        left=res_chars,
        right=df_expected_chars,
    )

    with pytest.raises(
        expected_exception=SQLSyntaxError,
        match="SQL subquery returns more than one column",
    ):
        ctx.execute(
            query="""
              SELECT
              df_chars.one
              FROM df_chars
              WHERE one IN (SELECT one, two FROM df_chars)
            """
        ).collect()


def test_subquery_20732() -> None:
    lf = pl.concat(
        [
            pl.LazyFrame([{"id": 1, "s": "a"}]),
            pl.LazyFrame([{"id": 2, "s": "b"}]),
        ]
    )
    res = pl.sql("SELECT * FROM lf WHERE id IN (SELECT MAX(id) FROM lf)", eager=True)
    assert res.to_dict(as_series=False) == {"id": [2], "s": ["b"]}


def test_unsupported_subquery_comparisons() -> None:
    """Test that using = with a subquery gives a helpful error message."""
    df = pl.DataFrame({"value": [2000, 2000]})

    for op, suggestion in (("=", "IN"), ("!=", "NOT IN")):
        with pytest.raises(
            expected_exception=SQLSyntaxError,
            match=rf"subquery comparisons with '{op}' are not supported; use '{suggestion}' instead",
        ):
            pl.sql(f"SELECT * FROM df WHERE value {op} (SELECT MAX(e) FROM df)")

    for op in ("<", "<=", ">", ">="):
        with pytest.raises(
            expected_exception=SQLSyntaxError,
            match=rf"subquery comparisons with '{op}' are not supported",
        ):
            pl.sql(f"SELECT * FROM df WHERE (SELECT MAX(e) FROM df) {op} value")

        with pytest.raises(
            expected_exception=SQLSyntaxError,
            match=rf"subquery comparisons with '{op}' are not supported",
        ):
            pl.sql(f"SELECT * FROM df WHERE value {op} (SELECT MAX(value) FROM df)")


def test_derived_table_without_alias() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # basic unaliased subquery
    with pl.SQLContext(df=df) as ctx:
        res = ctx.execute("SELECT * FROM (SELECT a, b FROM df) ORDER BY a", eager=True)
        assert_frame_equal(res, df)

        # set operation without subquery aliases
        res = ctx.execute(
            """
            SELECT * FROM (
                SELECT a, b FROM df WHERE a <= 2
                UNION ALL
                SELECT a, b FROM df WHERE a > 2
            )
            ORDER BY a
            """
        ).collect()
        assert_frame_equal(res, df)

        # unqualified (but unambiguous) column refs from unaliased derived table
        res = ctx.execute("SELECT a FROM (SELECT a, b FROM df) ORDER BY a", eager=True)
        assert_frame_equal(res, df.select("a"))


def test_derived_table_alias_errors() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # joining on unaliased derived table should raise
    for join_type in ("INNER", "LEFT", "CROSS"):
        constraint = "" if join_type == "CROSS" else "ON df.a = a2"
        with pytest.raises(
            expected_exception=SQLInterfaceError,
            match="cannot JOIN on unnamed relation",
        ):
            pl.sql(
                query=f"""
                  SELECT * FROM df
                  {join_type} JOIN (SELECT a AS a2 FROM df) {constraint}
                """
            ).collect()

    # unaliased derived tables in a join
    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="cannot JOIN on unnamed relation",
    ):
        pl.sql(
            query="""
              SELECT *
              FROM (SELECT a FROM df)
              INNER JOIN (SELECT b FROM df) ON a = b
            """,
        ).collect()

    # qualified wildcard on nonexistent alias
    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="no table or struct column named 'sq' found",
    ):
        pl.sql(
            query="SELECT sq.* FROM (SELECT a, b FROM df)",
            eager=True,
        )

    # qualified column reference on nonexistent alias
    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="no table or struct column named 'sq' found",
    ):
        pl.sql(
            query="SELECT sq.a FROM (SELECT a, b FROM df)",
            eager=True,
        )

    # qualified reference in different clauses
    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="no table or struct column named 'sq' found",
    ):
        pl.sql(
            query="SELECT a FROM (SELECT a, b FROM df) WHERE sq.a > 1",
            eager=True,
        )

    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="no table or struct column named 'sq' found",
    ):
        pl.sql(
            query="SELECT a, COUNT(*) FROM (SELECT a, b FROM df) GROUP BY sq.a",
            eager=True,
        )

    with pytest.raises(
        expected_exception=SQLInterfaceError,
        match="no table or struct column named 'sq' found",
    ):
        pl.sql(
            query="SELECT a FROM (SELECT a, b FROM df) ORDER BY sq.a",
            eager=True,
        )

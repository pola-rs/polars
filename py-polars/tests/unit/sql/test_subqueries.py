import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
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
        f"""
        SELECT {cols} FROM (SELECT * FROM df1) AS df1
        {join_type} JOIN (SELECT * FROM df2) AS df2
        ON df1.x = df2.y {constraint}
        """,
        eager=True,
    )
    assert sorted(res.to_series()) == [0, 1, 2, 3]


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

    sql = pl.SQLContext(df=df, df_other=df_other, df_chars=df_chars)
    res_same = sql.execute(
        """
        SELECT
        df.x as x
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

    res_double = sql.execute(
        """
        SELECT
        df.x as x
        FROM df
        WHERE x IN (SELECT y FROM df)
        AND y IN(SELECT w FROM df_other)
        """,
        eager=True,
    )
    df_expected_double = pl.DataFrame({"x": [2, 3, 4, 5]})
    assert_frame_equal(
        left=df_expected_double,
        right=res_double,
    )

    res_expressions = sql.execute(
        """
        SELECT
        df.x as x
        FROM df
        WHERE x+1 IN (SELECT y FROM df)
        AND y IN(SELECT w-1 FROM df_other)
        """,
        eager=True,
    )
    df_expected_expressions = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert_frame_equal(
        left=df_expected_expressions,
        right=res_expressions,
    )

    res_not_in = sql.execute(
        """
        SELECT
        df.x as x
        FROM df
        WHERE x NOT IN (SELECT y-5 FROM df)
        AND y NOT IN(SELECT w+5 FROM df_other)
        """,
        eager=True,
    )
    df_not_in = pl.DataFrame({"x": [3, 4]})
    assert_frame_equal(
        left=df_not_in,
        right=res_not_in,
    )

    res_chars = sql.execute(
        """
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
        SQLSyntaxError,
        match="SQL subquery returns more than one column",
    ):
        sql.execute(
            """
            SELECT
            df_chars.one
            FROM df_chars
            WHERE one IN (SELECT one, two FROM df_chars)
            """,
            eager=True,
        )

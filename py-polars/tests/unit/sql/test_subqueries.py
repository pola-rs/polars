import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

def test_sql_in_subquery() -> None:
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
            "one": ['a', 'b', 'c', 'd', 'e', 'f'],
            "two": ['b', 'c', 'd', 'e', 'f', 'g'],
        }
    )

    sql = pl.SQLContext(register_globals=True)
    res_same = sql.execute(
        """
        SELECT
        df.x as x
        FROM df
        WHERE x IN (SELECT y FROM df)
        """
        , eager = True)
    df_expected_same = pl.DataFrame({"x": [2,3,4,5,6]})
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
        """
        , eager = True)
    df_expected_double = pl.DataFrame({"x": [2,3,4,5]})
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
        """
        , eager = True)
    df_expected_expressions= pl.DataFrame({"x": [1,2,3,4]})
    assert_frame_equal(
        left=df_expected_expressions,
        right=res_expressions,
    )

    res_chars = sql.execute(
        """
        SELECT
        df_chars.one
        FROM df_chars
        WHERE one IN (SELECT two FROM df_chars)
        """
        , eager = True)
    df_expected_chars= pl.DataFrame({"one": ['b', 'c', 'd', 'e', 'f']})
    assert_frame_equal(
        left=res_chars,
        right=df_expected_chars,
    )

    with pytest.raises(
        pl.InvalidOperationError, match="SQL subquery will return more than one column"
    ):
        res_returns_two = sql.execute(
            """
            SELECT
            df_chars.one
            FROM df_chars
            WHERE one IN (SELECT one, two FROM df_chars)
            """
            , eager = True)
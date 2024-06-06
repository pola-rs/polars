from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


def test_array_literals() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              a1, a2, ARRAY_AGG(a1) AS a3, ARRAY_AGG(a2) AS a4
            FROM (
              SELECT
                [10,20,30] AS a1,
                ['a','b','c'] AS a2,
              FROM df
            ) tbl
            """
        )
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "a1": [[10, 20, 30]],
                    "a2": [["a", "b", "c"]],
                    "a3": [[[10, 20, 30]]],
                    "a4": [[["a", "b", "c"]]],
                }
            ),
        )


def test_array_to_string() -> None:
    data = {"values": [["aa", "bb"], [None, "cc"], ["dd", None]]}
    res = pl.DataFrame(data).sql(
        """
        SELECT
          ARRAY_TO_STRING(values, '') AS v1,
          ARRAY_TO_STRING(values, ':') AS v2,
          ARRAY_TO_STRING(values, ':', 'NA') AS v3
        FROM self
        """
    )
    assert_frame_equal(
        res,
        pl.DataFrame(
            {
                "v1": ["aabb", "cc", "dd"],
                "v2": ["aa:bb", "cc", "dd"],
                "v3": ["aa:bb", "NA:cc", "dd:NA"],
            }
        ),
    )


@pytest.mark.parametrize(
    "array_keyword",
    ["ARRAY", ""],
)
def test_unnest_table_function(array_keyword: str) -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        res = ctx.execute(
            f"""
            SELECT * FROM
              UNNEST(
                {array_keyword}[1, 2, 3, 4],
                {array_keyword}['ww','xx','yy','zz'],
                {array_keyword}[23.0, 24.5, 28.0, 27.5]
              ) AS tbl (x,y,z);
            """
        )
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "x": [1, 2, 3, 4],
                    "y": ["ww", "xx", "yy", "zz"],
                    "z": [23.0, 24.5, 28.0, 27.5],
                }
            ),
        )


def test_unnest_table_function_errors() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        with pytest.raises(
            SQLSyntaxError,
            match=r'UNNEST table alias must also declare column names, eg: "frame data" \(a,b,c\)',
        ):
            ctx.execute('SELECT * FROM UNNEST([1, 2, 3]) AS "frame data"')

        with pytest.raises(
            SQLSyntaxError,
            match="UNNEST table alias requires 1 column name, found 2",
        ):
            ctx.execute("SELECT * FROM UNNEST([1, 2, 3]) AS tbl (a, b)")

        with pytest.raises(
            SQLSyntaxError,
            match="UNNEST table alias requires 2 column names, found 1",
        ):
            ctx.execute("SELECT * FROM UNNEST([1,2,3], [3,4,5]) AS tbl (a)")

        with pytest.raises(
            SQLSyntaxError,
            match=r"UNNEST table must have an alias",
        ):
            ctx.execute("SELECT * FROM UNNEST([1, 2, 3])")

        with pytest.raises(
            SQLInterfaceError,
            match=r"UNNEST tables do not \(yet\) support WITH OFFSET/ORDINALITY",
        ):
            ctx.execute("SELECT * FROM UNNEST([1, 2, 3]) tbl (colx) WITH OFFSET")

        with pytest.raises(
            SQLInterfaceError,
            match="nested array literals are not yet supported",
        ):
            pl.sql_expr("[[1,2,3]] AS nested")

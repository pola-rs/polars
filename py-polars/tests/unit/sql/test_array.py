from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("sort_order", "limit", "expected"),
    [
        (None, None, [("a", ["x", "y"]), ("b", ["z", "X", "Y"])]),
        ("ASC", None, [("a", ["x", "y"]), ("b", ["z", "Y", "X"])]),
        ("DESC", None, [("a", ["y", "x"]), ("b", ["X", "Y", "z"])]),
        ("ASC", 2, [("a", ["x", "y"]), ("b", ["z", "Y"])]),
        ("DESC", 2, [("a", ["y", "x"]), ("b", ["X", "Y"])]),
        ("ASC", 1, [("a", ["x"]), ("b", ["z"])]),
        ("DESC", 1, [("a", ["y"]), ("b", ["X"])]),
    ],
)
def test_array_agg(sort_order: str | None, limit: int | None, expected: Any) -> None:
    order_by = "" if not sort_order else f" ORDER BY col0 {sort_order}"
    limit_clause = "" if not limit else f" LIMIT {limit}"

    res = pl.sql(
        f"""
        WITH data (col0, col1, col2) as (
          VALUES
            (1,'a','x'),
            (2,'a','y'),
            (4,'b','z'),
            (8,'b','X'),
            (7,'b','Y')
        )
        SELECT col1, ARRAY_AGG(col2{order_by}{limit_clause}) AS arrs
        FROM data
        GROUP BY col1
        ORDER BY col1
        """
    ).collect()

    assert res.rows() == expected


def test_array_literals() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              a1, a2,
              -- test some array ops
              ARRAY_AGG(a1) AS a3,
              ARRAY_AGG(a2) AS a4,
              ARRAY_CONTAINS(a1,20) AS i20,
              ARRAY_CONTAINS(a2,'zz') AS izz,
              ARRAY_REVERSE(a1) AS ar1,
              ARRAY_REVERSE(a2) AS ar2
            FROM (
              SELECT
                -- declare array literals
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
                    "i20": [True],
                    "izz": [False],
                    "ar1": [[30, 20, 10]],
                    "ar2": [["c", "b", "a"]],
                }
            ),
        )


@pytest.mark.parametrize(
    ("array_index", "expected"),
    [
        (-4, None),
        (-3, 99),
        (-2, 66),
        (-1, 33),
        (0, None),
        (1, 99),
        (2, 66),
        (3, 33),
        (4, None),
    ],
)
def test_array_indexing(array_index: int, expected: int | None) -> None:
    res = pl.sql(
        f"""
        SELECT
          arr[{array_index}] AS idx1,
          ARRAY_GET(arr,{array_index}) AS idx2,
        FROM (SELECT [99,66,33] AS arr) tbl
        """
    ).collect()

    assert_frame_equal(
        res,
        pl.DataFrame(
            {"idx1": [expected], "idx2": [expected]},
        ),
        check_dtypes=False,
    )


def test_array_indexing_by_expr() -> None:
    df = pl.DataFrame(
        {
            "idx": [-2, -1, 0, None, 1, 2, 3],
            "arr": [[0, 1, 2, 3], [4, 5], [6], [7, 8, 9], [8, 7], [6, 5, 4], [3, 2, 1]],
        }
    )
    res = df.sql(
        """
        SELECT
          arr[idx] AS idx1,
          ARRAY_GET(arr, idx) AS idx2
        FROM self
        """
    )
    expected = [2, 5, None, None, 8, 5, 1]
    assert_frame_equal(res, pl.DataFrame({"idx1": expected, "idx2": expected}))


def test_array_to_string() -> None:
    data = {
        "s_values": [["aa", "bb"], [None, "cc"], ["dd", None]],
        "n_values": [[999, 777], [None, 555], [333, None]],
    }
    res = pl.DataFrame(data).sql(
        """
        SELECT
          ARRAY_TO_STRING(s_values, '') AS vs1,
          ARRAY_TO_STRING(s_values, ':') AS vs2,
          ARRAY_TO_STRING(s_values, ':', 'NA') AS vs3,
          ARRAY_TO_STRING(n_values, '') AS vn1,
          ARRAY_TO_STRING(n_values, ':') AS vn2,
          ARRAY_TO_STRING(n_values, ':', 'NA') AS vn3
        FROM self
        """
    )
    assert_frame_equal(
        res,
        pl.DataFrame(
            {
                "vs1": ["aabb", "cc", "dd"],
                "vs2": ["aa:bb", "cc", "dd"],
                "vs3": ["aa:bb", "NA:cc", "dd:NA"],
                "vn1": ["999777", "555", "333"],
                "vn2": ["999:777", "555", "333"],
                "vn3": ["999:777", "NA:555", "333:NA"],
            }
        ),
    )
    with pytest.raises(
        SQLSyntaxError,
        match=r"ARRAY_TO_STRING expects 2-3 arguments \(found 1\)",
    ):
        pl.sql_expr("ARRAY_TO_STRING(arr)")


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
            match="nested array literals are not currently supported",
        ):
            pl.sql_expr("[[1,2,3]] AS nested")

from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql.asserts import assert_sql_matches

# ---------------------------------------------------------------------------------
# NOTE: 'UNNEST' is available as both a table function and a select-level function
# ---------------------------------------------------------------------------------


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
                {array_keyword}[23.0, 24.5, 28.0, 27.5],
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
            match=r"UNNEST tables do not \(yet\) support WITH OFFSET|ORDINALITY",
        ):
            ctx.execute("SELECT * FROM UNNEST([1, 2, 3]) tbl (colx) WITH OFFSET")


def test_unnest_select_expressions() -> None:
    # Multiple expressions should be exploded inline/together
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "data1": ["a,b", "c,d,e,f"],
            "data2": ["x,x", "y,y,z,z"],
        }
    )
    query = """
        SELECT
            id,
            UNNEST(STRING_TO_ARRAY(data1,',')) AS d1,
            UNNEST(STRING_TO_ARRAY(data2,',')) AS d2
        FROM self
        ORDER BY ALL
    """
    expected = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 2, 2],
            "d1": ["a", "b", "c", "d", "e", "f"],
            "d2": ["x", "x", "y", "y", "z", "z"],
        }
    )
    res = df.sql(query)
    assert_frame_equal(res, expected)

    assert_sql_matches(
        df,
        query=query,
        compare_with="duckdb",
        expected=expected,
    )


def test_unnest_aggregates() -> None:
    df = pl.DataFrame({"a": [i // 100 for i in range(1, 1000)]})
    query = """
        SELECT
            UNNEST(ARRAY_AGG(DISTINCT a)) AS x,
            UNNEST(ARRAY_AGG(DISTINCT a ORDER BY a)) AS y,
            UNNEST(ARRAY_AGG(DISTINCT a ORDER BY a DESC)) AS z
        FROM self
    """
    res = df.sql(query)
    assert res.to_dict(as_series=False) == {
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "z": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    }


def test_unnest_select_height_filter_order_by() -> None:
    # Note: SQL UNNEST equates to `pl.Dataframe.explode()`
    # (ordering is applied after the explode/unnest)
    df = pl.DataFrame(
        {
            "list_long": [[1, 2, 3], [4, 5, 6]],
            "sort_key": [2, 1],
            "filter_mask": [False, True],
            "filter_mask_all_true": True,
        }
    )

    # Unnest/explode is applied at the dataframe level, sort is applied afterward
    assert_frame_equal(
        df.sql("SELECT UNNEST(list_long) as list FROM self ORDER BY sort_key"),
        pl.Series("list", [4, 5, 6, 1, 2, 3]).to_frame(),
    )

    # No NULLS: since order is applied after explode on the dataframe level
    assert_frame_equal(
        df.sql(
            "SELECT UNNEST(list_long) as list FROM self ORDER BY sort_key NULLS FIRST"
        ),
        pl.Series("list", [4, 5, 6, 1, 2, 3]).to_frame(),
    )

    # Literals are broadcasted to output height of UNNEST:
    assert_frame_equal(
        df.sql("SELECT UNNEST(list_long) as list, 1 as x FROM self ORDER BY sort_key"),
        pl.select(pl.Series("list", [4, 5, 6, 1, 2, 3]), x=1),
    )

    # Note: Filter applies before projections in SQL
    assert_frame_equal(
        df.sql(
            "SELECT UNNEST(list_long) as list FROM self WHERE filter_mask ORDER BY sort_key"
        ),
        pl.Series("list", [4, 5, 6]).to_frame(),
    )
    assert_frame_equal(
        df.sql(
            "SELECT UNNEST(list_long) as list FROM self WHERE filter_mask_all_true ORDER BY sort_key"
        ),
        pl.Series("list", [4, 5, 6, 1, 2, 3]).to_frame(),
    )

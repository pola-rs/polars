from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ColumnNotFoundError, SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.utils.pycapsule_utils import PyCapsuleStreamHolder

if TYPE_CHECKING:
    from polars.datatypes import DataType


@pytest.fixture
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


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        ({"x": [1, 2, 3, 4]}, None),
        ({"x": [9, 8, 7, 6]}, {"x": pl.Int8}),
        ({"x": ["aa", "bb"]}, {"x": pl.Struct}),
        ({"x": [None, None], "y": [None, None]}, {"x": pl.Date, "y": pl.Float64}),
    ],
)
def test_boolean_where_clauses(
    data: dict[str, Any], schema: dict[str, DataType] | None
) -> None:
    df = pl.DataFrame(data=data, schema=schema)
    empty_df = df.clear()

    for true in ("TRUE", "1=1", "2 == 2", "'xx' = 'xx'", "TRUE AND 1=1"):
        assert_frame_equal(df, df.sql(f"SELECT * FROM self WHERE {true}"))

    for false in ("false", "1!=1", "2 != 2", "'xx' != 'xx'", "FALSE OR 1!=1"):
        assert_frame_equal(empty_df, df.sql(f"SELECT * FROM self WHERE {false}"))


def test_count() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 1, 22, 22, 333],
            "c": [1, 1, None, None, 2],
        }
    )
    res = df.sql(
        """
        SELECT
          -- count
          COUNT(a) AS count_a,
          COUNT(b) AS count_b,
          COUNT(c) AS count_c,
          COUNT(*) AS count_star,
          COUNT(NULL) AS count_null,
          -- count distinct
          COUNT(DISTINCT a) AS count_unique_a,
          COUNT(DISTINCT b) AS count_unique_b,
          COUNT(DISTINCT c) AS count_unique_c,
          COUNT(DISTINCT NULL) AS count_unique_null,
        FROM self
        """,
    )
    assert res.to_dict(as_series=False) == {
        "count_a": [5],
        "count_b": [5],
        "count_c": [3],
        "count_star": [5],
        "count_null": [0],
        "count_unique_a": [5],
        "count_unique_b": [3],
        "count_unique_c": [2],
        "count_unique_null": [0],
    }

    df = pl.DataFrame({"x": [None, None, None]})
    res = df.sql(
        """
        SELECT
          COUNT(x) AS count_x,
          COUNT(*) AS count_star,
          COUNT(DISTINCT x) AS count_unique_x
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "count_x": [0],
        "count_star": [3],
        "count_unique_x": [0],
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
    dfo = PyCapsuleStreamHolder(df)  # noqa: F841

    # run polars sql query against all frame types
    for dfs in (  # noqa: B007
        (df["a"] * 2).rename("c"),  # polars series
        (dfp["a"] * 2).rename("c"),  # pandas series
    ):
        res = pl.sql(
            """
            SELECT a, b, SUM(c) AS cc FROM (
              SELECT * FROM df               -- polars frame
                UNION ALL SELECT * FROM dfp  -- pandas frame
                UNION ALL SELECT * FROM dfa  -- pyarrow table
                UNION ALL SELECT * FROM dfb  -- pyarrow record batch
                UNION ALL SELECT * FROM dfo  -- arbitrary pycapsule object
            ) tbl
            INNER JOIN dfs ON dfs.c == tbl.b -- join on pandas/polars series
            GROUP BY "a", "b"
            ORDER BY "a", "b"
            """
        ).collect()

        expected = pl.DataFrame({"a": [1, 3], "b": [4, 6], "cc": [20, 30]})
        assert_frame_equal(left=expected, right=res)

    # register and operate on non-polars frames
    for obj in (dfa, dfp):
        with pl.SQLContext(obj=obj) as ctx:
            res = ctx.execute("SELECT * FROM obj", eager=True)
            assert_frame_equal(df, res)

    # don't register all compatible objects
    with pytest.raises(SQLInterfaceError, match="relation 'dfp' was not found"):
        pl.SQLContext(register_globals=True).execute("SELECT * FROM dfp")


def test_nested_cte_column_aliasing() -> None:
    # trace through nested CTEs with multiple levels of column & table aliasing
    df = pl.sql(
        """
        WITH
          x AS (SELECT w.* FROM (VALUES(1,2), (3,4)) AS w(a, b)),
          y (m, n) AS (
            WITH z(c, d) AS (SELECT a, b FROM x)
              SELECT d*2 AS d2, c*3 AS c3 FROM z
        )
        SELECT n, m FROM y
        """,
        eager=True,
    )
    assert df.to_dict(as_series=False) == {
        "n": [3, 9],
        "m": [4, 8],
    }


def test_invalid_derived_table_column_aliases() -> None:
    values_query = "SELECT * FROM (VALUES (1,2), (3,4))"

    with pytest.raises(
        SQLSyntaxError,
        match=r"columns \(5\) in alias 'tbl' does not match .* the table/query \(2\)",
    ):
        pl.sql(f"{values_query} AS tbl(a, b, c, d, e)")

    assert pl.sql(f"{values_query} tbl", eager=True).rows() == [(1, 2), (3, 4)]


def test_values_clause_table_registration() -> None:
    with pl.SQLContext(frames=None, eager=True) as ctx:
        # initially no tables are registered
        assert ctx.tables() == []

        # confirm that VALUES clause derived table is registered, post-query
        res1 = ctx.execute("SELECT * FROM (VALUES (-1,1)) AS tbl(x, y)")
        assert ctx.tables() == ["tbl"]

        # and confirm that we can select from it by the registered name
        res2 = ctx.execute("SELECT x, y FROM tbl")
        for res in (res1, res2):
            assert res.to_dict(as_series=False) == {"x": [-1], "y": [1]}


def test_read_csv(tmp_path: Path) -> None:
    # check empty string vs null, parsing of dates, etc
    df = pl.DataFrame(
        {
            "label": ["lorem", None, "", "ipsum"],
            "num": [-1, None, 0, 1],
            "dt": [
                date(1969, 7, 5),
                date(1999, 12, 31),
                date(2077, 10, 10),
                None,
            ],
        }
    )
    csv_target = tmp_path / "test_sql_read.csv"
    df.write_csv(csv_target)

    res = pl.sql(f"SELECT * FROM read_csv('{csv_target}')").collect()
    assert_frame_equal(df, res)

    with pytest.raises(
        SQLSyntaxError,
        match="`read_csv` expects a single file path; found 3 arguments",
    ):
        pl.sql("SELECT * FROM read_csv('a','b','c')")


def test_global_variable_inference_17398() -> None:
    users = pl.DataFrame({"id": "1"})

    res = pl.sql(
        query="""
          WITH user_by_email AS (SELECT id FROM users)
          SELECT * FROM user_by_email
        """,
        eager=True,
    )
    assert_frame_equal(res, users)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT invalid_column FROM self",
        "SELECT key, invalid_column FROM self",
        "SELECT invalid_column * 2 FROM self",
        "SELECT * FROM self ORDER BY invalid_column",
        "SELECT * FROM self WHERE invalid_column = 200",
        "SELECT * FROM self WHERE invalid_column = '200'",
        "SELECT key, SUM(n) AS sum_n FROM self GROUP BY invalid_column",
    ],
)
def test_invalid_cols(query: str) -> None:
    df = pl.DataFrame(
        {
            "key": ["xx", "xx", "yy"],
            "n": ["100", "200", "300"],
        }
    )
    with pytest.raises(ColumnNotFoundError, match="invalid_column"):
        df.sql(query)


@pytest.mark.parametrize("filter_expr", ["", "WHERE 1 = 1", "WHERE a == 1 OR a != 1"])
@pytest.mark.parametrize("order_expr", ["", "ORDER BY 1", "ORDER BY a"])
def test_select_output_heights_20058_21084(filter_expr: str, order_expr: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    # Queries that maintain original height

    assert_frame_equal(
        df.sql(f"SELECT 1 as a FROM self {filter_expr} {order_expr}").cast(pl.Int64),
        pl.select(a=pl.Series([1, 1, 1])),
    )

    assert_frame_equal(
        df.sql(f"SELECT 1 + 1 as a, 1 as b FROM self {filter_expr} {order_expr}").cast(
            pl.Int64
        ),
        pl.DataFrame({"a": [2, 2, 2], "b": [1, 1, 1]}),
    )

    # Queries that aggregate to unit height

    assert_frame_equal(
        df.sql(f"SELECT COUNT(*) as a FROM self {filter_expr} {order_expr}").cast(
            pl.Int64
        ),
        pl.DataFrame({"a": 3}),
    )

    assert_frame_equal(
        df.sql(
            f"SELECT COUNT(*) as a, 1 as b FROM self {filter_expr} {order_expr}"
        ).cast(pl.Int64),
        pl.DataFrame({"a": 3, "b": 1}),
    )

    assert_frame_equal(
        df.sql(
            f"SELECT FIRST(a) as a, 1 as b FROM self {filter_expr} {order_expr}"
        ).cast(pl.Int64),
        pl.DataFrame({"a": 1, "b": 1}),
    )

    assert_frame_equal(
        df.sql(f"SELECT SUM(a) as a, 1 as b FROM self {filter_expr} {order_expr}").cast(
            pl.Int64
        ),
        pl.DataFrame({"a": 6, "b": 1}),
    )

    assert_frame_equal(
        df.sql(
            f"SELECT FIRST(1) as a, 1 as b FROM self {filter_expr} {order_expr}"
        ).cast(pl.Int64),
        pl.DataFrame({"a": 1, "b": 1}),
    )

    assert_frame_equal(
        df.sql(
            f"SELECT FIRST(1) + 1 as a, 1 as b FROM self {filter_expr} {order_expr}"
        ).cast(pl.Int64),
        pl.DataFrame({"a": 2, "b": 1}),
    )

    assert_frame_equal(
        df.sql(
            f"SELECT FIRST(1 + 1) as a, 1 as b FROM self {filter_expr} {order_expr}"
        ).cast(pl.Int64),
        pl.DataFrame({"a": 2, "b": 1}),
    )


def test_select_explode_height_filter_order_by() -> None:
    # Note: `unnest()` from SQL equates to `Expr.explode()`
    df = pl.DataFrame(
        {
            "list_long": [[1, 2, 3], [4, 5, 6]],
            "sort_key": [2, 1],
            "filter_mask": [False, True],
            "filter_mask_all_true": True,
        }
    )

    # Height of unnest is larger than height of sort_key, the sort_key is
    # extended with NULLs.

    assert_frame_equal(
        df.sql("SELECT UNNEST(list_long) as list FROM self ORDER BY sort_key"),
        pl.Series("list", [2, 1, 3, 4, 5, 6]).to_frame(),
    )

    assert_frame_equal(
        df.sql(
            "SELECT UNNEST(list_long) as list FROM self ORDER BY sort_key NULLS FIRST"
        ),
        pl.Series("list", [3, 4, 5, 6, 2, 1]).to_frame(),
    )

    # Literals are broadcasted to output height of UNNEST:
    assert_frame_equal(
        df.sql("SELECT UNNEST(list_long) as list, 1 as x FROM self ORDER BY sort_key"),
        pl.select(pl.Series("list", [2, 1, 3, 4, 5, 6]), x=1),
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
        pl.Series("list", [2, 1, 3, 4, 5, 6]).to_frame(),
    )


@pytest.mark.parametrize(
    ("query", "result"),
    [
        (
            """SELECT a, COUNT(*) OVER (PARTITION BY a) AS b FROM self""",
            [3, 3, 3, 1, 3, 3, 3],
        ),
        (
            """SELECT a, COUNT() OVER (PARTITION BY a) AS b FROM self""",
            [3, 3, 3, 1, 3, 3, 3],
        ),
        (
            """SELECT a, COUNT(i) OVER (PARTITION BY a) AS b FROM self""",
            [3, 3, 3, 1, 1, 1, 1],
        ),
        (
            """SELECT a, COUNT(DISTINCT i) OVER (PARTITION BY a) AS b FROM self""",
            [2, 2, 2, 1, 1, 1, 1],
        ),
    ],
)
def test_count_partition_22665(query: str, result: list[Any]) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 3, 3, 3],
            "i": [0, 0, 1, 2, 3, None, None],
        }
    )
    out = df.sql(query).select("b")
    expected = pl.DataFrame({"b": result}).cast({"b": pl.UInt32})
    assert_frame_equal(out, expected)

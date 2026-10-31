from __future__ import annotations

import io
import re
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def test_frame() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "x": [1, 2, 3],
            "y": ["aaa", "bbb", "ccc"],
            "z": [date(2000, 12, 31), date(1978, 11, 15), date(2077, 10, 20)],
        },
        schema_overrides={"x": pl.UInt8},
    )


def test_create_table() -> None:
    with pl.SQLContext() as ctx:
        # test all three ways of creating a new table
        ctx.execute("CREATE TABLE tbl1(colx VARCHAR, coly DATE, colz ARRAY<DOUBLE>)")
        ctx.execute("CREATE TABLE tbl2 AS SELECT * FROM tbl1")
        ctx.execute("CREATE TABLE tbl3 LIKE tbl2")
        df = ctx.execute("SELECT * FROM tbl3", eager=True)

    df_expected = pl.DataFrame(
        schema={
            "colx": pl.String,
            "coly": pl.Date,
            "colz": pl.List(pl.Float64),
        }
    )
    assert_frame_equal(df_expected, df)


def test_create_table_from_file_io(io_files_path: Path) -> None:
    foods_csv = io_files_path / "foods*.csv"
    with pl.SQLContext() as ctx:
        ctx.execute(
            query=f"""
                CREATE TABLE foods AS
                SELECT * FROM READ_CSV('{foods_csv}')
            """,
            eager=True,
        )
        df = ctx.execute("SELECT * FROM foods", eager=True)
        assert df.schema == {
            "category": pl.String,
            "calories": pl.Int64,
            "fats_g": pl.Float64,
            "sugars_g": pl.Int64,
        }
        assert df.shape == (135, 4)


@pytest.mark.parametrize(
    ("delete_constraint", "expected_ids"),
    [
        # basic constraints
        ("WHERE id = 200", {100, 300}),
        ("WHERE id = 200 OR id = 300", {100}),
        ("WHERE id IN (200, 300, 400)", {100}),
        ("WHERE id NOT IN (200, 300, 400)", {200, 300}),
        # more involved constraints
        ("WHERE EXTRACT(year FROM dt) >= 2000", {200}),
        # null-handling (in the data)
        ("WHERE v1 < 0", {100, 300}),
        ("WHERE v1 > 0", {200, 300}),
        # null handling (in the constraint)
        ("WHERE v1 IS NULL", {100, 200}),
        ("WHERE v1 IS NOT NULL", {300}),
        # boolean handling (delete all/none)
        ("WHERE FALSE", {100, 200, 300}),
        ("WHERE TRUE", set()),
        # no constraint; equivalent to TRUNCATE (drop all rows)
        ("", set()),
    ],
)
def test_delete_clause(delete_constraint: str, expected_ids: set[int]) -> None:
    df = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "dt": [date(2020, 10, 10), date(1999, 1, 2), date(2001, 7, 5)],
            "v1": [3.5, -4.0, None],
            "v2": [10.0, 2.5, -1.5],
        }
    )
    res = df.sql(f"DELETE FROM self {delete_constraint}")
    assert set(res["id"]) == expected_ids


def test_drop_table(test_frame: pl.LazyFrame) -> None:
    # 'drop' completely removes the table from sql context
    expected = pl.DataFrame()

    with pl.SQLContext(frame=test_frame, eager=True) as ctx:
        res = ctx.execute("DROP TABLE frame")
        assert_frame_equal(res, expected)

        with pytest.raises(SQLInterfaceError, match="'frame' was not found"):
            ctx.execute("SELECT * FROM frame")


def test_explain_query(test_frame: pl.LazyFrame) -> None:
    # 'explain' returns the query plan for the given sql
    with pl.SQLContext(frame=test_frame) as ctx:
        plan = (
            ctx.execute("EXPLAIN SELECT * FROM frame")
            .select(pl.col("Logical Plan").str.join())
            .collect()
            .item()
        )
        assert (
            re.search(
                pattern=r"PROJECT.+?COLUMNS",
                string=plan,
                flags=re.IGNORECASE,
            )
            is not None
        )


def test_show_tables(test_frame: pl.LazyFrame) -> None:
    # 'show tables' lists all tables registered with the sql context in sorted order
    with pl.SQLContext(
        tbl3=test_frame,
        tbl2=test_frame,
        tbl1=test_frame,
    ) as ctx:
        res = ctx.execute("SHOW TABLES").collect()
        assert_frame_equal(res, pl.DataFrame({"name": ["tbl1", "tbl2", "tbl3"]}))


@pytest.mark.parametrize(
    "truncate_sql",
    [
        "TRUNCATE TABLE frame",
        "TRUNCATE frame",
    ],
)
def test_truncate_table(truncate_sql: str, test_frame: pl.LazyFrame) -> None:
    # 'truncate' preserves the table, but optimally drops all rows within it
    expected = pl.DataFrame(schema=test_frame.collect_schema())

    with pl.SQLContext(frame=test_frame, eager=True) as ctx:
        res = ctx.execute(truncate_sql)
        assert_frame_equal(res, expected)

        res = ctx.execute("SELECT * FROM frame")
        assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "query",
    [
        # unquoted identifiers are case-insensitive, so a relation need not be
        # referenced with the case it was declared with
        "SELECT s.x FROM (SELECT x FROM tbl) S",
        "SELECT S.x FROM (SELECT x FROM tbl) s",
        "SELECT SUB.x FROM (SELECT x FROM tbl) sub",
        "SELECT TBL.x FROM tbl",
        "SELECT t.x FROM tbl T",
        "WITH cte AS (SELECT x FROM tbl) SELECT c.x FROM CTE c",
    ],
)
def test_relation_name_case_insensitivity(query: str) -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3]})
    with pl.SQLContext(tbl=lf, eager=True) as ctx:
        assert ctx.execute(query).to_series().to_list() == [1, 2, 3]


def test_cte_shadows_registered_table() -> None:
    # a CTE takes precedence over a registered table of the same name, as does
    # a FROM alias naming that CTE
    registered = pl.LazyFrame({"x": [-1, -2]})
    source = pl.LazyFrame({"x": [1, 2, 3]})

    with pl.SQLContext(tbl=source, store=registered, eager=True) as ctx:
        assert ctx.execute(
            "WITH store AS (SELECT x FROM tbl) SELECT x FROM store"
        ).to_series().to_list() == [1, 2, 3]

        assert ctx.execute(
            "WITH c AS (SELECT x FROM tbl) SELECT store.x FROM c store"
        ).to_series().to_list() == [1, 2, 3]

        # the registered table is still reachable when nothing shadows it
        assert ctx.execute("SELECT x FROM store").to_series().to_list() == [-1, -2]


def test_cte_named_like_derived_table_from_earlier_query() -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3]})
    with pl.SQLContext(tbl=lf, eager=True) as ctx:
        ctx.execute("SELECT SUM(x) AS total FROM (SELECT x FROM tbl) agg")
        assert ctx.tables() == ["tbl"]

        res = ctx.execute("WITH agg AS (SELECT x FROM tbl) SELECT a.x FROM agg a")
        assert res.to_series().to_list() == [1, 2, 3]


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM range(2) AS lineitem",
        "SELECT * FROM UNNEST([1, 2]) AS lineitem(z)",
        "SELECT * FROM (SELECT 5 AS z) AS lineitem",
        "SELECT * FROM (SELECT 5 AS z) AS foo",
        "SELECT * FROM (SELECT 5 AS z) AS foo, lineitem",
    ],
)
def test_from_items_last_for_their_query(query: str) -> None:
    # #30070
    ctx = pl.SQLContext(lineitem=pl.LazyFrame({"k": [1]}))
    ctx.execute(query, eager=True)
    assert ctx.tables() == ["lineitem"]
    assert ctx.execute("SELECT * FROM lineitem", eager=True).columns == ["k"]


FROM_ITEMS_NAMED_LINEITEM = [
    "range(1, 3) AS lineitem(x)",
    "UNNEST([1, 2]) AS lineitem(x)",
    "(SELECT 1 AS x UNION ALL SELECT 2) AS lineitem",
]


@pytest.mark.parametrize("from_item", FROM_ITEMS_NAMED_LINEITEM)
@pytest.mark.parametrize(
    "query",
    [
        "SELECT lineitem.x FROM {from_item}",
        "SELECT x FROM {from_item} WHERE EXISTS (SELECT 1 FROM t WHERE t.k = lineitem.x)",
        "SELECT x FROM {from_item} WHERE x IN (SELECT x FROM lineitem)",
        "SELECT * FROM {from_item}, lineitem AS l2",
        "SELECT lineitem.x AS a, l2.x AS b FROM {from_item}, lineitem AS l2",
        "SELECT l2.x AS a, lineitem.x AS b FROM lineitem AS l2, {from_item}",
        "WITH lineitem AS (SELECT 7 AS x) SELECT * FROM {from_item}, lineitem AS l2",
    ],
)
def test_from_item_alias_is_not_a_table(from_item: str, query: str) -> None:
    assert_sql_matches(
        {"lineitem": pl.DataFrame({"x": [2]}), "t": pl.DataFrame({"k": [2]})},
        query=query.format(from_item=from_item),
        compare_with="duckdb",
        check_row_order=False,
        check_column_names=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        "WITH D AS (SELECT 7 AS c) SELECT d.a, D.c FROM (SELECT 1 AS a) AS d, D",
        "WITH D AS (SELECT 7 AS c) SELECT D.c, d.a FROM D, (SELECT 1 AS a) AS d",
        "SELECT t1.a, T1.b FROM (SELECT 1 AS a) AS t1, (SELECT 2 AS b) AS T1",
        "SELECT r.x, R.b FROM range(1, 2) AS r(x), (SELECT 2 AS b) AS R",
        "WITH U AS (SELECT 5 AS a) SELECT u.x, U.a FROM UNNEST([1]) AS u(x), U",
    ],
)
def test_from_item_names_differing_in_case(query: str) -> None:
    assert_sql_matches({}, query=query, compare_with="duckdb")


@pytest.mark.parametrize("from_item", FROM_ITEMS_NAMED_LINEITEM)
def test_from_item_alias_is_not_readable_as_a_table(from_item: str) -> None:
    ctx = pl.SQLContext()
    for query in [
        f"SELECT x FROM {from_item} WHERE x IN (SELECT x FROM lineitem)",
        f"SELECT * FROM {from_item} JOIN lineitem AS l2 ON lineitem.x = l2.x",
    ]:
        with pytest.raises(
            SQLInterfaceError, match="relation 'lineitem' was not found"
        ):
            ctx.execute(query, eager=True)


def _execute(ctx: pl.SQLContext[Any], query: str, *, eager_path: bool) -> pl.DataFrame:
    # `CREATE TABLE ... AS` resolves the query eagerly; a plain query is deferred
    if eager_path:
        ctx.execute(f"CREATE TABLE res AS {query}")
        query = "SELECT * FROM res"
    return ctx.execute(query, eager=True)


@pytest.mark.parametrize("eager_path", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT k FROM tpch.lineitem",
        'SELECT k FROM "tpch.lineitem"',
        "SELECT k FROM TPCH.LineItem",
        "SELECT lineitem.k FROM tpch.lineitem",
        "SELECT tpch.lineitem.k FROM tpch.lineitem",
        "SELECT li.k FROM tpch.lineitem AS li",
        "SELECT lineitem.* FROM tpch.lineitem",
        "SELECT tpch.lineitem.* FROM tpch.lineitem",
        "SELECT k FROM (SELECT k FROM tpch.lineitem) sub",
        "WITH c AS (SELECT k FROM tpch.lineitem) SELECT k FROM c",
        "SELECT k FROM tpch.lineitem WHERE k < 0 UNION ALL TABLE tpch.lineitem",
    ],
)
def test_qualified_table_name(query: str, eager_path: bool) -> None:
    # a dotted relation name resolves as a whole, and its trailing parts qualify columns
    lf = pl.LazyFrame({"k": [1, 2]})
    with pl.SQLContext({"tpch.lineitem": lf}) as ctx:
        res = _execute(ctx, query, eager_path=eager_path)
        assert sorted(res.to_series().to_list()) == [1, 2]


@pytest.mark.parametrize("eager_path", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT k FROM onelake.tpch.lineitem",
        "SELECT lineitem.k FROM onelake.tpch.lineitem",
        "SELECT tpch.lineitem.k FROM onelake.tpch.lineitem",
        "SELECT onelake.tpch.lineitem.k FROM onelake.tpch.lineitem",
        "SELECT onelake.tpch.lineitem.* FROM onelake.tpch.lineitem",
    ],
)
def test_three_part_table_name(query: str, eager_path: bool) -> None:
    lf = pl.LazyFrame({"k": [1, 2]})
    with pl.SQLContext({"onelake.tpch.lineitem": lf}) as ctx:
        res = _execute(ctx, query, eager_path=eager_path)
        assert res.to_series().to_list() == [1, 2]


def test_qualified_table_name_serialized_plan() -> None:
    lf = pl.LazyFrame({"k": [1, 2]})
    with pl.SQLContext({"tpch.lineitem": lf}) as ctx:
        res = ctx.execute("SELECT lineitem.k FROM tpch.lineitem WHERE k > 1")
    res = pl.LazyFrame.deserialize(io.BytesIO(res.serialize()))
    assert res.collect().to_series().to_list() == [2]


def test_relation_takes_precedence_over_struct_path() -> None:
    # `a.b.k` reads relation `a.b` when it is in FROM, else field `k` of struct `a.b`
    frames = {
        "a": pl.LazyFrame({"b": [{"k": 99}]}),
        "a.b": pl.LazyFrame({"k": [1]}),
    }
    with pl.SQLContext(frames) as ctx:
        assert ctx.execute("SELECT a.b.k FROM a, a.b", eager=True).item() == 1
        assert ctx.execute("SELECT a.b.k FROM a", eager=True).item() == 99


@pytest.mark.parametrize("eager_path", [False, True])
def test_qualified_table_name_not_resolved_by_first_part(eager_path: bool) -> None:
    frames = {
        "tpch": pl.LazyFrame({"v": ["tpch"]}),
        "tpch.lineitem": pl.LazyFrame({"v": ["tpch.lineitem"]}),
    }
    with pl.SQLContext(frames) as ctx:
        res = _execute(ctx, "SELECT v FROM tpch.lineitem", eager_path=eager_path)
        assert res.item() == "tpch.lineitem"


def test_qualified_table_name_hidden_by_alias() -> None:
    lf = pl.LazyFrame({"k": [1, 2]})
    with (
        pl.SQLContext({"tpch.lineitem": lf}) as ctx,
        pytest.raises(
            SQLInterfaceError, match=r'table "lineitem" is hidden by alias "li"'
        ),
    ):
        ctx.execute("SELECT lineitem.k FROM tpch.lineitem AS li", eager=True)


def test_create_table_if_not_exists() -> None:
    # an existing table (matched ignoring case, as in FROM) is left as it is
    with pl.SQLContext() as ctx:
        ctx.execute("CREATE TABLE IF NOT EXISTS frame AS SELECT 1 AS x")
        ctx.execute("CREATE TABLE IF NOT EXISTS frame AS SELECT 2 AS x")
        ctx.execute("CREATE TABLE IF NOT EXISTS FRAME AS SELECT 3 AS x")
        assert ctx.tables() == ["frame"]
        assert ctx.execute("SELECT x FROM frame", eager=True).item() == 1


@pytest.mark.parametrize("temp", ["TEMP", "TEMPORARY"])
def test_create_temp_table(temp: str) -> None:
    with pl.SQLContext() as ctx:
        ctx.execute(f"CREATE {temp} TABLE frame AS SELECT 1 AS x")
        assert ctx.execute("SELECT x FROM frame", eager=True).item() == 1

        # an existing table (matched ignoring case) is an error, unlike CREATE TABLE
        for sql in (
            f"CREATE {temp} TABLE frame AS SELECT 2 AS x",
            f"CREATE {temp} TABLE FRAME (x INT)",
        ):
            with pytest.raises(SQLInterfaceError, match="table 'frame' already exists"):
                ctx.execute(sql)
        # the query of a no-op does not run
        ctx.execute(f"CREATE {temp} TABLE IF NOT EXISTS Frame AS SELECT * FROM nosuch")
        assert ctx.execute("SELECT x FROM frame", eager=True).item() == 1

        ctx.execute(f"CREATE OR REPLACE {temp} TABLE FRAME AS SELECT 4 AS x")
        assert ctx.tables() == ["frame"]
        assert ctx.execute("SELECT x FROM frame", eager=True).item() == 4

        ctx.execute(f"CREATE {temp} TABLE other LIKE frame")
        assert ctx.execute("SELECT * FROM other", eager=True).columns == ["x"]
        ctx.execute(f"CREATE OR REPLACE {temp} TABLE new AS SELECT 5 AS x")
        assert sorted(ctx.tables()) == ["frame", "new", "other"]


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        (
            "CREATE OR REPLACE TABLE t AS SELECT 1 AS x",
            "`CREATE OR REPLACE TABLE` is not supported",
        ),
        (
            "CREATE OR REPLACE TEMP TABLE IF NOT EXISTS t AS SELECT 1 AS x",
            "cannot be combined",
        ),
        (
            "CREATE GLOBAL TEMPORARY TABLE t (x INT)",
            "`CREATE GLOBAL/LOCAL TABLE` is not supported",
        ),
    ],
)
def test_create_temp_table_errors(sql: str, error: str) -> None:
    with pytest.raises(SQLInterfaceError, match=error):
        pl.SQLContext().execute(sql)


def test_qualified_table_name_delete() -> None:
    # the WHERE of a DELETE qualifies by a trailing part, as in SELECT
    lf = pl.LazyFrame({"k": [1, 2]})
    for where in ["t.k = 1", "s.t.k = 1", "k = 1"]:
        with pl.SQLContext({"s.t": lf}) as ctx:
            res = ctx.execute(f"DELETE FROM s.t WHERE {where}", eager=True)
            assert res.to_series().to_list() == [2]


def test_dotted_name_shared_with_quoted_alias() -> None:
    # `s.t` and an alias `"s.t"` share a dotted name; a reference to either is ambiguous
    frames = {"s.t": pl.LazyFrame({"k": [1]}), "u": pl.LazyFrame({"k": [2]})}
    from_clause = 'FROM s.t CROSS JOIN u AS "s.t"'
    with pl.SQLContext(frames) as ctx:
        assert ctx.execute(f"SELECT * {from_clause}", eager=True).rows() == [(1, 2)]
        for query in [
            f"SELECT s.t.k {from_clause}",
            f'SELECT "s.t".k {from_clause}',
            'SELECT * FROM s.t JOIN u AS "s.t" ON s.t.k < "s.t".k',
        ]:
            with pytest.raises(SQLInterfaceError, match="same dotted name"):
                ctx.execute(query, eager=True)


def test_quoted_dotted_name_is_one_part() -> None:
    # `"tpch.lineitem"` finds the registered table, but as a qualifier it is one part
    frames = {
        "tpch.lineitem": pl.LazyFrame({"k": [1]}),
        "t": pl.LazyFrame({"k": [2]}),
        "u": pl.LazyFrame({"k": [3]}),
    }
    with pl.SQLContext(frames) as ctx:
        for query, expected in [
            ('SELECT "tpch.lineitem".k FROM "tpch.lineitem"', 1),
            ("SELECT lineitem.k FROM tpch.lineitem", 1),
            ('SELECT "a.b".k FROM t AS "a.b"', 2),
            ('SELECT x.k FROM t AS x CROSS JOIN u AS "a.x"', 2),
        ]:
            assert ctx.execute(query, eager=True).item() == expected

        for query, root in [
            ('SELECT lineitem.k FROM "tpch.lineitem"', "lineitem"),
            ('SELECT b.k FROM t AS "a.b"', "b"),
        ]:
            with pytest.raises(
                SQLInterfaceError, match=rf"no table or struct column named '{root}'"
            ):
                ctx.execute(query, eager=True)


def test_qualified_table_name_ddl() -> None:
    with pl.SQLContext() as ctx:
        ctx.execute("CREATE TABLE tpch.x AS SELECT 1 AS a")
        assert ctx.tables() == ["tpch.x"]
        assert ctx.execute("SELECT a FROM tpch.x", eager=True).item() == 1

        ctx.execute("TRUNCATE TABLE tpch.x")
        assert ctx.execute("SELECT a FROM tpch.x", eager=True).height == 0

        ctx.execute("DROP TABLE tpch.x")
        assert ctx.tables() == []


@pytest.mark.parametrize(
    ("name", "ddl"),
    [
        ("My Table", 'DROP TABLE "My Table"'),
        ("My Table", 'TRUNCATE TABLE "My Table"'),
        ("frame", "DROP TABLE FRAME"),
    ],
)
def test_ddl_table_name_quoting_and_case(name: str, ddl: str) -> None:
    # quotes are not part of the name, and lookups ignore case as in FROM
    with pl.SQLContext({name: pl.LazyFrame({"x": [1]})}) as ctx:
        ctx.execute(ddl)
        if ddl.startswith("DROP"):
            assert ctx.tables() == []
        else:
            assert ctx.execute(f'SELECT * FROM "{name}"', eager=True).height == 0


def test_relation_name_case_insensitive_ambiguity() -> None:
    # a name matching several tables only when ignoring case is an error
    frames = {
        "Foo": pl.LazyFrame({"x": ["Foo"]}),
        "FOO": pl.LazyFrame({"x": ["FOO"]}),
    }
    with pl.SQLContext(frames) as ctx:
        assert ctx.execute("SELECT x FROM Foo", eager=True).item() == "Foo"
        assert ctx.execute("SELECT x FROM FOO", eager=True).item() == "FOO"
        # without a FROM, only the eager path sees the registered tables
        for query, eager_path in (("SELECT x FROM foo", False), ("SELECT foo.x", True)):
            with pytest.raises(SQLInterfaceError, match=r"foo.*matches more than one"):
                _execute(ctx, query, eager_path=eager_path)


def test_create_table_matching_existing_name_ignoring_case() -> None:
    # CREATE TABLE finds an existing table as FROM does, ignoring case
    with pl.SQLContext({"frame": pl.LazyFrame({"x": [1]})}) as ctx:
        ctx.execute("CREATE TABLE FRAME AS SELECT 2 AS x")
        assert ctx.tables() == ["frame"]
        assert ctx.execute("SELECT x FROM Frame", eager=True).item() == 2

        ctx.execute("CREATE TABLE other LIKE FRAME")
        assert ctx.execute("SELECT * FROM other", eager=True).columns == ["x"]


@pytest.mark.parametrize(
    ("ddl", "error"),
    [
        ("TRUNCATE TABLE missing", "table 'missing' does not exist"),
        ("TRUNCATE TABLE IF EXISTS missing", None),
    ],
)
def test_truncate_missing_table(ddl: str, error: str | None) -> None:
    with pl.SQLContext({"frame": pl.LazyFrame({"x": [1]})}) as ctx:
        if error is None:
            ctx.execute(ddl)
        else:
            with pytest.raises(SQLInterfaceError, match=error):
                ctx.execute(ddl)
        assert ctx.execute("SELECT x FROM frame", eager=True).item() == 1


def test_qualifier_resolves_in_innermost_query_first() -> None:
    # the inner alias `tpch` (struct column `orders`) wins over the outer `tpch.orders`
    frames = {
        "tpch.orders": pl.LazyFrame({"o_orderkey": [1]}),
        "lineitem": pl.LazyFrame({"orders": [{"o_orderkey": 5}]}),
    }
    query = """
        SELECT o_orderkey FROM tpch.orders
        WHERE EXISTS (
          SELECT 1 FROM lineitem AS tpch WHERE tpch.orders.o_orderkey > {n}
        )
    """
    with pl.SQLContext(frames) as ctx:
        assert ctx.execute(query.format(n=2), eager=True).height == 1
        assert ctx.execute(query.format(n=9), eager=True).height == 0


def test_correlated_qualifier_naming_two_inner_relations() -> None:
    # an inner qualifier naming two relations is ambiguous
    frames = {
        "o": pl.LazyFrame({"k": [1, 2]}),
        "tpch.lineitem": pl.LazyFrame({"k": [1]}),
        "tpch_sf100.lineitem": pl.LazyFrame({"k": [2]}),
    }
    query = """
        SELECT * FROM o WHERE EXISTS (
          SELECT 1 FROM tpch.lineitem, tpch_sf100.lineitem WHERE lineitem.k = o.k
        )
    """
    with (
        pl.SQLContext(frames) as ctx,
        pytest.raises(
            SQLInterfaceError, match=r'ambiguous reference to table "lineitem"'
        ),
    ):
        ctx.execute(query, eager=True)


@pytest.mark.parametrize("eager_path", [False, True])
def test_from_table_name_is_not_an_earlier_alias(eager_path: bool) -> None:
    # a table name in FROM is a CTE or registered table, never another FROM item's alias
    frames = {
        "u": pl.LazyFrame({"k": [2]}),
        "t": pl.LazyFrame({"k": [1]}),
        "s.t": pl.LazyFrame({"k": [1]}),
    }
    with pl.SQLContext(frames) as ctx:
        for query in (
            "SELECT actual.k FROM u AS t CROSS JOIN t AS actual",
            'SELECT actual.k FROM u AS "s.t" CROSS JOIN s.t AS actual',
        ):
            res = _execute(ctx, query, eager_path=eager_path)
            assert res.to_series().to_list() == [1]


def test_ambiguous_column_hint_quotes_dotted_alias() -> None:
    frames = {"u": pl.LazyFrame({"k": [1]}), "v": pl.LazyFrame({"k": [2]})}
    with (
        pl.SQLContext(frames) as ctx,
        pytest.raises(SQLInterfaceError, match=r'use one of: "a\.b"\.k, z\.k'),
    ):
        ctx.execute('SELECT k FROM u AS "a.b" CROSS JOIN v AS z', eager=True)

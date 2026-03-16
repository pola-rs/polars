from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


def test_except_intersect() -> None:
    df1 = pl.DataFrame({"x": [1, 9, 1, 1], "y": [2, 3, 4, 4], "z": [5, 5, 5, 5]})
    df2 = pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4], "z": [7, 6, 5]})

    res_e = pl.sql("SELECT x, y, z FROM df1 EXCEPT SELECT * FROM df2", eager=True)
    res_i = pl.sql("SELECT * FROM df1 INTERSECT SELECT x, y, z FROM df2", eager=True)

    assert sorted(res_e.rows()) == [(1, 2, 5), (9, 3, 5)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]

    res_e = pl.sql("SELECT * FROM df2 EXCEPT TABLE df1", eager=True)
    res_i = pl.sql(
        """
        SELECT * FROM df2
        INTERSECT
        SELECT x::int8, y::int8, z::int8
          FROM (VALUES (1,2,5),(9,3,5),(1,4,5),(1,4,5)) AS df1(x,y,z)
        """,
        eager=True,
    )
    assert sorted(res_e.rows()) == [(1, 2, 7), (9, None, 6)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]

    # check null behaviour of nulls
    with pl.SQLContext(
        tbl1=pl.DataFrame({"x": [2, 9, 1], "y": [2, None, 4]}),
        tbl2=pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4]}),
    ) as ctx:
        res = ctx.execute("SELECT * FROM tbl1 EXCEPT SELECT * FROM tbl2", eager=True)
        assert_frame_equal(pl.DataFrame({"x": [2], "y": [2]}), res)


def test_except_intersect_by_name() -> None:
    df1 = pl.DataFrame(
        {
            "x": [1, 9, 1, 1],
            "y": [2, 3, 4, 4],
            "z": [5, 5, 5, 5],
        }
    )
    df2 = pl.DataFrame(
        {
            "y": [2, None, 4],
            "w": ["?", "!", "%"],
            "z": [7, 6, 5],
            "x": [1, 9, 1],
        }
    )
    res_e = pl.sql(
        "SELECT x, y, z FROM df1 EXCEPT BY NAME SELECT * FROM df2",
        eager=True,
    )
    res_i = pl.sql(
        "SELECT * FROM df1 INTERSECT BY NAME SELECT * FROM df2",
        eager=True,
    )
    assert sorted(res_e.rows()) == [(1, 2, 5), (9, 3, 5)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]
    assert res_e.columns == ["x", "y", "z"]
    assert res_i.columns == ["x", "y", "z"]


@pytest.mark.parametrize(
    ("op", "op_subtype"),
    [
        ("EXCEPT", "ALL"),
        ("EXCEPT", "ALL BY NAME"),
        ("INTERSECT", "ALL"),
        ("INTERSECT", "ALL BY NAME"),
    ],
)
def test_except_intersect_all_unsupported(op: str, op_subtype: str) -> None:
    df1 = pl.DataFrame({"n": [1, 1, 1, 2, 2, 2, 3]})
    df2 = pl.DataFrame({"n": [1, 1, 2, 2]})

    with pytest.raises(
        SQLInterfaceError,
        match=f"'{op} {op_subtype}' is not supported",
    ):
        pl.sql(f"SELECT * FROM df1 {op} {op_subtype} SELECT * FROM df2", eager=True)


def test_update_statement_error() -> None:
    df_large = pl.DataFrame(
        {
            "FQDN": ["c.ORG.na", "a.COM.na"],
            "NS1": ["ns1.c.org.na", "ns1.d.net.na"],
            "NS2": ["ns2.c.org.na", "ns2.d.net.na"],
            "NS3": ["ns3.c.org.na", "ns3.d.net.na"],
        }
    )
    df_small = pl.DataFrame(
        {
            "FQDN": ["c.org.na"],
            "NS1": ["ns1.c.org.na|127.0.0.1"],
            "NS2": ["ns2.c.org.na|127.0.0.1"],
            "NS3": ["ns3.c.org.na|127.0.0.1"],
        }
    )

    # Create a context and register the tables
    ctx = pl.SQLContext()
    ctx.register("large", df_large)
    ctx.register("small", df_small)

    with pytest.raises(
        SQLInterfaceError,
        match=r"'UPDATE large SET FQDN = .+ operation is currently unsupported",
    ):
        ctx.execute("""
            WITH u AS (
                SELECT
                    small.FQDN,
                    small.NS1,
                    small.NS2,
                    small.NS3
                FROM small
                INNER JOIN large ON small.FQDN = large.FQDN
            )
            UPDATE large
            SET
                FQDN = u.FQDN,
                NS1 = u.NS1,
                NS2 = u.NS2,
                NS3 = u.NS3
            FROM u
            WHERE large.FQDN = u.FQDN
        """)


@pytest.mark.parametrize("op", ["EXCEPT", "INTERSECT", "UNION"])
def test_except_intersect_union_errors(op: str) -> None:
    df1 = pl.DataFrame({"x": [1, 9, 1, 1], "y": [2, 3, 4, 4], "z": [5, 5, 5, 5]})
    df2 = pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4], "z": [7, 6, 5]})

    if op != "UNION":
        with pytest.raises(
            SQLInterfaceError,
            match=f"'{op} ALL' is not supported",
        ):
            pl.sql(
                f"SELECT * FROM df1 {op} ALL SELECT * FROM df2", eager=False
            ).collect()

    with pytest.raises(
        SQLInterfaceError,
        match=f"{op} requires equal number of columns in each table",
    ):
        pl.sql(f"SELECT x FROM df1 {op} SELECT x, y FROM df2", eager=False).collect()


@pytest.mark.parametrize(
    ("cols1", "cols2", "union_subtype", "expected"),
    [
        (
            ["*"],
            ["*"],
            "",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["frame2.*"],
            "ALL",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["frame1.*"],
            ["c1", "c2"],
            "DISTINCT",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["c2", "c1"],
            "ALL BY NAME",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["c1", "c2"],
            ["c1 AS x1", "c2 AS x2"],
            "",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["c1", "c2"],
            ["c2", "c1"],
            "BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        pytest.param(
            ["c1", "c2"],
            ["c2", "c1"],
            "DISTINCT BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
    ],
)
def test_union(
    cols1: list[str],
    cols2: list[str],
    union_subtype: str,
    expected: list[tuple[int, str]],
) -> None:
    with pl.SQLContext(
        frame1=pl.DataFrame({"c1": [1, 2], "c2": ["zz", "yy"]}),
        frame2=pl.DataFrame({"c1": [2, 3], "c2": ["yy", "xx"]}),
        eager=True,
    ) as ctx:
        query = f"""
            SELECT {", ".join(cols1)} FROM frame1
            UNION {union_subtype}
            SELECT {", ".join(cols2)} FROM frame2
        """
        assert sorted(ctx.execute(query).rows()) == expected


def test_union_nonmatching_colnames() -> None:
    # SQL allows "UNION" (aka: polars `concat`) on column names that don't match;
    # this behaves positionally, with column names coming from the first table
    with pl.SQLContext(
        df1=pl.DataFrame(
            data={"Value": [100, 200], "Tag": ["hello", "foo"]},
            schema_overrides={"Value": pl.Int16},
        ),
        df2=pl.DataFrame(
            data={"Number": [300, 400], "String": ["world", "bar"]},
            schema_overrides={"Number": pl.Int32},
        ),
        eager=True,
    ) as ctx:
        res = ctx.execute(
            query="""
                SELECT u.* FROM (
                    SELECT * FROM df1
                    UNION
                    SELECT * FROM df2
                ) u ORDER BY Value
            """
        )
        assert res.schema == {
            "Value": pl.Int32,
            "Tag": pl.String,
        }
        assert res.rows() == [
            (100, "hello"),
            (200, "foo"),
            (300, "world"),
            (400, "bar"),
        ]


def test_union_with_join_state_isolation() -> None:
    # confirm each branch of a UNION executes with isolated join state;
    # ensures that aliases from one branch don't leak into the other
    res = pl.sql(
        query="""
            -- start CTEs
            WITH
              a AS (SELECT 0 AS k),
              b AS (SELECT 1 AS k),
              c AS (SELECT 0 AS k)
            -- end of CTEs
            SELECT a.k FROM a JOIN c ON a.k = c.k
            UNION ALL
            SELECT b.k FROM b JOIN c ON b.k = c.k
        """,
        eager=True,
    )
    assert res.to_series().to_list() == [0]


def test_set_operations_order_by() -> None:
    df1 = pl.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]})
    df2 = pl.DataFrame({"id": [4, 5, 6], "value": [400, 500, 600]})
    df3 = pl.DataFrame({"id": [2, 3, 4], "value": [200, 300, 400]})

    # overall ORDER BY applies to the combined UNION result
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT * FROM df1
            UNION ALL
            SELECT * FROM df2
            ORDER BY id DESC
        """,
        expected={
            "id": [6, 5, 4, 3, 2, 1],
            "value": [600, 500, 400, 300, 200, 100],
        },
        compare_with="sqlite",
    )

    # ORDER BY with LIMIT on the final result
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT * FROM df1
            UNION ALL
            SELECT * FROM df2
            ORDER BY value DESC
            LIMIT 3
        """,
        expected={"id": [6, 5, 4], "value": [600, 500, 400]},
        compare_with="sqlite",
    )

    # ORDER BY with FETCH on the final result
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT * FROM df1
            UNION ALL
            SELECT * FROM df2
            ORDER BY value DESC
            FETCH FIRST 3 ROWS ONLY
        """,
        expected={"id": [6, 5, 4], "value": [600, 500, 400]},
        compare_with="duckdb",
    )

    # Nested ORDER BY in subqueries (top-N from each side) with LIMIT
    assert_sql_matches(
        frames={"df1": df1, "df2": df2},
        query="""
            SELECT * FROM (SELECT * FROM df1 ORDER BY value DESC LIMIT 2) AS top1
            UNION ALL
            SELECT * FROM (SELECT * FROM df2 ORDER BY value ASC LIMIT 2) AS top2
            ORDER BY id
        """,
        expected={"id": [2, 3, 4, 5], "value": [200, 300, 400, 500]},
        compare_with="sqlite",
    )

    # Nested ORDER BY in subqueries with LIMIT, with an outer ORDER BY/LIMIT
    assert_sql_matches(
        {"df1": df1, "df2": df2},
        query="""
            SELECT * FROM (
              SELECT * FROM (SELECT * FROM df1 ORDER BY value DESC LIMIT 2) t1
              UNION ALL
              SELECT * FROM (SELECT * FROM df2 ORDER BY value ASC LIMIT 2) t2
            ) t3
            ORDER BY id
            LIMIT 3
        """,
        expected={"id": [2, 3, 4], "value": [200, 300, 400]},
        compare_with="sqlite",
    )

    # EXCEPT with ORDER BY
    assert_sql_matches(
        {"df1": df1, "df3": df3},
        query="""
            SELECT * FROM df1
            EXCEPT
            SELECT * FROM df3
            ORDER BY id
        """,
        expected={"id": [1], "value": [100]},
        compare_with="sqlite",
    )

    # INTERSECT with ORDER BY
    assert_sql_matches(
        {"df1": df1, "df3": df3},
        query="""
            SELECT * FROM df1
            INTERSECT
            SELECT * FROM df3
            ORDER BY id DESC
        """,
        expected={"id": [3, 2], "value": [300, 200]},
        compare_with="sqlite",
    )

    # INTERSECT with ORDER BY and FETCH (df1 ∩ df3 = {(2,200), (3,300)})
    assert_sql_matches(
        {"df1": df1, "df2": df2, "df3": df3},
        query="""
            (
              SELECT * FROM df1
              UNION
              SELECT * FROM df2
              INTERSECT
              SELECT * FROM df3
            )
            ORDER BY id
            FETCH FIRST 4 ROWS ONLY
        """,
        expected={
            "id": [1, 2, 3, 4],
            "value": [100, 200, 300, 400],
        },
        compare_with="duckdb",
    )

    # Chained UNION with overall ORDER BY
    for open_paren, close_paren, compare_with in (
        ("", "", "sqlite"),
        ("", "", "duckdb"),
        ("(", ")", "duckdb"),
    ):
        assert_sql_matches(
            {"df1": df1, "df2": df2, "df3": df3},
            query=f"""
                {open_paren}
                SELECT * FROM df1
                UNION
                SELECT * FROM df2
                UNION
                SELECT * FROM df3
                {close_paren}
                ORDER BY value
            """,
            expected={
                "id": [1, 2, 3, 4, 5, 6],
                "value": [100, 200, 300, 400, 500, 600],
            },
            compare_with=compare_with,  # type: ignore[arg-type]
        )

    # UNION with ORDER BY on expression (wrapped in subquery)
    assert_sql_matches(
        {"df1": df1, "df2": df2},
        query="""
            SELECT * FROM (
                SELECT id, value FROM df1
                UNION ALL
                SELECT id, value FROM df2
            ) AS combined
            ORDER BY value % 200, id
        """,
        expected={
            "id": [2, 4, 6, 1, 3, 5],
            "value": [200, 400, 600, 100, 300, 500],
        },
        compare_with="sqlite",
    )

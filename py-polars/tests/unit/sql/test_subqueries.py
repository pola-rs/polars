import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


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


@pytest.mark.parametrize("op", ["=", "!=", "<", "<=", ">", ">="])
def test_scalar_subquery_comparison_operators(op: str) -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000, 4000]})

    assert_sql_matches(
        frames={"df": df},
        query=f"SELECT value FROM df WHERE value {op} (SELECT MAX(value) FROM df WHERE value < 4000)",
        compare_with="sqlite",
    )
    assert_sql_matches(
        frames={"df": df},
        query=f"SELECT value FROM df WHERE (SELECT MAX(value) FROM df WHERE value < 4000) {op} value",
        compare_with="sqlite",
    )


def test_scalar_subquery_arithmetic_operand() -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"df": df},
        query="SELECT value, value + (SELECT MAX(value) FROM df) AS z FROM df",
        compare_with="sqlite",
    )


def test_scalar_subquery_in_select_list() -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"df": df},
        query="SELECT value, (SELECT MAX(value) FROM df) AS mx FROM df",
        compare_with="sqlite",
    )


def test_scalar_subquery_in_having() -> None:
    df = pl.DataFrame(
        {
            "grp": ["a", "a", "b", "b", "c"],
            "value": [10, 20, 30, 40, 50],
        }
    )
    assert_sql_matches(
        frames={"df": df},
        query="""
            SELECT grp, SUM(value) AS s FROM df
            GROUP BY grp
            HAVING SUM(value) > (SELECT AVG(value) FROM df)
            ORDER BY grp
        """,
        compare_with="sqlite",
    )


def test_scalar_subquery_with_own_cte() -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000]})

    assert_sql_matches(
        frames={"df": df},
        query="""
            SELECT value FROM df
            WHERE value = (
                WITH cte AS (SELECT MAX(value) AS m FROM df)
                SELECT m FROM cte
            )
        """,
        compare_with="sqlite",
    )


def test_scalar_subquery_null_result() -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000]})

    # an empty subquery result compares as NULL, so it matches nothing
    assert_sql_matches(
        frames={"df": df},
        query="SELECT value FROM df WHERE value = (SELECT value FROM df WHERE value > 10000)",
        compare_with="sqlite",
    )
    assert_sql_matches(
        frames={"df": df},
        query="""
            SELECT value FROM df
            WHERE (SELECT value FROM df WHERE value > 10000) IS NULL
        """,
        compare_with="sqlite",
    )


def test_scalar_subquery_multi_row_errors() -> None:
    # PostgreSQL semantics: a scalar subquery returning more than one row is a
    # runtime error.
    df = pl.DataFrame({"value": [1000, 2000, 3000]})
    with pytest.raises(pl.exceptions.ComputeError, match="aggregation 'item'"):
        pl.sql(
            "SELECT value FROM df WHERE value = (SELECT value FROM df ORDER BY value)",
            eager=True,
        )


def test_unaliased_scalar_subquery_in_select_list() -> None:
    df = pl.DataFrame({"value": [1000, 2000, 3000]})
    res = pl.sql("SELECT (SELECT MAX(value) FROM df) FROM df", eager=True)
    assert res.height == 3
    assert res.to_series().to_list() == [3000, 3000, 3000]


def test_derived_table_without_alias() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # basic unaliased subquery
    with pl.SQLContext(df=df) as ctx:
        res = ctx.execute("SELECT * FROM (SELECT a, b FROM df) ORDER BY a", eager=True)
        assert_frame_equal(res, df)

        # set operation without subquery aliases
        res = ctx.execute(
            """
            FROM (
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


def _subquery_ctx() -> pl.SQLContext[pl.LazyFrame]:
    # One shared fixture for all subquery-to-join tests. The data placement is
    # deliberate: NULL keys on both sides (the NULL-semantics rows) and the
    # same-named composite columns a/b in both tables (the correlation-
    # resolution rows) — several expectations depend on exactly this layout.
    customer = pl.LazyFrame(
        {
            "c_custkey": [1, 2, 3, 4, 5, None],
            "c_acctbal": [10, 20, 30, 40, 50, 60],
            "a": [1, 1, 2, 1, 2, 2],
            "b": [10, 20, 10, 20, 20, 30],
        }
    )
    orders = pl.LazyFrame(
        {
            "o_custkey": [2, 3, 5, None],
            "o_amt": [20, 99, 99, 99],
            "a": [1, 2, 2, 9],
            "b": [20, 20, 30, 9],
        }
    )
    lineitem = pl.LazyFrame({"l_okey": [2, 5]})
    return pl.SQLContext(customer=customer, orders=orders, lineitem=lineitem)


@pytest.mark.parametrize(
    ("query", "expect_join", "expected"),
    [
        # Positive EXISTS must map to a semi join (every other EXISTS row
        # below is anti).
        (
            "SELECT c_custkey FROM customer WHERE EXISTS"
            " (SELECT 1 FROM orders WHERE o_custkey = c_custkey)",
            "SEMI JOIN",
            [2, 3, 5],
        ),
        # A subquery conjunct must rewrite even when AND-combined with
        # ordinary predicates (which must stay behind as a filter).
        (
            "SELECT c_custkey FROM customer WHERE c_custkey > 1 AND NOT EXISTS"
            " (SELECT 1 FROM orders WHERE o_custkey = c_custkey)",
            "ANTI JOIN",
            [4],
        ),
        # DISTINCT must not prevent the rewrite (only DISTINCT ON bails).
        (
            "SELECT c_custkey FROM customer"
            " WHERE c_custkey IN (SELECT DISTINCT o_custkey FROM orders)",
            "SEMI JOIN",
            [2, 3, 5],
        ),
        # Inner-only predicates must filter the inner relation before the join.
        (
            "SELECT c_custkey FROM customer WHERE NOT EXISTS"
            " (SELECT 1 FROM orders WHERE o_custkey = c_custkey AND o_amt = 99)",
            "ANTI JOIN",
            [1, 2, 4, None],
        ),
        # A correlated IN must join on both the membership key and the
        # correlation key.
        (
            "SELECT c_custkey FROM customer WHERE c_acctbal IN"
            " (SELECT o_amt FROM orders WHERE o_custkey = c_custkey)",
            "SEMI JOIN",
            [2],
        ),
        (
            "SELECT c_custkey FROM customer WHERE NOT EXISTS (SELECT 1 FROM"
            " orders JOIN lineitem ON o_custkey = l_okey WHERE o_custkey = c_custkey)",
            "ANTI JOIN",
            [1, 3, 4, None],
        ),
        # Same-named correlation columns must resolve via the qualifier
        # (schema membership alone is ambiguous), and composite correlation
        # must join on all key pairs, not just the first.
        (
            "SELECT a, b FROM customer c"
            " WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.a = c.a AND o.b = c.b)",
            "ANTI JOIN",
            [(1, 10), (2, 10)],
        ),
        # NOT IN should follow SQL three-valued logic (ref: issue #28433).
        (
            "SELECT c_custkey FROM customer"
            " WHERE c_custkey NOT IN (SELECT o_custkey FROM orders)",
            "ANTI JOIN",
            [],
        ),
        # NULL-free value set: unmatched keys are kept, but NULL outer key is dropped.
        (
            "SELECT c_custkey FROM customer"
            " WHERE c_custkey NOT IN (SELECT o_custkey FROM orders WHERE o_custkey IS NOT NULL)",
            "ANTI JOIN",
            [1, 4],
        ),
        # Correlated NOT IN applies 3VL per correlation group.
        (
            "SELECT c_custkey FROM customer c"
            " WHERE c_custkey NOT IN (SELECT o_custkey FROM orders o WHERE o.a = c.a)",
            "ANTI JOIN",
            [1, 4],
        ),
        # NOT EXISTS, by contrast, must keep a NULL outer key — even with a
        # NULL in the haystack (NULL = NULL must not count as a match).
        (
            "SELECT c_custkey FROM customer"
            " WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)",
            "ANTI JOIN",
            [1, 4, None],
        ),
        # DELETE must keep rows whose predicate is NULL (it drops only the
        # predicate-true rows; DELETE yields whole rows).
        (
            "DELETE FROM customer WHERE c_custkey IN (SELECT o_custkey FROM orders)",
            "ANTI JOIN",
            [(1, 10, 1, 10), (4, 40, 1, 20), (None, 60, 2, 30)],
        ),
        (
            "DELETE FROM customer"
            " WHERE EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)",
            "ANTI JOIN",
            [(1, 10, 1, 10), (4, 40, 1, 20), (None, 60, 2, 30)],
        ),
    ],
)
def test_sql_subquery_to_join(
    query: str,
    expect_join: str,
    expected: list[int | None | tuple[int | None, ...]],
) -> None:
    lf = _subquery_ctx().execute(query)

    plan = lf.explain(optimized=True)
    assert expect_join in plan, plan
    assert "implode" not in plan.lower(), plan

    result = lf.collect()
    expected_rows = [t if isinstance(t, tuple) else (t,) for t in expected]
    expected_df = pl.DataFrame(expected_rows, schema=result.schema, orient="row")
    assert_frame_equal(result, expected_df, check_row_order=False)


@pytest.mark.parametrize(
    "subquery",
    [
        # A nested subquery must not rewrite (its SubPlan node is only valid
        # after subquery lowering, not inside a plain filter expression).
        "SELECT 1 FROM orders WHERE o_custkey = c_custkey"
        " AND o_custkey IN (SELECT l_okey FROM lineitem)",
        # Clauses that change which rows the subquery yields must make the
        # rewrite bail, never be silently ignored. Each clause here empties
        # the subquery, so a rewrite that ignored it would get every row's
        # NOT EXISTS wrong.
        "SELECT 1 FROM orders WHERE o_custkey = c_custkey LIMIT 0",
        "SELECT TOP 0 1 FROM orders WHERE o_custkey = c_custkey",
        "SELECT 1 FROM orders WHERE o_custkey = c_custkey QUALIFY FALSE",
    ],
)
def test_sql_subquery_not_rewritten(subquery: str) -> None:
    sql = f"SELECT c_custkey FROM customer WHERE NOT EXISTS ({subquery})"
    with pytest.raises(SQLInterfaceError, match="not currently supported"):
        _subquery_ctx().execute(sql, eager=True)

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


def _frames() -> dict[str, pl.DataFrame]:
    return {
        "t1": pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}),
        "t2": pl.DataFrame({"g": [10, 10, 20], "w": [1, 2, 3]}),
    }


def test_equality_correlated_exists_still_works() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )
    # an equality correlation must still be lowered to a semi join, not routed
    # through the general (decorrelated flag column) EXISTS path
    with pl.SQLContext(frames=_frames()) as ctx:
        plan = ctx.execute(
            "SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)"
        ).explain()
        assert "SEMI JOIN" in plan, plan


def test_equality_correlated_not_exists_still_works() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE NOT EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [3]},
    )
    # an equality correlation must still be lowered to an anti join, not
    # routed through the general (decorrelated flag column) EXISTS path
    with pl.SQLContext(frames=_frames()) as ctx:
        plan = ctx.execute(
            "SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)"
        ).explain()
        assert "ANTI JOIN" in plan, plan


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("<", [2, 3]),
        ("<=", [1, 2, 3]),
        (">", [1, 2]),
        (">=", [1, 2, 3]),
        ("<>", [1, 2, 3]),
    ],
)
def test_exists_inequality_correlation(op: str, expected: list[int]) -> None:
    # Correlation through a self-referencing alias (`t1 AS x` vs outer `t1`).
    assert_sql_matches(
        frames=_frames(),
        query=(
            f"SELECT a FROM t1 WHERE EXISTS "
            f"(SELECT 1 FROM t1 AS x WHERE x.b {op} t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": expected},
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("<", [1]),
        ("<=", []),
        (">", [3]),
        (">=", []),
        ("<>", []),
    ],
)
def test_not_exists_inequality_correlation(op: str, expected: list[int]) -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            f"SELECT a FROM t1 WHERE NOT EXISTS "
            f"(SELECT 1 FROM t1 AS x WHERE x.b {op} t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": expected},
    )


def test_exists_empty_inner_result() -> None:
    # No `x.b` value is ever both less than and greater than the same
    # `t1.b`, so EXISTS is false for every outer row.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": []},
    )


def test_not_exists_all_rows_match() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE NOT EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3]},
    )


def test_exists_inequality_combined_with_other_where_conjunct() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a > 1 AND EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_exists_inequality_with_inner_local_filter() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE EXISTS "
            "(SELECT 1 FROM t2 WHERE t2.g < t1.b AND t2.w > 1) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_not_exists_inequality_null_edge_case() -> None:
    # A NULL correlation column makes every comparison NULL (never true), so
    # the subquery matches zero rows and NOT EXISTS is true.
    frames = {"t3": pl.DataFrame({"a": [1, 2, 3], "b": [10, None, 30]})}
    assert_sql_matches(
        frames=frames,
        query=(
            "SELECT a FROM t3 WHERE NOT EXISTS "
            "(SELECT 1 FROM t3 AS x WHERE x.b < t3.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )


# --- EXISTS in general expression position (OR / CASE / SELECT list) -------
#
# None of these shapes are a whole WHERE filter or a top-level AND-conjunct
# of it, so they can't be lowered to a semi/anti join or count-filter; they
# exercise the decorrelated boolean flag column path instead.


def test_exists_or_predicate() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [2, 3]},
    )


def test_not_exists_or_predicate() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1]},
    )


def test_equality_correlated_exists_in_or_position() -> None:
    # An equality correlation, but not a top-level AND-conjunct, so this must
    # go through the general decorrelation path rather than the semi-join
    # fast path.
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 99 "
            "OR EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2]},
    )


def test_exists_in_case_expression() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, CASE WHEN EXISTS "
            "(SELECT 1 FROM t1 AS x WHERE x.b < t1.b) THEN 1 ELSE 0 END AS flag "
            "FROM t1 ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "flag": [0, 1, 1]},
    )


def test_exists_in_select_list() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e "
            "FROM t1 ORDER BY a"
        ),
        compare_with=None,
        expected={"a": [1, 2, 3], "e": [False, True, True]},
    )


def test_not_exists_in_select_list() -> None:
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a, NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e "
            "FROM t1 ORDER BY a"
        ),
        compare_with=None,
        expected={"a": [1, 2, 3], "e": [True, False, False]},
    )


def test_uncorrelated_exists_in_expression_position() -> None:
    # Uncorrelated EXISTS is a constant boolean, broadcast onto every row.
    assert_sql_matches(
        frames=_frames(),
        query="SELECT a FROM t1 WHERE a = 1 OR EXISTS (SELECT 1 FROM t2) ORDER BY a",
        compare_with="duckdb",
        expected={"a": [1, 2, 3]},
    )
    assert_sql_matches(
        frames=_frames(),
        query=(
            "SELECT a FROM t1 WHERE a = 1 "
            "OR EXISTS (SELECT 1 FROM t2 WHERE g > 1000) ORDER BY a"
        ),
        compare_with="duckdb",
        expected={"a": [1]},
    )


def test_decorrelated_exists_is_order_independent() -> None:
    # Needs several morsels and a joined outer frame: a smaller or unjoined
    # fixture has a stable row order and passes either way.
    n = 200_000
    fact = pl.DataFrame(
        {
            "ord": [i // 2 for i in range(n)],
            "wh": [(i % 2) if (i // 2) % 5 else 0 for i in range(n)],
            "dk": [i % 500 for i in range(n)],
            "ak": [i % 300 for i in range(n)],
            "v": [float(i % 97) for i in range(n)],
        }
    )
    dim_d = pl.DataFrame(
        {"dk": list(range(500)), "keep_d": [i % 2 == 0 for i in range(500)]}
    )
    dim_a = pl.DataFrame(
        {
            "ak": list(range(300)),
            "st": ["GA" if i % 3 == 0 else "XX" for i in range(300)],
        }
    )
    query = """
        SELECT count(DISTINCT ord) AS n_ord, sum(v) AS sv
        FROM fact, dim_d, dim_a
        WHERE fact.dk = dim_d.dk AND dim_d.keep_d = TRUE
          AND fact.ak = dim_a.ak AND dim_a.st = 'GA'
          AND EXISTS (SELECT 1 FROM fact AS f2
                      WHERE fact.ord = f2.ord AND fact.wh <> f2.wh)
        ORDER BY count(DISTINCT ord)
    """
    ctx = pl.SQLContext(fact=fact, dim_d=dim_d, dim_a=dim_a)
    expected = ctx.execute(query).collect(engine="in-memory").row(0)
    results = {ctx.execute(query).collect(engine="streaming").row(0) for _ in range(8)}
    assert results == {expected}


def test_exists_subquery_multi_relation_inner_from() -> None:
    frames = {
        "customer": pl.DataFrame({"c_customer_sk": [1, 2, 3]}),
        "store_sales": pl.DataFrame(
            {"ss_customer_sk": [1, 1, 2], "ss_sold_date_sk": [1, 2, 2]}
        ),
        "date_dim": pl.DataFrame({"d_date_sk": [1, 2], "d_year": [2000, 2001]}),
    }
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT c_customer_sk FROM customer c
            WHERE EXISTS (
                SELECT * FROM store_sales, date_dim
                WHERE c.c_customer_sk = ss_customer_sk
                  AND ss_sold_date_sk = d_date_sk
                  AND d_year = 2001
            )
            ORDER BY c_customer_sk
        """,
        compare_with="duckdb",
    )


def test_exists_and_not_exists_multi_relation_inner_from() -> None:
    frames = {
        "customer": pl.DataFrame({"c_customer_sk": [1, 2, 3, 4]}),
        "store_sales": pl.DataFrame(
            {"ss_customer_sk": [1, 2], "ss_sold_date_sk": [2, 2]}
        ),
        "web_sales": pl.DataFrame({"ws_customer_sk": [2], "ws_sold_date_sk": [2]}),
        "date_dim": pl.DataFrame({"d_date_sk": [1, 2], "d_year": [2000, 2001]}),
    }
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT c_customer_sk FROM customer c
            WHERE EXISTS (
                SELECT * FROM store_sales, date_dim
                WHERE c.c_customer_sk = ss_customer_sk
                  AND ss_sold_date_sk = d_date_sk AND d_year = 2001
            )
            AND NOT EXISTS (
                SELECT * FROM web_sales, date_dim
                WHERE c.c_customer_sk = ws_customer_sk
                  AND ws_sold_date_sk = d_date_sk AND d_year = 2001
            )
            ORDER BY c_customer_sk
        """,
        compare_with="duckdb",
    )


def _q16_frames() -> dict[str, pl.DataFrame]:
    # the TPC-DS q16 shape: outer predicates alongside a self-correlated
    # EXISTS whose correlation is not equality-only
    return {
        "sales": pl.DataFrame(
            {
                "order_no": [1, 1, 2, 2, 3, 4],
                "warehouse": [10, 20, 10, 10, 30, 40],
                "state": ["GA", "GA", "GA", "GA", "CA", "GA"],
                "cost": [5, 6, 7, 8, 9, 10],
            }
        ),
    }


_Q16_QUERY = """
    SELECT sum(cost) AS total FROM sales s1
    WHERE s1.state = 'GA'
      AND EXISTS (
          SELECT * FROM sales s2
          WHERE s1.order_no = s2.order_no
            AND s1.warehouse <> s2.warehouse
      )
"""


def test_exists_does_not_block_predicate_pushdown() -> None:
    # the outer predicate must be applied below the row index the EXISTS
    # introduces, or it never reaches the scan
    with pl.SQLContext(frames=_q16_frames()) as ctx:
        plan = ctx.execute(_Q16_QUERY).explain()

    assert "ROW INDEX" in plan, plan
    assert plan.index("ROW INDEX") < plan.index('col("state") == "GA"'), plan


def test_exists_with_outer_predicates_matches_duckdb() -> None:
    assert_sql_matches(
        frames=_q16_frames(),
        query=_Q16_QUERY,
        compare_with="duckdb",
        expected={"total": [11]},
    )


def test_exists_outer_predicate_in_disjunction_not_split() -> None:
    # an OR is a single conjunct, so nothing may be applied early
    assert_sql_matches(
        frames=_q16_frames(),
        query="""
            SELECT sum(cost) AS total FROM sales s1
            WHERE (s1.state = 'CA' OR s1.warehouse = 40)
              AND EXISTS (
                  SELECT * FROM sales s2
                  WHERE s1.order_no = s2.order_no
                    AND s1.warehouse <> s2.warehouse
              )
        """,
        compare_with="duckdb",
    )


def test_exists_outer_predicate_with_nulls() -> None:
    # a NULL outer predicate is neither true nor false
    frames = {
        "sales": pl.DataFrame(
            {
                "order_no": [1, 1, 2, 2, 3],
                "warehouse": [10, 20, 10, 20, 30],
                "state": ["GA", None, "GA", "GA", None],
                "cost": [5, 6, 7, 8, 9],
            }
        ),
    }
    assert_sql_matches(
        frames=frames,
        query="""
            SELECT order_no, warehouse FROM sales s1
            WHERE s1.state = 'GA'
              AND EXISTS (
                  SELECT * FROM sales s2
                  WHERE s1.order_no = s2.order_no
                    AND s1.warehouse <> s2.warehouse
              )
            ORDER BY order_no, warehouse
        """,
        compare_with="duckdb",
    )


def test_delete_with_exists_and_outer_predicate() -> None:
    # DELETE negates the conjunction as a whole, so no conjunct may be applied
    # on its own
    with pl.SQLContext(frames=_q16_frames()) as ctx:
        remaining = ctx.execute(
            """
            DELETE FROM sales
            WHERE state = 'GA'
              AND EXISTS (
                  SELECT * FROM sales s2
                  WHERE sales.order_no = s2.order_no
                    AND sales.warehouse <> s2.warehouse
              )
            """
        ).collect()

    # this DELETE also leaks an internal correlation column, so compare the
    # table's own columns
    assert remaining.select("order_no", "warehouse", "state", "cost").to_dict(
        as_series=False
    ) == {
        "order_no": [2, 2, 3, 4],
        "warehouse": [10, 10, 30, 40],
        "state": ["GA", "GA", "CA", "GA"],
        "cost": [7, 8, 9, 10],
    }


def test_in_subquery_does_not_block_predicate_pushdown() -> None:
    # `IN (subquery)` reaches the same rewrite as EXISTS
    frames = {
        "sales": pl.DataFrame(
            {
                "cust": [1, 2, 3, 4],
                "state": ["GA", "GA", "CA", "GA"],
                "amt": [5, 6, 7, 8],
            }
        ),
        "returns": pl.DataFrame({"cust": [2, 3, 4]}),
    }
    query = """
        SELECT sum(amt) AS total FROM sales
        WHERE state = 'GA'
          AND cust IN (SELECT cust FROM returns)
    """
    assert_sql_matches(
        frames=frames,
        query=query,
        compare_with="duckdb",
        expected={"total": [14]},
    )

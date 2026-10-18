from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars._typing import EngineType

ENGINES: list[EngineType] = ["in-memory", "streaming"]


def assert_grouping_sets_matches(
    frames: pl.LazyFrame | Mapping[str, pl.LazyFrame],
    query: str,
    *,
    expected: dict[str, Sequence[Any]] | None = None,
    compare_with_duckdb: bool = True,
) -> None:
    """Check `query` on both engines against DuckDB and/or `expected`."""
    assert_sql_matches(
        frames,
        query=query,
        compare_with="duckdb" if compare_with_duckdb else None,
        expected=expected,
        engines=ENGINES,
    )


@pytest.fixture
def sales() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "category": ["a", "a", "a", "b", "b", "b", None],
            "class": ["x", "y", "y", "x", "x", "y", "x"],
            "value": [10, 20, 30, 40, 50, 60, 70],
        }
    )


@pytest.mark.parametrize(
    "group_by",
    [
        "ROLLUP(category, class)",
        "CUBE(category, class)",
        "GROUPING SETS ((category, class), (category), (class), ())",
        "GROUPING SETS ((category, class), ())",
        "GROUPING SETS ((category), (class))",
        "GROUPING SETS (category, class)",
        "GROUPING SETS ((category, class), (category), (category), (), ())",
        "ROLLUP((category, class))",
        "ROLLUP((category, class), category)",
        "category, ROLLUP(class)",
        "ROLLUP(category), ROLLUP(class)",
        "CUBE((category, class), class)",
    ],
)
def test_grouping_sets_constructs(sales: pl.LazyFrame, group_by: str) -> None:
    assert_grouping_sets_matches(
        sales,
        f"""
        SELECT category, class, SUM(value) AS total, COUNT(*) AS n,
               GROUPING(category) AS gc, GROUPING(class) AS gk
        FROM self
        GROUP BY {group_by}
        ORDER BY gc, gk, category NULLS LAST, class NULLS LAST, total
        """,
    )


@pytest.mark.parametrize(
    "group_by",
    [
        "(), category",
        "GROUPING SETS (())",
        "GROUPING SETS ((), ())",
        "GROUPING SETS ((category), (category))",
        "ROLLUP(category, category)",
    ],
)
def test_grouping_sets_single_key_constructs(
    sales: pl.LazyFrame, group_by: str
) -> None:
    assert_grouping_sets_matches(
        sales,
        f"""
        SELECT SUM(value) AS total, COUNT(*) AS n
        FROM self
        GROUP BY {group_by}
        ORDER BY total, n
        """,
    )


def test_rollup_key_as_aggregate_input() -> None:
    # Omitted keys are NULL in the output, but aggregates still see the input values.
    lf = pl.LazyFrame({"a": [1, 1, 2, None, None]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, SUM(a) AS s, COUNT(a) AS c, COUNT(*) AS n, GROUPING(a) AS g
        FROM self
        GROUP BY ROLLUP(a)
        ORDER BY g, a NULLS LAST
        """,
        expected={
            "a": [1, 2, None, None],
            "s": [2, 2, None, 4],
            "c": [2, 1, 0, 3],
            "n": [2, 1, 2, 5],
            "g": [0, 0, 0, 1],
        },
    )


def test_grouping_sets_filter_and_distinct() -> None:
    lf = pl.LazyFrame(
        {
            "k": ["a", "a", "b", "b", "b"],
            "b": [1, -1, 1, 2, 2],
        }
    )
    assert_grouping_sets_matches(
        lf,
        """
        SELECT k, b,
               SUM(b) FILTER (WHERE b > 0) AS positive,
               SUM(b) FILTER (WHERE b > 100) AS none_match,
               COUNT(DISTINCT b) AS nd
        FROM self
        GROUP BY ROLLUP(k, b)
        ORDER BY k NULLS LAST, b NULLS LAST
        """,
        expected={
            "k": ["a", "a", "a", "b", "b", "b", None],
            "b": [-1, 1, None, 1, 2, None, None],
            "positive": [None, 1, 1, 1, 4, 5, 6],
            "none_match": [None] * 7,
            "nd": [1, 1, 2, 1, 1, 2, 3],
        },
    )


@pytest.mark.parametrize("empty", [True, False])
def test_grouping_sets_empty_and_global(empty: bool) -> None:
    data = {"k": [], "v": []} if empty else {"k": ["a", "b"], "v": [1, 2]}
    lf = pl.LazyFrame(data, schema={"k": pl.String, "v": pl.Int64})

    # Empty input: keyed sets give no rows, () gives exactly one row.
    assert_grouping_sets_matches(
        lf,
        """
        SELECT k, COUNT(*) AS n, SUM(v) AS s, AVG(v) AS a, COUNT(v) AS c
        FROM self
        GROUP BY GROUPING SETS ((k), ())
        ORDER BY k NULLS LAST
        """,
        expected=(
            {"k": [None], "n": [0], "s": [None], "a": [None], "c": [0]}
            if empty
            else {
                "k": ["a", "b", None],
                "n": [1, 1, 2],
                "s": [1, 2, 3],
                "a": [1.0, 2.0, 1.5],
                "c": [1, 1, 2],
            }
        ),
    )
    # Two empty sets give two rows regardless of the input height.
    assert_grouping_sets_matches(
        lf,
        "SELECT 1 AS one FROM self GROUP BY GROUPING SETS ((), ())",
        expected={"one": [1, 1]},
        compare_with_duckdb=False,
    )
    assert_grouping_sets_matches(
        lf,
        "SELECT COUNT(*) AS n FROM self GROUP BY GROUPING SETS ((), ())",
        expected={"n": [0, 0] if empty else [2, 2]},
    )
    assert_grouping_sets_matches(
        lf,
        "SELECT COUNT(*) AS n FROM self GROUP BY ()",
        expected={"n": [0 if empty else 2]},
    )
    # A constant key is still an ordinary key: no group on empty input.
    assert_grouping_sets_matches(
        lf,
        "SELECT 1 AS one, COUNT(*) AS n FROM self GROUP BY GROUPING SETS ((1))",
        expected={"one": [] if empty else [1], "n": [] if empty else [2]},
    )


def test_grouping_sets_computed_keys_and_aliases(sales: pl.LazyFrame) -> None:
    assert_grouping_sets_matches(
        sales,
        """
        SELECT UPPER(category) AS cat, value % 2 AS parity, SUM(value) AS total,
               GROUPING(UPPER(category)) AS g1, GROUPING(UPPER(category), value % 2) AS g2
        FROM self
        GROUP BY ROLLUP(cat, value % 2)
        ORDER BY g2, cat NULLS LAST, parity NULLS LAST
        """,
    )
    # SELECT aliases are accepted as GROUPING() arguments.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT UPPER(category) AS cat, SUM(value) AS total, GROUPING(cat) AS g
        FROM self
        GROUP BY ROLLUP(cat)
        ORDER BY g, cat NULLS LAST
        """,
        compare_with_duckdb=False,
        expected={
            "cat": ["A", "B", None, None],
            "total": [60, 150, 70, 280],
            "g": [0, 0, 0, 1],
        },
    )
    # Ordinal and qualified references resolve to the same keys.
    assert_grouping_sets_matches(
        {"t": sales},
        """
        SELECT t.category, class, SUM(value) AS total, GROUPING(category, t.class) AS g
        FROM t
        GROUP BY CUBE(1, t.class)
        ORDER BY g, category NULLS LAST, class NULLS LAST
        """,
    )
    # A computed key and an aggregate over the same source column stay independent.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT value % 2 AS parity, SUM(value) AS total, MAX(value) AS mx
        FROM self
        GROUP BY ROLLUP(value % 2)
        ORDER BY parity NULLS LAST
        """,
        expected={
            "parity": [0, None],
            "total": [280, 280],
            "mx": [70, 70],
        },
    )


def test_grouping_sets_repeated_keys_in_set(sales: pl.LazyFrame) -> None:
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, SUM(value) AS total, GROUPING(category) AS g
        FROM self
        GROUP BY GROUPING SETS ((category, category), (category), ())
        ORDER BY g, category NULLS LAST
        """,
    )


def test_grouping_function_multi_arg(sales: pl.LazyFrame) -> None:
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, class,
               GROUPING(category, class) AS g_cc,
               GROUPING(class, category) AS g_kc,
               GROUPING_ID(class) AS gid,
               GROUPING(category) + GROUPING(class) AS lochierarchy
        FROM self
        GROUP BY CUBE(category, class)
        ORDER BY g_cc, category NULLS LAST, class NULLS LAST
        """,
    )


def test_grouping_function_width_boundary() -> None:
    n = 63
    lf = pl.LazyFrame({f"c{i}": [1] for i in range(n)})
    keys = ", ".join(f"c{i}" for i in range(n))
    out = lf.sql(
        f"SELECT GROUPING({keys}) AS g FROM self GROUP BY GROUPING SETS (({keys}), ())"
    ).collect()
    assert sorted(out["g"].to_list()) == [0, 2**63 - 1]
    assert out.schema["g"] == pl.Int64

    with pytest.raises(SQLSyntaxError, match="between 1 and 63 arguments"):
        lf.sql(
            f"SELECT GROUPING({keys}, c0) FROM self GROUP BY ROLLUP({keys})"
        ).collect()


def test_grouping_in_having_order_by_and_window(sales: pl.LazyFrame) -> None:
    # HAVING keeps a subtotal that pre-filtering the input would have lost.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, class, SUM(value) AS total
        FROM self
        GROUP BY ROLLUP(category, class)
        HAVING GROUPING(class) = 1 OR SUM(value) > 50
        ORDER BY category NULLS LAST, class NULLS LAST, total
        """,
        expected={
            "category": ["a", "b", "b", "b", None, None, None],
            "class": [None, "x", "y", None, "x", None, None],
            "total": [60, 90, 60, 150, 70, 70, 280],
        },
    )
    # GROUPING() restated in ORDER BY rather than referenced by alias.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, class, SUM(value) AS total
        FROM self
        GROUP BY ROLLUP(category, class)
        ORDER BY GROUPING(category) + GROUPING(class),
                 CASE WHEN GROUPING(class) = 0 THEN category END NULLS LAST,
                 category NULLS LAST, class NULLS LAST, total
        """,
    )
    # A window over the combined rows, partitioned by GROUPING() (TPC-DS q70 shape).
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, class, SUM(value) AS total,
               GROUPING(category) + GROUPING(class) AS lochierarchy,
               RANK() OVER (
                   PARTITION BY GROUPING(category) + GROUPING(class),
                                CASE WHEN GROUPING(class) = 0 THEN category END
                   ORDER BY SUM(value) DESC
               ) AS rank_within_parent
        FROM self
        GROUP BY ROLLUP(category, class)
        ORDER BY lochierarchy DESC, category NULLS LAST, class NULLS LAST, rank_within_parent
        """,
    )


def test_aggregates_only_in_order_by_and_having(sales: pl.LazyFrame) -> None:
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, class
        FROM self
        GROUP BY ROLLUP(category, class)
        HAVING COUNT(*) > 1
        ORDER BY SUM(value) DESC, category NULLS LAST, class NULLS LAST
        """,
    )
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, GROUPING(category) * SUM(value) AS weighted
        FROM self
        GROUP BY ROLLUP(category)
        ORDER BY category NULLS LAST, weighted
        """,
    )


def test_grouping_sets_outer_query(sales: pl.LazyFrame) -> None:
    # Internal columns do not leak to SELECT * in the outer query, and the outer
    # window ranks the whole rollup (TPC-DS q67 shape).
    assert_grouping_sets_matches(
        sales,
        """
        SELECT * FROM (
            SELECT category, class, SUM(value) AS total,
                   RANK() OVER (PARTITION BY category ORDER BY SUM(value) DESC) AS rk
            FROM self
            GROUP BY ROLLUP(category, class)
        ) sub
        WHERE rk <= 2
        ORDER BY category NULLS LAST, class NULLS LAST, total
        """,
    )
    # DISTINCT and LIMIT apply to the combined result.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT DISTINCT GROUPING(category) AS g
        FROM self
        GROUP BY ROLLUP(category)
        ORDER BY g
        """,
        expected={"g": [0, 1]},
    )
    out = sales.sql(
        """
        SELECT category, SUM(value) AS total
        FROM self GROUP BY CUBE(category, class)
        ORDER BY total DESC LIMIT 2
        """
    ).collect()
    assert out.to_dict(as_series=False) == {
        "category": [None, None],
        "total": [280, 170],
    }


def test_grouping_sets_nested_scopes(sales: pl.LazyFrame) -> None:
    # Each query block binds its own GROUPING() calls.
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, GROUPING(category) AS g, SUM(value) AS total
        FROM self
        WHERE class IN (
            SELECT class FROM self GROUP BY ROLLUP(class) HAVING GROUPING(class) = 0
        )
        GROUP BY ROLLUP(category)
        ORDER BY g, category NULLS LAST
        """,
    )


def test_grouping_function_plain_group_by(sales: pl.LazyFrame) -> None:
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, GROUPING(category) AS g, SUM(value) AS total
        FROM self GROUP BY category ORDER BY category NULLS LAST
        """,
        expected={
            "category": ["a", "b", None],
            "g": [0, 0, 0],
            "total": [60, 150, 70],
        },
    )
    assert_grouping_sets_matches(
        sales,
        """
        SELECT category, GROUPING(category) AS g, SUM(value) AS total
        FROM self GROUP BY ALL ORDER BY category NULLS LAST
        """,
        compare_with_duckdb=False,
        expected={
            "category": ["a", "b", None],
            "g": [0, 0, 0],
            "total": [60, 150, 70],
        },
    )


def test_grouping_sets_array_agg_and_string_agg(sales: pl.LazyFrame) -> None:
    out = sales.sql(
        """
        SELECT category, ARRAY_AGG(value ORDER BY value) AS vals, COUNT(*) AS n
        FROM self
        GROUP BY GROUPING SETS ((category), ())
        ORDER BY category NULLS LAST, n
        """
    ).collect()
    assert out.to_dict(as_series=False) == {
        "category": ["a", "b", None, None],
        "vals": [[10, 20, 30], [40, 50, 60], [70], [10, 20, 30, 40, 50, 60, 70]],
        "n": [3, 3, 1, 7],
    }


def test_grouping_sets_shared_input(sales: pl.LazyFrame) -> None:
    # One cached input feeds every branch of the union.
    plan = sales.sql(
        """
        SELECT category, class, SUM(value) AS total
        FROM self WHERE value > 10
        GROUP BY ROLLUP(category, class)
        """
    ).explain()
    assert plan.count("CACHE") >= 3
    assert "UNION" in plan

    plan = sales.sql(
        "SELECT category, SUM(value) AS total FROM self GROUP BY category"
    ).explain()
    assert "CACHE" not in plan
    assert "UNION" not in plan


@pytest.mark.parametrize(
    "optimizations",
    [
        pl.QueryOptFlags.none(),
        pl.QueryOptFlags(comm_subplan_elim=False),
        pl.QueryOptFlags(projection_pushdown=False, predicate_pushdown=False),
    ],
)
def test_grouping_sets_optimizations(
    sales: pl.LazyFrame, optimizations: pl.QueryOptFlags
) -> None:
    query = """
        SELECT category, class, SUM(value) AS total, GROUPING(category, class) AS g
        FROM self WHERE value > 10
        GROUP BY CUBE(category, class)
        ORDER BY g, category NULLS LAST, class NULLS LAST
    """
    # The optimized result is checked against DuckDB; unoptimized runs must match it.
    assert_grouping_sets_matches(sales, query)
    expected = sales.sql(query).collect()
    for engine in ENGINES:
        out = sales.sql(query).collect(engine=engine, optimizations=optimizations)
        assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("query", "match"),
    [
        (
            "SELECT category, GROUPING(value) FROM self GROUP BY ROLLUP(category)",
            "GROUPING.. argument 'value' does not appear in the GROUP BY clause",
        ),
        (
            "SELECT GROUPING(category) FROM self",
            "GROUPING.. requires a GROUP BY clause",
        ),
        (
            "SELECT category FROM self WHERE GROUPING(category) = 0 GROUP BY ROLLUP(category)",
            "not allowed in the WHERE clause",
        ),
        (
            "SELECT category FROM self GROUP BY ROLLUP(GROUPING(category))",
            "not allowed in the GROUP BY clause",
        ),
        (
            "SELECT GROUPING(category) OVER () FROM self GROUP BY ROLLUP(category)",
            "cannot be used as a window function",
        ),
        (
            "SELECT GROUPING(category) FILTER (WHERE value > 1) FROM self GROUP BY ROLLUP(category)",
            "does not support a FILTER clause",
        ),
        (
            "SELECT GROUPING(*) FROM self GROUP BY ROLLUP(category)",
            "expects column expressions",
        ),
        (
            "SELECT value FROM self GROUP BY ROLLUP(category)",
            "should participate in the GROUP BY",
        ),
        (
            "SELECT SUM(GROUPING(category)) FROM self GROUP BY ROLLUP(category)",
            "cannot be used inside an aggregate function",
        ),
    ],
)
def test_grouping_sets_errors(sales: pl.LazyFrame, query: str, match: str) -> None:
    with pytest.raises(SQLSyntaxError, match=match):
        sales.sql(query).collect()


def test_grouping_in_delete_where(sales: pl.LazyFrame) -> None:
    with (
        pl.SQLContext(frames={"self": sales}) as ctx,
        pytest.raises(SQLSyntaxError, match="not allowed in the WHERE clause"),
    ):
        ctx.execute("DELETE FROM self WHERE GROUPING(category) = 1").collect()


def test_grouping_sets_expansion_limit() -> None:
    keys = ", ".join(f"c{i}" for i in range(13))
    lf = pl.LazyFrame({f"c{i}": [1] for i in range(13)})
    with pytest.raises(pl.exceptions.SQLInterfaceError, match="grouping set limit"):
        lf.sql(f"SELECT COUNT(*) FROM self GROUP BY CUBE({keys})").collect()
    with pytest.raises(pl.exceptions.SQLInterfaceError, match="more than 4096"):
        lf.sql(
            "SELECT COUNT(*) FROM self GROUP BY CUBE(c0, c1, c2, c3, c4, c5, c6),"
            " CUBE(c7, c8, c9, c10, c11, c12)"
        ).collect()


def test_count_star_mixed_with_grouping() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, COUNT(*) + GROUPING(a) AS n
        FROM self GROUP BY ROLLUP(a) ORDER BY GROUPING(a), a
        """,
        expected={"a": [1, 2, None], "n": [2, 1, 4]},
    )


@pytest.mark.parametrize("having", ["a + 1 = 2", "k = 2"])
def test_computed_key_in_having_and_projections(having: str) -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [10, 20]})
    assert_grouping_sets_matches(
        lf,
        f"""
        SELECT a + 1 AS k, (a + 1) * 10 AS k10, SUM(b) AS s
        FROM self GROUP BY ROLLUP(a + 1) HAVING {having}
        """,
        expected={"k": [2], "k10": [20], "s": [20]},
    )


def test_order_by_aggregate_mixed_with_grouping() -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [10, 20]})
    assert_grouping_sets_matches(
        lf,
        "SELECT a FROM self GROUP BY ROLLUP(a) ORDER BY SUM(b) + GROUPING(a)",
        expected={"a": [0, 1, None]},
    )


@pytest.mark.parametrize("select_grouping", ["", "GROUPING(a) AS g,"])
def test_grouping_in_qualify(select_grouping: str) -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [10, 20]})
    out = lf.sql(
        f"""
        SELECT a, {select_grouping} SUM(b) AS s, RANK() OVER (ORDER BY SUM(b)) AS rn
        FROM self GROUP BY ROLLUP(a)
        QUALIFY rn >= 1 AND GROUPING(a) = 1
        """
    ).collect()
    assert out.select("a", "s", "rn").to_dict(as_series=False) == {
        "a": [None],
        "s": [30],
        "rn": [3],
    }
    assert "__POLARS" not in "".join(out.columns)


def test_user_column_with_internal_prefix() -> None:
    lf = pl.LazyFrame({"__POLARS_GROUPING_user": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT __POLARS_GROUPING_user, COUNT(*) AS n
        FROM self GROUP BY __POLARS_GROUPING_user ORDER BY __POLARS_GROUPING_user
        """,
        expected={"__POLARS_GROUPING_user": [1, 2], "n": [2, 1]},
    )


def test_grouping_argument_is_not_an_ordinal(sales: pl.LazyFrame) -> None:
    with pytest.raises(SQLSyntaxError, match="does not appear in the GROUP BY clause"):
        sales.sql(
            "SELECT category, GROUPING(1) FROM self GROUP BY ROLLUP(category)"
        ).collect()


def test_group_count_versus_window_count() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, GROUPING(a) + COUNT(*) OVER () AS n
        FROM self GROUP BY ROLLUP(a) ORDER BY GROUPING(a), a
        """,
        expected={"a": [1, 2, None], "n": [3, 3, 4]},
    )
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, COUNT(*) + COUNT(*) OVER (PARTITION BY GROUPING(a)) + GROUPING(a) AS n
        FROM self GROUP BY ROLLUP(a) ORDER BY GROUPING(a), a
        """,
        expected={"a": [1, 2, None], "n": [4, 3, 5]},
    )
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, a DESC) AS rn
        FROM self GROUP BY ROLLUP(a) ORDER BY rn
        """,
        expected={"a": [None, 1, 2], "rn": [1, 2, 3]},
    )


@pytest.mark.parametrize("order_by", ["1", "ALL", "s"])
def test_order_by_ordinal_and_all_use_select_list(order_by: str) -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [20, 10]})
    assert_grouping_sets_matches(
        lf,
        f"SELECT SUM(b) AS s, a FROM self GROUP BY ROLLUP(a) ORDER BY {order_by}",
        expected={"s": [10, 20, 30], "a": [1, 0, None]},
    )


def test_order_by_ordinal_out_of_range() -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [20, 10]})
    with pytest.raises(pl.exceptions.SQLInterfaceError, match="ordinal value"):
        lf.sql("SELECT SUM(b) AS s FROM self GROUP BY ROLLUP(a) ORDER BY 2").collect()


def test_aggregate_combined_with_key() -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": [20, 10]})
    assert_grouping_sets_matches(
        lf,
        "SELECT a, SUM(b) + a AS s FROM self GROUP BY ROLLUP(a) ORDER BY a",
        expected={"a": [0, 1, None], "s": [20, 11, None]},
    )
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a + 1 AS k, SUM(b) AS s
        FROM self GROUP BY ROLLUP(a + 1) ORDER BY SUM(b) + (a + 1)
        """,
        expected={"k": [2, 1, None], "s": [10, 20, 30]},
    )
    # The same split applies to an ordinary GROUP BY.
    assert_grouping_sets_matches(
        lf,
        "SELECT a, SUM(b) + a AS s FROM self GROUP BY a ORDER BY a",
        expected={"a": [0, 1], "s": [20, 11]},
    )


def test_empty_window_with_literal_key() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT 1 AS k, COUNT(*) OVER () AS n, GROUPING(1) AS g
        FROM self GROUP BY ROLLUP(1) ORDER BY g
        """,
        expected={"k": [1, None], "n": [2, 2], "g": [0, 1]},
    )


@pytest.mark.parametrize(
    "expr", ["ABS(SUM(b) + GROUPING(a))", "COALESCE(SUM(b) + GROUPING(a), 0)"]
)
def test_elementwise_functions_around_aggregates(expr: str) -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    assert_grouping_sets_matches(
        lf,
        f"SELECT a, {expr} AS s FROM self GROUP BY ROLLUP(a) ORDER BY a",
        expected={"a": [1, 2, None], "s": [30, 30, 61]},
    )


def test_elementwise_functions_around_aggregates_in_order_by_and_with_key() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    assert_grouping_sets_matches(
        lf,
        "SELECT a FROM self GROUP BY ROLLUP(a) ORDER BY ABS(SUM(b) + GROUPING(a)), a",
        expected={"a": [1, 2, None]},
    )
    assert_grouping_sets_matches(
        lf,
        "SELECT a, ABS(SUM(b) + a) AS s FROM self GROUP BY ROLLUP(a) ORDER BY a",
        expected={"a": [1, 2, None], "s": [31, 32, None]},
    )


@pytest.mark.parametrize(
    ("order_by", "a"),
    [
        ("COUNT(*) OVER (PARTITION BY GROUPING(a)), a", [None, 1, 2]),
        ("COUNT(*) OVER (), a", [1, 2, None]),
        ("RANK() OVER (ORDER BY SUM(b)), a", [1, 2, None]),
    ],
)
def test_unselected_window_in_order_by(order_by: str, a: list[int | None]) -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    assert_grouping_sets_matches(
        lf,
        f"SELECT a, SUM(b) AS s FROM self GROUP BY ROLLUP(a) ORDER BY {order_by}",
        expected={"a": a, "s": [60 if x is None else 30 for x in a]},
    )


def test_count_distinct_mixed_with_key_or_grouping() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    assert_grouping_sets_matches(
        lf,
        "SELECT a, COUNT(DISTINCT b) + GROUPING(a) AS n FROM self GROUP BY ROLLUP(a) ORDER BY a",
        expected={"a": [1, 2, None], "n": [2, 1, 4]},
    )
    assert_grouping_sets_matches(
        lf,
        "SELECT a, a + COUNT(DISTINCT b) AS n FROM self GROUP BY a ORDER BY a",
        expected={"a": [1, 2], "n": [3, 3]},
    )


def test_literal_key_referenced_in_having() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT 1 AS k, COUNT(*) AS n
        FROM self GROUP BY ROLLUP(k) HAVING k IS NOT NULL
        """,
        expected={"k": [1], "n": [3]},
    )


@pytest.mark.parametrize(
    ("agg", "value"), [("AVG(1)", 1.0), ("MIN(1)", 1), ("COUNT(DISTINCT 1)", 1)]
)
def test_constant_input_aggregates_mixed_with_grouping(agg: str, value: float) -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        f"SELECT a, {agg} + GROUPING(a) AS v FROM self GROUP BY ROLLUP(a) ORDER BY a",
        expected={"a": [1, 2, None], "v": [value, value, value + 1]},
    )


def test_constant_input_aggregate_mixed_with_key() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2]})
    assert_grouping_sets_matches(
        lf,
        "SELECT a, AVG(1) + a AS v FROM self GROUP BY a ORDER BY a",
        expected={"a": [1, 2], "v": [2.0, 3.0]},
    )


def test_whole_frame_window_marker_is_private() -> None:
    lf = pl.LazyFrame({"__POLARS_WHOLE_FRAME_WINDOW": ["x", "y"], "v": [1, 2]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT __POLARS_WHOLE_FRAME_WINDOW,
               COUNT(*) OVER (PARTITION BY __POLARS_WHOLE_FRAME_WINDOW) AS n
        FROM self GROUP BY __POLARS_WHOLE_FRAME_WINDOW, v ORDER BY 1
        """,
        expected={"__POLARS_WHOLE_FRAME_WINDOW": ["x", "y"], "n": [1, 1]},
    )


def test_group_by_all_without_keys_keeps_rows() -> None:
    lf = pl.LazyFrame({"v": [1, 2]})
    out = lf.sql("SELECT ROW_NUMBER() OVER () AS n FROM self GROUP BY ALL ORDER BY n")
    assert out.collect()["n"].to_list() == [1, 2]


def test_window_over_selected_aggregate() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    assert_grouping_sets_matches(
        lf,
        """
        SELECT a, MIN(b) AS m, MAX(MIN(b)) OVER () AS w
        FROM self GROUP BY ROLLUP(a) ORDER BY a
        """,
        expected={"a": [1, 2, None], "m": [10, 30, 10], "w": [30, 30, 30]},
    )

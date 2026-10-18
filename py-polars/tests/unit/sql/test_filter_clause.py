from __future__ import annotations

import math
from typing import Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, SchemaError, SQLInterfaceError
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "grp": ["a", "b", "a", "b", "a", "b"],
            "x": [1, 2, 3, 4, None, 6],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.mark.parametrize(
    ("agg", "values"),
    [
        ("SUM(x) FILTER (WHERE y > 20)", [3, 10]),
        ("AVG(x) FILTER (WHERE y > 20)", [3.0, 5.0]),
        ("MIN(x) FILTER (WHERE grp = 'a')", [1, None]),
        ("MAX(x) FILTER (WHERE grp = 'a')", [3, None]),
        ("COUNT(*) FILTER (WHERE grp = 'a')", [3, 0]),
        ("COUNT(1) FILTER (WHERE grp = 'a')", [3, 0]),
        ("COUNT(x) FILTER (WHERE grp = 'a')", [2, 0]),
        ("COUNT(x) FILTER (WHERE y > 20)", [1, 2]),
        ("COUNT(DISTINCT x) FILTER (WHERE y > 20)", [1, 2]),
    ],
)
def test_filter_clause_grouped(lf: pl.LazyFrame, agg: str, values: list[Any]) -> None:
    assert_sql_matches(
        frames=lf,
        query=f"SELECT grp, {agg} AS v FROM self GROUP BY grp ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "b"], "v": values},
    )


@pytest.mark.parametrize(
    ("agg", "values"),
    [
        ("MEDIAN(x) FILTER (WHERE y > 20)", [3.0, 5.0]),
        ("STDDEV_SAMP(x) FILTER (WHERE y > 20)", [None, math.sqrt(2.0)]),
        ("VAR_SAMP(x) FILTER (WHERE y > 20)", [None, 2.0]),
        ("QUANTILE_CONT(x, 0.5) FILTER (WHERE y > 20)", [3.0, 5.0]),
    ],
)
def test_filter_clause_misc_aggfuncs(
    lf: pl.LazyFrame, agg: str, values: list[Any]
) -> None:
    assert_sql_matches(
        frames=lf,
        query=f"SELECT grp, {agg} AS v FROM self GROUP BY grp ORDER BY grp",
        compare_with="duckdb",
        expected={"grp": ["a", "b"], "v": values},
    )


def test_filter_clause_approx_quantile(lf: pl.LazyFrame) -> None:
    # not compared against a reference backend: other engines use a different sketch
    assert_sql_matches(
        frames=lf,
        query="""
            SELECT grp, APPROX_QUANTILE(x, 0.5) FILTER (WHERE y > 20) AS v
            FROM self GROUP BY grp ORDER BY grp
        """,
        compare_with=None,
        expected={"grp": ["a", "b"], "v": [3, 6]},
    )


@pytest.mark.parametrize(
    ("agg", "value"),
    [
        ("SUM(x) FILTER (WHERE y > 20)", 13),
        ("AVG(x) FILTER (WHERE y > 20)", 13.0 / 3.0),
        ("COUNT(*) FILTER (WHERE grp = 'a')", 3),
        ("COUNT(x) FILTER (WHERE y > 20)", 3),
        ("COUNT(DISTINCT x) FILTER (WHERE grp = 'b')", 3),
    ],
)
def test_filter_clause_no_group_by(lf: pl.LazyFrame, agg: str, value: Any) -> None:
    assert_sql_matches(
        frames=lf,
        query=f"SELECT {agg} AS v FROM self",
        compare_with="sqlite",
        expected={"v": [value]},
    )


def test_filter_clause_multiple_aggs(lf: pl.LazyFrame) -> None:
    assert_sql_matches(
        frames=lf,
        query="""
            SELECT
                grp,
                SUM(x) FILTER (WHERE y > 20) AS sum_high,
                COUNT(*) FILTER (WHERE x IS NOT NULL) AS n_not_null,
                AVG(y) FILTER (WHERE grp = 'a') AS avg_a
            FROM self
            GROUP BY grp
            ORDER BY grp
        """,
        compare_with="sqlite",
        expected={
            "grp": ["a", "b"],
            "sum_high": [3, 10],
            "n_not_null": [2, 3],
            "avg_a": [30.0, None],
        },
    )


def test_filter_clause_multi_parameter_func() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 4, 10, 8, 9, 13],
            "c": ["a", "b", "a", "a", "b", "b"],
        }
    )
    expected_b = 165.0 / math.sqrt(78.0 * 366.0)
    assert_sql_matches(
        frames=lf,
        query="""
            SELECT c, CORR(a, b) FILTER (WHERE a > 1) AS r
            FROM self GROUP BY c
            ORDER BY c
        """,
        compare_with="duckdb",
        expected={"c": ["a", "b"], "r": [-1.0, expected_b]},
    )


@pytest.mark.parametrize(
    "agg", ["SUM(x)", "COUNT(*)", "COUNT(x)", "MIN(x)", "MAX(x)", "AVG(x)"]
)
@pytest.mark.parametrize(
    "over",
    ["PARTITION BY grp", "PARTITION BY grp ORDER BY y", "ORDER BY y ROWS 1 PRECEDING"],
)
def test_filter_clause_with_over(agg: str, over: str) -> None:
    df = pl.DataFrame(
        {
            "grp": ["a", "a", "b", "b", "b"],
            "x": [1, None, 2, 3, 4],
            "y": [10, 30, 20, 40, 50],
        }
    )
    assert_sql_matches(
        df,
        query=f"SELECT y, {agg} FILTER (WHERE y > 15) OVER ({over}) AS v FROM self ORDER BY y",
        compare_with="duckdb",
        engines=["in-memory", "streaming"],
    )


def test_filter_clause_with_over_unsupported() -> None:
    df = pl.DataFrame({"grp": ["a", "b"], "x": [1, 2], "y": [10, 30]})
    with pytest.raises(
        SQLInterfaceError,
        match="'FILTER' combined with 'OVER' is not supported for STDDEV",
    ):
        pl.sql(
            "SELECT STDDEV(x) FILTER (WHERE y > 20) OVER (PARTITION BY grp) FROM df"
        ).collect()


@pytest.mark.parametrize("agg", ["SUM(2)", "COUNT(*)", "SUM(x)"])
def test_filter_clause_non_boolean_error(agg: str) -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    for pred in (
        "x",
        "x + (SELECT 0)",
        "x + CAST(x IN (SELECT x FROM self) AS INT)",
        "x + CAST(x = ANY (SELECT x FROM self) AS INT)",
    ):
        with pytest.raises(InvalidOperationError, match="must be of type `Boolean`"):
            df.sql(f"SELECT {agg} FILTER (WHERE {pred}) FROM self")
    # Without a schema, the predicate is checked when it runs.
    for engine in ("in-memory", "streaming"):
        with pytest.raises(SchemaError, match="`Boolean`"):
            df.lazy().select(pl.sql_expr(f"{agg} FILTER (WHERE x)")).collect(
                engine=engine
            )


def test_filter_clause_subquery_non_boolean_cast_error() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(InvalidOperationError, match="casting from"):
        df.sql("SELECT COUNT(*) FILTER (WHERE (SELECT 'abc')) FROM self")


@pytest.mark.parametrize(
    "query",
    [
        """
        SELECT
          SUM(2) FILTER (WHERE (SELECT TRUE)) AS a,
          COUNT(*) FILTER (WHERE (SELECT TRUE)) AS b,
          STDDEV(1) FILTER (WHERE (SELECT TRUE)) AS c,
          SUM(x) FILTER (WHERE x > (SELECT 1)) AS d
        FROM self
        """,
        """
        SELECT
          g,
          SUM(2) FILTER (WHERE (SELECT TRUE)) AS a,
          COUNT(*) FILTER (WHERE x > (SELECT 1)) AS b
        FROM self GROUP BY g ORDER BY g
        """,
        # A condition that reads no input is cast to boolean.
        """
        SELECT
          COUNT(*) FILTER (WHERE (SELECT NULL)) AS a,
          COUNT(*) FILTER (WHERE (SELECT 0)) AS b,
          SUM(2) FILTER (WHERE (SELECT 1)) AS c,
          SUM(2) FILTER (WHERE 1 + CAST(1 IN (SELECT x FROM self) AS INT)) AS d,
          SUM(x) FILTER (WHERE 1 + CAST(1 = ANY (SELECT x FROM self) AS INT)) AS e
        FROM self
        """,
    ],
)
def test_filter_clause_subquery(query: str) -> None:
    # The predicate reads a subquery value once per row.
    df = pl.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3]})
    assert_sql_matches(
        df, query=query, compare_with="duckdb", engines=["in-memory", "streaming"]
    )

from __future__ import annotations

import math
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
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

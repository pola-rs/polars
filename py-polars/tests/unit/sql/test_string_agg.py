from __future__ import annotations

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_test() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "grp": ["a", "b", "a", "b", "a", "b"],
            "x": ["x1", "y1", "x2", "y2", "x3", "y3"],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.mark.parametrize(
    ("agg", "values"),
    [
        # plain aggregation, no per-group ordering
        ("STRING_AGG(x, ',')", ["x1,x2,x3", "y1,y2,y3"]),
        # separator is optional and defaults to ","
        ("STRING_AGG(x)", ["x1,x2,x3", "y1,y2,y3"]),
        # in-arg ORDER BY (asc/desc): sorts the input before concatenation
        ("STRING_AGG(x, '|' ORDER BY y DESC)", ["x3|x2|x1", "y3|y2|y1"]),
        ("STRING_AGG(x, '%' ORDER BY y ASC)", ["x1%x2%x3", "y1%y2%y3"]),
        # DISTINCT + ORDER BY: dedupes then sorts
        ("STRING_AGG(DISTINCT x, ',' ORDER BY x)", ["x1,x2,x3", "y1,y2,y3"]),
        # FILTER composes with ORDER BY (predicate restricts
        # the row set, ORDER BY operates on what's left)
        (
            "STRING_AGG(x, ',' ORDER BY y) FILTER (WHERE y > 25)",
            ["x2,x3", "y2,y3"],
        ),
    ],
)
def test_string_agg_grouped(df_test: pl.LazyFrame, agg: str, values: list[str]) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"SELECT grp, {agg} AS v FROM df GROUP BY grp ORDER BY grp",
        compare_with="duckdb",
        expected={"grp": ["a", "b"], "v": values},
    )


@pytest.mark.parametrize(
    "func_alias",
    ["STRING_AGG", "GROUP_CONCAT", "LISTAGG"],
)
def test_string_agg_aliases(df_test: pl.LazyFrame, func_alias: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT grp, {func_alias}(x, ',' ORDER BY x) AS v
            FROM df GROUP BY grp
            ORDER BY grp
        """,
        compare_with=("sqlite" if func_alias != "LISTAGG" else None),
        expected={"grp": ["a", "b"], "v": ["x1,x2,x3", "y1,y2,y3"]},
    )


def test_string_agg_no_group_by(df_test: pl.LazyFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="SELECT STRING_AGG(x, ',' ORDER BY y) AS v FROM df",
        compare_with="duckdb",
        expected={"v": ["x1,y1,x2,y2,x3,y3"]},
    )


def test_string_agg_limit(df_test: pl.LazyFrame) -> None:
    out = pl.SQLContext(df=df_test).execute(
        query="""
            SELECT grp, STRING_AGG(x, ',' ORDER BY y DESC LIMIT 2) AS v
            FROM df GROUP BY grp
            ORDER BY grp
        """,
        eager=True,
    )
    assert out.to_dict(as_series=False) == {
        "grp": ["a", "b"],
        "v": ["x3,x2", "y3,y2"],
    }


def test_string_agg_all_null_returns_null() -> None:
    # `STRING_AGG` over an all-null input returns NULL (not an empty string),
    # matching standard SQL. A non-null empty string is still concatenated.
    df = pl.DataFrame({"a": [None, None]}, schema={"a": pl.String})
    assert_sql_matches(
        {"df": df},
        query="SELECT STRING_AGG(a, ',') AS v FROM df",
        compare_with="duckdb",
        expected={"v": [None]},
    )

    # an all-null group yields NULL; a group with values concatenates as normal
    grp = pl.DataFrame(
        {"g": [1, 1, 2, 2], "a": [None, None, "p", "q"]},
        schema={"g": pl.Int64, "a": pl.String},
    )
    assert_sql_matches(
        {"df": grp},
        query="SELECT g, STRING_AGG(a, ',') AS v FROM df GROUP BY g ORDER BY g",
        compare_with="duckdb",
        expected={"g": [1, 2], "v": [None, "p,q"]},
    )

    # a non-null empty string is preserved (distinct from the all-null case)
    empty = pl.DataFrame({"a": ["", None]}, schema={"a": pl.String})
    assert_sql_matches(
        {"df": empty},
        query="SELECT STRING_AGG(a, ',') AS v FROM df",
        compare_with="duckdb",
        expected={"v": [""]},
    )

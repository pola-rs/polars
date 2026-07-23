from __future__ import annotations

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


def test_negated_count() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) AS c FROM self",
        compare_with="duckdb",
        expected={"c": [-3]},
    )
    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) * -1 AS c FROM self",
        compare_with="duckdb",
        expected={"c": [3]},
    )


def test_count_dtype() -> None:
    # `COUNT(*)` must return Int64, regardless of the input frame's own dtypes
    df = pl.DataFrame({"a": [1, 2, 3]})
    res = df.sql("SELECT COUNT(*) AS c FROM self")
    assert res.schema["c"] == pl.Int64


def test_negated_count_group_by() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT g, -COUNT(*) AS c FROM self GROUP BY g ORDER BY g",
        compare_with="duckdb",
        expected={"g": ["a", "b"], "c": [-2, -1]},
    )


def test_sum_literal() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT SUM(5) AS s, SUM(-3) AS t FROM self",
        compare_with="duckdb",
        expected={"s": [15], "t": [-9]},
    )


def test_sum_literal_group_by() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT g, SUM(3) AS s FROM self GROUP BY g ORDER BY g",
        compare_with="duckdb",
        expected={"g": ["a", "b"], "s": [6, 3]},
    )


def test_sum_literal_empty_table() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(5) AS s FROM self",
        compare_with="duckdb",
        expected={"s": [None]},
    )


def test_sum_min_max_empty_group_global() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT SUM(a) AS s, MIN(a) AS mn, MAX(a) AS mx FROM self WHERE a > 100",
        compare_with="duckdb",
        expected={"s": [None], "mn": [None], "mx": [None]},
    )


def test_sum_min_max_all_null_group_global() -> None:
    df = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(a) AS s, MIN(a) AS mn, MAX(a) AS mx FROM self",
        compare_with="duckdb",
        expected={"s": [None], "mn": [None], "mx": [None]},
    )


def test_sum_min_max_empty_and_null_group_by() -> None:
    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [10, 20, None, None],
        }
    )

    assert_sql_matches(
        df,
        query="""
            SELECT g, SUM(v) AS s, MIN(v) AS mn, MAX(v) AS mx
            FROM self
            GROUP BY g
            ORDER BY g
        """,
        compare_with="duckdb",
        expected={
            "g": ["a", "b"],
            "s": [30, None],
            "mn": [10, None],
            "mx": [20, None],
        },
    )


def test_sum_where_empties_group() -> None:
    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [10, 20, 1, 2],
        }
    )

    assert_sql_matches(
        df,
        query="""
            SELECT g, SUM(v) AS s
            FROM self
            WHERE g = 'a'
            GROUP BY g
            ORDER BY g
        """,
        compare_with="duckdb",
        expected={"g": ["a"], "s": [30]},
    )


def test_negated_count_and_sum_interaction() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [10, 20, None, None]})

    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) + SUM(v) AS r FROM self WHERE g = 'b'",
        compare_with="duckdb",
        expected={"r": [None]},
    )


def test_min_max_distinct_is_noop() -> None:
    df = pl.DataFrame({"v": [3, 1, 1, None, 3]})

    assert_sql_matches(
        df,
        query="SELECT MIN(DISTINCT v) AS mn, MAX(DISTINCT v) AS mx FROM self",
        compare_with="duckdb",
        expected={"mn": [1], "mx": [3]},
    )


def test_sum_avg_distinct_dedup() -> None:
    df = pl.DataFrame({"v": [1, 1, 2, 2, 3, None]})

    assert_sql_matches(
        df,
        query="SELECT SUM(DISTINCT v) AS s, AVG(DISTINCT v) AS a FROM self",
        compare_with="duckdb",
        expected={"s": [6], "a": [2.0]},
    )


def test_sum_avg_distinct_dedup_group_by() -> None:
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "a", "a", "b", "b"],
            "v": [1, 1, 2, 2, 3, None, None],
        }
    )

    assert_sql_matches(
        df,
        query="""
            SELECT g, SUM(DISTINCT v) AS s, AVG(DISTINCT v) AS a
            FROM self
            GROUP BY g
            ORDER BY g
        """,
        compare_with="duckdb",
        expected={"g": ["a", "b"], "s": [6, None], "a": [2.0, None]},
    )


def test_sum_distinct_literal() -> None:
    # a broadcast literal has a single distinct value, so `SUM(DISTINCT <lit>)`
    # must not behave like plain `SUM(<lit>)` (which scales with the row count).
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT SUM(DISTINCT 5) AS s FROM self",
        compare_with="duckdb",
        expected={"s": [5]},
    )


def test_sum_avg_distinct_empty_table() -> None:
    df = pl.DataFrame({"v": []}, schema={"v": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(DISTINCT v) AS s, AVG(DISTINCT v) AS a FROM self",
        compare_with="duckdb",
        expected={"s": [None], "a": [None]},
    )


def test_group_concat_distinct_with_separator_errors() -> None:
    df = pl.DataFrame({"a": [1, 1, 2]})

    # DISTINCT + explicit separator: not supported (matches SQLite's
    # "DISTINCT aggregates must have exactly one argument").
    with pytest.raises(pl.exceptions.SQLSyntaxError, match="DISTINCT"):
        df.sql("SELECT GROUP_CONCAT(DISTINCT a, ':') AS s FROM self")

    # DISTINCT with a single argument (no separator) must still work.
    # (ORDER BY pins the concatenation order, which DISTINCT alone does not
    # guarantee -- needed for a deterministic comparison against DuckDB.)
    assert_sql_matches(
        df,
        query="SELECT GROUP_CONCAT(DISTINCT a ORDER BY a) AS s FROM self",
        compare_with="duckdb",
        expected={"s": ["1,2"]},
    )

    # Non-DISTINCT with an explicit separator must still work.
    assert_sql_matches(
        df,
        query="SELECT GROUP_CONCAT(a, ':') AS s FROM self",
        compare_with="duckdb",
        expected={"s": ["1:1:2"]},
    )

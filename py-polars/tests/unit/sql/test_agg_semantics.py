from __future__ import annotations

import polars as pl
from tests.unit.sql import assert_sql_matches


def test_negated_count() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) AS c FROM self",
        compare_with="sqlite",
        expected={"c": [-3]},
    )
    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) * -1 AS c FROM self",
        compare_with="sqlite",
        expected={"c": [3]},
    )


def test_negated_count_group_by() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT g, -COUNT(*) AS c FROM self GROUP BY g ORDER BY g",
        compare_with="sqlite",
        expected={"g": ["a", "b"], "c": [-2, -1]},
    )


def test_sum_literal() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT SUM(5) AS s, SUM(-3) AS t FROM self",
        compare_with="sqlite",
        expected={"s": [15], "t": [-9]},
    )


def test_sum_literal_group_by() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT g, SUM(3) AS s FROM self GROUP BY g ORDER BY g",
        compare_with="sqlite",
        expected={"g": ["a", "b"], "s": [6, 3]},
    )


def test_sum_literal_empty_table() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(5) AS s FROM self",
        compare_with="sqlite",
        expected={"s": [None]},
    )


def test_sum_min_max_empty_group_global() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert_sql_matches(
        df,
        query="SELECT SUM(a) AS s, MIN(a) AS mn, MAX(a) AS mx FROM self WHERE a > 100",
        compare_with="sqlite",
        expected={"s": [None], "mn": [None], "mx": [None]},
    )


def test_sum_min_max_all_null_group_global() -> None:
    df = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(a) AS s, MIN(a) AS mn, MAX(a) AS mx FROM self",
        compare_with="sqlite",
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
        compare_with="sqlite",
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
        compare_with="sqlite",
        expected={"g": ["a"], "s": [30]},
    )


def test_negated_count_and_sum_interaction() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [10, 20, None, None]})

    assert_sql_matches(
        df,
        query="SELECT -COUNT(*) + SUM(v) AS r FROM self WHERE g = 'b'",
        compare_with="sqlite",
        expected={"r": [None]},
    )

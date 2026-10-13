from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.meta import get_index_type
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
        engines=["in-memory", "streaming"],
        expected={"s": [None], "mn": [None], "mx": [None]},
    )


def test_sum_min_max_all_null_group_global() -> None:
    df = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Int64})

    assert_sql_matches(
        df,
        query="SELECT SUM(a) AS s, MIN(a) AS mn, MAX(a) AS mx FROM self",
        compare_with="duckdb",
        engines=["in-memory", "streaming"],
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
        engines=["in-memory", "streaming"],
        expected={
            "g": ["a", "b"],
            "s": [30, None],
            "mn": [10, None],
            "mx": [20, None],
        },
    )


def test_sum_plan_has_no_count() -> None:
    lf = pl.LazyFrame({"g": ["a", "b"], "v": [1, None]})
    plan = lf.sql("SELECT g, SUM(v) AS s, SUM(DISTINCT v) AS d FROM self GROUP BY g")
    explained = plan.explain()
    assert explained.count("sum(null_on_empty=true)") == 2
    assert "count()" not in explained


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


@pytest.mark.parametrize(
    "agg",
    [
        "MIN(1)",
        "MAX(1)",
        "AVG(1)",
        "MEDIAN(1)",
        "STDDEV(1)",
        "VARIANCE(1)",
        "FIRST(1)",
        "SUM(1 + 1)",
        "COUNT(DISTINCT 1)",
        "QUANTILE_CONT(1, 0.5)",
        "STRING_AGG('a', ',')",
        "SUM(-1000000000)",
        "SUM(2000000000)",
        "SUM(DISTINCT 2000000000)",
        "SUM(2000000000) FILTER (WHERE g > 0)",
        "SUM(1) FILTER (WHERE g > 5)",
        "SUM(TRUE)",
        "SUM(NULL)",
        "SUM(CAST(NULL AS INT))",
        "SUM(DISTINCT CAST(NULL AS INT))",
        "MAX(CAST(NULL AS INT))",
        "COUNT(CAST(NULL AS INT))",
        "STDDEV(CAST(NULL AS DOUBLE))",
        "COUNT(-1)",
        "COUNT(NULL)",
        "COUNT(DISTINCT 1) FILTER (WHERE g > 1)",
        "AVG(1) FILTER (WHERE g > 1)",
        "STDDEV(1) FILTER (WHERE g > 1)",
        "STDDEV(NULL)",
        "MAX(NULL)",
        "SUM(2) FILTER (WHERE TRUE)",
        "STDDEV(1) FILTER (WHERE TRUE)",
        "MAX(1) FILTER (WHERE 1 = 2)",
        "COUNT(*) FILTER (WHERE NULL)",
        "SUM(DISTINCT CAST(20000 AS SMALLINT)) * 2",
        "SUM(DISTINCT TRUE)",
    ],
)
@pytest.mark.parametrize("n_rows", [0, 3])
def test_aggregate_of_constant(agg: str, n_rows: int) -> None:
    # A constant is read once per row, as a column is.
    df = pl.DataFrame({"g": [1, 2, 2][:n_rows]}, schema={"g": pl.Int64})
    for query in [
        f"SELECT {agg} AS a FROM self",
        f"SELECT g, {agg} AS a FROM self GROUP BY g ORDER BY g",
    ]:
        assert_sql_matches(
            df,
            query=query,
            compare_with="duckdb",
            engines=["in-memory", "streaming"],
        )


def test_aggregate_of_constant_not_read_per_row() -> None:
    # The aggregate of a constant follows from the number of rows read, so the
    # constant is not repeated for each row.
    lf = pl.LazyFrame({"g": [1, 2, 2]})
    for agg in ["MAX(1)", "STDDEV(1)", "COUNT(-1)", "SUM(1) FILTER (WHERE g > 1)"]:
        plan = lf.sql(f"SELECT g, {agg} AS a FROM self GROUP BY g").explain()
        assert "repeat([len()])" not in plan


@pytest.mark.parametrize(
    ("group_by", "has_check"),
    [
        ("GROUP BY g", False),
        ("GROUP BY g + 1", True),
        ("GROUP BY 1 + 1", True),
        ("", True),
    ],
)
def test_aggregate_of_constant_in_groups(group_by: str, has_check: bool) -> None:
    # A group of a column key has a row, so the aggregate does not check that rows
    # are read.
    query = f"SELECT MAX(1) AS m, COUNT(DISTINCT 1) AS d FROM self {group_by}"
    for n_rows in [0, 3]:
        lf = pl.LazyFrame({"g": [1, 2, 2][:n_rows]}, schema={"g": pl.Int64})
        assert ("> 0" in lf.sql(query).explain()) == has_check
        # Over no rows, a GROUP BY on constant keys gives a row in Polars.
        if n_rows or group_by != "GROUP BY 1 + 1":
            assert_sql_matches(
                lf,
                query=query,
                compare_with="duckdb",
                check_row_order=False,
                engines=["in-memory", "streaming"],
            )


def test_sum_of_constant_types() -> None:
    # The sum of a constant has the dtype of SUM over a column of it.
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_sql_matches(
        df,
        query="""
            SELECT
              SUM(CAST(20000 AS SMALLINT)) AS a,
              SUM(DISTINCT TRUE) AS b,
              SUM(CAST(1.5 AS REAL)) AS c
            FROM self
        """,
        compare_with=None,
        check_dtypes=True,
        expected=pl.DataFrame(
            {"a": [60000], "b": [1], "c": [4.5]},
            schema={"a": pl.Int64, "b": get_index_type(), "c": pl.Float32},
        ),
        engines=["in-memory", "streaming"],
    )
    with pytest.raises(InvalidOperationError, match="`sum` operation not supported"):
        df.sql("SELECT SUM(DISTINCT 'a') AS a FROM self")


def test_spread_of_non_finite_constant() -> None:
    # Over two or more rows, the spread of a constant is that of a column of it.
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_sql_matches(
        df,
        query="""
            SELECT
              STDDEV(CAST('NaN' AS DOUBLE)) AS a,
              VARIANCE(CAST('Infinity' AS DOUBLE)) AS b
            FROM self
        """,
        compare_with=None,
        expected={"a": [float("nan")], "b": [float("nan")]},
        engines=["in-memory", "streaming"],
    )


def test_sum_of_integer_literal() -> None:
    # Integer literals are summed as Int64.
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert_sql_matches(
        df,
        query="""
            SELECT
              SUM(-1000000000) AS s,
              SUM(1000000000) FILTER (WHERE x > 1) AS f,
              TOTAL(1000000000) AS t
            FROM self
        """,
        compare_with=None,
        check_dtypes=True,
        expected=pl.DataFrame({"s": [-3000000000], "f": [2000000000], "t": [3e9]}),
        engines=["in-memory", "streaming"],
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

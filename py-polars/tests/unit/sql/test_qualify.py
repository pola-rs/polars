from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "category": ["A", "A", "A", "B", "B", "B"],
            "value": [100, 200, 150, 300, 250, 400],
        }
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category)",
            id="above_avg",
        ),
        pytest.param(
            "value = MAX(value) OVER (PARTITION BY category)",
            id="equals_max",
        ),
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category) AND value < 500",
            id="compound_expr",
        ),
    ],
)
def test_qualify_constraints(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY category, value
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
        },
    )


def test_qualify_distinct() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [100, 100, 200, 200, 300, 300],
        }
    )
    assert_sql_matches(
        {"df": df},
        query="""
            SELECT DISTINCT category, value
            FROM df
            QUALIFY value = MAX(value) OVER (PARTITION BY category)
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "category": ["A", "B", "C"],
            "value": [100, 200, 300],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "400 < SUM(value) OVER (PARTITION BY category)",
            id="sum_window",
        ),
        pytest.param(
            "COUNT(*) OVER (PARTITION BY category) = 3",
            id="count_window",
        ),
    ],
)
def test_qualify_matches_all_rows(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY id DESC
        """,
        compare_with="duckdb",
        expected={
            "id": [6, 5, 4, 3, 2, 1],
            "category": ["B", "B", "B", "A", "A", "A"],
            "value": [400, 250, 300, 150, 200, 100],
        },
    )


def test_qualify_multiple_clauses(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df
            QUALIFY
              value >= 300
              AND SUM(value) OVER (PARTITION BY category) > 500
            ORDER BY value
        """,
        compare_with="duckdb",
        expected={
            "id": [4, 6],
            "category": ["B", "B"],
            "value": [300, 400],
        },
    )
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df
            QUALIFY
              value = MAX(value) OVER (PARTITION BY category)
              OR value = MIN(value) OVER (PARTITION BY category)
            ORDER BY id
        """,
        compare_with="duckdb",
        expected={
            "id": [1, 2, 5, 6],
            "category": ["A", "A", "B", "B"],
            "value": [100, 200, 250, 400],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > MAX(value) OVER (PARTITION BY category)",
            id="greater_than_max",
        ),
        pytest.param(
            "value < MIN(value) OVER (PARTITION BY category)",
            id="less_than_min",
        ),
    ],
)
def test_qualify_returns_no_rows(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT id, category, value
            FROM df QUALIFY {qualify_clause}
        """,
        compare_with="duckdb",
        expected={"id": [], "category": [], "value": []},
    )


def test_qualify_using_select_alias(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT
              id,
              category,
              value,
              MAX(value) OVER (PARTITION BY category) as max_value
            FROM df
            QUALIFY value = max_value
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
            "max_value": [200, 400],
        },
    )


@pytest.mark.parametrize(
    "qualify_clause",
    [
        pytest.param(
            "value > avg_value AND COUNT(*) OVER (PARTITION BY category) = 3",
            id="mixed_alias_and_explicit",
        ),
        pytest.param(
            "value > AVG(value) OVER (PARTITION BY category)",
            id="window_in_select",
        ),
    ],
)
def test_qualify_miscellaneous(df_test: pl.DataFrame, qualify_clause: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=f"""
            SELECT
              id,
              category,
              value,
              AVG(value) OVER (PARTITION BY category) as avg_value
            FROM df
            QUALIFY {qualify_clause}
            ORDER BY category
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 6],
            "category": ["A", "B"],
            "value": [200, 400],
            "avg_value": [150.0, 316.6666666666667],
        },
    )


def test_qualify_with_internal_cumulative_sum() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 3, 4, 2, 5],
            "value": [10, 30, 40, 20, 50],
        }
    )
    assert_sql_matches(
        {"df": df},
        query="""
            SELECT id, value
            FROM df
            QUALIFY SUM(value) OVER (ORDER BY id) <= 60
            ORDER BY id
        """,
        compare_with="duckdb",
        expected={
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        },
    )


def test_qualify_with_alias_and_comparison(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, SUM(value) OVER (PARTITION BY category) as total
            FROM df QUALIFY total > 500
            ORDER BY id DESC
        """,
        compare_with="duckdb",
        expected={
            "id": [6, 5, 4],
            "total": [950, 950, 950],
        },
    )


def test_qualify_with_where_clause(df_test: pl.DataFrame) -> None:
    assert_sql_matches(
        {"df": df_test},
        query="""
            SELECT id, category, value
            FROM df WHERE value > 200
            QUALIFY value != MAX(value) OVER (PARTITION BY category)
            ORDER BY value
        """,
        compare_with="duckdb",
        expected={
            "id": [5, 4],
            "category": ["B", "B"],
            "value": [250, 300],
        },
    )


def test_qualify_expected_errors(df_test: pl.DataFrame) -> None:
    ctx = pl.SQLContext(df=df_test, eager=True)
    with pytest.raises(
        SQLSyntaxError,
        match="QUALIFY clause must reference window functions",
    ):
        ctx.execute("SELECT id, category, value FROM df QUALIFY value > 200")


def test_qualify_with_subquery_does_not_leak_placeholder() -> None:
    frames = {
        "t1": pl.DataFrame({"k": [1, 2, 3]}),
        "t2": pl.DataFrame({"k": [1, 1, 2]}),
    }
    res = pl.SQLContext(frames=frames, eager=True).execute(
        "SELECT k, ROW_NUMBER() OVER (ORDER BY k) AS rn FROM t1 "
        "QUALIFY rn > (SELECT MIN(k) FROM t2)"
    )
    assert res.columns == ["k", "rn"]
    assert res["k"].to_list() == [2, 3]


@pytest.mark.parametrize(
    "query",
    [
        # columns that are not selected
        """
        SELECT id FROM df
        QUALIFY ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) = 1
        ORDER BY id
        """,
        "SELECT id FROM df QUALIFY value = MAX(value) OVER (PARTITION BY category)",
        # an input column comes before a SELECT alias of the same name
        "SELECT value AS id, id AS value FROM df QUALIFY id = MAX(id) OVER ()",
        # windows over grouped rows
        """
        SELECT category, COUNT(*) AS n FROM df GROUP BY category
        QUALIFY RANK() OVER (ORDER BY SUM(value) DESC) = 1
        """,
        """
        SELECT category FROM df GROUP BY category
        QUALIFY SUM(SUM(value)) OVER (ORDER BY category) > 500
        ORDER BY category
        """,
        "SELECT COUNT(*) AS n FROM df QUALIFY ROW_NUMBER() OVER () = 1",
        # aliases of aggregates in a window
        """
        SELECT category, COUNT(*) AS n FROM df GROUP BY category
        QUALIFY LAG(n) OVER (ORDER BY category) > 2
        """,
        """
        SELECT category, SUM(value) AS s FROM df GROUP BY category
        QUALIFY RANK() OVER (ORDER BY s DESC) = 1
        """,
        """
        SELECT category, SUM(value) AS s FROM df GROUP BY category
        QUALIFY s * 2 > SUM(s) OVER ()
        """,
        # a renamed column
        """
        SELECT * RENAME (value AS v) FROM df
        QUALIFY ROW_NUMBER() OVER (PARTITION BY category ORDER BY v) = 1
        ORDER BY id
        """,
    ],
)
def test_qualify_before_projection(df_test: pl.DataFrame, query: str) -> None:
    assert_sql_matches(
        {"df": df_test},
        query=query,
        compare_with="duckdb",
        engines=["in-memory", "streaming"],
    )


def test_qualify_column_named_like_internal_column() -> None:
    df = pl.DataFrame({"g": [1, 1, 2], "__POLARS_QUALIFY": [7, 8, 9]})
    for query in [
        'SELECT "__POLARS_QUALIFY", g FROM self QUALIFY ROW_NUMBER() OVER (ORDER BY g) = 1',
        """
        SELECT g, MAX("__POLARS_QUALIFY") AS "__POLARS_QUALIFY" FROM self GROUP BY g
        QUALIFY RANK() OVER (ORDER BY g) = 1
        """,
    ]:
        res = df.sql(query)
        assert sorted(res.columns) == ["__POLARS_QUALIFY", "g"]
        assert res.height == 1


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, s AS v FROM t QUALIFY v = 1 AND ROW_NUMBER() OVER (ORDER BY id) > 0",
        "SELECT id, s AS v FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY v = 1 ORDER BY id) = 1",
        """
        SELECT g, MAX(s) AS m FROM t GROUP BY g
        QUALIFY m = 1 AND RANK() OVER (ORDER BY g) > 0
        """,
    ],
)
def test_qualify_alias_types(query: str) -> None:
    # The type of an alias is known where QUALIFY compares it with a literal.
    df = pl.DataFrame({"id": [1, 2, 3], "s": ["1", "2", "1"], "g": [1, 1, 2]})
    assert_sql_matches(
        {"t": df},
        query=f"{query} ORDER BY 1",
        compare_with="duckdb",
        engines=["in-memory", "streaming"],
    )


def test_qualify_renamed_column_type() -> None:
    df = pl.DataFrame({"id": [1, 2, 3], "s": ["1", "2", "1"]})
    res = df.sql(
        """
        SELECT * RENAME (s AS v) FROM self
        QUALIFY v = 1 AND ROW_NUMBER() OVER (ORDER BY id) > 0
        ORDER BY id
        """
    )
    assert res.to_dict(as_series=False) == {"id": [1, 3], "v": ["1", "1"]}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, (SELECT 2) AS a FROM t QUALIFY a = 2 AND ROW_NUMBER() OVER (ORDER BY id) > 0",
        "SELECT id, (SELECT MAX(s) FROM t) AS a FROM t QUALIFY a = 2 AND ROW_NUMBER() OVER (ORDER BY id) > 0",
        "SELECT id, (SELECT 2) AS a FROM t QUALIFY SUM(a) OVER () = 6",
        "SELECT id, (SELECT 2) AS a FROM t QUALIFY LAG(a) OVER (ORDER BY id) = 2",
        """
        SELECT g, (SELECT 2) AS a, COUNT(*) AS n FROM t GROUP BY g
        QUALIFY SUM(a * n) OVER () = 6
        """,
        """
        SELECT g, COUNT(*) + (SELECT 2) AS a FROM t GROUP BY g
        QUALIFY SUM(a) OVER () > 6
        """,
        """
        SELECT g, MAX(g + ARRAY_LENGTH(ARRAY_REVERSE((SELECT ARRAY[1, 2])))) AS a
        FROM t GROUP BY g
        QUALIFY SUM(a) OVER () > 6
        """,
    ],
)
def test_qualify_subquery_alias(query: str) -> None:
    df = pl.DataFrame({"id": [1, 2, 3], "g": [1, 1, 2], "s": ["1", "2", "1"]})
    assert_sql_matches(
        {"t": df},
        query=f"{query} ORDER BY 1",
        compare_with="duckdb",
        engines=["in-memory", "streaming"],
    )

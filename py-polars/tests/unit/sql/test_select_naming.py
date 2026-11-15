from __future__ import annotations

import pytest

import polars as pl
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_naming() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 5],
            "c": [100, 200, 8],
        }
    )


def test_binary_expr_collides_with_bare_column(df_naming: pl.DataFrame) -> None:
    """`a-b` derives the output name "a", colliding with the bare column `a`."""
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a-b, a FROM t1 ORDER BY a-b")

    # first occurrence keeps the natural name; later collisions get a
    # deterministic `name:n` suffix (pinning the disambiguation scheme)
    assert out.columns == ["a", "a:1"]
    assert out["a"].to_list() == [-18, -9, -2]
    assert out["a:1"].to_list() == [2, 1, 3]


def test_function_call_collides_with_bare_column(df_naming: pl.DataFrame) -> None:
    """`abs(a)` derives the output name "a", colliding with the bare column `a`."""
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a, abs(a) FROM t1 ORDER BY a")

    assert out.columns == ["a", "a:1"]
    assert out["a"].to_list() == [1, 2, 3]
    assert out["a:1"].to_list() == [1, 2, 3]


def test_two_derived_expressions_collide(df_naming: pl.DataFrame) -> None:
    """`a-b` and `a-c` both derive the output name "a"."""
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a-b, a-c FROM t1 ORDER BY a-b")

    assert out.columns == ["a", "a:1"]


def test_explicit_alias_is_never_renamed(df_naming: pl.DataFrame) -> None:
    """An explicit `AS` alias always wins, even if declared after the collision."""
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a-b, 5 AS a FROM t1 ORDER BY a")

    assert out.columns == ["a:1", "a"]
    assert out["a"].to_list() == [5, 5, 5]


def test_explicit_duplicate_alias_still_errors(df_naming: pl.DataFrame) -> None:
    """Two explicit aliases sharing a name remain a genuine ambiguity."""
    with (
        pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx,
        pytest.raises(pl.exceptions.ComputeError),
    ):
        ctx.execute("SELECT 1 AS x, 2 AS x FROM t1")


def test_verbatim_duplicate_selection_still_errors(df_naming: pl.DataFrame) -> None:
    """Re-selecting the exact same unaliased column twice stays an error.

    This is a genuine duplicate column selection, not an incidental name clash.
    """
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        with pytest.raises(pl.exceptions.DuplicateError):
            ctx.execute("SELECT a, a FROM t1")
        with pytest.raises(pl.exceptions.DuplicateError):
            ctx.execute("SELECT *, a FROM t1")


def test_repeated_constants_are_renamed(df_naming: pl.DataFrame) -> None:
    """A repeated unaliased constant gets a new name, like distinct constants."""
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a, NULL, NULL, 1, 1 FROM t1 ORDER BY a")
        assert out.columns == ["a", "literal", "literal:1", "literal:2", "literal:3"]
        assert out.row(0) == (1, None, None, 1, 1)

        out = ctx.execute(
            "SELECT a, NULL, SUM(b) AS s, NULL FROM t1 GROUP BY a ORDER BY a"
        )
        assert out.columns == ["a", "literal", "s", "literal:1"]
        assert out.rows() == [
            (1, None, 10, None),
            (2, None, 20, None),
            (3, None, 5, None),
        ]

        out = ctx.execute("SELECT NULL, SUM(b) AS s, NULL FROM t1")
        assert out.rows() == [(None, 35, None)]


def test_no_collision_output_names_are_unchanged(df_naming: pl.DataFrame) -> None:
    """Queries without a naming collision keep their pre-existing output names."""
    assert_sql_matches(
        df_naming,
        query="SELECT a, b, a + b AS c2 FROM self ORDER BY a",
        compare_with="duckdb",
        expected={"a": [1, 2, 3], "b": [10, 20, 5], "c2": [11, 22, 8]},
    )


def test_collision_with_where_and_group_by(df_naming: pl.DataFrame) -> None:
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT a-b, a FROM t1 WHERE a > 1 ORDER BY a")
        assert out.columns == ["a", "a:1"]
        assert out["a:1"].to_list() == [2, 3]

        out = ctx.execute("SELECT a, abs(a) FROM t1 GROUP BY a ORDER BY a")
        assert out.columns == ["a", "a:1"]
        assert out["a:1"].to_list() == [1, 2, 3]


def test_collision_with_distinct(df_naming: pl.DataFrame) -> None:
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT DISTINCT a-b, a FROM t1 ORDER BY a-b")

    assert out.columns == ["a", "a:1"]
    assert out.height == 3


def test_collision_with_wildcard(df_naming: pl.DataFrame) -> None:
    with pl.SQLContext(frames={"t1": df_naming}, eager=True) as ctx:
        out = ctx.execute("SELECT *, a-b FROM t1 ORDER BY a")

    assert out.columns == ["a", "b", "c", "a:1"]


def test_unaliased_scalar_subquery_name_is_stable() -> None:
    """An unnamed scalar subquery gets Polars' usual name for unnamed expressions.

    It previously took an internal name that differed on every execution.
    """
    frames = {"t": pl.DataFrame({"v": [1, 2, 3]})}
    query = "SELECT (SELECT MAX(v) FROM t) FROM t"

    names = [
        pl.SQLContext(frames=frames, eager=True).execute(query).columns
        for _ in range(3)
    ]
    assert names == [["literal"], ["literal"], ["literal"]]

    with pl.SQLContext(frames=frames, eager=True) as ctx:
        # collisions disambiguate like any other repeated derived name
        assert ctx.execute(
            "SELECT (SELECT MAX(v) FROM t), (SELECT MIN(v) FROM t) FROM t"
        ).columns == ["literal", "literal:1"]
        # an explicit alias still wins
        assert ctx.execute("SELECT (SELECT MAX(v) FROM t) AS mx FROM t").columns == [
            "mx"
        ]


@pytest.mark.parametrize(
    "query",
    [
        "SELECT a * 2 AS x, x + 1 AS y, y * x AS z FROM self ORDER BY a",
        # an input column comes before an alias of the same name
        "SELECT a * 2 AS b, b + 1 AS y FROM self ORDER BY a",
        "SELECT a AS x, x AS a, a + 1 AS c FROM self ORDER BY x",
        "SELECT a * 2 AS x, CASE WHEN x > 3 THEN 'big' END AS size FROM self ORDER BY a",
        "SELECT MAX(a) AS m, MIN(a) AS n, m - n AS spread FROM self",
        "SELECT g, MAX(a) AS m, MIN(a) AS n, m - n AS d FROM self GROUP BY g HAVING d > 0",
        "SELECT g AS k, k || '!' AS k2, COUNT(*) AS n FROM self GROUP BY k ORDER BY k",
        "SELECT a % 2 AS x, x * 10 AS y, COUNT(*) AS n FROM self GROUP BY x ORDER BY x",
        "SELECT g AS p, MAX(b) OVER (PARTITION BY p) AS m FROM self ORDER BY a",
        "SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn, rn * 10 AS r10 FROM self "
        "QUALIFY r10 > 10 ORDER BY a",
        "SELECT DISTINCT a % 2 AS x, x + 1 AS y FROM self ORDER BY x",
        "SELECT 2 AS x, SUM(x) AS s, COUNT(x) AS n FROM self",
        "SELECT g, 2 AS x, SUM(x) AS s FROM self GROUP BY g ORDER BY g",
    ],
)
def test_select_alias_reuse(query: str) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 5], "g": ["x", "x", "y"]})
    assert_sql_matches(df, query=query, compare_with="duckdb")


def test_select_alias_reuse_random() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    res = df.sql("SELECT RANDOM() AS r, r AS r2, a * 2 AS x, x FROM self")
    assert res.columns == ["r", "r2", "x", "x:1"]
    assert res["r"].to_list() == res["r2"].to_list()
    assert res["x:1"].to_list() == [2, 4, 6]


def test_select_alias_reuse_types() -> None:
    df = pl.DataFrame({"a": [1, 2], "s": [{"b": 1, "c": "p"}, {"b": 2, "c": "q"}]})
    res = df.sql(
        """
        SELECT
          s AS x, x.b AS xb, x.*,
          (SELECT 1) AS one, TYPEOF(one) AS t1, TO_TIMESTAMP(one) AS ts,
          SUM(a) OVER () AS w, TYPEOF(w) AS tw
        FROM self
        """
    )
    assert res.select("xb", "b", "c", "t1", "w", "tw").rows() == [
        (1, 1, "p", "i32", 3, "i64"),
        (2, 2, "q", "i32", 3, "i64"),
    ]
    assert res["ts"].dtype == pl.Datetime("us", "UTC")


def test_select_alias_reuse_errors() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    # an alias is read by the items after it only
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='"x"'):
        df.sql("SELECT x + 1 AS y, a * 2 AS x FROM self")
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='"x"'):
        df.sql("SELECT a * 2 AS x FROM self WHERE x > 2")

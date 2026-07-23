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


def test_no_collision_output_names_are_unchanged(df_naming: pl.DataFrame) -> None:
    """Queries without a naming collision keep their pre-existing output names."""
    assert_sql_matches(
        df_naming,
        query="SELECT a, b, a + b AS c2 FROM self ORDER BY a",
        compare_with="sqlite",
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

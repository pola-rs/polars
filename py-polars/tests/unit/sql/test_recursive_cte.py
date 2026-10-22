from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError


@pytest.fixture
def ctx() -> pl.SQLContext[pl.LazyFrame]:
    return pl.SQLContext(a=pl.DataFrame({"x": [1, 2, 3]}))


def test_recursive_with_without_self_reference(
    ctx: pl.SQLContext[pl.LazyFrame],
) -> None:
    res = ctx.execute(
        "WITH RECURSIVE c AS (SELECT x FROM a) SELECT * FROM c ORDER BY x", eager=True
    )
    assert res["x"].to_list() == [1, 2, 3]


def test_recursive_with_forward_reference(ctx: pl.SQLContext[pl.LazyFrame]) -> None:
    # Under RECURSIVE a CTE may read a CTE defined after it.
    res = ctx.execute(
        """
        WITH RECURSIVE
          b AS (SELECT x + 1 AS y FROM C),
          c AS (SELECT x FROM a)
        SELECT * FROM b ORDER BY y
        """,
        eager=True,
    )
    assert res["y"].to_list() == [2, 3, 4]


def test_recursive_with_forward_reference_in_subquery(
    ctx: pl.SQLContext[pl.LazyFrame],
) -> None:
    res = ctx.execute(
        """
        WITH RECURSIVE
          b AS (SELECT x FROM a WHERE x > (SELECT MIN(x) FROM c)),
          c AS (SELECT x FROM a)
        SELECT * FROM b ORDER BY x
        """,
        eager=True,
    )
    assert res["x"].to_list() == [2, 3]


def test_recursive_with_nested_with_scopes(ctx: pl.SQLContext[pl.LazyFrame]) -> None:
    # The inner `b` hides the outer one, so `b` doesn't read itself.
    res = ctx.execute(
        """
        WITH RECURSIVE b AS (WITH b AS (SELECT x FROM a) SELECT * FROM b)
        SELECT * FROM b ORDER BY x
        """,
        eager=True,
    )
    assert res["x"].to_list() == [1, 2, 3]

    # The inner `c` is defined after `z`, so `z` reads the outer `c`.
    res = ctx.execute(
        """
        WITH RECURSIVE
          b AS (
            WITH z AS (SELECT * FROM c), c AS (SELECT 0 AS x)
            SELECT * FROM z
          ),
          c AS (SELECT x FROM a)
        SELECT * FROM b ORDER BY x
        """,
        eager=True,
    )
    assert res["x"].to_list() == [1, 2, 3]

    # The inner `b` is also visible to the ORDER BY wrapping its query.
    res = ctx.execute(
        """
        WITH RECURSIVE b AS (
          (WITH b AS (SELECT 1 AS x) SELECT * FROM b) ORDER BY (SELECT x FROM b)
        )
        SELECT * FROM b
        """,
        eager=True,
    )
    assert res["x"].to_list() == [1]


@pytest.mark.parametrize(
    "query",
    [
        "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) "
        "SELECT * FROM t",
        "WITH RECURSIVE t AS (SELECT x FROM a WHERE x IN (SELECT x FROM T)) "
        "SELECT * FROM t",
    ],
)
def test_recursive_with_self_reference_unsupported(
    ctx: pl.SQLContext[pl.LazyFrame], query: str
) -> None:
    with pytest.raises(SQLInterfaceError, match="CTE 't' reads itself"):
        ctx.execute(query, eager=True)


@pytest.mark.parametrize(
    ("query", "name"),
    [
        (
            "WITH RECURSIVE b AS (SELECT * FROM c), c AS (SELECT 1 AS x), "
            "C AS (SELECT 2 AS x) SELECT * FROM b",
            "C",
        ),
        (
            "WITH RECURSIVE b AS (SELECT 1 AS x), b AS (SELECT x + 1 AS x FROM b) "
            "SELECT * FROM b",
            "b",
        ),
    ],
)
def test_recursive_with_duplicate_names(
    ctx: pl.SQLContext[pl.LazyFrame], query: str, name: str
) -> None:
    with pytest.raises(
        SQLInterfaceError, match=f"CTE name '{name}' is used more than once"
    ):
        ctx.execute(query, eager=True)


def test_recursive_with_cycle_unsupported(ctx: pl.SQLContext[pl.LazyFrame]) -> None:
    with pytest.raises(SQLInterfaceError, match=r"mutually recursive CTEs .*\(b, c\)"):
        ctx.execute(
            """
            WITH RECURSIVE
              b AS (SELECT * FROM c),
              c AS (SELECT * FROM b),
              d AS (SELECT x FROM a)
            SELECT * FROM d
            """,
            eager=True,
        )

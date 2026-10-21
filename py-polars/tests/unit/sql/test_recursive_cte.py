from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError

if TYPE_CHECKING:
    from polars._typing import EngineType


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


@pytest.mark.parametrize("engine", ["auto", "in-memory", "streaming"])
def test_recursive_count(ctx: pl.SQLContext[pl.LazyFrame], engine: EngineType) -> None:
    res = ctx.execute(
        """
        WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 5)
        SELECT * FROM t ORDER BY n
        """
    ).collect(engine=engine)
    assert res["n"].to_list() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Walk a hierarchy with a join.
        (
            """
            WITH RECURSIVE h AS (
              SELECT id, 0 AS lvl FROM emp WHERE boss IS NULL
              UNION ALL
              SELECT e.id, h.lvl + 1 FROM emp e JOIN h ON e.boss = h.id
            )
            SELECT * FROM h ORDER BY id
            """,
            [(1, 0), (2, 1), (3, 1), (4, 2), (5, 3)],
        ),
        # An outer join in the step.
        (
            """
            WITH RECURSIVE h AS (
              SELECT id, boss FROM emp WHERE id = 5
              UNION ALL
              SELECT e.id, e.boss FROM h LEFT JOIN emp e ON e.id = h.boss
              WHERE h.boss IS NOT NULL
            )
            SELECT * FROM h ORDER BY id
            """,
            [(1, None), (2, 1), (4, 2), (5, 4)],
        ),
        # A column list.
        (
            """
            WITH RECURSIVE t(a, b) AS (
              SELECT 1, 10 UNION ALL SELECT a + 1, b * 2 FROM t WHERE a < 3
            )
            SELECT * FROM t ORDER BY a
            """,
            [(1, 10), (2, 20), (3, 40)],
        ),
        # An empty anchor: the step never runs.
        (
            """
            WITH RECURSIVE t AS (
              SELECT id FROM emp WHERE id > 10 UNION ALL SELECT id + 1 FROM t
            )
            SELECT count(*) AS n FROM t
            """,
            [(0,)],
        ),
        # An aggregate in the step.
        (
            """
            WITH RECURSIVE t AS (
              SELECT id FROM emp WHERE id < 3
              UNION ALL
              SELECT max(id) + 1 FROM t HAVING max(id) < 4
            )
            SELECT * FROM t ORDER BY id
            """,
            [(1,), (2,), (3,), (4,)],
        ),
        # A NULL anchor column takes the type of the step, also through another column.
        (
            """
            WITH RECURSIVE t AS (
              SELECT NULL AS a, NULL AS b, 0 AS n
              UNION ALL
              SELECT b, 'x', n + 1 FROM t WHERE n < 2
            )
            SELECT * FROM t ORDER BY n
            """,
            [(None, None, 0), (None, "x", 1), ("x", "x", 2)],
        ),
        # Both terms are cast to the types of the anchor.
        (
            """
            WITH RECURSIVE t AS (
              SELECT CAST(1 AS SMALLINT) AS n
              UNION ALL
              SELECT CAST(n AS BIGINT) * 2 FROM t WHERE n < 4
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1,), (2,), (4,)],
        ),
        # A CTE defined in the step and read twice is computed in every round.
        (
            """
            WITH RECURSIVE t AS (
              SELECT 1 AS n
              UNION ALL
              (
                WITH s AS (SELECT n + 1 AS m FROM t)
                SELECT s1.m FROM s s1 JOIN s s2 ON s1.m = s2.m WHERE s1.m <= 3
              )
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1,), (2,), (3,)],
        ),
        # GROUPING SETS over the rows of the round before.
        (
            """
            WITH RECURSIVE t AS (
              SELECT 1 AS n, 1 AS k
              UNION ALL
              SELECT max(n) + 1, k FROM t WHERE n < 3 GROUP BY GROUPING SETS ((k))
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1, 1), (2, 1), (3, 1)],
        ),
        # A union in the step whose second part only gives rows in the first round.
        (
            """
            WITH RECURSIVE t AS (
              SELECT 1 AS n
              UNION ALL
              SELECT * FROM (
                SELECT n + 1 AS n FROM t WHERE n < 3
                UNION ALL
                SELECT 100 AS n FROM t WHERE n = 1
              )
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1,), (2,), (3,), (100,)],
        ),
        # Two reads of the rows of the round before.
        (
            """
            WITH RECURSIVE t AS (
              SELECT 1 AS n UNION ALL SELECT t1.n + t2.n FROM t t1, t t2 WHERE t1.n < 8
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1,), (2,), (4,), (8,)],
        ),
        # A top-k in the step must not keep its bound between rounds.
        (
            """
            WITH RECURSIVE t AS (
              SELECT id AS n FROM emp WHERE id IN (1, 5)
              UNION ALL
              SELECT n + 10 FROM (SELECT n FROM t ORDER BY n LIMIT 1) WHERE n < 30
            )
            SELECT * FROM t ORDER BY n
            """,
            [(1,), (5,), (11,), (21,), (31,)],
        ),
        # A recursive CTE read twice, and one read in the step of another.
        (
            """
            WITH RECURSIVE
              a AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM a WHERE n < 3),
              b AS (SELECT 0 AS m UNION ALL SELECT b.m + 1 FROM b JOIN a ON a.n = b.m + 1)
            SELECT b.m, a.n FROM b LEFT JOIN a ON a.n = b.m ORDER BY m
            """,
            [(0, None), (1, 1), (2, 2), (3, 3)],
        ),
    ],
)
@pytest.mark.parametrize("engine", ["auto", "in-memory"])
def test_recursive_queries(
    query: str, expected: list[tuple[Any, ...]], engine: EngineType
) -> None:
    emp = pl.DataFrame({"id": [1, 2, 3, 4, 5], "boss": [None, 1, 1, 2, 4]})
    with pl.SQLContext(emp=emp) as ctx:
        assert ctx.execute(query).collect(engine=engine).rows() == expected


def test_recursive_shared_cte_computed_once() -> None:
    # `base` is read in the anchor, in the step and outside: it must have the same
    # values everywhere.
    src = pl.LazyFrame().select(i=pl.int_range(5), x=pl.int_range(5).shuffle())
    with pl.SQLContext(src=src) as ctx:
        res = ctx.execute(
            """
            WITH RECURSIVE
              base AS (SELECT i, x FROM src),
              t AS (
                SELECT i, x FROM base WHERE i = 0
                UNION ALL
                SELECT b.i, b.x FROM t JOIN base b ON b.i = t.i + 1
              )
            SELECT count(*) AS n, sum(CASE WHEN t.x = b.x THEN 1 ELSE 0 END) AS same
            FROM t JOIN base b ON b.i = t.i
            """,
            eager=True,
        )
    assert res.rows() == [(5, 5)]


def test_recursive_in_memory_engine_uses_streaming_optimizations(
    ctx: pl.SQLContext[pl.LazyFrame],
) -> None:
    res = ctx.execute(
        """
        WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3)
        SELECT SUM(n) AS s, SUM(n) + 1 AS s1 FROM t
        """
    ).collect(engine="in-memory")
    assert res.rows() == [(6, 7)]


def test_recursive_collect_all(ctx: pl.SQLContext[pl.LazyFrame]) -> None:
    lf = ctx.execute(
        "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) "
        "SELECT * FROM t"
    )
    other = pl.LazyFrame({"y": [1, 2]}).select(pl.col("y").sum())
    a, b = pl.collect_all([lf.sort("n"), other], engine="in-memory")
    assert a["n"].to_list() == [1, 2, 3]
    assert b["y"].to_list() == [3]


def test_recursive_explain_and_serialize(ctx: pl.SQLContext[pl.LazyFrame]) -> None:
    lf = ctx.execute(
        "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) "
        "SELECT * FROM t"
    )
    plan = lf.explain()
    assert "RECURSIVE" in plan
    assert "WORKING TABLE" in plan

    lf2 = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))
    assert lf2.sort("n").collect()["n"].to_list() == [1, 2, 3]


@pytest.mark.parametrize(
    ("query", "match"),
    [
        (
            "WITH RECURSIVE t AS (SELECT x FROM a WHERE x IN (SELECT x FROM T)) "
            "SELECT * FROM t",
            "must have the form `anchor UNION ALL step`",
        ),
        (
            "WITH RECURSIVE t AS (SELECT 1 AS n UNION SELECT n + 1 FROM t WHERE n < 3) "
            "SELECT * FROM t",
            "only supports UNION ALL",
        ),
        (
            "WITH RECURSIVE t AS (SELECT * FROM t UNION ALL SELECT 1 AS n) "
            "SELECT * FROM t",
            "the anchor of recursive CTE 't' may not read 't'",
        ),
        (
            "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n, n AS m FROM t WHERE n < 3) "
            "SELECT * FROM t",
            "different number of columns",
        ),
    ],
)
def test_recursive_errors(
    ctx: pl.SQLContext[pl.LazyFrame], query: str, match: str
) -> None:
    with pytest.raises(SQLInterfaceError, match=match):
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

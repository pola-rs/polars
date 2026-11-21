from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError


@pytest.fixture
def ctx() -> pl.SQLContext[pl.DataFrame]:
    t = pl.DataFrame({"x": range(2000), "g": [i % 3 for i in range(2000)]})
    u = pl.DataFrame({"x": range(10)})
    return pl.SQLContext(t=t, u=u, eager=True)


def count(ctx: pl.SQLContext[pl.DataFrame], query: str) -> int:
    n: int = ctx.execute(query).item()
    return n


@pytest.mark.parametrize(
    "sample",
    [
        "TABLESAMPLE (10 PERCENT)",
        "TABLESAMPLE 10%",
        "TABLESAMPLE BERNOULLI (10)",
        "TABLESAMPLE SYSTEM (10 PERCENT)",
        "TABLESAMPLE reservoir(10%)",
        "AS a TABLESAMPLE 10 PERCENT",
        "USING SAMPLE 10%",
        "USING SAMPLE 10 PERCENT (bernoulli)",
        "USING SAMPLE bernoulli(10%)",
    ],
)
def test_tablesample_forms(ctx: pl.SQLContext[pl.DataFrame], sample: str) -> None:
    n = count(ctx, f"SELECT COUNT(*) FROM t {sample}")
    assert 100 < n < 300


def test_tablesample_seed(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    query = "SELECT x FROM t TABLESAMPLE (20 PERCENT) REPEATABLE (42) ORDER BY x"
    first = ctx.execute(query)
    assert 200 < first.height < 600
    assert first.equals(ctx.execute(query))
    other = ctx.execute("SELECT x FROM t USING SAMPLE 20% (bernoulli, 42) ORDER BY x")
    assert first.equals(other)
    other = ctx.execute("SELECT x FROM t USING SAMPLE 20% REPEATABLE (43) ORDER BY x")
    assert not first.equals(other)

    assert count(ctx, "SELECT COUNT(*) FROM t TABLESAMPLE (0 PERCENT)") == 0
    assert count(ctx, "SELECT COUNT(*) FROM t TABLESAMPLE (100 PERCENT)") == 2000


def test_tablesample_in_queries(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    # the sample is taken before WHERE and GROUP BY
    res = ctx.execute(
        """
        SELECT g, COUNT(*) AS n FROM t WHERE x % 2 = 0
        GROUP BY g USING SAMPLE 100% ORDER BY g
        """
    )
    assert res.rows() == [(0, 334), (1, 333), (2, 333)]
    assert (
        count(
            ctx, "SELECT COUNT(*) FROM (SELECT * FROM t) AS s TABLESAMPLE (0 PERCENT)"
        )
        == 0
    )
    assert (
        count(
            ctx,
            """
        SELECT COUNT(*) FROM t TABLESAMPLE (100 PERCENT)
        JOIN u TABLESAMPLE (0 PERCENT) ON t.x = u.x
        """,
        )
        == 0
    )
    res = ctx.execute(
        """
        SELECT (SELECT COUNT(*) FROM u USING SAMPLE 100%) AS a,
          x % 3 AS m
        FROM u ORDER BY x LIMIT 2
        """
    )
    assert res.rows() == [(10, 0), (10, 1)]
    assert ctx.execute(
        """
        SELECT COUNT(*) FROM t USING SAMPLE 0%
        UNION ALL SELECT COUNT(*) FROM u USING SAMPLE 100%
        """
    ).rows() == [(0,), (10,)]


def test_tablesample_correlated_subquery() -> None:
    # a subquery over a sampled table must not reuse the unsampled outer rows
    ctx = pl.SQLContext(
        t=pl.DataFrame({"x": [1, 2, 3]}), u=pl.DataFrame({"y": [1, 2, 3]}), eager=True
    )
    query = """
        SELECT x, y FROM t, u
        WHERE x = y AND y = (SELECT MIN(y) FROM u TABLESAMPLE (0 PERCENT) WHERE y = t.x)
    """
    assert ctx.execute(query).rows() == []


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ("SELECT * FROM t TABLESAMPLE (10 ROWS)", "number of rows is not supported"),
        ("SELECT * FROM t USING SAMPLE 10", "number of rows is not supported"),
        (
            "SELECT * FROM t TABLESAMPLE reservoir(10)",
            "number of rows is not supported",
        ),
        ("SELECT * FROM t TABLESAMPLE (150 PERCENT)", "between 0 and 100"),
        ("SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 4)", "only a percentage"),
        ("SELECT * FROM t, u USING SAMPLE 10%", "one table in FROM"),
        ("SELECT * FROM t USING SAMPLE nope", "invalid `USING SAMPLE` clause"),
    ],
)
def test_tablesample_errors(
    ctx: pl.SQLContext[pl.DataFrame], query: str, match: str
) -> None:
    with pytest.raises(SQLInterfaceError, match=match):
        ctx.execute(query)


def test_tablesample_streaming() -> None:
    lf = pl.LazyFrame({"x": range(1000)})
    res = lf.sql("SELECT x FROM self TABLESAMPLE (30 PERCENT) REPEATABLE (1)")
    assert res.collect(engine="streaming").equals(res.collect(engine="in-memory"))

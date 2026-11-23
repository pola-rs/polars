from __future__ import annotations

from typing import Literal

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def frames() -> dict[str, pl.DataFrame]:
    return {
        "t": pl.DataFrame(
            {"a": [1, 2, 3, None], "b": [1, 5, None, 2], "c": ["1", "x", "3", None]}
        ),
        "u": pl.DataFrame({"p": [10, 20, 20], "q": [1, 3, 4]}),
        "s": pl.DataFrame({"st": [{"a": 7}, {"a": 8}]}),
    }


@pytest.mark.parametrize(
    "query",
    [
        # IN and NOT IN over an empty and a non-empty subquery, NULL on either side.
        "SELECT a FROM t WHERE a IN (SELECT b FROM u WHERE p > 10)",
        "SELECT a FROM t WHERE a IN (SELECT b FROM u WHERE p > 100)",
        "SELECT a FROM t WHERE a NOT IN (SELECT b FROM u WHERE p > 10)",
        "SELECT a FROM t WHERE a NOT IN (SELECT b FROM u WHERE p > 100)",
        "SELECT a FROM t WHERE a IN (SELECT t.b FROM u WHERE p > 10)",
        "SELECT a FROM t AS x WHERE a IN (SELECT x.b - 4 FROM u)",
        "SELECT a, a IN (SELECT b FROM u) AS m, a NOT IN (SELECT b FROM u) AS n FROM t",
        "SELECT a, a IN (SELECT b FROM u WHERE FALSE) AS m FROM t",
        # The rows of the subquery decide only whether it is empty.
        "SELECT a FROM t WHERE a IN (SELECT DISTINCT b FROM u)",
        "SELECT a FROM t WHERE a IN (SELECT b FROM u GROUP BY q HAVING q > 3)",
        "SELECT a FROM t WHERE a IN (SELECT b FROM u ORDER BY q LIMIT 1 OFFSET 2)",
        "SELECT a FROM t WHERE a IN (SELECT b FROM u ORDER BY q LIMIT 1 OFFSET 3)",
        # A correlated WHERE in the subquery.
        "SELECT a FROM t WHERE a IN (SELECT b FROM u WHERE q = t.b)",
        "SELECT a FROM t WHERE a NOT IN (SELECT b FROM u WHERE q = t.b)",
        # Scalar subqueries over no row and one row.
        "SELECT a, (SELECT b FROM u WHERE p = 10) AS v FROM t",
        "SELECT a, (SELECT t.b + a FROM u WHERE p = 30) AS v FROM t",
        "SELECT a, (SELECT DISTINCT b FROM u WHERE p = 20) AS v FROM t",
        "SELECT a FROM t WHERE a = (SELECT b FROM u WHERE p = 10)",
        "SELECT a, (SELECT b FROM u WHERE q = t.a) AS v FROM t",
        # The item is not evaluated when the subquery has no rows.
        "SELECT a, a IN (SELECT CAST(t.c AS INT) FROM u WHERE FALSE) AS m FROM t",
        "SELECT a, a NOT IN (SELECT CAST(t.c AS INT) FROM u WHERE q = t.a) AS m FROM t",
        # Without FROM.
        "SELECT a, (SELECT t.b) AS v, a IN (SELECT t.b) AS m FROM t",
        # Other positions.
        "SELECT b, COUNT(*) AS n FROM t GROUP BY b HAVING b IN (SELECT b FROM u)",
        "SELECT t.a, u.q FROM t JOIN u ON t.a = u.q AND t.b IN (SELECT t.b FROM s)",
        "SELECT a FROM (SELECT a FROM t WHERE a IN (SELECT b FROM u)) AS d",
        # A column of the subquery's own relation shadows the outer one.
        "SELECT a FROM t WHERE a IN (SELECT q FROM u AS t)",
        "SELECT a FROM t WHERE a IN (SELECT a FROM t AS x WHERE x.a > 1)",
    ],
)
def test_subquery_outer_select_item(
    query: str, frames: dict[str, pl.DataFrame]
) -> None:
    assert_sql_matches(
        frames=frames, query=query, compare_with="duckdb", check_row_order=False
    )
    lf = pl.SQLContext(frames).execute(query)
    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=False,
    )


def test_subquery_struct_field_stays_inner(frames: dict[str, pl.DataFrame]) -> None:
    res = pl.SQLContext(frames).execute(
        "SELECT a FROM t WHERE a + 6 IN (SELECT st.a FROM s) ORDER BY a", eager=True
    )
    assert res["a"].to_list() == [1, 2]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_subquery_outer_select_item_multiple_rows(
    engine: Literal["in-memory", "streaming"], frames: dict[str, pl.DataFrame]
) -> None:
    with pytest.raises(Exception, match="single value"):
        pl.SQLContext(frames).execute(
            "SELECT a, (SELECT b FROM u WHERE p = 20) AS v FROM t"
        ).collect(engine=engine)


@pytest.mark.parametrize(
    "query",
    [
        # The alias is read by HAVING.
        "SELECT a FROM t WHERE b IN (SELECT t.b AS v FROM u HAVING v > 1)",
        # A grouping set null-extends the item.
        "SELECT a FROM t WHERE 99 IN (SELECT t.b FROM u GROUP BY ROLLUP(1))",
        # The item reads the subquery's columns through a selector.
        "SELECT (SELECT COLUMNS('^p$') + t.b FROM u LIMIT 1) AS v FROM t",
        # The item numbers the rows of the subquery.
        "SELECT b, (SELECT ROW_NUMBER() + t.b FROM u LIMIT 1) AS v FROM t",
        # Without FROM, a clause can remove the only row.
        "SELECT b, (SELECT t.b WHERE FALSE) AS v FROM t",
        "SELECT b, b IN (SELECT t.b WHERE b IS NULL) AS v FROM t",
        # The item depends on the grouping of the subquery.
        "SELECT b, (SELECT GROUPING(t.b) FROM u GROUP BY t.b) AS s FROM t GROUP BY b",
    ],
)
def test_subquery_outer_select_item_not_moved(
    query: str, frames: dict[str, pl.DataFrame]
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        pl.SQLContext(frames).execute(query, eager=True)

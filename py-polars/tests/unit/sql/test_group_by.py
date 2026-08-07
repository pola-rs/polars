from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from typing import Literal


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_group_by(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(eager=True)
    ctx.register("foods", lf)

    out = ctx.execute(
        """
        SELECT
            count(category) as n,
            category,
            max(calories) as max_cal,
            median(calories) as median_cal,
            min(fats_g) as min_fats
        FROM foods
        GROUP BY category
        HAVING n > 5
        ORDER BY n, category DESC
        """
    )
    assert out.to_dict(as_series=False) == {
        "n": [7, 7, 8],
        "category": ["vegetables", "fruit", "seafood"],
        "max_cal": [45, 130, 200],
        "median_cal": [25.0, 50.0, 145.0],
        "min_fats": [0.0, 0.0, 1.5],
    }

    lf = pl.LazyFrame(
        {
            "grp": ["a", "b", "c", "c", "b"],
            "att": ["x", "y", "x", "y", "y"],
        }
    )
    assert ctx.tables() == ["foods"]

    ctx.register("test", lf)
    assert ctx.tables() == ["foods", "test"]

    out = ctx.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(as_series=False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_group_by_all() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx", "yy", "xx", "zz"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [99, 99, 66, 66, 66, 66],
        }
    )

    # basic group/agg
    res = df.sql(
        """
        SELECT
            a,
            SUM(b),
            SUM(c),
            COUNT(*) AS n
        FROM self
        GROUP BY ALL
        ORDER BY ALL
        """
    )
    expected = pl.DataFrame(
        {
            "a": ["xx", "yy", "zz"],
            "b": [9, 6, 6],
            "c": [231, 165, 66],
            "n": [3, 2, 1],
        }
    )
    assert_frame_equal(expected, res, check_dtypes=False)

    # more involved determination of agg/group columns
    res = df.sql(
        """
        SELECT
            SUM(b) AS sum_b,
            SUM(c) AS sum_c,
            (SUM(b) + SUM(c)) / 2.0 AS sum_bc_over_2,  -- nested agg
            a as grp, --aliased group key
        FROM self
        GROUP BY ALL
        ORDER BY grp
        """
    )
    expected = pl.DataFrame(
        {
            "sum_b": [9, 6, 6],
            "sum_c": [231, 165, 66],
            "sum_bc_over_2": [120.0, 85.5, 36.0],
            "grp": ["xx", "yy", "zz"],
        }
    )
    assert_frame_equal(expected, res.sort(by="grp"))


def test_group_by_all_multi() -> None:
    dt1 = date(1999, 12, 31)
    dt2 = date(2028, 7, 5)

    df = pl.DataFrame(
        {
            "key": ["xx", "yy", "xx", "yy", "xx", "xx"],
            "dt": [dt1, dt1, dt1, dt2, dt2, dt2],
            "value": [10.5, -5.5, 20.5, 8.0, -3.0, 5.0],
        }
    )
    expected = pl.DataFrame(
        {
            "dt": [dt1, dt1, dt2, dt2],
            "key": ["xx", "yy", "xx", "yy"],
            "sum_value": [31.0, -5.5, 2.0, 8.0],
            "ninety_nine": [99, 99, 99, 99],
        },
        schema_overrides={"ninety_nine": pl.Int16},
    )

    # the following groupings should all be equivalent
    for group in (
        "ALL",
        "1, 2",
        "dt, key",
    ):
        res = df.sql(
            f"""
            SELECT dt, key, sum_value, ninety_nine::int2 FROM
            (
                SELECT
                  dt,
                  key,
                  SUM(value) AS sum_value,
                  99 AS ninety_nine
                FROM self
                GROUP BY {group}
                ORDER BY dt, key
            ) AS grp
            """
        )
        assert_frame_equal(expected, res)


def test_group_by_ordinal_position() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx", "yy", "xx", "zz"],
            "b": [1, None, 3, 4, 5, 6],
            "c": [99, 99, 66, 66, 66, 66],
        }
    )
    expected = pl.LazyFrame(
        {
            "c": [66, 99],
            "total_b": [18, 1],
            "count_b": [4, 1],
            "count_star": [4, 2],
        }
    )

    with pl.SQLContext(frame=df) as ctx:
        res1 = ctx.execute(
            """
            SELECT
              c,
              SUM(b) AS total_b,
              COUNT(b) AS count_b,
              COUNT(*) AS count_star
            FROM frame
            GROUP BY 1
            ORDER BY c
            """
        )
        assert_frame_equal(res1, expected, check_dtypes=False)

        res2 = ctx.execute(
            """
            WITH "grp" AS (
              SELECT NULL::date as dt, c, SUM(b) AS total_b
              FROM frame
              GROUP BY 2, 1
            )
            SELECT c, total_b FROM grp ORDER BY c"""
        )
        assert_frame_equal(res2, expected.select(pl.nth(0, 1)))


def test_group_by_errors() -> None:
    df = pl.DataFrame(
        {
            "a": ["xx", "yy", "xx"],
            "b": [10, 20, 30],
            "c": [99, 99, 66],
        }
    )
    with pytest.raises(
        SQLSyntaxError,
        match=r"negative ordinal values are invalid for GROUP BY; found -99",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY -99, a")

    with pytest.raises(
        SQLSyntaxError,
        match=r"GROUP BY requires a valid expression or positive ordinal; found '!!!'",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY a, '!!!'")

    with pytest.raises(
        SQLSyntaxError,
        match=r"'a' should participate in the GROUP BY clause or an aggregate function",
    ):
        df.sql("SELECT a, SUM(b) FROM self GROUP BY b")

    with pytest.raises(
        SQLSyntaxError,
        match=r"HAVING clause not valid outside of GROUP BY",
    ):
        df.sql("SELECT a, COUNT(a) AS n FROM self HAVING n > 1")


def test_group_by_having_aggregate_not_in_select() -> None:
    """Test HAVING with aggregate functions not present in SELECT."""
    df = pl.DataFrame(
        {"grp": ["a", "a", "a", "b", "b", "c"], "val": [1, 2, 3, 4, 5, 6]}
    )
    # COUNT(*) not in SELECT - only group 'a' has 3 rows
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING COUNT(*) > 2",
        compare_with="sqlite",
        expected={"grp": ["a"]},
    )

    # SUM not in SELECT
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING SUM(val) > 5 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "b", "c"]},
    )

    # AVG not in SELECT
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING AVG(val) > 4 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["b", "c"]},
    )

    # MIN/MAX not in SELECT
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING MIN(val) >= 4 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["b", "c"]},
    )


def test_group_by_having_aggregate_in_select() -> None:
    """Test HAVING properly references an aggregate already computed in SELECT."""
    df = pl.DataFrame(
        {"grp": ["a", "a", "a", "b", "b", "c"], "val": [1, 2, 3, 4, 5, 6]}
    )
    # COUNT(*) in SELECT and HAVING
    for count_expr in ("COUNT(*)", "cnt"):
        assert_sql_matches(
            df,
            query=f"SELECT grp, COUNT(*) AS cnt FROM self GROUP BY grp HAVING {count_expr} > 2",
            compare_with="sqlite",
            expected={"grp": ["a"], "cnt": [3]},
        )

    # SUM in SELECT and HAVING
    for sum_expr in ("total", "SUM(val)"):
        assert_sql_matches(
            df,
            query=f"SELECT grp, SUM(val) AS total FROM self GROUP BY grp HAVING {sum_expr} > 5 ORDER BY grp",
            compare_with="sqlite",
            expected={"grp": ["a", "b", "c"], "total": [6, 9, 6]},
        )


def test_group_by_having_multiple_aggregates() -> None:
    """Test HAVING with multiple aggregate conditions."""
    df = pl.DataFrame(
        {"grp": ["a", "a", "a", "b", "b", "c"], "val": [1, 2, 3, 4, 5, 6]}
    )
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING COUNT(*) >= 2 AND SUM(val) > 5 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "b"]},
    )
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING COUNT(*) = 1 OR SUM(val) >= 9 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["b", "c"]},
    )


def test_group_by_having_compound_expressions() -> None:
    """Test HAVING with compound expressions involving aggregates."""
    df = pl.DataFrame(
        {"grp": ["a", "a", "c", "b", "b"], "val": [10, 20, 100, 5, 15]},
    )
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING SUM(val) / COUNT(*) > 10 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "c"]},
    )
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING MAX(val) - MIN(val) > 5 ORDER BY grp DESC",
        compare_with="sqlite",
        expected={"grp": ["b", "a"]},
    )
    for sum_expr, count_expr in (
        ("SUM(val)", "COUNT(*)"),
        ("total", "COUNT(*)"),
        ("SUM(val)", "n"),
        ("total", "n"),
    ):
        assert_sql_matches(
            df,
            query=f"""
                SELECT grp, SUM(val) AS total, COUNT(*) AS n
                FROM self
                GROUP BY grp
                HAVING {sum_expr} / {count_expr} > 10 ORDER BY grp
            """,
            compare_with="sqlite",
            expected={
                "grp": ["a", "c"],
                "total": [30, 100],
                "n": [2, 1],
            },
        )


def test_group_by_having_with_nulls() -> None:
    """Test HAVING behaviour with NULL values."""
    df = pl.DataFrame(
        {"grp": ["a", "b", "a", "b", "c"], "val": [None, None, 1, None, 5]}
    )
    # COUNT(*) counts all rows, including NULLs...
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING COUNT(*) > 1 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "b"]},
    )

    # ...whereas COUNT(col) excludes NULLs
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING COUNT(val) > 0 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "c"]},
    )


def test_group_by_having_composite_aggregates() -> None:
    df = pl.DataFrame(
        {
            "grp": ["a", "a", "b", "b", "c"],
            "val": [10, 20, 5, 15, 100],
        }
    )
    # SUM (null-guarded wrapper) as a bare HAVING predicate operand
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING SUM(val) > 25 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "c"]},
    )
    # SUM wrapped in a further scalar function
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING ABS(SUM(val)) > 25 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "c"]},
    )
    # composite SUM combined with another aggregate
    assert_sql_matches(
        df,
        query="SELECT grp FROM self GROUP BY grp HAVING SUM(val) / COUNT(*) > 10 ORDER BY grp",
        compare_with="sqlite",
        expected={"grp": ["a", "c"]},
    )
    # STRING_AGG (implode+join composite) as a HAVING predicate
    txt = pl.DataFrame({"grp": ["a", "a", "b"], "s": ["x", "y", "z"]})
    assert_sql_matches(
        txt,
        query="SELECT grp FROM self GROUP BY grp HAVING STRING_AGG(s, ',') = 'x,y'",
        compare_with="duckdb",
        expected={"grp": ["a"]},
    )
    # CORR (null-guarded composite) as a HAVING predicate
    corr = pl.DataFrame(
        {
            "grp": [1, 1, 1, 2, 2, 2],
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0, 6.0, 4.0, 2.0],
        }
    )
    assert_sql_matches(
        corr,
        query="SELECT grp FROM self GROUP BY grp HAVING CORR(x, y) > 0 ORDER BY grp",
        compare_with="duckdb",
        expected={"grp": [1]},
    )


@pytest.mark.parametrize(
    ("having_clause", "expected"),
    [
        # basic count conditions
        ("COUNT(*) > 2", [1]),
        ("COUNT(*) >= 2 AND COUNT(*) <= 3", [1, 2]),
        ("(COUNT(*) > 1)", [1, 2]),
        ("NOT COUNT(*) < 2", [1, 2]),
        # range / membership
        ("COUNT(*) BETWEEN 2 AND 3", [1, 2]),
        ("COUNT(*) NOT BETWEEN 1 AND 2", [1]),
        ("COUNT(*) IN (1, 3)", [1, 3]),
        ("COUNT(*) NOT IN (1, 2)", [1]),
        # conditional
        ("CASE WHEN COUNT(*) > 2 THEN 1 ELSE 0 END = 1", [1]),
    ],
)
def test_group_by_having_misc_01(
    having_clause: str,
    expected: list[int],
) -> None:
    df = pl.DataFrame({"a": [1, 1, 1, 2, 2, 3]})
    assert_sql_matches(
        df,
        query=f"SELECT a FROM self GROUP BY a HAVING {having_clause} ORDER BY a",
        compare_with="sqlite",
        expected={"a": expected},
    )


@pytest.mark.parametrize(
    ("having_clause", "expected"),
    [
        ("SUM(b) > 50", [1, 3]),
        ("AVG(b) > 15", [1, 3]),
        ("ABS(SUM(b)) > 50", [1, 3]),
        ("ROUND(ABS(AVG(b))) > 15", [1, 3]),
        ("ABS(SUM(b)) + ABS(AVG(b)) > 100", [3]),
        ("CASE WHEN SUM(b) < 10 THEN 0 ELSE SUM(b) END > 50", [1, 3]),
    ],
)
def test_group_by_having_misc_02(
    having_clause: str,
    expected: list[int],
) -> None:
    df = pl.DataFrame({"a": [1, 1, 1, 2, 2, 3], "b": [10, 20, 30, 5, 15, 100]})
    assert_sql_matches(
        df,
        query=f"SELECT a FROM self GROUP BY a HAVING {having_clause} ORDER BY a",
        compare_with="sqlite",
        expected={"a": expected},
    )


@pytest.mark.parametrize(
    ("having_clause", "expected"),
    [
        ("MAX(b) IS NULL", [1]),
        ("MAX(b) IS NOT NULL", [2]),
    ],
)
def test_group_by_having_misc_03(
    having_clause: str,
    expected: list[int],
) -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [None, None, 5]})
    assert_sql_matches(
        df,
        query=f"SELECT a FROM self GROUP BY a HAVING {having_clause}",
        compare_with="sqlite",
        expected={"a": expected},
    )


def test_group_by_output_struct() -> None:
    df = pl.DataFrame({"g": [1], "x": [2], "y": [3]})
    out = df.group_by("g").agg(pl.struct(pl.col.x.min(), pl.col.y.sum()))
    assert out.rows() == [(1, {"x": 2, "y": 3})]


@pytest.mark.parametrize(
    "maintain_order",
    [False, True],
)
def test_group_by_list_cat_24049(maintain_order: bool) -> None:
    df = pl.DataFrame(
        {
            "x": [["a"], ["b", "c"], ["a"], ["a"], ["d"], ["b", "c"]],
            "y": [1, 2, 3, 4, 5, 10],
        },
        schema={"x": pl.List(pl.Categorical), "y": pl.Int32},
    )

    expected = pl.DataFrame(
        {"x": [["a"], ["b", "c"], ["d"]], "y": [8, 12, 5]},
        schema={"x": pl.List(pl.Categorical), "y": pl.Int32},
    )
    assert_frame_equal(
        df.group_by("x", maintain_order=maintain_order).agg(pl.col.y.sum()),
        expected,
        check_row_order=maintain_order,
    )


@pytest.mark.parametrize(
    "maintain_order",
    [False, True],
)
def test_group_by_struct_cat_24049(maintain_order: bool) -> None:
    a = {"k1": "a2", "k2": "a2"}
    b = {"k1": "b2", "k2": "b2"}
    c = {"k1": "c2", "k2": "c2"}
    s = pl.Struct({"k1": pl.Categorical, "k2": pl.Categorical})

    df = pl.DataFrame(
        {
            "x": [a, b, a, a, c, b],
            "y": [1, 2, 3, 4, 5, 10],
        },
        schema={"x": s, "y": pl.Int32},
    )

    expected = pl.DataFrame(
        {"x": [a, b, c], "y": [8, 12, 5]},
        schema={"x": s, "y": pl.Int32},
    )
    assert_frame_equal(
        df.group_by("x", maintain_order=maintain_order).agg(pl.col.y.sum()),
        expected,
        check_row_order=maintain_order,
    )


def test_group_by_aggregate_name_is_group_key() -> None:
    """Unaliased aggregation with a column that's also used in the GROUP BY key."""
    df = pl.DataFrame({"c0": [1, 2]})

    # 'COUNT(col)' where 'col' is also part of the group key
    for query in (
        "SELECT COUNT(c0) FROM self GROUP BY c0",
        "SELECT COUNT(c0) AS c0 FROM self GROUP BY c0",
    ):
        assert_sql_matches(
            df,
            query=query,
            compare_with="sqlite",
            check_column_names=False,
            expected={"c0": [1, 1]},
        )

    # Same condition with a table prefix (and a different aggfunc)
    query = "SELECT SUM(self.c0) FROM self GROUP BY self.c0"
    assert_sql_matches(
        df,
        query=query,
        compare_with="sqlite",
        check_row_order=False,
        check_column_names=False,
        expected={"c0": [1, 2]},
    )


@pytest.mark.parametrize(
    "query",
    [
        # GROUP BY referencing SELECT alias for arithmetic expression
        "SELECT COUNT(*) AS n, value / 10.0 AS bucket FROM self GROUP BY bucket ORDER BY bucket",
        # Multiple aliased expressions in GROUP BY
        "SELECT COUNT(*) AS n, value / 10.0 AS tens, value % 3 AS rem FROM self GROUP BY tens, rem ORDER BY tens, rem",
        # GROUP BY alias with additional aggregation
        "SELECT SUM(id) AS total, value / 20.0 AS grp FROM self GROUP BY grp ORDER BY grp",
        # GROUP BY ordinal position with aliased column
        "SELECT value / 10 AS bucket, COUNT(*) AS n FROM self GROUP BY 1 ORDER BY 1",
        # GROUP BY ordinal with multiple aliased columns
        "SELECT id % 2 AS parity, value / 10.0 AS tens, SUM(id) AS total FROM self GROUP BY 1, 2 ORDER BY 1, 2",
    ],
)
def test_group_by_select_alias(query: str) -> None:
    """Test GROUP BY can reference SELECT aliases for computed expressions."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        }
    )
    assert_sql_matches(df, query=query, compare_with="sqlite")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT (x >= 5) AS x_gt_5, COUNT(*) AS n FROM self GROUP BY (x >= 5) ORDER BY x_gt_5",
        "SELECT (x * 10) AS x10, COUNT(*) AS n FROM self GROUP BY (x * 10) ORDER BY x10",
        "SELECT (x = 5) AS x_eq_5, COUNT(*) AS n FROM self GROUP BY (x = 5) ORDER BY x_eq_5",
        "SELECT (x >= 5) AS x_gt_5, COUNT(*) AS n FROM self GROUP BY ALL ORDER BY x_gt_5",
    ],
)
def test_group_by_computed_key_repeated_27735(query: str) -> None:
    comparison_backend: Literal["sqlite", "duckdb"] = (
        "duckdb" if "GROUP BY ALL" in query else "sqlite"
    )
    df = pl.DataFrame({"x": [3, 5, 7]})
    assert_sql_matches(
        frames=df,
        query=query,
        compare_with=comparison_backend,
    )


def test_group_by_empty_or_scalar_key_exprs_23397() -> None:
    lf = pl.LazyFrame({"a": [0, 1, 2, 3, 4]})

    q = lf.group_by().agg(pl.len())
    plan = q.explain()

    assert plan.startswith("SELECT")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"len": pl.Series([5], dtype=pl.get_index_type())}),
    )

    q = lf.group_by().agg("a")
    plan = q.explain()

    assert plan.startswith("SELECT")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": pl.Series([[0, 1, 2, 3, 4]])}),
    )

    q = lf.group_by().agg("a")
    plan = q.explain()

    assert plan.startswith("SELECT")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": pl.Series([[0, 1, 2, 3, 4]])}),
    )

    q = lf.group_by().agg("a", a_sum=pl.sum("a"))
    plan = q.explain()

    assert plan.startswith("SELECT")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {"a": pl.Series([[0, 1, 2, 3, 4]]), "a_sum": 10},
            schema_overrides={"a_sum": pl.Int64},
        ),
    )

    q = lf.group_by(
        pl.lit(1).alias("1"),
        pl.lit(2).alias("2"),
        a_max=pl.max("a"),
    ).agg(pl.len())
    plan = q.explain()

    assert plan.startswith("SELECT")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "1": 1,
                "2": 2,
                "a_max": 4,
                "len": pl.Series([5], dtype=pl.get_index_type()),
            },
            schema_overrides={"a_max": pl.Int64},
        ),
    )

    q = lf.group_by().map_groups(lambda df: df, schema=lf.collect_schema())
    with pytest.raises(pl.exceptions.ComputeError, match="not implemented"):
        q.collect()

    q = lf.group_by().having(pl.len() != 5).agg(pl.len())
    plan = q.explain()

    assert "AGGREGATE" not in plan
    assert q.collect().shape == (0, 1)

    q = lf.group_by().having(pl.len() == 5).agg(pl.len())
    plan = q.explain()

    assert "AGGREGATE" not in plan
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"len": pl.Series([5], dtype=pl.get_index_type())}),
    )


def test_sum_and_total_28434() -> None:
    # `SUM` over an empty/all-null input should return NULL (SQL standard).
    # `TOTAL` is the (SQLite) non-standard counterpart that returns zero.
    all_null = pl.DataFrame({"a": [None, None]}, schema={"a": pl.Int64})
    grp = pl.DataFrame(
        {"g": [1, 1, 2, 2], "a": [None, None, 3, 4]},
        schema={"g": pl.Int64, "a": pl.Int64},
    )
    mixed = pl.DataFrame({"a": [1, None, 2]}, schema={"a": pl.Int64})

    # scalar: all-null -> SUM is NULL, TOTAL is 0
    expected = pl.DataFrame(
        data={"s": [None], "t": [0]},
        schema={"s": pl.Int64, "t": pl.Int64},
    )
    assert_frame_equal(
        all_null.sql("SELECT SUM(a) AS s, TOTAL(a) AS t FROM self"), expected
    )
    assert_frame_equal(
        all_null.sql("""
            SELECT
              SUM(a) OVER () AS s,
              TOTAL(a) OVER () AS t,
            FROM self
        """),
        expected,
    )

    # all-null group -> (NULL, 0); a group with values sums identically for both
    assert_frame_equal(
        grp.sql("SELECT g, SUM(a) AS s, TOTAL(a) AS t FROM self GROUP BY g ORDER BY g"),
        pl.DataFrame(
            {"g": [1, 2], "s": [None, 7], "t": [0, 7]},
            schema={"g": pl.Int64, "s": pl.Int64, "t": pl.Int64},
        ),
    )
    # a mix of null and non-null values sums identically for both
    assert_frame_equal(
        mixed.sql("SELECT SUM(a) AS s, TOTAL(a) AS t FROM self"),
        pl.DataFrame({"s": [3], "t": [3]}, schema={"s": pl.Int64, "t": pl.Int64}),
    )
    # dtype is preserved across numeric types (float stays float)
    all_null_f32 = pl.DataFrame({"a": [None, None]}, schema={"a": pl.Float32})
    assert_frame_equal(
        all_null_f32.sql("SELECT SUM(a) AS s, TOTAL(a) AS t FROM self"),
        pl.DataFrame(
            {"s": [None], "t": [0.0]}, schema={"s": pl.Float32, "t": pl.Float32}
        ),
    )

    # `SUM`-specific NULL conformance (no `TOTAL` analog), checked against sqlite
    assert_sql_matches(
        all_null,
        query="SELECT COALESCE(SUM(a), -99) AS s FROM self",
        compare_with="sqlite",
        expected={"s": [-99]},
    )
    assert_sql_matches(
        all_null,
        query="SELECT SUM(a) AS s, AVG(a) AS m FROM self",
        compare_with="sqlite",
        expected={"s": [None], "m": [None]},
    )


def test_corr_no_complete_pairs_returns_null() -> None:
    # `CORR` returns NULL when there are no complete (both-non-null) pairs
    allnull = pl.DataFrame(
        {"a": [None, None], "b": [None, None]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    assert_sql_matches(
        allnull,
        query="SELECT CORR(a, b) AS c FROM self",
        compare_with="duckdb",
        expected={"c": [None]},
    )

    # rows exist, but no single row has both values non-null -> NULL
    no_pairs = pl.DataFrame(
        {"a": [1.0, None], "b": [None, 2.0]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    assert_sql_matches(
        no_pairs,
        query="SELECT CORR(a, b) AS c FROM self",
        compare_with="duckdb",
        expected={"c": [None]},
    )

    # a well-defined correlation is unaffected by the fix
    corr = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    assert_sql_matches(
        corr,
        query="SELECT CORR(a, b) AS c FROM self",
        compare_with="duckdb",
        expected={"c": [1.0]},
    )

    # an all-null group yields NULL; a group with a real correlation is computed
    grp = pl.DataFrame(
        {
            "g": [1, 1, 2, 2, 2],
            "a": [None, None, 1.0, 2.0, 3.0],
            "b": [None, None, 2.0, 4.0, 6.0],
        },
        schema={"g": pl.Int64, "a": pl.Float64, "b": pl.Float64},
    )
    assert_sql_matches(
        grp,
        query="SELECT g, CORR(a, b) AS c FROM self GROUP BY g ORDER BY g",
        compare_with="duckdb",
        expected={"g": [1, 2], "c": [None, 1.0]},
    )

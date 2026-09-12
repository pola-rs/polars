from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


@pytest.mark.parametrize(
    ("sort_order", "limit", "expected"),
    [
        (None, None, [("a", ["x", "y"]), ("b", ["z", "X", "Y"])]),
        ("ASC", None, [("a", ["x", "y"]), ("b", ["z", "Y", "X"])]),
        ("DESC", None, [("a", ["y", "x"]), ("b", ["X", "Y", "z"])]),
        ("ASC", 2, [("a", ["x", "y"]), ("b", ["z", "Y"])]),
        ("DESC", 2, [("a", ["y", "x"]), ("b", ["X", "Y"])]),
        ("ASC", 1, [("a", ["x"]), ("b", ["z"])]),
        ("DESC", 1, [("a", ["y"]), ("b", ["X"])]),
    ],
)
def test_array_agg(sort_order: str | None, limit: int | None, expected: Any) -> None:
    order_by = "" if not sort_order else f" ORDER BY col0 {sort_order}"
    limit_clause = "" if not limit else f" LIMIT {limit}"

    res = pl.sql(
        f"""
        WITH data (col0, col1, col2) as (
          VALUES
            (1,'a','x'),
            (2,'a','y'),
            (4,'b','z'),
            (8,'b','X'),
            (7,'b','Y')
        )
        SELECT col1, ARRAY_AGG(col2{order_by}{limit_clause}) AS arrs
        FROM data
        GROUP BY col1
        ORDER BY col1
        """
    ).collect()

    assert res.rows() == expected


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_array_agg_scalar_input(engine: EngineType) -> None:
    df = pl.LazyFrame(
        {
            "group": ["a", "a", "b"],
            "order": [2, 1, 0],
            "keep": [True, False, False],
        }
    )
    result = df.sql(
        """
        SELECT
          group,
          ARRAY_AGG(1) AS plain,
          ARRAY_AGG(CAST(NULL AS INTEGER)) AS nulls,
          ARRAY_AGG(1 + 2 ORDER BY order) AS composed,
          ARRAY_AGG(DISTINCT 1) AS distinct_values,
          ARRAY_AGG(1 ORDER BY order LIMIT 1) AS limited,
          ARRAY_AGG(1) FILTER (WHERE keep) AS filtered,
          ARRAY_AGG(1) FILTER (WHERE TRUE) AS scalar_filter,
          ARRAY_AGG([1, 2]) AS lists,
          ARRAY_AGG(([1, 2])) AS parenthesized_lists,
          ARRAY_AGG(CAST(([1, 2]) AS SMALLINT[])) AS cast_lists,
          ARRAY_AGG(DISTINCT 1 ORDER BY (1)) FILTER (WHERE keep) AS distinct_filtered,
          ARRAY_AGG([[1, 2], [3, 4]]) AS nested_lists
        FROM self
        GROUP BY group
        ORDER BY group
        """
    ).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "group": ["a", "b"],
            "plain": [[1, 1], [1]],
            "nulls": [[None, None], [None]],
            "composed": [[3, 3], [3]],
            "distinct_values": [[1], [1]],
            "limited": [[1], [1]],
            "filtered": [[1], []],
            "scalar_filter": [[1, 1], [1]],
            "lists": [[[1, 2], [1, 2]], [[1, 2]]],
            "parenthesized_lists": [[[1, 2], [1, 2]], [[1, 2]]],
            "cast_lists": [[[1, 2], [1, 2]], [[1, 2]]],
            "distinct_filtered": [[1], []],
            "nested_lists": [
                [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                [[[1, 2], [3, 4]]],
            ],
        },
        schema={
            "group": pl.String,
            "plain": pl.List(pl.Int32),
            "nulls": pl.List(pl.Int32),
            "composed": pl.List(pl.Int32),
            "distinct_values": pl.List(pl.Int32),
            "limited": pl.List(pl.Int32),
            "filtered": pl.List(pl.Int32),
            "scalar_filter": pl.List(pl.Int32),
            "lists": pl.List(pl.List(pl.Int64)),
            "parenthesized_lists": pl.List(pl.List(pl.Int64)),
            "cast_lists": pl.List(pl.List(pl.Int16)),
            "distinct_filtered": pl.List(pl.Int32),
            "nested_lists": pl.List(pl.List(pl.List(pl.Int64))),
        },
    )
    assert_frame_equal(result, expected)


def test_array_agg_dtype_dependent_expression() -> None:
    result = pl.LazyFrame({"values": [[1, 2], [3, 4]]}).sql(
        "SELECT ARRAY_AGG(ARRAY_REVERSE(values)) AS reversed FROM self"
    )
    expected = pl.LazyFrame(
        {"reversed": [[[2, 1], [4, 3]]]},
        schema={"reversed": pl.List(pl.List(pl.Int64))},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("rows", [0, 3])
def test_array_agg_scalar_input_global(rows: int, engine: EngineType) -> None:
    df = pl.LazyFrame({"value": range(rows)})
    result = df.sql(
        """
        SELECT
          ARRAY_AGG(1) AS plain,
          ARRAY_AGG(1 ORDER BY 1) AS ordered,
          ARRAY_AGG([1, 2] ORDER BY [1, 2]) AS ordered_lists,
          ARRAY_AGG(CAST([1, 2] AS SMALLINT[]) ORDER BY CAST([1, 2] AS BIGINT[])) AS cast_lists,
          ARRAY_AGG(DISTINCT 1 ORDER BY (1)) AS distinct_values,
          ARRAY_AGG(1) FILTER (WHERE TRUE) AS filtered
        FROM self
        """
    ).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "plain": [[1] * rows],
            "ordered": [[1] * rows],
            "ordered_lists": [[[1, 2]] * rows],
            "cast_lists": [[[1, 2]] * rows],
            "distinct_values": [[1] if rows else []],
            "filtered": [[1] * rows],
        },
        schema={
            "plain": pl.List(pl.Int32),
            "ordered": pl.List(pl.Int32),
            "ordered_lists": pl.List(pl.List(pl.Int64)),
            "cast_lists": pl.List(pl.List(pl.Int16)),
            "distinct_values": pl.List(pl.Int32),
            "filtered": pl.List(pl.Int32),
        },
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "expression",
    [
        "ARRAY_AGG(x ORDER BY CAST('bad' AS INTEGER), x)",
        "ARRAY_AGG(CAST(['bad'] AS INTEGER[]))",
    ],
)
def test_array_agg_scalar_cast_error(engine: EngineType, expression: str) -> None:
    df = pl.LazyFrame({"x": [2, 1]})
    with pytest.raises(InvalidOperationError, match=r"conversion from .* failed"):
        df.sql(f"SELECT {expression} FROM self").collect(engine=engine)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("grouped", [False, True])
def test_array_agg_mixed_order_by(engine: EngineType, grouped: bool) -> None:
    df = pl.LazyFrame(
        {"g": [0, 0, 0, 0], "x": [3, None, 1, 2], "keep": [True, True, False, True]}
    )
    group_by = " GROUP BY g" if grouped else ""
    result = df.sql(
        "SELECT ARRAY_AGG(x ORDER BY 1 ASC, x DESC NULLS LAST, [1, 2] ASC) "
        f"FILTER (WHERE keep) AS values FROM self{group_by}"
    ).collect(engine=engine)
    expected = pl.DataFrame(
        {"values": [[3, 2, None]]}, schema={"values": pl.List(pl.Int64)}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_array_agg_scalar_input_window(engine: EngineType) -> None:
    df = pl.LazyFrame({"id": [0, 1, 2], "group": ["a", "a", "b"]})
    result = df.sql(
        """
        SELECT id, ARRAY_AGG(1) OVER (PARTITION BY group) AS values
        FROM self
        ORDER BY id
        """
    ).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "values": [[1, 1], [1, 1], [1]],
        },
        schema={"id": pl.Int64, "values": pl.List(pl.Int32)},
    )
    assert_frame_equal(result, expected)


def test_array_literals() -> None:
    with pl.SQLContext(df=None, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              a1, a2,
              -- test some array ops
              ARRAY_AGG(a1) AS a3,
              ARRAY_AGG(a2) AS a4,
              ARRAY_CONTAINS(a1,20) AS i20,
              ARRAY_CONTAINS(a2,'zz') AS izz,
              ARRAY_REVERSE(a1) AS ar1,
              ARRAY_REVERSE(a2) AS ar2
            FROM (
              SELECT
                -- declare array literals
                ARRAY[10,20,30] AS a1,
                ['a','b','c'] AS a2,
              FROM df
            ) tbl
            """
        )
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "a1": [[10, 20, 30]],
                    "a2": [["a", "b", "c"]],
                    "a3": [[[10, 20, 30]]],
                    "a4": [[["a", "b", "c"]]],
                    "i20": [True],
                    "izz": [False],
                    "ar1": [[30, 20, 10]],
                    "ar2": [["c", "b", "a"]],
                }
            ),
        )


@pytest.mark.parametrize("function_name", ["ARRAY_INNER_PRODUCT", "ARRAY_DOT_PRODUCT"])
def test_array_inner_product(function_name: str) -> None:
    df = pl.DataFrame(
        {
            "lhs": [[1, 2, 3], [1, None, 3], None, [None, None, None]],
            "rhs": [
                [4.0, 5.0, 6.0],
                [4.0, 5.0, None],
                [1.0, 2.0, 3.0],
                [None, None, None],
            ],
        },
        schema={
            "lhs": pl.Array(pl.Int32, 3),
            "rhs": pl.Array(pl.Float32, 3),
        },
    )

    result = df.sql(f"SELECT {function_name}(lhs, rhs) AS dot FROM self")
    expected = df.select(pl.col("lhs").arr.dot("rhs").alias("dot"))

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("literal", "literal_on_left"),
    [
        ("[4.0, 5.0, 6.0]", False),
        ("ARRAY[4.0, 5.0, 6.0]", True),
        ("([4.0, 5.0, 6.0])", False),
    ],
)
def test_array_inner_product_literal(literal: str, literal_on_left: bool) -> None:
    df = pl.DataFrame(
        {"arr": [[1, 2, 3], [4, 5, 6], None]},
        schema={"arr": pl.Array(pl.Int32, 3)},
    )
    arguments = f"{literal}, arr" if literal_on_left else f"arr, {literal}"

    result = df.sql(f"SELECT ARRAY_INNER_PRODUCT({arguments}) AS dot FROM self")
    literal_expr = pl.lit([4.0, 5.0, 6.0], dtype=pl.Array(pl.Float64, 3))
    expected = df.select(
        (
            literal_expr.arr.dot("arr")
            if literal_on_left
            else pl.col("arr").arr.dot(literal_expr)
        ).alias("dot")
    )

    assert_frame_equal(result, expected)


def test_array_inner_product_null_literal() -> None:
    df = pl.DataFrame(
        {"values": [[1, 2, 3], [4, None, 6], None]},
        schema={"values": pl.Array(pl.Int32, 3)},
    )

    result = df.sql(
        "SELECT ARRAY_INNER_PRODUCT(values, [NULL, NULL, NULL]) AS dot FROM self"
    )

    assert_frame_equal(
        result,
        pl.DataFrame({"dot": [0, 0, None]}, schema={"dot": pl.Int32}),
    )


def test_array_inner_product_literal_inherits_projection_height() -> None:
    df = pl.DataFrame({"row": [1, 2, 3]})

    result = df.sql("SELECT ARRAY_INNER_PRODUCT([1, 2], ARRAY[3, 4]) AS dot FROM self")

    assert_frame_equal(result, pl.DataFrame({"dot": [11, 11, 11]}))


def test_array_inner_product_non_array_receiver() -> None:
    df = pl.DataFrame({"lhs": [[1, 2]]})

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="expected Array datatype",
    ):
        df.sql("SELECT ARRAY_INNER_PRODUCT(lhs, [3, 4]) FROM self")


def test_array_inner_product_uses_native_array_dot_plan() -> None:
    df = pl.DataFrame(
        {
            "lhs": [[1, 2]],
            "rhs": [[3, 4]],
        },
        schema={
            "lhs": pl.Array(pl.Int64, 2),
            "rhs": pl.Array(pl.Int64, 2),
        },
    )

    with pl.SQLContext(df=df) as ctx:
        plan = ctx.execute(
            "SELECT ARRAY_INNER_PRODUCT(lhs, rhs) AS dot FROM df"
        ).explain(optimized=False)

    assert ".arr.dot([" in plan
    assert ").arr.sum()" not in plan
    assert " * " not in plan


@pytest.mark.parametrize(
    "arguments",
    ["lhs", "lhs, rhs, lhs"],
)
def test_array_inner_product_arity(arguments: str) -> None:
    df = pl.DataFrame(
        {"lhs": [[1, 2]], "rhs": [[3, 4]]},
        schema={"lhs": pl.Array(pl.Int64, 2), "rhs": pl.Array(pl.Int64, 2)},
    )

    with pytest.raises(SQLInterfaceError, match="no function matches"):
        df.sql(f"SELECT ARRAY_INNER_PRODUCT({arguments}) FROM self")


def test_array_inner_product_unequal_widths() -> None:
    df = pl.DataFrame(
        {"lhs": [[1, 2]], "rhs": [[3, 4, 5]]},
        schema={"lhs": pl.Array(pl.Int64, 2), "rhs": pl.Array(pl.Int64, 3)},
    )

    with pytest.raises(pl.exceptions.ShapeError, match="equal array widths"):
        df.sql("SELECT ARRAY_INNER_PRODUCT(lhs, rhs) FROM self")


@pytest.mark.parametrize(
    ("array_index", "expected"),
    [
        (-4, None),
        (-3, 99),
        (-2, 66),
        (-1, 33),
        (0, None),
        (1, 99),
        (2, 66),
        (3, 33),
        (4, None),
    ],
)
def test_array_indexing(array_index: int, expected: int | None) -> None:
    res = pl.sql(
        f"""
        SELECT
          arr[{array_index}] AS idx1,
          ARRAY_GET(arr,{array_index}) AS idx2,
        FROM (SELECT [99,66,33] AS arr) tbl
        """
    ).collect()

    assert_frame_equal(
        res,
        pl.DataFrame(
            {"idx1": [expected], "idx2": [expected]},
        ),
        check_dtypes=False,
    )


def test_array_indexing_by_expr() -> None:
    df = pl.DataFrame(
        {
            "idx": [-2, -1, 0, None, 1, 2, 3],
            "arr": [[0, 1, 2, 3], [4, 5], [6], [7, 8, 9], [8, 7], [6, 5, 4], [3, 2, 1]],
        }
    )
    res = df.sql(
        """
        SELECT
          arr[idx] AS idx1,
          ARRAY_GET(arr, idx) AS idx2
        FROM self
        """
    )
    expected = [2, 5, None, None, 8, 5, 1]
    assert_frame_equal(res, pl.DataFrame({"idx1": expected, "idx2": expected}))


def test_array_to_string() -> None:
    data = {
        "s_values": [["aa", "bb"], [None, "cc"], ["dd", None]],
        "n_values": [[999, 777], [None, 555], [333, None]],
    }
    res = pl.DataFrame(data).sql(
        """
        SELECT
          ARRAY_TO_STRING(s_values, '') AS vs1,
          ARRAY_TO_STRING(s_values, ':') AS vs2,
          ARRAY_TO_STRING(s_values, ':', 'NA') AS vs3,
          ARRAY_TO_STRING(n_values, '') AS vn1,
          ARRAY_TO_STRING(n_values, ':') AS vn2,
          ARRAY_TO_STRING(n_values, ':', 'NA') AS vn3
        FROM self
        """
    )
    assert_frame_equal(
        res,
        pl.DataFrame(
            {
                "vs1": ["aabb", "cc", "dd"],
                "vs2": ["aa:bb", "cc", "dd"],
                "vs3": ["aa:bb", "NA:cc", "dd:NA"],
                "vn1": ["999777", "555", "333"],
                "vn2": ["999:777", "555", "333"],
                "vn3": ["999:777", "NA:555", "333:NA"],
            }
        ),
    )
    with pytest.raises(
        SQLSyntaxError,
        match=r"ARRAY_TO_STRING expects 2-3 arguments \(found 1\)",
    ):
        pl.sql_expr("ARRAY_TO_STRING(arr)")


def test_array_typed_literals() -> None:
    res = pl.sql(
        """
        SELECT
          -- typed temporal literals
          ARRAY[DATE '2024-01-01', DATE '1969-07-20'] AS dt,
          ARRAY[TIME '08:30:00', TIME '23:59:59'] AS tm,
          ARRAY[TIMESTAMP(3) '2024-01-01 12:00:00'] AS dtm,
          -- cast syntax (::type and CAST)
          ARRAY['2024-01-01'::date, '1969-07-20'::date] AS dt_cast,
          ARRAY['08:30:00'::time, '23:59:59'::time] AS tm_cast,
          ARRAY[CAST('2024-01-01' AS DATE)] AS dt_explicit,
          -- numeric literal casts
          ARRAY[100::bigint, -50::bigint] AS i64_cast,
          ARRAY[1.5::double, -2.7::double] AS f64_cast,
          ARRAY[['42'::int16], ['-7'::int16]] AS str_to_nested_int16,
        FROM (VALUES (0)) tbl (x)
        """,
        eager=True,
    )
    # values are typed properly
    assert res.to_dict(as_series=False) == {
        "dt": [[date(2024, 1, 1), date(1969, 7, 20)]],
        "tm": [[time(8, 30), time(23, 59, 59)]],
        "dtm": [[datetime(2024, 1, 1, 12, 0)]],
        "dt_cast": [[date(2024, 1, 1), date(1969, 7, 20)]],
        "tm_cast": [[time(8, 30), time(23, 59, 59)]],
        "dt_explicit": [[date(2024, 1, 1)]],
        "i64_cast": [[100, -50]],
        "f64_cast": [[1.5, -2.7]],
        "str_to_nested_int16": [[[42], [-7]]],
    }
    # schema exactly matches the casts
    assert res.schema == {
        "dt": pl.List(pl.Date),
        "tm": pl.List(pl.Time),
        "dtm": pl.List(pl.Datetime("ms", time_zone=None)),
        "dt_cast": pl.List(pl.Date),
        "tm_cast": pl.List(pl.Time),
        "dt_explicit": pl.List(pl.Date),
        "i64_cast": pl.List(pl.Int64),
        "f64_cast": pl.List(pl.Float64),
        "str_to_nested_int16": pl.List(pl.List(pl.Int16)),
    }


def test_array_typed_literals_mixed_error() -> None:
    with pytest.raises(
        SQLInterfaceError,
        match="expected consistent dtypes",
    ):
        pl.sql("SELECT ARRAY[DATE '2024-01-01', TIME '12:00:00']").collect()

from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import DuplicateError
from polars.testing import assert_frame_equal, assert_series_equal


def test_struct_args_kwargs() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["a", "b"]})

    # Single input
    result = df.select(r=pl.struct((pl.col("a") + pl.col("b")).alias("p")))
    expected = pl.DataFrame({"r": [{"p": 4}, {"p": 6}]})
    assert_frame_equal(result, expected)

    # List input
    result = df.select(r=pl.struct([pl.col("a").alias("p"), pl.col("b").alias("q")]))
    expected = pl.DataFrame({"r": [{"p": 1, "q": 3}, {"p": 2, "q": 4}]})
    assert_frame_equal(result, expected)

    # Positional input
    result = df.select(r=pl.struct(pl.col("a").alias("p"), pl.col("b").alias("q")))
    assert_frame_equal(result, expected)

    # Keyword input
    result = df.select(r=pl.struct(p="a", q="b"))
    assert_frame_equal(result, expected)


def test_struct_with_lit() -> None:
    expr = pl.struct([pl.col("a"), pl.lit(1).alias("b")])

    assert (
        pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        .select(expr)
        .to_dict(as_series=False)
    ) == {"a": []}

    assert (
        pl.DataFrame({"a": pl.Series([1], dtype=pl.Int64)})
        .select(expr)
        .to_dict(as_series=False)
    ) == {"a": [{"a": 1, "b": 1}]}

    assert (
        pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64)})
        .select(expr)
        .to_dict(as_series=False)
    ) == {"a": [{"a": 1, "b": 1}, {"a": 2, "b": 1}]}


def test_eager_struct() -> None:
    with pytest.raises(DuplicateError, match="multiple fields with name '' found"):
        s = pl.struct([pl.Series([1, 2, 3]), pl.Series(["a", "b", "c"])], eager=True)

    s = pl.struct(
        [pl.Series("a", [1, 2, 3]), pl.Series("b", ["a", "b", "c"])], eager=True
    )
    assert s.dtype == pl.Struct


def test_struct_from_schema_only() -> None:
    # Workaround for new streaming engine.
    # we create a dataframe with default types
    df = pl.DataFrame(
        {
            "str": ["a", "b", "c", "d", "e"],
            "u8": [1, 2, 3, 4, 5],
            "i32": [1, 2, 3, 4, 5],
            "f64": [1, 2, 3, 4, 5],
            "cat": ["a", "b", "c", "d", "e"],
            "datetime": pl.Series(
                [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 5),
                ]
            ),
            "bool": [1, 0, 1, 1, 0],
            "list[u8]": [[1], [2], [3], [4], [5]],
        }
    )

    # specify a schema with specific dtypes
    s = df.select(
        pl.struct(
            schema={
                "str": pl.String,
                "u8": pl.UInt8,
                "i32": pl.Int32,
                "f64": pl.Float64,
                "cat": pl.Categorical,
                "datetime": pl.Datetime("ms"),
                "bool": pl.Boolean,
                "list[u8]": pl.List(pl.UInt8),
            }
        ).alias("s")
    )["s"]

    # check dtypes
    assert s.dtype == pl.Struct(
        [
            pl.Field("str", pl.String),
            pl.Field("u8", pl.UInt8),
            pl.Field("i32", pl.Int32),
            pl.Field("f64", pl.Float64),
            pl.Field("cat", pl.Categorical),
            pl.Field("datetime", pl.Datetime("ms")),
            pl.Field("bool", pl.Boolean),
            pl.Field("list[u8]", pl.List(pl.UInt8)),
        ]
    )

    # check values
    assert s.to_list() == [
        {
            "str": "a",
            "u8": 1,
            "i32": 1,
            "f64": 1.0,
            "cat": "a",
            "datetime": datetime(2023, 1, 1, 0, 0),
            "bool": True,
            "list[u8]": [1],
        },
        {
            "str": "b",
            "u8": 2,
            "i32": 2,
            "f64": 2.0,
            "cat": "b",
            "datetime": datetime(2023, 1, 2, 0, 0),
            "bool": False,
            "list[u8]": [2],
        },
        {
            "str": "c",
            "u8": 3,
            "i32": 3,
            "f64": 3.0,
            "cat": "c",
            "datetime": datetime(2023, 1, 3, 0, 0),
            "bool": True,
            "list[u8]": [3],
        },
        {
            "str": "d",
            "u8": 4,
            "i32": 4,
            "f64": 4.0,
            "cat": "d",
            "datetime": datetime(2023, 1, 4, 0, 0),
            "bool": True,
            "list[u8]": [4],
        },
        {
            "str": "e",
            "u8": 5,
            "i32": 5,
            "f64": 5.0,
            "cat": "e",
            "datetime": datetime(2023, 1, 5, 0, 0),
            "bool": False,
            "list[u8]": [5],
        },
    ]


def test_struct_broadcasting() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2],
            "col2": [10, 20],
        }
    )

    assert (
        df.select(
            pl.struct(
                [
                    pl.lit("a").alias("a"),
                    pl.col("col1").alias("col1"),
                ]
            ).alias("my_struct")
        )
    ).to_dict(as_series=False) == {
        "my_struct": [{"a": "a", "col1": 1}, {"a": "a", "col1": 2}]
    }


def test_struct_list_cat_8235() -> None:
    df = pl.DataFrame(
        {"values": [["a", "b", "c"]]}, schema={"values": pl.List(pl.Categorical)}
    )
    assert df.select(pl.struct("values")).to_dict(as_series=False) == {
        "values": [{"values": ["a", "b", "c"]}]
    }


def test_struct_lit_cast() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    schema = {"a": pl.Int64, "b": pl.List(pl.Int64)}

    out = df.select(
        pl.struct(pl.col("a"), pl.lit(None).alias("b"), schema=schema)  # type: ignore[arg-type]
    ).get_column("a")

    expected = pl.Series(
        "a",
        [
            {"a": 1, "b": None},
            {"a": 2, "b": None},
            {"a": 3, "b": None},
        ],
        dtype=pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.List(pl.Int64))]),
    )
    assert_series_equal(out, expected)

    out = df.select(
        pl.struct([pl.col("a"), pl.lit(pl.Series([[]])).alias("b")], schema=schema)  # type: ignore[arg-type]
    ).get_column("a")

    expected = pl.Series(
        "a",
        [
            {"a": 1, "b": []},
            {"a": 2, "b": []},
            {"a": 3, "b": []},
        ],
        dtype=pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.List(pl.Int64))]),
    )
    assert_series_equal(out, expected)


def test_suffix_in_struct_creation() -> None:
    assert (
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        ).select(pl.struct(pl.col(["a", "c"]).name.suffix("_foo")).alias("bar"))
    ).unnest("bar").to_dict(as_series=False) == {"a_foo": [1, 2], "c_foo": [5, 6]}


def test_resolved_names_15442() -> None:
    df = pl.DataFrame(
        {
            "x": [206.0],
            "y": [225.0],
        }
    )
    center = pl.struct(
        x=pl.col("x"),
        y=pl.col("y"),
    )

    left = 0
    right = 1000
    in_x = (left < center.struct.field("x")) & (center.struct.field("x") <= right)
    assert df.lazy().filter(in_x).collect().shape == (1, 2)


def test_error_on_duplicate_field_name_22959() -> None:
    with pytest.raises(DuplicateError, match="'literal'"):
        pl.select(
            pl.struct(
                pl.lit(1),
                pl.lit(2),
            )
        )


def test_struct_nested_naming_in_group_by_23701() -> None:
    df = pl.LazyFrame({"ID": [1], "SOURCE_FIELD": ["some value"]})

    expr_inner_struct = pl.struct(
        pl.col("SOURCE_FIELD").alias("INNER_FIELD"),
    )

    expr_outer_struct = pl.struct(
        pl.lit(date(2026, 1, 1))
        .dt.offset_by(pl.int_ranges(0, 3).list.last().cast(pl.String) + "d")
        .alias("OUTER_FIELD"),
        expr_inner_struct.alias("INNER_STRUCT"),
    ).alias("OUTER_STRUCT")

    agg_df = df.group_by("ID").agg(expr_outer_struct)

    assert agg_df.collect_schema() == agg_df.collect().schema


# parametric tuples: (expr, is_scalar)
agg_expressions = [
    (pl.lit(1, pl.Int64), True),  # LiteralScalar
    (pl.col("n"), False),  # NotAggregated
    (pl.int_range(1, pl.len() + 1), False),  # AggregatedList
    (pl.col("n").first(), True),  # AggregatedScalar
]


@pytest.mark.parametrize("lhs", agg_expressions)
@pytest.mark.parametrize("rhs", agg_expressions)
@pytest.mark.parametrize("n_rows", [0, 1, 2, 3])
@pytest.mark.parametrize("maintain_order", [True, False])
def test_struct_schema_in_group_by_apply_expr_24168(
    lhs: tuple[pl.Expr, bool],
    rhs: tuple[pl.Expr, bool],
    n_rows: int,
    maintain_order: bool,
) -> None:
    df = pl.DataFrame({"g": [10, 10, 20], "n": [1, 2, 3]})
    lf = df.head(n_rows).lazy()
    expr = pl.struct(lhs[0].alias("lhs"), rhs[0].alias("rhs")).alias("expr")
    q = lf.group_by("g", maintain_order=maintain_order).agg(expr)
    out = q.collect()

    # check schema
    assert q.collect_schema() == out.schema

    # check output against ground truth (single group only)
    if n_rows in [0, 1]:
        if lhs[1] and rhs[1]:
            expected = pl.DataFrame({"g": [10], "expr": [{"lhs": 1, "rhs": 1}]})
            expected = expected.head(n_rows)
            assert_frame_equal(out, expected, check_row_order=maintain_order)
        else:
            expected = pl.DataFrame({"g": [10], "expr": [[{"lhs": 1, "rhs": 1}]]})
            expected = expected.head(n_rows)
            assert_frame_equal(out, expected, check_row_order=maintain_order)

    if n_rows == 2:
        if lhs[1] and rhs[1]:
            expected = pl.DataFrame({"g": [10], "expr": [{"lhs": 1, "rhs": 1}]})
            assert_frame_equal(out, expected, check_row_order=maintain_order)
        else:
            expected = pl.DataFrame(
                {
                    "g": [10],
                    "expr": [
                        [
                            {"lhs": 1, "rhs": 1},
                            {"lhs": 1 if lhs[1] else 2, "rhs": 1 if rhs[1] else 2},
                        ]
                    ],
                }
            )
            assert_frame_equal(out, expected, check_row_order=maintain_order)

    # check output against non_aggregated expression evaluation
    if n_rows in [1, 2, 3]:
        grouped = df.head(n_rows).group_by("g", maintain_order=maintain_order)

        out_non_agg = pl.DataFrame({})
        for df_group in grouped:
            df = df_group[1]
            if lhs[1] and rhs[1]:
                df = df.head(1)
                df = df.select(["g", expr])
            else:
                df = df.select(["g", expr.implode()]).head(1)
            out_non_agg = out_non_agg.vstack(df)

        assert_frame_equal(out, out_non_agg, check_row_order=maintain_order)

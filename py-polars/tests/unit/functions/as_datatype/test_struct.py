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

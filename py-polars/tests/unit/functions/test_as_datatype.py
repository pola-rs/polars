from datetime import date, datetime

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_date_datetime() -> None:
    df = pl.DataFrame(
        {
            "year": [2001, 2002, 2003],
            "month": [1, 2, 3],
            "day": [1, 2, 3],
            "hour": [23, 12, 8],
        }
    )
    out = df.select(
        pl.all(),
        pl.datetime("year", "month", "day", "hour").dt.hour().cast(int).alias("h2"),
        pl.date("year", "month", "day").dt.day().cast(int).alias("date"),
    )
    assert_series_equal(out["date"], df["day"].rename("date"))
    assert_series_equal(out["h2"], df["hour"].rename("h2"))


def test_time() -> None:
    df = pl.DataFrame(
        {
            "hour": [7, 14, 21],
            "min": [10, 20, 30],
            "sec": [15, 30, 45],
            "micro": [123456, 555555, 987654],
        }
    )
    out = df.select(
        pl.all(),
        pl.time("hour", "min", "sec", "micro").dt.hour().cast(int).alias("h2"),
        pl.time("hour", "min", "sec", "micro").dt.minute().cast(int).alias("m2"),
        pl.time("hour", "min", "sec", "micro").dt.second().cast(int).alias("s2"),
        pl.time("hour", "min", "sec", "micro").dt.microsecond().cast(int).alias("ms2"),
    )
    assert_series_equal(out["h2"], df["hour"].rename("h2"))
    assert_series_equal(out["m2"], df["min"].rename("m2"))
    assert_series_equal(out["s2"], df["sec"].rename("s2"))
    assert_series_equal(out["ms2"], df["micro"].rename("ms2"))


def test_empty_duration() -> None:
    s = pl.DataFrame([], {"days": pl.Int32}).select(pl.duration(days="days"))
    assert s.dtypes == [pl.Duration("ns")]
    assert s.shape == (0, 1)


def test_list_concat() -> None:
    s0 = pl.Series("a", [[1, 2]])
    s1 = pl.Series("b", [[3, 4, 5]])
    expected = pl.Series("a", [[1, 2, 3, 4, 5]])

    out = s0.list.concat([s1])
    assert_series_equal(out, expected)

    out = s0.list.concat(s1)
    assert_series_equal(out, expected)

    df = pl.DataFrame([s0, s1])
    assert_series_equal(df.select(pl.concat_list(["a", "b"]).alias("a"))["a"], expected)
    assert_series_equal(
        df.select(pl.col("a").list.concat("b").alias("a"))["a"], expected
    )
    assert_series_equal(
        df.select(pl.col("a").list.concat(["b"]).alias("a"))["a"], expected
    )


def test_concat_list_with_lit() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert df.select(pl.concat_list([pl.col("a"), pl.lit(1)]).alias("a")).to_dict(
        False
    ) == {"a": [[1, 1], [2, 1], [3, 1]]}

    assert df.select(pl.concat_list([pl.lit(1), pl.col("a")]).alias("a")).to_dict(
        False
    ) == {"a": [[1, 1], [1, 2], [1, 3]]}


def test_concat_list_empty_raises() -> None:
    with pytest.raises(pl.ComputeError):
        pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.concat_list([]))


def test_list_concat_nulls() -> None:
    assert pl.DataFrame(
        {
            "a": [["a", "b"], None, ["c", "d", "e"], None],
            "t": [["x"], ["y"], None, None],
        }
    ).with_columns(pl.concat_list(["a", "t"]).alias("concat"))["concat"].to_list() == [
        ["a", "b", "x"],
        None,
        None,
        None,
    ]


def test_concat_list_in_agg_6397() -> None:
    df = pl.DataFrame({"group": [1, 2, 2, 3], "value": ["a", "b", "c", "d"]})

    # single list
    assert df.groupby("group").agg(
        [
            # this casts every element to a list
            pl.concat_list(pl.col("value")),
        ]
    ).sort("group").to_dict(False) == {
        "group": [1, 2, 3],
        "value": [[["a"]], [["b"], ["c"]], [["d"]]],
    }

    # nested list
    assert df.groupby("group").agg(
        [
            pl.concat_list(pl.col("value").implode()).alias("result"),
        ]
    ).sort("group").to_dict(False) == {
        "group": [1, 2, 3],
        "result": [[["a"]], [["b", "c"]], [["d"]]],
    }


def test_list_concat_supertype() -> None:
    df = pl.DataFrame(
        [pl.Series("a", [1, 2], pl.UInt8), pl.Series("b", [10000, 20000], pl.UInt16)]
    )
    assert df.with_columns(pl.concat_list(pl.col(["a", "b"])).alias("concat_list"))[
        "concat_list"
    ].to_list() == [[1, 10000], [2, 20000]]


def test_categorical_list_concat_4762() -> None:
    df = pl.DataFrame({"x": "a"})
    expected = {"x": [["a", "a"]]}

    q = df.lazy().select([pl.concat_list([pl.col("x").cast(pl.Categorical)] * 2)])
    with pl.StringCache():
        assert q.collect().to_dict(False) == expected


def test_list_concat_rolling_window() -> None:
    # inspired by:
    # https://stackoverflow.com/questions/70377100/use-the-rolling-function-of-polars-to-get-a-list-of-all-values-in-the-rolling-wi
    # this tests if it works without specifically creating list dtype upfront. note that
    # the given answer is preferred over this snippet as that reuses the list array when
    # shifting
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        }
    )
    out = df.with_columns(
        [pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)]
    ).select(
        [pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")]
    )
    assert out.shape == (5, 1)

    s = out.to_series()
    assert s.dtype == pl.List
    assert s.to_list() == [
        [None, None, 1.0],
        [None, 1.0, 2.0],
        [1.0, 2.0, 9.0],
        [2.0, 9.0, 2.0],
        [9.0, 2.0, 13.0],
    ]

    # this test proper null behavior of concat list
    out = (
        df.with_columns(pl.col("A").reshape((-1, 1)))  # first turn into a list
        .with_columns(
            [
                pl.col("A").shift(i).alias(f"A_lag_{i}")
                for i in range(3)  # slice the lists to a lag
            ]
        )
        .select(
            [
                pl.all(),
                pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias(
                    "A_rolling"
                ),
            ]
        )
    )
    assert out.shape == (5, 5)

    l64 = pl.List(pl.Float64)
    assert out.schema == {
        "A": l64,
        "A_lag_0": l64,
        "A_lag_1": l64,
        "A_lag_2": l64,
        "A_rolling": l64,
    }


def test_concat_list_reverse_struct_fields() -> None:
    df = pl.DataFrame({"nums": [1, 2, 3, 4], "letters": ["a", "b", "c", "d"]}).select(
        [
            pl.col("nums"),
            pl.struct(["letters", "nums"]).alias("combo"),
            pl.struct(["nums", "letters"]).alias("reverse_combo"),
        ]
    )
    result1 = df.select(pl.concat_list(["combo", "reverse_combo"]))
    result2 = df.select(pl.concat_list(["combo", "combo"]))
    assert_frame_equal(result1, result2)


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
        pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}).select(expr).to_dict(False)
    ) == {"a": []}

    assert (
        pl.DataFrame({"a": pl.Series([1], dtype=pl.Int64)}).select(expr).to_dict(False)
    ) == {"a": [{"a": 1, "b": 1}]}

    assert (
        pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64)})
        .select(expr)
        .to_dict(False)
    ) == {"a": [{"a": 1, "b": 1}, {"a": 2, "b": 1}]}


def test_eager_struct() -> None:
    with pytest.raises(pl.DuplicateError, match="multiple fields with name '' found"):
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
                "str": pl.Utf8,
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
            pl.Field("str", pl.Utf8),
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
    ).to_dict(False) == {"my_struct": [{"a": "a", "col1": 1}, {"a": "a", "col1": 2}]}


def test_struct_list_cat_8235() -> None:
    df = pl.DataFrame(
        {"values": [["a", "b", "c"]]}, schema={"values": pl.List(pl.Categorical)}
    )
    assert df.select(pl.struct("values")).to_dict(False) == {
        "values": [{"values": ["a", "b", "c"]}]
    }


def test_struct_lit_cast() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    schema = {"a": pl.Int64, "b": pl.List(pl.Int64)}

    for lit in [pl.lit(None), pl.lit([[]])]:
        s = df.select(pl.struct([pl.col("a"), lit.alias("b")], schema=schema))["a"]  # type: ignore[arg-type]
        assert s.dtype == pl.Struct(
            [pl.Field("a", pl.Int64), pl.Field("b", pl.List(pl.Int64))]
        )
        assert s.to_list() == [
            {"a": 1, "b": None},
            {"a": 2, "b": None},
            {"a": 3, "b": None},
        ]


def test_suffix_in_struct_creation() -> None:
    assert (
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        ).select(pl.struct(pl.col(["a", "c"]).suffix("_foo")).alias("bar"))
    ).unnest("bar").to_dict(False) == {"a_foo": [1, 2], "c_foo": [5, 6]}


def test_concat_str() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.concat_str(["a", "b"], separator="-")])
    assert out["a"].to_list() == ["a-1", "b-2", "c-3"]


def test_concat_str_wildcard_expansion() -> None:
    # one function requires wildcard expansion the other need
    # this tests the nested behavior
    # see: #2867

    df = pl.DataFrame({"a": ["x", "Y", "z"], "b": ["S", "o", "S"]})
    assert df.select(
        pl.concat_str(pl.all()).str.to_lowercase()
    ).to_series().to_list() == ["xs", "yo", "zs"]


def test_format() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    out = df.select([pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt")])
    assert out["fmt"].to_list() == ["foo_a_bar_1", "foo_b_bar_2", "foo_c_bar_3"]


def test_struct_deprecation_exprs_keyword() -> None:
    with pytest.deprecated_call():
        result = pl.select(pl.struct(exprs=1.0))

    expected = pl.DataFrame(
        {"literal": [{"literal": 1.0}]},
        schema={"literal": pl.Struct({"literal": pl.Float64})},
    )
    assert_frame_equal(result, expected)

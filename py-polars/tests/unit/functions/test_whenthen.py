from datetime import datetime

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_when_then() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    expr = pl.when(pl.col("a") < 3).then(pl.lit("x"))

    result = df.select(
        expr.otherwise(pl.lit("y")).alias("a"),
        expr.alias("b"),
    )

    expected = pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "y"],
            "b": ["x", "x", None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_chained() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    expr = (
        pl.when(pl.col("a") < 3)
        .then(pl.lit("x"))
        .when(pl.col("a") > 4)
        .then(pl.lit("z"))
    )

    result = df.select(
        expr.otherwise(pl.lit("y")).alias("a"),
        expr.alias("b"),
    )

    expected = pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "z"],
            "b": ["x", "x", None, None, "z"],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_invalid_chains() -> None:
    with pytest.raises(AttributeError):
        pl.when("a").when("b")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").otherwise(2)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).then(2)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).otherwise(2).otherwise(3)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").when("c")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").otherwise("2")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").then(2).when("c").when("d")  # type: ignore[attr-defined]


def test_when_then_implicit_none() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "C"],
            "points": [11, 8, 10, 6, 6, 5],
        }
    )

    result = df.select(
        pl.when(pl.col("points") > 7).then(pl.lit("Foo")),
        pl.when(pl.col("points") > 7).then(pl.lit("Foo")).alias("bar"),
    )

    expected = pl.DataFrame(
        {
            "literal": ["Foo", "Foo", "Foo", None, None, None],
            "bar": ["Foo", "Foo", "Foo", None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_empty_list_5547() -> None:
    out = pl.DataFrame({"a": []}).select([pl.when(pl.col("a") > 1).then([1])])
    assert out.shape == (0, 1)
    assert out.dtypes == [pl.List(pl.Int64)]


def test_nested_when_then_and_wildcard_expansion_6284() -> None:
    df = pl.DataFrame(
        {
            "1": ["a", "b"],
            "2": ["c", "d"],
        }
    )

    out0 = df.with_columns(
        pl.when(pl.any_horizontal(pl.all() == "a"))
        .then(pl.lit("a"))
        .otherwise(
            pl.when(pl.any_horizontal(pl.all() == "d"))
            .then(pl.lit("d"))
            .otherwise(None)
        )
        .alias("result")
    )

    out1 = df.with_columns(
        pl.when(pl.any_horizontal(pl.all() == "a"))
        .then(pl.lit("a"))
        .when(pl.any_horizontal(pl.all() == "d"))
        .then(pl.lit("d"))
        .otherwise(None)
        .alias("result")
    )

    assert_frame_equal(out0, out1)
    assert out0.to_dict(as_series=False) == {
        "1": ["a", "b"],
        "2": ["c", "d"],
        "result": ["a", "d"],
    }


def test_list_zip_with_logical_type() -> None:
    df = pl.DataFrame(
        {
            "start": [datetime(2023, 1, 1, 1, 1, 1), datetime(2023, 1, 1, 1, 1, 1)],
            "stop": [datetime(2023, 1, 1, 1, 3, 1), datetime(2023, 1, 1, 1, 4, 1)],
            "use": [1, 0],
        }
    )

    df = df.with_columns(
        pl.datetime_ranges(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_1"),
        pl.datetime_ranges(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_2"),
    )

    out = df.select(
        pl.when(pl.col("use") == 1)
        .then(pl.col("interval_2"))
        .otherwise(pl.col("interval_1"))
        .alias("interval_new")
    )
    assert out.dtypes == [pl.List(pl.Datetime(time_unit="us", time_zone=None))]


def test_type_coercion_when_then_otherwise_2806() -> None:
    out = (
        pl.DataFrame({"names": ["foo", "spam", "spam"], "nrs": [1, 2, 3]})
        .select(
            [
                pl.when(pl.col("names") == "spam")
                .then(pl.col("nrs") * 2)
                .otherwise(pl.lit("other"))
                .alias("new_col"),
            ]
        )
        .to_series()
    )
    expected = pl.Series("new_col", ["other", "4", "6"])
    assert out.to_list() == expected.to_list()

    # test it remains float32
    assert (
        pl.Series("a", [1.0, 2.0, 3.0], dtype=pl.Float32)
        .to_frame()
        .select(pl.when(pl.col("a") > 2.0).then(pl.col("a")).otherwise(0.0))
    ).to_series().dtype == pl.Float32


def test_when_then_edge_cases_3994() -> None:
    df = pl.DataFrame(data={"id": [1, 1], "type": [2, 2]})

    # this tests if lazy correctly assigns the list schema to the column aggregation
    assert (
        df.lazy()
        .group_by(["id"])
        .agg(pl.col("type"))
        .with_columns(
            pl.when(pl.col("type").list.len() == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("type"))
            .name.keep()
        )
        .collect()
    ).to_dict(as_series=False) == {"id": [1], "type": [[2, 2]]}

    # this tests ternary with an empty argument
    assert (
        df.filter(pl.col("id") == 42)
        .group_by(["id"])
        .agg(pl.col("type"))
        .with_columns(
            pl.when(pl.col("type").list.len() == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("type"))
            .name.keep()
        )
    ).to_dict(as_series=False) == {"id": [], "type": []}


def test_object_when_then_4702() -> None:
    # please don't ever do this
    x = pl.DataFrame({"Row": [1, 2], "Type": [pl.Date, pl.UInt8]})

    assert x.with_columns(
        pl.when(pl.col("Row") == 1)
        .then(pl.lit(pl.UInt16, allow_object=True))
        .otherwise(pl.lit(pl.UInt8, allow_object=True))
        .alias("New_Type")
    ).to_dict(as_series=False) == {
        "Row": [1, 2],
        "Type": [pl.Date, pl.UInt8],
        "New_Type": [pl.UInt16, pl.UInt8],
    }


def test_comp_categorical_lit_dtype() -> None:
    df = pl.DataFrame(
        data={"column": ["a", "b", "e"], "values": [1, 5, 9]},
        schema=[("column", pl.Categorical), ("more", pl.Int32)],
    )

    assert df.with_columns(
        pl.when(pl.col("column") == "e")
        .then(pl.lit("d"))
        .otherwise(pl.col("column"))
        .alias("column")
    ).dtypes == [pl.Categorical, pl.Int32]


def test_when_then_deprecated_string_input() -> None:
    df = pl.DataFrame(
        {
            "a": [True, False],
            "b": [1, 2],
            "c": [3, 4],
        }
    )

    with pytest.deprecated_call():
        result = df.select(pl.when("a").then("b").otherwise("c").alias("when"))

    expected = pl.Series("when", ["b", "c"])
    assert_series_equal(result.to_series(), expected)

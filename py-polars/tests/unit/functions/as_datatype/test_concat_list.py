import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal


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

    result = df.select(pl.concat_list([pl.col("a"), pl.lit(1)]).alias("a"))
    expected = {"a": [[1, 1], [2, 1], [3, 1]]}
    assert result.to_dict(as_series=False) == expected

    result = df.select(pl.concat_list([pl.lit(1), pl.col("a")]).alias("a"))
    expected = {"a": [[1, 1], [1, 2], [1, 3]]}
    assert result.to_dict(as_series=False) == expected


def test_concat_list_empty_raises() -> None:
    with pytest.raises(ComputeError):
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
    assert df.group_by("group").agg(
        [
            # this casts every element to a list
            pl.concat_list(pl.col("value")),
        ]
    ).sort("group").to_dict(as_series=False) == {
        "group": [1, 2, 3],
        "value": [[["a"]], [["b"], ["c"]], [["d"]]],
    }

    # nested list
    assert df.group_by("group").agg(
        [
            pl.concat_list(pl.col("value").implode()).alias("result"),
        ]
    ).sort("group").to_dict(as_series=False) == {
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
        assert q.collect().to_dict(as_series=False) == expected


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
        pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)
    ).select(pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling"))
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
        df.with_columns(
            pl.col("A").reshape((-1, 1)).arr.to_list()  # first turn into a list
        )
        .with_columns(
            pl.col("A").shift(i).alias(f"A_lag_{i}")
            for i in range(3)  # slice the lists to a lag
        )
        .select(
            pl.all(),
            pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling"),
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
        pl.col("nums"),
        pl.struct(["letters", "nums"]).alias("combo"),
        pl.struct(["nums", "letters"]).alias("reverse_combo"),
    )
    result1 = df.select(pl.concat_list(["combo", "reverse_combo"]))
    result2 = df.select(pl.concat_list(["combo", "combo"]))
    assert_frame_equal(result1, result2)


def test_concat_list_empty() -> None:
    df = pl.DataFrame({"a": []})
    df.select(pl.concat_list("a"))


def test_concat_list_empty_struct() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.Struct({"b": pl.Boolean})})
    df.select(pl.concat_list("a"))

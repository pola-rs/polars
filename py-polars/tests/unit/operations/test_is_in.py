from __future__ import annotations

from datetime import date

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_struct_logical_is_in() -> None:
    df1 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 1), date(2022, 1, 7), eager=True),
            "y": [0, 4, 6, 2, 3, 4, 5],
        }
    )
    df2 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 3), date(2022, 1, 9), eager=True),
            "y": [6, 2, 3, 4, 5, 0, 1],
        }
    )

    s1 = df1.select(pl.struct(["x", "y"])).to_series()
    s2 = df2.select(pl.struct(["x", "y"])).to_series()

    assert s1.is_in(s2).to_list() == [False, False, True, True, True, True, True]


def test_is_in_bool() -> None:
    vals = [True, None]
    df = pl.DataFrame({"A": [True, False, None]})
    assert df.select(pl.col("A").is_in(vals)).to_dict(False) == {
        "A": [True, False, None]
    }


def test_is_in_empty_list_4559() -> None:
    assert pl.Series(["a"]).is_in([]).to_list() == [False]


def test_is_in_empty_list_4639() -> None:
    df = pl.DataFrame({"a": [1, None]})
    empty_list: list[int] = []

    assert df.with_columns([pl.col("a").is_in(empty_list).alias("a_in_list")]).to_dict(
        False
    ) == {"a": [1, None], "a_in_list": [False, None]}


def test_is_in_struct() -> None:
    df = pl.DataFrame(
        {
            "struct_elem": [{"a": 1, "b": 11}, {"a": 1, "b": 90}],
            "struct_list": [
                [{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}],
                [{"a": 3, "b": 3}],
            ],
        }
    )

    assert df.filter(pl.col("struct_elem").is_in("struct_list")).to_dict(False) == {
        "struct_elem": [{"a": 1, "b": 11}],
        "struct_list": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}]],
    }


def test_is_in_null_prop() -> None:
    assert pl.Series([None], dtype=pl.Float32).is_in(pl.Series([42])).item() is None
    assert (
        pl.Series([{"a": None}], dtype=pl.Struct({"a": pl.Float32}))
        .is_in(pl.Series([{"a": 42}]))
        .item()
        is None
    )
    assert pl.Series([None], dtype=pl.Boolean).is_in(pl.Series([42])).item() is None
    assert (
        pl.Series([{"a": None}], dtype=pl.Struct({"a": pl.Boolean}))
        .is_in(pl.Series([{"a": 42}]))
        .item()
        is None
    )


def test_is_in_9070() -> None:
    assert not pl.Series([1]).is_in(pl.Series([1.99])).item()


def test_is_in_float_list_10764() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1.0, 2.0, 3.0, 4.0, 5.0], [3.14, 5.28]],
            "n": [3.0, 2.0],
        }
    )

    assert df.select(pl.col("n").is_in("lst").alias("is_in")).to_dict(False) == {
        "is_in": [True, False]
    }


def test_is_in_df() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(pl.col("a").is_in([1, 2]))["a"].to_list() == [
        True,
        True,
        False,
    ]


def test_is_in_series() -> None:
    s = pl.Series(["a", "b", "c"])

    out = s.is_in(["a", "b"])
    assert out.to_list() == [True, True, False]

    # Check if empty list is converted to pl.Utf8.
    out = s.is_in([])
    assert out.to_list() == [False] * out.len()

    for x_y_z in (["x", "y", "z"], {"x", "y", "z"}):
        out = s.is_in(x_y_z)
        assert out.to_list() == [False, False, False]

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4], "c": ["e", "d"]})
    assert df.select(pl.col("a").is_in(pl.col("b"))).to_series().to_list() == [
        True,
        False,
    ]
    assert df.select(pl.col("b").is_in([])).to_series().to_list() == [False] * df.height

    with pytest.raises(pl.ComputeError, match=r"cannot compare"):
        df.select(pl.col("b").is_in(["x", "x"]))

    # check we don't shallow-copy and accidentally modify 'a' (see: #10072)
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [1, 3]).is_in(a)

    assert a.name == "a"
    assert_series_equal(b, pl.Series("b", [True, False]))


def test_is_in_invalid_shape() -> None:
    with pytest.raises(pl.ComputeError):
        pl.Series("a", [1, 2, 3]).is_in([[]])

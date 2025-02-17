from __future__ import annotations

import polars as pl
from polars import col
from polars.testing import assert_frame_equal


def test_col_select() -> None:
    df = pl.DataFrame(
        {
            "ham": [1, 2, 3],
            "hamburger": [11, 22, 33],
            "foo": [3, 2, 1],
            "bar": ["a", "b", "c"],
        }
    )

    # Single column
    assert df.select(pl.col("foo")).columns == ["foo"]
    # Regex
    assert df.select(pl.col("*")).columns == ["ham", "hamburger", "foo", "bar"]
    assert df.select(pl.col("^ham.*$")).columns == ["ham", "hamburger"]
    assert df.select(pl.col("*").exclude("ham")).columns == ["hamburger", "foo", "bar"]
    # Multiple inputs
    assert df.select(pl.col(["hamburger", "foo"])).columns == ["hamburger", "foo"]
    assert df.select(pl.col("hamburger", "foo")).columns == ["hamburger", "foo"]
    assert df.select(pl.col(pl.Series(["ham", "foo"]))).columns == ["ham", "foo"]
    # Dtypes
    assert df.select(pl.col(pl.String)).columns == ["bar"]
    assert df.select(pl.col(pl.Int64, pl.Float64)).columns == [
        "ham",
        "hamburger",
        "foo",
    ]


def test_col_series_selection() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [1], "c": [1]})
    srs = pl.Series(["b", "c"])

    assert ldf.select(pl.col(srs)).collect_schema().names() == ["b", "c"]


def test_col_dot_style() -> None:
    df = pl.DataFrame({"lower": 1, "UPPER": 2, "_underscored": 3})

    result = df.select(
        col.lower,
        col.UPPER,
        col._underscored,
    )

    expected = df.select("lower", "UPPER", "_underscored")
    assert_frame_equal(result, expected)

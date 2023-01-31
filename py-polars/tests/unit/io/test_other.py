from __future__ import annotations

import copy
from typing import cast

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    assert_frame_equal(copy.copy(df), df)
    assert_frame_equal(copy.deepcopy(df), df)

    a = pl.Series("a", [1, 2])
    assert_series_equal(copy.copy(a), a)
    assert_series_equal(copy.deepcopy(a), a)


def test_categorical_round_trip() -> None:
    df = pl.DataFrame({"ints": [1, 2, 3], "cat": ["a", "b", "c"]})
    df = df.with_columns(pl.col("cat").cast(pl.Categorical))

    tbl = df.to_arrow()
    assert "dictionary" in str(tbl["cat"].type)

    df2 = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert df2.dtypes == [pl.Int64, pl.Categorical]


def test_date_list_fmt() -> None:
    df = pl.DataFrame(
        {
            "mydate": ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-05"],
            "index": [1, 2, 5, 5],
        }
    )

    df = df.with_columns(pl.col("mydate").str.strptime(pl.Date, "%Y-%m-%d"))
    assert (
        str(df.groupby("index", maintain_order=True).agg(pl.col("mydate"))["mydate"])
        == """shape: (3,)
Series: 'mydate' [list[date]]
[
	[2020-01-01]
	[2020-01-02]
	[2020-01-05, 2020-01-05]
]"""
    )


def test_from_different_chunks() -> None:
    s0 = pl.Series("a", [1, 2, 3, 4, None])
    s1 = pl.Series("b", [1, 2])
    s11 = pl.Series("b", [1, 2, 3])
    s1.append(s11)

    # check we don't panic
    df = pl.DataFrame([s0, s1])
    df.to_arrow()
    df = pl.DataFrame([s0, s1])
    out = df.to_pandas()
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (5, 2)

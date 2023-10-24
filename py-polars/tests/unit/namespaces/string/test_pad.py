from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_str_pad_start() -> None:
    df = pl.DataFrame({"a": ["foo", "longer_foo", "longest_fooooooo", "hi"]})

    result = df.select(
        pl.col("a").str.pad_start(10).alias("padded"),
        pl.col("a").str.pad_start(10).str.len_bytes().alias("padded_len"),
    )

    expected = pl.DataFrame(
        {
            "padded": ["       foo", "longer_foo", "longest_fooooooo", "        hi"],
            "padded_len": [10, 10, 16, 10],
        },
        schema_overrides={"padded_len": pl.UInt32}
    )
    assert_frame_equal(result, expected)


def test_str_pad_end() -> None:
    df = pl.DataFrame({"a": ["foo", "longer_foo", "longest_fooooooo", "hi"]})

    result = df.select(
        pl.col("a").str.pad_end(10).alias("padded"),
        pl.col("a").str.pad_end(10).str.len_bytes().alias("padded_len"),
    )

    expected = pl.DataFrame(
        {
            "padded": ["foo       ", "longer_foo", "longest_fooooooo", "hi        "],
            "padded_len": [10, 10, 16, 10],
        },
        schema_overrides={"padded_len": pl.UInt32}
    )
    assert_frame_equal(result, expected)


def test_str_zfill() -> None:
    df = pl.DataFrame(
        {
            "num": [-10, -1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000, None],
        }
    )
    out = [
        "-0010",
        "-0001",
        "00000",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "100000",
        "1000000",
        None,
    ]
    assert (
        df.with_columns(pl.col("num").cast(str).str.zfill(5)).to_series().to_list()
        == out
    )
    assert df["num"].cast(str).str.zfill(5).to_list() == out


def test_str_ljust_deprecated() -> None:
    s = pl.Series(["a", "bc", "def"])

    with pytest.deprecated_call():
        result = s.str.ljust(5)

    expected = pl.Series(["a    ", "bc   ", "def  "])
    assert_series_equal(result, expected)


def test_str_rjust_deprecated() -> None:
    s = pl.Series(["a", "bc", "def"])

    with pytest.deprecated_call():
        result = s.str.rjust(5)

    expected = pl.Series(["    a", "   bc", "  def"])
    assert_series_equal(result, expected)

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


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
        schema_overrides={"padded_len": pl.UInt32},
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
        schema_overrides={"padded_len": pl.UInt32},
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


def test_str_zfill_expr() -> None:
    df = pl.DataFrame(
        {
            "num": ["-10", "-1", "0", "1", "10", None, "1"],
            "len": [3, 4, 3, 2, 5, 3, None],
        }
    )
    out = df.select(
        all_expr=pl.col("num").str.zfill(pl.col("len")),
        str_lit=pl.lit("10").str.zfill(pl.col("len")),
        len_lit=pl.col("num").str.zfill(5),
    )
    expected = pl.DataFrame(
        {
            "all_expr": ["-10", "-001", "000", "01", "00010", None, None],
            "str_lit": ["010", "0010", "010", "10", "00010", "010", None],
            "len_lit": ["-0010", "-0001", "00000", "00001", "00010", None, "00001"],
        }
    )
    assert_frame_equal(out, expected)


def test_pad_end_unicode() -> None:
    lf = pl.LazyFrame({"a": ["Café", "345", "東京", None]})

    result = lf.select(pl.col("a").str.pad_end(6, "日"))

    expected = pl.LazyFrame({"a": ["Café日日", "345日日日", "東京日日日日", None]})
    assert_frame_equal(result, expected)


def test_pad_start_unicode() -> None:
    lf = pl.LazyFrame({"a": ["Café", "345", "東京", None]})

    result = lf.select(pl.col("a").str.pad_start(6, "日"))

    expected = pl.LazyFrame({"a": ["日日Café", "日日日345", "日日日日東京", None]})
    assert_frame_equal(result, expected)


def test_str_zfill_unicode_not_respected() -> None:
    lf = pl.LazyFrame({"a": ["Café", "345", "東京", None]})

    result = lf.select(pl.col("a").str.zfill(6))

    expected = pl.LazyFrame({"a": ["0Café", "000345", "東京", None]})
    assert_frame_equal(result, expected)

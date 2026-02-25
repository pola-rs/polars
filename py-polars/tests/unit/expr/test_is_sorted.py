from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_is_sorted_ascending() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    result = df.select(pl.col("a").is_sorted())
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_is_sorted_descending() -> None:
    df = pl.DataFrame({"a": [4, 3, 2, 1]})
    result = df.select(pl.col("a").is_sorted(descending=True))
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_is_sorted_unsorted() -> None:
    df = pl.DataFrame({"a": [3, 1, 4, 2]})
    result = df.select(pl.col("a").is_sorted())
    expected = pl.DataFrame({"a": [False]})
    assert_frame_equal(result, expected)


def test_is_sorted_descending_none_accepts_either() -> None:
    df = pl.DataFrame({"a": [4, 3, 2, 1]})
    result = df.select(pl.col("a").is_sorted(descending=None))
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_is_sorted_with_nulls_last() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None]})
    result = df.select(pl.col("a").is_sorted(nulls_last=True))
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)

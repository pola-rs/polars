from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_is_sorted_ascending() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    result = df.lazy().select(pl.col("a").is_sorted()).collect(engine="streaming")
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_streaming_is_sorted_descending() -> None:
    df = pl.DataFrame({"a": [4, 3, 2, 1]})
    result = (
        df.lazy()
        .select(pl.col("a").is_sorted(descending=True))
        .collect(engine="streaming")
    )
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_streaming_is_sorted_unsorted() -> None:
    df = pl.DataFrame({"a": [3, 1, 4, 2]})
    result = df.lazy().select(pl.col("a").is_sorted()).collect(engine="streaming")
    expected = pl.DataFrame({"a": [False]})
    assert_frame_equal(result, expected)


def test_streaming_is_sorted_descending_none() -> None:
    df = pl.DataFrame({"a": [4, 3, 2, 1]})
    result = (
        df.lazy()
        .select(pl.col("a").is_sorted(descending=None))
        .collect(engine="streaming")
    )
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)


def test_streaming_is_sorted_with_nulls_last() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None]})
    result = (
        df.lazy()
        .select(pl.col("a").is_sorted(nulls_last=True))
        .collect(engine="streaming")
    )
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected)

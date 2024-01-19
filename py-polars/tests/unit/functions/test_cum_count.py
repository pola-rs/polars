from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize(("reverse", "output"), [(False, [1, 2, 3]), (True, [3, 2, 1])])
def test_cum_count_no_args(reverse: bool, output: list[int]) -> None:
    df = pl.DataFrame({"a": [5, 5, None]})
    with pytest.deprecated_call():
        result = df.select(pl.cum_count(reverse=reverse))
    expected = pl.Series("cum_count", output, dtype=pl.UInt32).to_frame()
    assert_frame_equal(result, expected)


def test_cum_count_single_arg() -> None:
    df = pl.DataFrame({"a": [5, 5, None]})
    result = df.select(pl.cum_count("a"))
    expected = pl.Series("a", [1, 2, 2], dtype=pl.UInt32).to_frame()
    assert_frame_equal(result, expected)


def test_cum_count_multi_arg() -> None:
    df = pl.DataFrame(
        {
            "a": [5, 5, 5],
            "b": [None, 5, 5],
            "c": [5, None, 5],
            "d": [5, 5, None],
            "e": [None, None, None],
        }
    )
    result = df.select(pl.cum_count("a", "b", "c", "d", "e"))
    expected = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], dtype=pl.UInt32),
            pl.Series("b", [0, 1, 2], dtype=pl.UInt32),
            pl.Series("c", [1, 1, 2], dtype=pl.UInt32),
            pl.Series("d", [1, 2, 2], dtype=pl.UInt32),
            pl.Series("e", [0, 0, 0], dtype=pl.UInt32),
        ]
    )
    assert_frame_equal(result, expected)


def test_cum_count_multi_arg_reverse() -> None:
    df = pl.DataFrame(
        {
            "a": [5, 5, 5],
            "b": [None, 5, 5],
            "c": [5, None, 5],
            "d": [5, 5, None],
            "e": [None, None, None],
        }
    )
    result = df.select(pl.cum_count("a", "b", "c", "d", "e", reverse=True))
    expected = pl.DataFrame(
        [
            pl.Series("a", [3, 2, 1], dtype=pl.UInt32),
            pl.Series("b", [2, 2, 1], dtype=pl.UInt32),
            pl.Series("c", [2, 1, 1], dtype=pl.UInt32),
            pl.Series("d", [2, 1, 0], dtype=pl.UInt32),
            pl.Series("e", [0, 0, 0], dtype=pl.UInt32),
        ]
    )
    assert_frame_equal(result, expected)


def test_cum_count() -> None:
    df = pl.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], schema=["A"])

    out = df.group_by("A", maintain_order=True).agg(
        pl.col("A").cum_count().alias("foo")
    )

    assert out["foo"][0].to_list() == [1, 2, 3, 4]
    assert out["foo"][1].to_list() == [1, 2]


def test_cumcount_deprecated() -> None:
    df = pl.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], schema=["A"])

    with pytest.deprecated_call():
        out = df.group_by("A", maintain_order=True).agg(
            pl.col("A").cumcount().alias("foo")
        )

    assert out["foo"][0].to_list() == [1, 2, 3, 4]
    assert out["foo"][1].to_list() == [1, 2]


def test_series_cum_count() -> None:
    s = pl.Series(["x", "k", None, "d"])
    result = s.cum_count()
    expected = pl.Series([1, 2, 2, 3], dtype=pl.UInt32)
    assert_series_equal(result, expected)

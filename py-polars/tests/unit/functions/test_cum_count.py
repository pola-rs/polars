from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(("reverse", "output"), [(False, [0, 1, 2]), (True, [2, 1, 0])])
def test_cum_count_no_args(reverse: bool, output: list[int]) -> None:
    df = pl.DataFrame({"a": [5, 5, None]})
    result = df.select(pl.cum_count(reverse=reverse))
    expected = pl.Series("cum_count", output, dtype=pl.UInt32).to_frame()
    assert_frame_equal(result, expected)


def test_cum_count_single_arg() -> None:
    df = pl.DataFrame({"a": [5, 5, None]})
    result = df.select(pl.cum_count("a"))
    expected = pl.Series("a", [0, 1, 2], dtype=pl.UInt32).to_frame()
    assert_frame_equal(result, expected)


def test_cum_count_multi_arg() -> None:
    df = pl.DataFrame({"a": [5, 5, None], "b": [5, None, None], "c": [1, 2, 3]})
    result = df.select(pl.cum_count("a", "b"))
    expected = pl.DataFrame(
        [
            pl.Series("a", [0, 1, 2], dtype=pl.UInt32),
            pl.Series("b", [0, 1, 2], dtype=pl.UInt32),
        ]
    )
    assert_frame_equal(result, expected)

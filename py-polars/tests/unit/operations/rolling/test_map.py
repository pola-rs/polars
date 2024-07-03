from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    ("input", "output"),
    [
        ([1, 5], [1, 6]),
        ([1], [1]),
    ],
)
def test_rolling_map_window_size_9160(input: list[int], output: list[int]) -> None:
    s = pl.Series(input)
    result = s.rolling_map(lambda x: sum(x), window_size=2, min_periods=1)
    expected = pl.Series(output)
    assert_series_equal(result, expected)


def testing_rolling_map_window_size_with_nulls() -> None:
    s = pl.Series([0, 1, None, 3, 4, 5])
    result = s.rolling_map(lambda x: sum(x), window_size=3, min_periods=3)
    expected = pl.Series([None, None, None, None, None, 12])
    assert_series_equal(result, expected)


def test_rolling_map_clear_reuse_series_state_10681() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 1, 2, 2, 2, 2],
            "b": [0.0, 1.0, 11.0, 7.0, 4.0, 2.0, 3.0, 8.0],
        }
    )

    result = df.select(
        pl.col("b")
        .rolling_map(lambda s: s.min(), window_size=3, min_periods=2)
        .over("a")
        .alias("min")
    )

    expected = pl.Series("min", [None, 0.0, 0.0, 1.0, None, 2.0, 2.0, 2.0])
    assert_series_equal(result.to_series(), expected)


def test_rolling_map_np_nansum() -> None:
    s = pl.Series("a", [11.0, 2.0, 9.0, float("nan"), 8.0])

    result = s.rolling_map(np.nansum, 3)

    expected = pl.Series("a", [None, None, 22.0, 11.0, 17.0])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_rolling_map_std(dtype: PolarsDataType) -> None:
    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=dtype)
    result = s.rolling_map(function=lambda s: s.std(), window_size=3)

    expected = pl.Series("A", [None, None, 4.358899, 4.041452, 5.567764], dtype=dtype)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_rolling_map_std_weights(dtype: PolarsDataType) -> None:
    s = pl.Series("A", [1.0, 2.0, 9.0, 2.0, 13.0], dtype=dtype)

    result = s.rolling_map(
        function=lambda s: s.std(), window_size=3, weights=[1.0, 2.0, 3.0]
    )

    expected = pl.Series("A", [None, None, 14.224392, 8.326664, 18.929694], dtype=dtype)
    assert_series_equal(result, expected)


def test_rolling_map_sum_int() -> None:
    s = pl.Series("A", [1, 2, 9, 2, 13], dtype=pl.Int32)

    result = s.rolling_map(function=lambda s: s.sum(), window_size=3)

    expected = pl.Series("A", [None, None, 12, 13, 24], dtype=pl.Int32)
    assert_series_equal(result, expected)


def test_rolling_map_sum_int_cast_to_float() -> None:
    s = pl.Series("A", [1, 2, 9, None, 13], dtype=pl.Int32)

    result = s.rolling_map(
        function=lambda s: s.sum(), window_size=3, weights=[1.0, 2.0, 3.0]
    )

    expected = pl.Series("A", [None, None, 32.0, None, None], dtype=pl.Float64)
    assert_series_equal(result, expected)


def test_rolling_map_rolling_sum() -> None:
    s = pl.Series("A", list(range(5)), dtype=pl.Float64)

    result = s.rolling_map(
        function=lambda s: s.sum(),
        window_size=3,
        weights=[1.0, 2.1, 3.2],
        min_periods=2,
        center=True,
    )

    expected = s.rolling_sum(
        window_size=3, weights=[1.0, 2.1, 3.2], min_periods=2, center=True
    )
    assert_series_equal(result, expected)


def test_rolling_map_rolling_std() -> None:
    s = pl.Series("A", list(range(6)), dtype=pl.Float64)

    result = s.rolling_map(
        function=lambda s: s.std(),
        window_size=4,
        min_periods=3,
        center=False,
    )

    expected = s.rolling_std(window_size=4, min_periods=3, center=False)
    assert_series_equal(result, expected)

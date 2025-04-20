import pytest

import polars as pl
from polars.testing.asserts.series import assert_series_equal


@pytest.fixture
def values() -> pl.Series:
    return pl.Series([7, 4, 1, 8, 5, 2, 9, 6])


@pytest.fixture
def by_col() -> pl.Series:
    return pl.Series([0, 0, 1, 1, 2, 2, 3, 3])


def test_series_rolling_min_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([4, 4, 1, 1, 1, 1, 2, 2])
    actual = values.rolling_min_by(by_col, "2i")
    assert_series_equal(expected, actual)


def test_series_rolling_max_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([7, 7, 8, 8, 8, 8, 9, 9])
    actual = values.rolling_max_by(by_col, "2i")
    assert_series_equal(expected, actual)


def test_series_rolling_mean_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([5.5, 5.5, 5.0, 5.0, 4.0, 4.0, 5.5, 5.5])
    actual = values.rolling_mean_by(by_col, "2i")
    assert_series_equal(expected, actual)


def test_series_rolling_sum_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([11, 11, 20, 20, 16, 16, 22, 22])
    actual = values.rolling_sum_by(by_col, "2i")
    assert_series_equal(expected, actual)


def test_series_rolling_std_by(values: pl.Series, by_col: pl.Series) -> None:
    sqrt7p5 = 2.7386127875258
    expected = pl.Series([1.5, 1.5, sqrt7p5, sqrt7p5, sqrt7p5, sqrt7p5, 2.5, 2.5])
    actual = values.rolling_std_by(by_col, "2i", ddof=0)
    assert_series_equal(expected, actual)


def test_series_rolling_var_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([2.25, 2.25, 7.5, 7.5, 7.5, 7.5, 6.25, 6.25])
    actual = values.rolling_var_by(by_col, "2i", ddof=0)
    assert_series_equal(expected, actual)


def test_series_rolling_median_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([5.5, 5.5, 5.5, 5.5, 3.5, 3.5, 5.5, 5.5])
    actual = values.rolling_median_by(by_col, "2i")
    assert_series_equal(expected, actual)


def test_series_rolling_quantile_by(values: pl.Series, by_col: pl.Series) -> None:
    expected = pl.Series([5.5, 5.5, 5.5, 5.5, 3.5, 3.5, 5.5, 5.5])
    actual = values.rolling_quantile_by(
        by_col, "2i", quantile=0.5, interpolation="linear"
    )
    assert_series_equal(expected, actual)

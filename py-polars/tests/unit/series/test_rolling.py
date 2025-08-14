from datetime import datetime

import pytest

import polars as pl
from polars.testing.asserts.series import assert_series_equal


@pytest.fixture
def values() -> pl.Series:
    return pl.Series([6, 9, 2, 5, 8, 1, 4, 7])


@pytest.fixture
def by_col() -> pl.Series:
    return pl.Series([0, 0, 1, 1, 2, 2, 3, 3])


@pytest.fixture
def by_col_temporal() -> pl.Series:
    return pl.Series(
        pl.datetime_range(
            datetime(2025, 1, 1), datetime(2025, 1, 2), "1h", eager=True
        ).head(8)
    )


def test_series_rolling_min_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_min_by(by_col, "2i")
    expected = pl.Series([6, 6, 2, 2, 1, 1, 1, 1])
    assert_series_equal(actual, expected)


def test_series_rolling_max_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_max_by(by_col, "2i")
    expected = pl.Series([9, 9, 9, 9, 8, 8, 8, 8])
    assert_series_equal(actual, expected)


def test_series_rolling_sum_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_sum_by(by_col, "2i")
    expected = pl.Series([15, 15, 22, 22, 16, 16, 20, 20])
    assert_series_equal(actual, expected)


def test_series_rolling_mean_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_mean_by(by_col, "2i")
    expected = pl.Series([7.5, 7.5, 5.5, 5.5, 4.0, 4.0, 5.0, 5.0])
    assert_series_equal(actual, expected)


def test_series_rolling_median_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_median_by(by_col, "2i")
    expected = pl.Series([7.5, 7.5, 5.5, 5.5, 3.5, 3.5, 5.5, 5.5])
    assert_series_equal(actual, expected)


def test_series_rolling_std_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_std_by(by_col, "2i")
    expected = pl.Series([2.12, 2.12, 2.88, 2.88, 3.16, 3.16, 3.16, 3.16])
    assert_series_equal(actual, expected, abs_tol=1e-2)


def test_series_rolling_var_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_var_by(by_col, "2i")
    expected = pl.Series([4.5, 4.5, 8.33, 8.33, 10.0, 10.0, 10.0, 10.0])
    assert_series_equal(actual, expected, abs_tol=1e-2)


def test_series_rolling_quantile_by(values: pl.Series, by_col: pl.Series) -> None:
    actual = values.rolling_quantile_by(by_col, "2i", quantile=0.5)
    expected = pl.Series([9.0, 9.0, 6.0, 6.0, 5.0, 5.0, 7.0, 7.0])
    assert_series_equal(actual, expected)


def test_series_rolling_min_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_min_by(by_col_temporal, "2h")
    expected = pl.Series([6, 6, 2, 2, 5, 1, 1, 4])
    assert_series_equal(actual, expected)


def test_series_rolling_max_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_max_by(by_col_temporal, "2h")
    expected = pl.Series([6, 9, 9, 5, 8, 8, 4, 7])
    assert_series_equal(actual, expected)


def test_series_rolling_sum_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_sum_by(by_col_temporal, "2h")
    expected = pl.Series([6, 15, 11, 7, 13, 9, 5, 11])
    assert_series_equal(actual, expected)


def test_series_rolling_mean_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_mean_by(by_col_temporal, "2h")
    expected = pl.Series([6.0, 7.5, 5.5, 3.5, 6.5, 4.5, 2.5, 5.5])
    assert_series_equal(actual, expected)


def test_series_rolling_median_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_median_by(by_col_temporal, "3h")
    expected = pl.Series([6.0, 7.5, 6.0, 5.0, 5.0, 5.0, 4.0, 4.0])
    assert_series_equal(actual, expected)


def test_series_rolling_std_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_std_by(by_col_temporal, "2h")
    expected = pl.Series([None, 2.12, 4.94, 2.12, 2.12, 4.94, 2.12, 2.12])
    assert_series_equal(actual, expected, abs_tol=1e-2)


def test_series_rolling_var_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_var_by(by_col_temporal, "2h")
    expected = pl.Series([None, 4.5, 24.5, 4.5, 4.5, 24.5, 4.5, 4.5])
    assert_series_equal(actual, expected, abs_tol=1e-2)


def test_series_rolling_quantile_by_temporal(
    values: pl.Series, by_col_temporal: pl.Series
) -> None:
    actual = values.rolling_quantile_by(by_col_temporal, "2h", quantile=0.5)
    expected = pl.Series([6.0, 9.0, 9.0, 5.0, 8.0, 8.0, 4.0, 7.0])
    assert_series_equal(actual, expected)

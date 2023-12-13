from datetime import date

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_series_describe_int() -> None:
    s = pl.Series([1, 2, 3])
    result = s.describe()

    stats = {
        "count": 3.0,
        "null_count": 0.0,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "25%": 1.0,
        "50%": 2.0,
        "75%": 3.0,
        "max": 3.0,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_float() -> None:
    s = pl.Series([1.3, 4.6, 8.9])
    result = s.describe()

    stats = {
        "count": 3.0,
        "null_count": 0.0,
        "mean": 4.933333333333334,
        "std": 3.8109491381194442,
        "min": 1.3,
        "25%": 1.3,
        "50%": 4.6,
        "75%": 8.9,
        "max": 8.9,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_string() -> None:
    s = pl.Series(["abc", "pqr", "xyz"])
    result = s.describe()

    stats = {
        "count": 3,
        "null_count": 0,
        "unique": 3,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_boolean() -> None:
    s = pl.Series([True, False, None, True, True])
    result = s.describe()

    stats = {
        "count": 4,
        "null_count": 1,
        "sum": 3,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_date() -> None:
    s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
    result = s.describe()

    stats = {
        "count": "3",
        "null_count": "0",
        "min": "2021-01-01",
        "50%": "2021-01-02 00:00:00",
        "max": "2021-01-03",
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_empty() -> None:
    s = pl.Series(dtype=pl.Float64)
    result = s.describe()
    print(result)
    stats = {
        "count": 0.0,
        "null_count": 0.0,
        "mean": None,
        "std": None,
        "min": None,
        "25%": None,
        "50%": None,
        "75%": None,
        "max": None,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_unsupported_dtype() -> None:
    s = pl.Series(dtype=pl.List(pl.Int64))
    with pytest.raises(
        TypeError, match="cannot describe Series of data type List\\(Int64\\)"
    ):
        s.describe()

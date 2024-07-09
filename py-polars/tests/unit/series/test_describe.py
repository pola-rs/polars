from datetime import date

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
        "25%": 2.0,
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
        "25%": 4.6,
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
        "count": "3",
        "null_count": "0",
        "min": "abc",
        "max": "xyz",
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_boolean() -> None:
    s = pl.Series([True, False, None, True, True])
    result = s.describe()

    stats = {
        "count": 4,
        "null_count": 1,
        "mean": 0.75,
        "min": False,
        "max": True,
    }
    expected = pl.DataFrame(
        data={"statistic": stats.keys(), "value": stats.values()},
        schema_overrides={"value": pl.Float64},
    )
    assert_frame_equal(expected, result)


def test_series_describe_date() -> None:
    s = pl.Series([date(1999, 12, 31), date(2011, 3, 11), date(2021, 1, 18)])
    result = s.describe(interpolation="linear")

    stats = {
        "count": "3",
        "null_count": "0",
        "mean": "2010-09-29 16:00:00",
        "min": "1999-12-31",
        "25%": "2005-08-05",
        "50%": "2011-03-11",
        "75%": "2016-02-13",
        "max": "2021-01-18",
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_empty() -> None:
    s = pl.Series(dtype=pl.Float64)
    result = s.describe()
    stats = {
        "count": 0.0,
        "null_count": 0.0,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_null() -> None:
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.describe()
    stats = {
        "count": 0.0,
        "null_count": 2.0,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)


def test_series_describe_nested_list() -> None:
    s = pl.Series(
        values=[[10e10, 10e15], [10e12, 10e13], [10e10, 10e15]],
        dtype=pl.List(pl.Float64),
    )
    result = s.describe()
    stats = {
        "count": 3.0,
        "null_count": 0.0,
    }
    expected = pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})
    assert_frame_equal(expected, result)

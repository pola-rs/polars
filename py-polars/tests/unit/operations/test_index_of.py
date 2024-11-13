from __future__ import annotations

import pytest

import numpy as np
import polars as pl


# Nulls
# NaN
# inf
# -inf
# sorted (both directions)
# all numeric dtypes
# multiple chunks
# dtype of value doesn't match dtype of series
# empty list


def assert_index_of(series: pl.Series, value: object) -> None:
    """
    ``Series.index_of()`` returns the index of the value, or None if it can't
    be found.
    """
    if value is not None and np.isnan(value):
        expected_index = None
        for i, o in enumerate(series.to_list()):
            if o is not None and np.isnan(o):
                expected_index = i
                break
    else:
        try:
            expected_index = series.to_list().index(value)
        except ValueError:
            expected_index = None
    if expected_index == -1:
        expected_index = None
    # Eager API:
    assert series.index_of(value) == expected_index
    # Lazy API:
    pl.LazyFrame({"series": series}).select(
        pl.col("series").index_of(value)
    ).collect().get_column("series").to_list() == [expected_index]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float32])
def test_float(dtype):
    values = [1.5, np.nan, np.inf, 3.0, None, -np.inf]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat([pl.Series([1, 7], dtype=dtype), series], rechunk=False)

    for value in values + [
        np.int8(3),
        np.int64(2**42),
        np.float64(1.5),
        np.float32(1.5),
        np.float32(2**37),
        np.float64(2**100),
    ]:
        for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
            assert_index_of(s, value)


def test_null():
    series = pl.Series([None, None], dtype=pl.Null)
    assert_index_of(series, None)


def test_empty():
    series = pl.Series([], dtype=pl.Null)
    assert_index_of(series, None)
    series = pl.Series([], dtype=pl.Int64)
    assert_index_of(series, None)
    assert_index_of(series, 12)
    assert_index_of(series.sort(descending=True), 12)
    assert_index_of(series.sort(descending=False), 12)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
def test_integer(dtype):
    values = [51, 3, None, 4]
    series = pl.Series(values, dtype=dtype)
    sorted_series_asc = series.sort(descending=False)
    sorted_series_desc = series.sort(descending=True)
    chunked_series = pl.concat(
        [pl.Series([100, 7], dtype=dtype), series], rechunk=False
    )

    for value in values + [
        np.int8(3),
        np.int64(2**42),
        np.float64(3.0),
        np.float32(3.0),
        np.uint64(2**63),
        np.int8(-7),
        np.float32(1.5),
        np.float64(1.5),
        # Catch a bug where rounding happens:
        np.float32(3.1),
        np.float64(3.1),
    ]:
        for s in [series, sorted_series_asc, sorted_series_desc, chunked_series]:
            assert_index_of(s, value)

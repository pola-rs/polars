from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing.asserts.series import assert_series_equal


def test_series_mixed_dtypes_list() -> None:
    values = [[0.1, 1]]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.List(pl.Float64)
    assert s.to_list() == [[0.1, 1.0]]


def test_series_mixed_dtypes_string() -> None:
    values = [[12], "foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.String
    assert s.to_list() == ["[12]", "foo", "9"]
    assert s[1] == "foo"


def test_series_mixed_dtypes_object() -> None:
    values = [[12], b"foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.Object
    assert s.to_list() == values
    assert s[1] == b"foo"


# https://github.com/pola-rs/polars/issues/15139
@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), None])
def test_sequence_of_series_with_dtype(dtype: pl.PolarsDataType | None) -> None:
    values = [1, 2, 3]
    int_series = pl.Series(values)
    list_series = pl.Series([int_series], dtype=dtype)

    assert list_series.to_list() == [values]
    assert list_series.dtype == pl.List(pl.Int64)


@pytest.mark.parametrize(
    ("values", "dtype", "expected_dtype"),
    [
        ([1, 1.0, 1], None, pl.Float64),
        ([1, 1, "1.0"], None, pl.String),
        ([1, 1.0, "1.0"], None, pl.String),
        ([True, 1], None, pl.Int64),
        ([True, 1.0], None, pl.Float64),
        ([True, 1], pl.Boolean, pl.Boolean),
        ([True, 1.0], pl.Boolean, pl.Boolean),
        ([False, "1.0"], None, pl.String),
    ],
)
def test_upcast_primitive_and_strings(
    values: list[Any], dtype: pl.PolarsDataType, expected_dtype: pl.PolarsDataType
) -> None:
    assert pl.Series(values, dtype=dtype).dtype == expected_dtype


def test_preserve_decimal_precision() -> None:
    dtype = pl.Decimal(None, 1)
    s = pl.Series(dtype=dtype)
    assert s.dtype == dtype


@pytest.mark.parametrize("dtype", [None, pl.Duration("ms")])
def test_large_timedelta(dtype: pl.DataType | None) -> None:
    values = [timedelta.min, timedelta.max]
    s = pl.Series(values, dtype=dtype)
    assert s.dtype == pl.Duration("ms")

    # Microsecond precision is lost
    expected = [timedelta.min, timedelta.max - timedelta(microseconds=999)]
    assert s.to_list() == expected


def test_array_large_u64() -> None:
    u64_max = 2**64 - 1
    values = [[u64_max]]
    dtype = pl.Array(pl.UInt64, 1)
    s = pl.Series(values, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == values


def test_series_init_ambiguous_datetime() -> None:
    value = datetime(2001, 10, 28, 2)
    dtype = pl.Datetime(time_zone="Europe/Belgrade")

    result = pl.Series([value], dtype=dtype, strict=True)
    expected = pl.Series([datetime(2001, 10, 28, 3)]).dt.replace_time_zone(
        "Europe/Belgrade"
    )
    assert_series_equal(result, expected)

    result = pl.Series([value], dtype=dtype, strict=False)
    assert_series_equal(result, expected)


def test_series_init_nonexistent_datetime() -> None:
    value = datetime(2024, 3, 31, 2, 30)
    dtype = pl.Datetime(time_zone="Europe/Amsterdam")

    result = pl.Series([value], dtype=dtype, strict=True)
    expected = pl.Series([datetime(2024, 3, 31, 4, 30)]).dt.replace_time_zone(
        "Europe/Amsterdam"
    )
    assert_series_equal(result, expected)

    result = pl.Series([value], dtype=dtype, strict=False)
    assert_series_equal(result, expected)


# https://github.com/pola-rs/polars/issues/15518
def test_series_init_np_temporal_with_nat_15518() -> None:
    arr = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], "datetime64[D]")
    arr[1] = np.datetime64("NaT")

    result = pl.Series(arr)

    expected = pl.Series([date(2020, 1, 1), None, date(2020, 1, 3)])
    assert_series_equal(result, expected)


def test_series_init_np_2d_zero_zero_shape() -> None:
    arr = np.array([]).reshape(0, 0)
    with pytest.raises(
        pl.InvalidOperationError,
        match=re.escape("cannot reshape empty array into shape (0, 0)"),
    ):
        pl.Series(arr)

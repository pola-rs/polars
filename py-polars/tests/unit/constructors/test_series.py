from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


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
def test_sequence_of_series_with_dtype(dtype: PolarsDataType | None) -> None:
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
    values: list[Any], dtype: PolarsDataType, expected_dtype: PolarsDataType
) -> None:
    with pytest.raises(TypeError):
        pl.Series(values, dtype=dtype, strict=True)

    assert pl.Series(values, dtype=dtype, strict=False).dtype == expected_dtype


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


def test_series_init_pandas_timestamp_18127() -> None:
    result = pl.Series([pd.Timestamp("2000-01-01T00:00:00.123456789", tz="UTC")])
    # Note: time unit is not (yet) respected, it should be Datetime('ns', 'UTC').
    assert result.dtype == pl.Datetime("us", "UTC")


def test_series_init_np_2d_zero_zero_shape() -> None:
    arr = np.array([]).reshape(0, 0)
    with pytest.raises(
        InvalidOperationError,
        match=re.escape("cannot reshape empty array into shape (0, 0)"),
    ):
        pl.Series(arr)


def test_list_null_constructor_schema() -> None:
    expected = pl.List(pl.Null)
    assert pl.Series([[]]).dtype == expected
    assert pl.Series([[]], dtype=pl.List).dtype == expected

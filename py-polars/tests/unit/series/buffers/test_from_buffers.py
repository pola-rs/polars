from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(
    s=series(
        allowed_dtypes=(pl.INTEGER_DTYPES | pl.FLOAT_DTYPES | {pl.Boolean}),
        chunked=False,
    )
)
def test_series_from_buffers_numeric_with_validity(s: pl.Series) -> None:
    validity = s.is_not_null()
    result = pl.Series._from_buffers(s.dtype, data=s, validity=validity)
    assert_series_equal(s, result)


@given(
    s=series(
        allowed_dtypes=(pl.INTEGER_DTYPES | pl.FLOAT_DTYPES | {pl.Boolean}),
        chunked=False,
        null_probability=0.0,
    )
)
def test_series_from_buffers_numeric(s: pl.Series) -> None:
    result = pl.Series._from_buffers(s.dtype, data=s)
    assert_series_equal(s, result)


@given(s=series(allowed_dtypes=pl.TEMPORAL_DTYPES, chunked=False))
def test_series_from_buffers_temporal_with_validity(s: pl.Series) -> None:
    validity = s.is_not_null()
    physical = pl.Int32 if s.dtype == pl.Date else pl.Int64
    data = s.cast(physical)
    result = pl.Series._from_buffers(s.dtype, data=data, validity=validity)
    assert_series_equal(s, result)


def test_series_from_buffers_int() -> None:
    dtype = pl.UInt16
    data = pl.Series([97, 98, 99, 195], dtype=dtype)
    validity = pl.Series([True, True, False, True])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series([97, 98, None, 195], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_float() -> None:
    dtype = pl.Float64
    data = pl.Series([0.0, 1.0, -1.0, float("nan"), float("inf")], dtype=dtype)
    validity = pl.Series([True, True, False, True, True])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series([0.0, 1.0, None, float("nan"), float("inf")], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_boolean() -> None:
    dtype = pl.Boolean
    data = pl.Series([True, False, True])
    validity = pl.Series([True, True, False])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series([True, False, None])
    assert_series_equal(result, expected)


def test_series_from_buffers_datetime() -> None:
    dtype = pl.Datetime(time_zone="Europe/Amsterdam")
    data = pl.Series(
        [
            datetime(2022, 2, 10, 6),
            datetime(2022, 2, 11, 12),
            datetime(2022, 2, 12, 18),
        ],
        dtype=dtype,
    ).cast(pl.Int64)
    validity = pl.Series([True, False, True])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series(
        [
            datetime(2022, 2, 10, 6),
            None,
            datetime(2022, 2, 12, 18),
        ],
        dtype=dtype,
    )
    assert_series_equal(result, expected)


def test_series_from_buffers_string() -> None:
    dtype = pl.Utf8
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    validity = pl.Series([True, True, False, True])
    offsets = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)

    result = pl.Series._from_buffers(dtype, data=[data, offsets], validity=validity)

    expected = pl.Series(["a", "bc", None, "éâç"], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_enum() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    data = pl.Series([0, 1, 0, 2], dtype=pl.UInt32)
    validity = pl.Series([True, True, False, True])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series(["a", "b", None, "c"], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_sliced() -> None:
    dtype = pl.Int64
    data = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    data = data[5:]
    validity = pl.Series([True, True, True, True, False, True, False, False, True])
    validity = validity[5:]

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series([6, None, None, 9], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_unsupported_validity() -> None:
    s = pl.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="validity buffer must have data type Boolean, got Int64",
    ):
        pl.Series._from_buffers(pl.Date, data=s, validity=s)


def test_series_from_buffers_unsupported_offsets() -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    offsets = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int8)

    with pytest.raises(
        TypeError,
        match="offsets buffer must have data type Int64, got Int8",
    ):
        pl.Series._from_buffers(pl.Utf8, data=[data, offsets])


def test_series_from_buffers_offsets_do_not_match_data() -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    offsets = pl.Series([0, 1, 3, 3, 9, 11], dtype=pl.Int64)

    with pytest.raises(
        pl.PolarsPanicError,
        match="offsets must not exceed the values length",
    ):
        pl.Series._from_buffers(pl.Utf8, data=[data, offsets])


def test_series_from_buffers_no_buffers() -> None:
    with pytest.raises(
        TypeError,
        match="`data` input to `from_buffers` must contain at least one buffer",
    ):
        pl.Series._from_buffers(pl.Int32, data=[])

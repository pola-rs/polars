from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from hypothesis import given

import polars as pl
from polars.exceptions import PanicException
from polars.testing import assert_series_equal
from polars.testing.parametric import series
from tests.unit.conftest import NUMERIC_DTYPES

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


@given(
    s=series(
        allowed_dtypes=[*NUMERIC_DTYPES, pl.Boolean],
        allow_chunks=False,
    )
)
def test_series_from_buffers_numeric_with_validity(s: pl.Series) -> None:
    validity = s.is_not_null()
    result = pl.Series._from_buffers(s.dtype, data=s, validity=validity)
    assert_series_equal(s, result)


@given(
    s=series(
        allowed_dtypes=[*NUMERIC_DTYPES, pl.Boolean],
        allow_chunks=False,
        allow_null=False,
    )
)
def test_series_from_buffers_numeric(s: pl.Series) -> None:
    result = pl.Series._from_buffers(s.dtype, data=s)
    assert_series_equal(s, result)


@given(
    s=series(
        allowed_dtypes=[pl.Date, pl.Time, pl.Datetime, pl.Duration],
        allow_chunks=False,
    )
)
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
    tzinfo = ZoneInfo("Europe/Amsterdam")
    data = pl.Series(
        [
            datetime(2022, 2, 10, 6, tzinfo=tzinfo),
            datetime(2022, 2, 11, 12, tzinfo=tzinfo),
            datetime(2022, 2, 12, 18, tzinfo=tzinfo),
        ],
        dtype=dtype,
    ).cast(pl.Int64)
    validity = pl.Series([True, False, True])

    result = pl.Series._from_buffers(dtype, data=data, validity=validity)

    expected = pl.Series(
        [
            datetime(2022, 2, 10, 6, tzinfo=tzinfo),
            None,
            datetime(2022, 2, 12, 18, tzinfo=tzinfo),
        ],
        dtype=dtype,
    )
    assert_series_equal(result, expected)


def test_series_from_buffers_string() -> None:
    dtype = pl.String
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

    msg = "validity buffer must have data type Boolean, got Int64"
    with pytest.raises(TypeError, match=msg):
        pl.Series._from_buffers(pl.Date, data=s, validity=s)


def test_series_from_buffers_unsupported_offsets() -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    offsets = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int8)

    msg = "offsets buffer must have data type Int64, got Int8"
    with pytest.raises(TypeError, match=msg):
        pl.Series._from_buffers(pl.String, data=[data, offsets])


def test_series_from_buffers_offsets_do_not_match_data() -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    offsets = pl.Series([0, 1, 3, 3, 9, 11], dtype=pl.Int64)

    msg = "offsets must not exceed the values length"
    with pytest.raises(PanicException, match=msg):
        pl.Series._from_buffers(pl.String, data=[data, offsets])


def test_series_from_buffers_no_buffers() -> None:
    msg = "`data` input to `_from_buffers` must contain at least one buffer"
    with pytest.raises(TypeError, match=msg):
        pl.Series._from_buffers(pl.Int32, data=[])

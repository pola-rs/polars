from __future__ import annotations

from datetime import date, datetime

import pytest
from hypothesis import given

import polars as pl
from polars.polars import PySeries
from polars.testing import assert_series_equal
from polars.testing.parametric import series
from polars.utils._wrap import wrap_s


@given(
    s=series(
        allowed_dtypes=(pl.INTEGER_DTYPES | pl.FLOAT_DTYPES | {pl.Boolean}),
        chunked=False,
    )
)
def test_series_from_buffers_numeric_with_validity(s: pl.Series) -> None:
    validity = s.is_not_null()
    result = wrap_s(PySeries._from_buffers(s.dtype, data=s._s, validity=validity._s))
    assert_series_equal(s, result)


@given(
    s=series(
        allowed_dtypes=(pl.INTEGER_DTYPES | pl.FLOAT_DTYPES | {pl.Boolean}),
        chunked=False,
        null_probability=0.0,
    )
)
def test_series_from_buffers_numeric(s: pl.Series) -> None:
    result = wrap_s(PySeries._from_buffers(s.dtype, data=s._s))
    assert_series_equal(s, result)


@given(s=series(allowed_dtypes=pl.TEMPORAL_DTYPES, chunked=False, null_probability=0.0))
def test_series_from_buffers_temporal(s: pl.Series) -> None:
    print(s)
    physical = pl.Int32 if s.dtype == pl.Date else pl.Int64
    data = s.cast(physical)
    result = wrap_s(PySeries._from_buffers(s.dtype, data=data._s))
    assert_series_equal(s, result)


def test_series_from_buffers_int() -> None:
    dtype = pl.UInt16
    data = pl.Series([97, 98, 99, 195], dtype=dtype)
    validity = pl.Series([True, True, False, True])

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

    expected = pl.Series([97, 98, None, 195], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_float() -> None:
    dtype = pl.Float64
    data = pl.Series([0.0, 1.0, -1.0, float("nan"), float("inf")], dtype=dtype)
    validity = pl.Series([True, True, False, True, True])

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

    expected = pl.Series([0.0, 1.0, None, float("nan"), float("inf")], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_boolean() -> None:
    dtype = pl.Boolean
    data = pl.Series([True, False, True])
    validity = pl.Series([True, True, False])

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

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

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

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

    result = wrap_s(PySeries._from_buffers(dtype, data._s, validity._s, offsets._s))

    expected = pl.Series(["a", "bc", None, "éâç"], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_enum() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    data = pl.Series([0, 1, 0, 2], dtype=pl.UInt32)
    validity = pl.Series([True, True, False, True])

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

    expected = pl.Series(["a", "b", None, "c"], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_sliced() -> None:
    dtype = pl.Int64
    data = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    data = data[5:]
    validity = pl.Series([True, True, True, True, False, True, False, False, True])
    validity = validity[5:]

    result = wrap_s(PySeries._from_buffers(dtype, data=data._s, validity=validity._s))

    expected = pl.Series([6, None, None, 9], dtype=dtype)
    assert_series_equal(result, expected)


def test_series_from_buffers_unsupported_data() -> None:
    s = pl.Series([date(2020, 1, 1), date(2020, 2, 5)])

    with pytest.raises(
        TypeError,
        match="data buffer must have a physical data type, got date",
    ):
        PySeries._from_buffers(pl.Date, data=s._s)


def test_series_from_buffers_unsupported_validity() -> None:
    s = pl.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="validity buffer must have data type Boolean, got Int64",
    ):
        wrap_s(PySeries._from_buffers(pl.Date, data=s._s, validity=s._s))


def test_series_from_buffers_unsupported_offsets() -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    offsets = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int8)

    with pytest.raises(
        TypeError,
        match="offsets buffer must have data type Int64, got Int8",
    ):
        PySeries._from_buffers(pl.Date, data=data._s, offsets=offsets._s)


@pytest.mark.parametrize(
    "offsets",
    [
        pl.Series([0, 1, 3, 3, 9, 11], dtype=pl.Int64),
        pl.Series([0, 1, 3, 3], dtype=pl.Int64),
    ],
)
def test_series_from_buffers_offsets_do_not_match_data(offsets: pl.Series) -> None:
    data = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)

    with pytest.raises(
        TypeError,
        match="offsets buffer does not match data buffer length",
    ):
        PySeries._from_buffers(pl.Date, data=data._s, offsets=offsets._s)

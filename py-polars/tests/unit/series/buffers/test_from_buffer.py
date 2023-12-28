from __future__ import annotations

from datetime import date

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
def test_series_from_buffer(s: pl.Series) -> None:
    buffer_info = s._get_buffer_info()
    result = pl.Series._from_buffer(s.dtype, buffer_info, base=s)
    assert_series_equal(s, result)


def test_series_from_buffer_numeric() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.UInt16)
    buffer_info = s._get_buffer_info()
    result = pl.Series._from_buffer(s.dtype, buffer_info, base=s)
    assert_series_equal(s, result)


def test_series_from_buffer_sliced_bitmask() -> None:
    s = pl.Series([True] * 9, dtype=pl.Boolean)[5:]
    buffer_info = s._get_buffer_info()
    result = pl.Series._from_buffer(s.dtype, buffer_info, base=s)
    assert_series_equal(s, result)


def test_series_from_buffer_unsupported() -> None:
    s = pl.Series([date(2020, 1, 1), date(2020, 2, 5)])
    buffer_info = s._get_buffer_info()

    with pytest.raises(
        TypeError,
        match="`from_buffer` requires a physical type as input for `dtype`, got date",
    ):
        pl.Series._from_buffer(pl.Date, buffer_info, base=s)

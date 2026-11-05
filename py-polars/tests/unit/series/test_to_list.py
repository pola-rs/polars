from __future__ import annotations

import datetime

import pytest
from hypothesis import example, given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(
    s=series(
        # Roundtrip doesn't work with time zones:
        # https://github.com/pola-rs/polars/issues/16297
        allow_time_zones=False,
    )
)
@example(s=pl.Series(dtype=pl.Array(pl.Date, 1)))
def test_to_list(s: pl.Series) -> None:
    values = s.to_list()
    result = pl.Series(values, dtype=s.dtype)
    assert_series_equal(s, result, categorical_as_str=True)


def test_series_to_list_out_of_range_datetime() -> None:
    s = pl.Series([datetime.datetime(9999, 12, 31)]).dt.offset_by("1d")
    with pytest.raises(ValueError, match="year"):
        s.to_list()

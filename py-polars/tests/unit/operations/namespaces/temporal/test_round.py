from __future__ import annotations

from datetime import datetime, timedelta

import hypothesis.strategies as st
from hypothesis import given

import polars as pl
from polars._utils.convert import parse_as_duration_string
from polars.testing import assert_series_equal


@given(
    datetimes=st.lists(
        st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
        min_size=1,
        max_size=3,
    ),
    every=st.timedeltas(
        min_value=timedelta(microseconds=1), max_value=timedelta(days=1)
    ).map(parse_as_duration_string),
)
def test_fast_path_vs_slow_path(datetimes: list[datetime], every: str) -> None:
    s = pl.Series(datetimes)
    # Might use fastpath:
    result = s.dt.round(every)
    # Definitely uses slowpath:
    expected = s.dt.round(pl.Series([every] * len(datetimes)))
    assert_series_equal(result, expected)

from __future__ import annotations

import datetime as dt

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, reject

import polars as pl
from polars._utils.various import parse_version


@given(
    start=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    end=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    week_mask=st.lists(
        st.sampled_from([True, False]),
        min_size=7,
        max_size=7,
    ),
    holidays=st.lists(
        st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
        min_size=0,
        max_size=100,
    ),
)
def test_against_np_busday_count(
    start: dt.date, end: dt.date, week_mask: tuple[bool, ...], holidays: list[dt.date]
) -> None:
    assume(any(week_mask))
    result = (
        pl.DataFrame({"start": [start], "end": [end]})
        .select(
            n=pl.business_day_count(
                "start", "end", week_mask=week_mask, holidays=holidays
            )
        )["n"]
        .item()
    )
    expected = np.busday_count(start, end, weekmask=week_mask, holidays=holidays)
    if start > end and parse_version(np.__version__) < parse_version("1.25"):
        # Bug in old versions of numpy
        reject()
    assert result == expected

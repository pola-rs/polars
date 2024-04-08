from __future__ import annotations

import datetime as dt

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, reject

import polars as pl
from polars._utils.various import parse_version


@given(
    start=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    end=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
)
def test_against_np_busday_count(
    start: dt.date,
    end: dt.date,
) -> None:
    result = (
        pl.DataFrame({"start": [start], "end": [end]})
        .select(n=pl.business_day_count("start", "end"))["n"]
        .item()
    )
    expected = np.busday_count(start, end)
    if start > end and parse_version(np.__version__) < parse_version("1.25"):
        # Bug in old versions of numpy
        reject()
    assert result == expected

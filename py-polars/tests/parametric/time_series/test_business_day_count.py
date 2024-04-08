from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, reject

import polars as pl
from polars._utils.various import parse_version
from polars.functions.business import _make_week_mask

if TYPE_CHECKING:
    from polars.type_aliases import DayOfWeek


@given(
    start=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    end=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    weekend=st.lists(
        st.sampled_from(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
        min_size=0,
        max_size=6,
        unique=True,
    ),
)
def test_against_np_busday_count(
    start: dt.date,
    end: dt.date,
    weekend: list[DayOfWeek],
) -> None:
    result = (
        pl.DataFrame({"start": [start], "end": [end]})
        .select(n=pl.business_day_count("start", "end", weekend=weekend))["n"]
        .item()
    )
    expected = np.busday_count(start, end, weekmask=_make_week_mask(weekend))
    if start > end and parse_version(np.__version__) < parse_version("1.25"):
        # Bug in old versions of numpy
        reject()
    assert result == expected

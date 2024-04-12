from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import Roll


@given(
    start=st.dates(min_value=dt.date(1969, 1, 1), max_value=dt.date(1970, 12, 31)),
    n=st.integers(min_value=-100, max_value=100),
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
    roll=st.sampled_from(["forward", "backward"]),
)
def test_against_np_busday_offset(
    start: dt.date,
    n: int,
    week_mask: tuple[bool, ...],
    holidays: list[dt.date],
    roll: Roll,
) -> None:
    assume(any(week_mask))
    result = (
        pl.DataFrame({"start": [start]})
        .select(
            res=pl.col("start").dt.add_business_days(
                n, week_mask=week_mask, holidays=holidays, roll=roll
            )
        )["res"]
        .item()
    )
    expected = np.busday_offset(
        start, n, weekmask=week_mask, holidays=holidays, roll=roll
    )
    assert result == expected

import datetime as dt

import hypothesis.strategies as st
from hypothesis import given

import polars as pl


@given(
    value=st.datetimes(
        min_value=dt.datetime(1000, 1, 1),
        max_value=dt.datetime(3000, 1, 1),
    ),
    n=st.integers(min_value=1, max_value=100),
)
def test_truncate_monthly(value: dt.date, n: int) -> None:
    result = pl.Series([value]).dt.truncate(f"{n}mo").item()
    # manual calculation
    total = value.year * 12 + value.month - 1
    remainder = total % n
    total -= remainder
    year, month = (total // 12), ((total % 12) + 1)
    expected = dt.datetime(year, month, 1)
    assert result == expected

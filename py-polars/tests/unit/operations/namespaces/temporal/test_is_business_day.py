from __future__ import annotations

from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal


@pytest.mark.parametrize(
    ("holidays", "week_mask", "expected_values"),
    [
        ((), (True, True, True, True, True, False, False), [True, True]),
        (
            (date(2020, 1, 1),),
            (True, True, True, True, True, False, False),
            [False, True],
        ),
        (
            (date(2020, 1, 2),),
            (True, True, True, True, True, False, False),
            [True, False],
        ),
        (
            (date(2020, 1, 2), date(2020, 1, 1)),
            (True, True, True, True, True, False, False),
            [False, False],
        ),
        ((), (True, True, False, True, True, True, True), [False, True]),
        ((), (True, True, False, False, True, True, True), [False, False]),
        ((), (True, True, True, False, True, True, True), [True, False]),
        (
            (),
            (True, True, True, True, True, True, True),
            [True, True],
        ),  # no holidays :sob:
        (
            (date(2020, 1, 1),),
            (True, True, True, True, True, True, True),
            [False, True],
        ),
        (
            (date(2020, 1, 1),),
            (True, True, False, True, True, True, True),
            [False, True],
        ),
    ],
)
def test_is_business_day(
    holidays: tuple[date, ...], week_mask: tuple[bool, ...], expected_values: list[bool]
) -> None:
    # Date
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)]})
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    expected = pl.Series("date", expected_values)
    assert_series_equal(result, expected)
    # Datetime
    df = pl.DataFrame({"date": [datetime(2020, 1, 1, 3), datetime(2020, 1, 2, 1)]})
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    assert_series_equal(result, expected)
    # Datetime tz-aware
    df = pl.DataFrame(
        {"date": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}
    ).with_columns(pl.col("date").dt.replace_time_zone("Asia/Kathmandu"))
    result = df.select(
        pl.col("date").dt.is_business_day(holidays=holidays, week_mask=week_mask)
    )["date"]
    assert_series_equal(result, expected)


def test_is_business_day_invalid() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)]})
    with pytest.raises(ComputeError):
        df.select(pl.col("date").dt.is_business_day(week_mask=[False] * 7))


def test_is_business_day_repr() -> None:
    assert "is_business_day" in repr(pl.col("date").dt.is_business_day())

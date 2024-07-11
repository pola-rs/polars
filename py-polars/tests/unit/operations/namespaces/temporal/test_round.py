from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars._utils.convert import parse_as_duration_string
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo


else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


@pytest.mark.parametrize("time_zone", [None, "Asia/Kathmandu"])
def test_round_by_day_datetime(time_zone: str | None) -> None:
    ser = pl.Series([datetime(2021, 11, 7, 3)]).dt.replace_time_zone(time_zone)
    result = ser.dt.round("1d")
    expected = pl.Series([datetime(2021, 11, 7)]).dt.replace_time_zone(time_zone)
    assert_series_equal(result, expected)


def test_round_ambiguous() -> None:
    t = (
        pl.datetime_range(
            date(2020, 10, 25),
            datetime(2020, 10, 25, 2),
            "30m",
            eager=True,
            time_zone="Europe/London",
        )
        .alias("datetime")
        .dt.offset_by("15m")
    )
    result = t.dt.round("30m")
    expected = (
        pl.Series(
            [
                "2020-10-25T00:30:00+0100",
                "2020-10-25T01:00:00+0100",
                "2020-10-25T01:30:00+0100",
                "2020-10-25T01:00:00+0000",
                "2020-10-25T01:30:00+0000",
                "2020-10-25T02:00:00+0000",
                "2020-10-25T02:30:00+0000",
            ]
        )
        .str.to_datetime()
        .dt.convert_time_zone("Europe/London")
        .rename("datetime")
    )
    assert_series_equal(result, expected)

    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                date(2020, 10, 25),
                datetime(2020, 10, 25, 2),
                "30m",
                eager=True,
                time_zone="Europe/London",
            ).dt.offset_by("15m")
        }
    )

    df = df.select(pl.col("date").dt.round("30m"))
    assert df.to_dict(as_series=False) == {
        "date": [
            datetime(2020, 10, 25, 0, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 2, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 2, 30, tzinfo=ZoneInfo("Europe/London")),
        ]
    }


def test_round_by_week() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [
                    # Sunday and Monday
                    "1998-04-12",
                    "2022-11-28",
                ]
            ).str.strptime(pl.Date, "%Y-%m-%d")
        }
    )

    assert (
        df.select(
            pl.col("date").dt.round("7d").alias("7d"),
            pl.col("date").dt.round("1w").alias("1w"),
        )
    ).to_dict(as_series=False) == {
        "7d": [date(1998, 4, 9), date(2022, 12, 1)],
        "1w": [date(1998, 4, 13), date(2022, 11, 28)],
    }


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
def test_dt_round_fast_path_vs_slow_path(datetimes: list[datetime], every: str) -> None:
    s = pl.Series(datetimes)
    # Might use fastpath:
    result = s.dt.round(every)
    # Definitely uses slowpath:
    expected = s.dt.round(pl.Series([every] * len(datetimes)))
    assert_series_equal(result, expected)

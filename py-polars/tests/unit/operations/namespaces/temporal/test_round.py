from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars._utils.convert import parse_as_duration_string
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars.type_aliases import TimeUnit


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


def test_round_date() -> None:
    # n vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 19)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.round(pl.col("b")))["a"]
    expected = pl.Series("a", [None, None, date(2020, 2, 1)])
    assert_series_equal(result, expected)

    # n vs 1
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.round("1mo"))["a"]
    expected = pl.Series("a", [date(2020, 1, 1), None, date(2020, 1, 1)])
    assert_series_equal(result, expected)

    # n vs missing
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(pl.col("a").dt.round(pl.lit(None, dtype=pl.String)))["a"]
    expected = pl.Series("a", [None, None, None], dtype=pl.Date)
    assert_series_equal(result, expected)

    # 1 vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(a=pl.date(2020, 1, 1).dt.round(pl.col("b")))["a"]
    expected = pl.Series("a", [None, date(2020, 1, 1), date(2020, 1, 1)])
    assert_series_equal(result, expected)

    # missing vs n
    df = pl.DataFrame(
        {"a": [date(2020, 1, 1), None, date(2020, 1, 3)], "b": [None, "1mo", "1mo"]}
    )
    result = df.select(a=pl.lit(None, dtype=pl.Date).dt.round(pl.col("b")))["a"]
    expected = pl.Series("a", [None, None, None], dtype=pl.Date)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_round_datetime_simple(time_unit: TimeUnit) -> None:
    s = pl.Series([datetime(2020, 1, 2, 6)], dtype=pl.Datetime(time_unit))
    result = s.dt.round("1mo").item()
    assert result == datetime(2020, 1, 1)
    result = s.dt.round("1d").item()
    assert result == datetime(2020, 1, 2)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_round_datetime_w_expression(time_unit: TimeUnit) -> None:
    df = pl.DataFrame(
        {"a": [datetime(2020, 1, 2, 6), datetime(2020, 1, 20, 21)], "b": ["1mo", "1d"]},
        schema_overrides={"a": pl.Datetime(time_unit)},
    )
    result = df.select(pl.col("a").dt.round(pl.col("b")))["a"]
    assert result[0] == datetime(2020, 1, 1)
    assert result[1] == datetime(2020, 1, 21)


@pytest.mark.parametrize(
    ("time_unit", "every"),
    [
        ("ms", "1h"),
        ("us", "1h0m0s"),
        ("ns", timedelta(hours=1)),
    ],
    ids=["milliseconds", "microseconds", "nanoseconds"],
)
def test_round(
    time_unit: TimeUnit,
    every: str | timedelta,
) -> None:
    start, stop = datetime(2022, 1, 1), datetime(2022, 1, 2)
    s = pl.datetime_range(
        start,
        stop,
        timedelta(minutes=30),
        time_unit=time_unit,
        eager=True,
    ).alias(f"dates[{time_unit}]")

    # can pass strings and time-deltas
    out = s.dt.round(every)
    assert out.dt[0] == start
    assert out.dt[1] == start + timedelta(hours=1)
    assert out.dt[2] == start + timedelta(hours=1)
    assert out.dt[3] == start + timedelta(hours=2)
    # ...
    assert out.dt[-3] == stop - timedelta(hours=1)
    assert out.dt[-2] == stop
    assert out.dt[-1] == stop


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_round_duration(time_unit: TimeUnit) -> None:
    durations = pl.Series(
        [
            timedelta(seconds=21),
            timedelta(seconds=35),
            timedelta(seconds=59),
            None,
            timedelta(seconds=-35),
        ]
    ).dt.cast_time_unit(time_unit)

    expected = pl.Series(
        [
            timedelta(seconds=20),
            timedelta(seconds=40),
            timedelta(seconds=60),
            None,
            timedelta(seconds=-40),
        ]
    ).dt.cast_time_unit(time_unit)

    assert_series_equal(durations.dt.round("10s"), expected)


def test_round_duration_zero() -> None:
    """Rounding to the nearest zero should raise a descriptive error."""
    durations = pl.Series([timedelta(seconds=21), timedelta(seconds=35)])

    with pytest.raises(
        InvalidOperationError,
        match="cannot round a Duration to a non-positive Duration",
    ):
        durations.dt.round("0s")


@pytest.mark.parametrize("every", ["mo", "q", "y"])
def test_round_duration_non_constant(every: str) -> None:
    # Duration series can't be rounded to non-constant durations
    durations = pl.Series([timedelta(seconds=21)])

    with pytest.raises(InvalidOperationError):
        durations.dt.round("1" + every)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_round_duration_half(time_unit: TimeUnit) -> None:
    # Values at halfway points should round away from zero
    durations = pl.Series(
        [timedelta(minutes=-30), timedelta(minutes=30), timedelta(minutes=90)]
    ).dt.cast_time_unit(time_unit)

    expected = pl.Series(
        [timedelta(hours=-1), timedelta(hours=1), timedelta(hours=2)]
    ).dt.cast_time_unit(time_unit)

    assert_series_equal(durations.dt.round("1h"), expected)


def test_round_expr() -> None:
    df = pl.DataFrame(
        {
            "date": [
                datetime(2022, 11, 14),
                datetime(2023, 10, 11),
                datetime(2022, 3, 20, 5, 7, 18),
                datetime(2022, 4, 3, 13, 30, 32),
                None,
                datetime(2022, 12, 1),
            ],
            "every": ["1y", "1mo", "1m", "1m", "1mo", None],
        }
    )

    output = df.select(
        all_expr=pl.col("date").dt.round(every=pl.col("every")),
        date_lit=pl.lit(datetime(2022, 4, 3, 13, 30, 32)).dt.round(
            every=pl.col("every")
        ),
        every_lit=pl.col("date").dt.round("1d"),
    )

    expected = pl.DataFrame(
        {
            "all_expr": [
                datetime(2023, 1, 1),
                datetime(2023, 10, 1),
                datetime(2022, 3, 20, 5, 7),
                datetime(2022, 4, 3, 13, 31),
                None,
                None,
            ],
            "date_lit": [
                datetime(2022, 1, 1),
                datetime(2022, 4, 1),
                datetime(2022, 4, 3, 13, 31),
                datetime(2022, 4, 3, 13, 31),
                datetime(2022, 4, 1),
                None,
            ],
            "every_lit": [
                datetime(2022, 11, 14),
                datetime(2023, 10, 11),
                datetime(2022, 3, 20),
                datetime(2022, 4, 4),
                None,
                datetime(2022, 12, 1),
            ],
        }
    )

    assert_frame_equal(output, expected)

    all_lit = pl.select(all_lit=pl.lit(datetime(2022, 3, 20, 5, 7)).dt.round("1h"))
    assert all_lit.to_dict(as_series=False) == {"all_lit": [datetime(2022, 3, 20, 5)]}


def test_round_negative() -> None:
    """Test that rounding to a negative duration gives a helpful error message."""
    with pytest.raises(
        InvalidOperationError, match="cannot round a Date to a non-positive Duration"
    ):
        pl.Series([date(1895, 5, 7)]).dt.round("-1m")

    with pytest.raises(
        InvalidOperationError,
        match="cannot round a Datetime to a non-positive Duration",
    ):
        pl.Series([datetime(1895, 5, 7)]).dt.round("-1m")

    with pytest.raises(
        InvalidOperationError,
        match="cannot round a Duration to a non-positive Duration",
    ):
        pl.Series([timedelta(days=1)]).dt.round("-1m")

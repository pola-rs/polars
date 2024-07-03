from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


def test_empty_duration() -> None:
    s = pl.DataFrame([], {"days": pl.Int32}).select(pl.duration(days="days"))
    assert s.dtypes == [pl.Duration("us")]
    assert s.shape == (0, 1)


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [
        ("ms", timedelta(days=1, minutes=2, seconds=3, milliseconds=4)),
        ("us", timedelta(days=1, minutes=2, seconds=3, milliseconds=4, microseconds=5)),
        ("ns", timedelta(days=1, minutes=2, seconds=3, milliseconds=4, microseconds=5)),
    ],
)
def test_duration_time_units(time_unit: TimeUnit, expected: timedelta) -> None:
    result = pl.LazyFrame().select(
        duration=pl.duration(
            days=1,
            minutes=2,
            seconds=3,
            milliseconds=4,
            microseconds=5,
            nanoseconds=6,
            time_unit=time_unit,
        )
    )
    assert result.collect_schema()["duration"] == pl.Duration(time_unit)
    assert result.collect()["duration"].item() == expected
    if time_unit == "ns":
        assert (
            result.collect()["duration"].dt.total_nanoseconds().item() == 86523004005006
        )


def test_datetime_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(1999, 1, 1, 7),
                datetime(2022, 1, 2, 14),
                datetime(3000, 12, 31, 21),
            ],
            "add": [1, 2, -1],
        }
    )
    out = df.select(
        (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
        (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
        (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
        (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
        (pl.col("datetime") + pl.duration(microseconds=pl.col("add") * 1000)).alias(
            "add_usecs"
        ),
    )
    expected = pl.DataFrame(
        {
            "add_weeks": [
                datetime(1999, 1, 8, 7),
                datetime(2022, 1, 16, 14),
                datetime(3000, 12, 24, 21),
            ],
            "add_days": [
                datetime(1999, 1, 2, 7),
                datetime(2022, 1, 4, 14),
                datetime(3000, 12, 30, 21),
            ],
            "add_hours": [
                datetime(1999, 1, 1, hour=8),
                datetime(2022, 1, 2, hour=16),
                datetime(3000, 12, 31, hour=20),
            ],
            "add_seconds": [
                datetime(1999, 1, 1, 7, second=1),
                datetime(2022, 1, 2, 14, second=2),
                datetime(3000, 12, 31, 20, 59, 59),
            ],
            "add_usecs": [
                datetime(1999, 1, 1, 7, microsecond=1000),
                datetime(2022, 1, 2, 14, microsecond=2000),
                datetime(3000, 12, 31, 20, 59, 59, 999000),
            ],
        }
    )
    assert_frame_equal(out, expected)


def test_date_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "date": [date(10, 1, 1), date(2000, 7, 5), date(9990, 12, 31)],
            "offset": [365, 7, -31],
        }
    )
    out = df.select(
        (pl.col("date") + pl.duration(days="offset")).alias("add_days"),
        (pl.col("date") - pl.duration(days="offset")).alias("sub_days"),
        (pl.col("date") + pl.duration(weeks="offset")).alias("add_weeks"),
        (pl.col("date") - pl.duration(weeks="offset")).alias("sub_weeks"),
    )
    assert out.to_dict(as_series=False) == {
        "add_days": [date(11, 1, 1), date(2000, 7, 12), date(9990, 11, 30)],
        "sub_days": [date(9, 1, 1), date(2000, 6, 28), date(9991, 1, 31)],
        "add_weeks": [date(16, 12, 30), date(2000, 8, 23), date(9990, 5, 28)],
        "sub_weeks": [date(3, 1, 3), date(2000, 5, 17), date(9991, 8, 5)],
    }


def test_add_duration_3786() -> None:
    df = pl.DataFrame(
        {
            "datetime": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
            "add": [1, 2],
        }
    )
    assert df.slice(0, 1).with_columns(
        (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
        (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
        (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
        (pl.col("datetime") + pl.duration(milliseconds="add")).alias(
            "add_milliseconds"
        ),
        (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
    ).to_dict(as_series=False) == {
        "datetime": [datetime(2022, 1, 1, 0, 0)],
        "add": [1],
        "add_weeks": [datetime(2022, 1, 8, 0, 0)],
        "add_days": [datetime(2022, 1, 2, 0, 0)],
        "add_seconds": [datetime(2022, 1, 1, 0, 0, 1)],
        "add_milliseconds": [datetime(2022, 1, 1, 0, 0, 0, 1000)],
        "add_hours": [datetime(2022, 1, 1, 1, 0)],
    }


@pytest.mark.parametrize(
    ("time_unit", "ms", "us", "ns"),
    [
        ("ms", 11, 0, 0),
        ("us", 0, 11_007, 0),
        ("ns", 0, 0, 11_007_003),
    ],
)
def test_duration_subseconds_us(time_unit: TimeUnit, ms: int, us: int, ns: int) -> None:
    result = pl.duration(
        milliseconds=6, microseconds=4_005, nanoseconds=1_002_003, time_unit=time_unit
    )
    expected = pl.duration(
        milliseconds=ms, microseconds=us, nanoseconds=ns, time_unit=time_unit
    )
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_ns() -> None:
    result = pl.duration(milliseconds=4, microseconds=3_000, nanoseconds=10)
    expected = pl.duration(
        milliseconds=4, microseconds=3_000, nanoseconds=10, time_unit="ns"
    )
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_us() -> None:
    result = pl.duration(milliseconds=4, microseconds=3_000)
    expected = pl.duration(milliseconds=4, microseconds=3_000, time_unit="us")
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_ms() -> None:
    result = pl.duration(milliseconds=4)
    expected = pl.duration(milliseconds=4, time_unit="us")
    assert_frame_equal(pl.select(result), pl.select(expected))

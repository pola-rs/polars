from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    SchemaError,
    ShapeError,
)
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import DateDiffUnit, TimeUnit

UNITS: list[DateDiffUnit] = [
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
]


def date_diffs(lf: pl.LazyFrame, units: list[DateDiffUnit]) -> pl.DataFrame:
    q = lf.select(pl.date_diff("start", "end", unit).alias(unit) for unit in units)
    out = q.collect()
    assert_frame_equal(q.collect(engine="streaming"), out)
    assert out.schema == dict.fromkeys(units, pl.Int64)
    return out


def test_date_diff_dates() -> None:
    lf = pl.LazyFrame(
        {
            # 2024-01-07 is a Sunday
            "start": [date(2024, 1, 31), date(2024, 1, 1), date(2024, 1, 7), None],
            "end": [date(2024, 2, 1), date(2024, 1, 31), date(2023, 12, 31), None],
        }
    )
    out = date_diffs(lf, UNITS)

    def sub_day(days: int) -> tuple[int, ...]:
        seconds = days * 86_400
        return (days * 24, days * 1440, seconds, *(seconds * 10**k for k in (3, 6, 9)))

    assert out.rows() == [
        (0, 0, 1, 0, 1, *sub_day(1)),
        (0, 0, 0, 4, 30, *sub_day(30)),
        (-1, -1, -1, -1, -7, *sub_day(-7)),
        (None,) * 11,
    ]

    # the start of a new week (Monday) is a boundary
    lf = pl.LazyFrame({"start": [date(2024, 1, 7)], "end": [date(2024, 1, 8)]})
    assert date_diffs(lf, ["week", "day"]).rows() == [(1, 1)]


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_date_diff_datetimes(time_unit: TimeUnit) -> None:
    lf = pl.LazyFrame(
        {
            "start": [
                datetime(2024, 1, 1, 10, 59, 59, 999_000),
                datetime(2023, 12, 31, 23, 59),
                datetime(1969, 12, 31, 23, 30),
            ],
            "end": [
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 0, 1),
                datetime(1970, 1, 1, 0, 30),
            ],
        },
        schema={"start": pl.Datetime(time_unit), "end": pl.Datetime(time_unit)},
    )
    assert date_diffs(lf, UNITS).rows() == [
        (0, 0, 0, 0, 0, 1, 1, 1, 1, 1_000, 1_000_000),
        (1, 1, 1, 1, 1, 1, 2, 120, 120_000, 120_000_000, 120_000_000_000),
        (1, 1, 1, 0, 1, 1, 60, 3_600, 3_600_000, 3_600_000_000, 3_600_000_000_000),
    ]


def test_date_diff_mixed_dtypes() -> None:
    lf = pl.LazyFrame(
        {
            "start": [date(2024, 1, 1)],
            "end": pl.Series(
                [datetime(2024, 1, 2, 3, 0, 0, 1)], dtype=pl.Datetime("ns")
            ),
        }
    )
    assert date_diffs(lf, ["day", "hour", "nanosecond"]).rows() == [
        (1, 27, 97_200_000_001_000)
    ]

    lf = pl.LazyFrame(
        {
            "start": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ms")),
            "end": pl.Series([datetime(2024, 1, 1, 0, 0, 1)], dtype=pl.Datetime("ns")),
        }
    )
    assert date_diffs(lf, ["second", "nanosecond"]).rows() == [(1, 1_000_000_000)]

    # the finer time unit is used
    lf = pl.LazyFrame(
        {
            "start": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ms")),
            "end": pl.Series(
                [datetime(2024, 1, 1, 0, 0, 0, 999)], dtype=pl.Datetime("us")
            ),
        }
    )
    assert date_diffs(lf, ["millisecond", "microsecond"]).rows() == [(0, 999)]

    # days beyond the range of Date
    day_ms = 86_400_000
    lf = pl.LazyFrame(
        {"start": [2**31 * day_ms], "end": [(2**31 + 7) * day_ms]},
        schema={"start": pl.Datetime("ms"), "end": pl.Datetime("ms")},
    )
    assert date_diffs(lf, ["week", "day"]).rows() == [(1, 7)]


def test_date_diff_time_zones() -> None:
    ams = ZoneInfo("Europe/Amsterdam")
    lf = pl.LazyFrame(
        {
            "start": [
                # 01:30 CET to 03:30 CEST is one hour
                datetime(2024, 3, 31, 1, 30, tzinfo=ams),
                # second 02:30 is one hour after the first
                datetime(2024, 10, 27, 0, 30, tzinfo=ZoneInfo("UTC")),
                datetime(2024, 1, 1, 23, 30, tzinfo=ams),
            ],
            "end": [
                datetime(2024, 3, 31, 3, 30, tzinfo=ams),
                datetime(2024, 10, 27, 1, 30, tzinfo=ZoneInfo("UTC")),
                datetime(2024, 1, 2, 0, 30, tzinfo=ams),
            ],
        },
        schema={"start": pl.Datetime("us", ams.key), "end": pl.Datetime("us", ams.key)},
    )
    assert date_diffs(lf, ["day", "hour", "minute"]).rows() == [
        (0, 1, 60),
        (0, 1, 60),
        (1, 1, 60),
    ]

    # boundaries are those of the local time
    kolkata = "Asia/Kolkata"
    lf = pl.LazyFrame(
        {
            "start": [datetime(2024, 1, 1, 15, 50), datetime(2024, 1, 1, 18, 0)],
            "end": [datetime(2024, 1, 1, 16, 10), datetime(2024, 1, 1, 18, 40)],
        },
        schema={"start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC")},
    ).with_columns(pl.all().dt.replace_time_zone(None).dt.replace_time_zone(kolkata))
    assert date_diffs(lf, ["day", "hour"]).rows() == [(0, 1), (0, 0)]

    # 18:00 UTC to 18:40 UTC crosses midnight in Kolkata
    lf = lf.with_columns(
        pl.all().dt.replace_time_zone(None).dt.replace_time_zone("UTC")
    )
    lf = lf.with_columns(pl.all().dt.convert_time_zone(kolkata))
    assert date_diffs(lf, ["day", "hour"]).rows() == [(0, 0), (1, 1)]


def test_date_diff_times() -> None:
    lf = pl.LazyFrame(
        {
            "start": [time(10, 59, 59), time(23, 0), None],
            "end": [time(11, 0, 1), time(0, 30), time(1)],
        }
    )
    assert date_diffs(lf, ["hour", "minute", "second", "nanosecond"]).rows() == [
        (1, 1, 2, 2_000_000_000),
        (-23, -1350, -81_000, -81_000_000_000_000),
        (None, None, None, None),
    ]


def test_date_diff_broadcast_and_group_by() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 2],
            "d": [date(2024, 1, 15), date(2024, 5, 1), date(2025, 1, 1)],
        }
    )
    assert df.select(
        a=pl.date_diff(date(2024, 1, 1), "d", "month"),
        b=pl.date_diff("d", datetime(2024, 1, 1, 12), "hour"),
        c=pl.date_diff(pl.lit(None), "d", "day"),
    ).rows() == [(0, -324, None), (4, -2892, None), (12, -8772, None)]

    out = df.group_by("g", maintain_order=True).agg(
        pl.date_diff(pl.col("d").first(), "d", "month")
    )
    assert out.rows() == [(1, [0, 4]), (2, [0])]


def test_date_diff_errors() -> None:
    with pytest.raises(ValueError, match="`unit` must be one of"):
        pl.date_diff("a", "b", "days")  # type: ignore[arg-type]

    with pytest.raises(
        InvalidOperationError, match="unit 'day' is not supported for Time"
    ):
        pl.select(pl.date_diff(pl.lit(time(1)), pl.lit(time(2)), "day"))

    with pytest.raises(InvalidOperationError, match="expects Date, Datetime or Time"):
        pl.select(pl.date_diff(pl.lit(1), pl.lit(2), "day"))

    with pytest.raises(SchemaError, match="supertype"):
        pl.select(pl.date_diff(pl.lit(date(2024, 1, 1)), pl.lit(time(2)), "hour"))

    with pytest.raises(ComputeError, match="overflow in `date_diff`"):
        pl.select(
            pl.date_diff(
                pl.lit(datetime(1700, 1, 1)), pl.lit(datetime(2200, 1, 1)), "nanosecond"
            )
        )

    # a date outside the range of the other input's time unit
    end = pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ns"))
    with pytest.raises(InvalidOperationError, match="conversion from `date`"):
        pl.select(pl.date_diff(date(1600, 1, 1), end, "day"))

    df = pl.DataFrame({"a": [date(2024, 1, 1)] * 3})
    with pytest.raises(ShapeError):
        df.select(pl.date_diff("a", pl.Series([date(2024, 1, 1)] * 2), "day"))

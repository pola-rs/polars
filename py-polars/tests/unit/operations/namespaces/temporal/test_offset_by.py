from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import Ambiguous, TimeUnit


@pytest.mark.parametrize(
    ("inputs", "offset", "outputs"),
    [
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "1d",
            [date(2020, 1, 2), date(2020, 1, 3)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-1d",
            [date(2019, 12, 31), date(2020, 1, 1)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "3d",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "72h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "+2d24h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-2mo",
            [date(2019, 11, 1), date(2019, 11, 2)],
        ),
    ],
)
def test_date_offset_by(inputs: list[date], offset: str, outputs: list[date]) -> None:
    result = pl.Series(inputs).dt.offset_by(offset)
    expected = pl.Series(outputs)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("inputs", "offset", "outputs"),
    [
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "1d",
            [date(2020, 1, 2), date(2020, 1, 3)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-1d",
            [date(2019, 12, 31), date(2020, 1, 1)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "3d",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "72h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "2d24h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "7m",
            [datetime(2020, 1, 1, 0, 7), datetime(2020, 1, 2, 0, 7)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-3m",
            [datetime(2019, 12, 31, 23, 57), datetime(2020, 1, 1, 23, 57)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "2mo",
            [datetime(2020, 3, 1), datetime(2020, 3, 2)],
        ),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", ["Europe/London", "Asia/Kathmandu", None])
def test_datetime_offset_by(
    inputs: list[date],
    offset: str,
    outputs: list[datetime],
    time_unit: TimeUnit,
    time_zone: str | None,
) -> None:
    result = (
        pl.Series(inputs, dtype=pl.Datetime(time_unit))
        .dt.replace_time_zone(time_zone)
        .dt.offset_by(offset)
    )
    expected = pl.Series(outputs, dtype=pl.Datetime(time_unit)).dt.replace_time_zone(
        time_zone
    )
    assert_series_equal(result, expected)


def test_offset_by_unique_29_feb_19608() -> None:
    df20 = pl.select(
        t=pl.datetime_range(
            pl.datetime(2020, 2, 28),
            pl.datetime(2020, 3, 1),
            closed="left",
            time_unit="ms",
            interval="8h",
            time_zone="UTC",
        ),
    ).with_columns(x=pl.int_range(pl.len()))
    df19 = df20.with_columns(pl.col("t").dt.offset_by("-1y"))
    result = df19.unique("t", keep="first").sort("t")
    expected = pl.DataFrame(
        {
            "t": [
                datetime(2019, 2, 28),
                datetime(2019, 2, 28, 8),
                datetime(2019, 2, 28, 16),
            ],
            "x": [0, 1, 2],
        },
        schema_overrides={"t": pl.Datetime("ms", "UTC")},
    )
    assert_frame_equal(result, expected)


def test_month_then_day_21283() -> None:
    series_vienna = pl.Series(
        [datetime(2024, 5, 15, 8, 0)], dtype=pl.Datetime(time_zone="Europe/Vienna")
    )
    result = series_vienna.dt.offset_by("2y1mo1q1h")[0]
    expected = datetime.strptime("2026-09-15 11:00:00+02:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    result = series_vienna.dt.offset_by("2y1mo1q1h1d")[0]
    expected = datetime.strptime("2026-09-16 11:00:00+02:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    series_utc = pl.Series(
        [datetime(2024, 5, 15, 8, 0)], dtype=pl.Datetime(time_zone="UTC")
    )
    result = series_utc.dt.offset_by("2y1mo1q1h")[0]
    expected = datetime.strptime("2026-09-15 09:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    result = series_utc.dt.offset_by("2y1mo1q1h1d")[0]
    expected = datetime.strptime("2026-09-16 09:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected


def test_offset_by_unequal_length_22018() -> None:
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([datetime(2088, 8, 8, 8, 8, 8, 8)] * 2).dt.offset_by(
            pl.Series([f"{h}y" for h in range(3)])
        )


@pytest.mark.parametrize(
    ("start", "by", "ambiguous"),
    [
        (datetime(2025, 10, 25, 2), "1d", "earliest"),
        (datetime(2025, 10, 27, 2), "-1d", "latest"),
        (datetime(2025, 10, 19, 2), "1w", "earliest"),
        (datetime(2025, 11, 2, 2), "-1w", "latest"),
        (datetime(2024, 10, 26, 2), "1y", "earliest"),
        (datetime(2026, 10, 26, 2), "-1y", "latest"),
        (datetime(2025, 9, 26, 2), "1mo", "earliest"),
        (datetime(2025, 11, 26, 2), "-1mo", "latest"),
    ],
)
def test_offset_by_rfc_5545_boundaries(
    start: datetime, by: str, ambiguous: Ambiguous
) -> None:
    s = pl.Series([start]).dt.replace_time_zone("Europe/Amsterdam")
    result = s.dt.offset_by(by)
    expected = pl.Series([datetime(2025, 10, 26, 2)]).dt.replace_time_zone(
        "Europe/Amsterdam", ambiguous=ambiguous
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("start", "by", "expected_dt"),
    [
        (datetime(2025, 3, 29, 2, 30), "1d", datetime(2025, 3, 30, 1, 30)),
        (datetime(2025, 3, 31, 2, 30), "-1d", datetime(2025, 3, 30, 3, 30)),
        (datetime(2025, 1, 30, 2, 30), "2mo", datetime(2025, 3, 30, 1, 30)),
        (datetime(2025, 4, 30, 2, 30), "-1mo", datetime(2025, 3, 30, 3, 30)),
    ],
)
def test_offset_by_rfc_5545_boundaries_non_existest(
    start: datetime, by: str, expected_dt: datetime
) -> None:
    s = pl.Series([start]).dt.replace_time_zone("Europe/Amsterdam")
    result = s.dt.offset_by(by)
    expected = pl.Series([expected_dt]).dt.replace_time_zone("Europe/Amsterdam")
    assert_series_equal(result, expected)

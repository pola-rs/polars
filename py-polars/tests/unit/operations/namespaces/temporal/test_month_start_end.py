from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2022, 3, 15, 3), datetime(2022, 3, 1, 3)),
        (datetime(2022, 3, 15, 3, 2, 1, 123000), datetime(2022, 3, 1, 3, 2, 1, 123000)),
        (datetime(2022, 3, 15), datetime(2022, 3, 1)),
        (datetime(2022, 3, 1), datetime(2022, 3, 1)),
    ],
)
@pytest.mark.parametrize(
    ("tzinfo", "time_zone"),
    [
        (None, None),
        (ZoneInfo("Asia/Kathmandu"), "Asia/Kathmandu"),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_month_start_datetime(
    dt: datetime,
    expected: datetime,
    time_unit: TimeUnit,
    tzinfo: ZoneInfo | None,
    time_zone: str | None,
) -> None:
    ser = pl.Series([dt]).dt.replace_time_zone(time_zone).dt.cast_time_unit(time_unit)
    result = ser.dt.month_start().item()
    assert result == expected.replace(tzinfo=tzinfo)


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (date(2022, 3, 15), date(2022, 3, 1)),
        (date(2022, 3, 31), date(2022, 3, 1)),
    ],
)
def test_month_start_date(dt: date, expected: date) -> None:
    ser = pl.Series([dt])
    result = ser.dt.month_start().item()
    assert result == expected


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2022, 3, 15, 3), datetime(2022, 3, 31, 3)),
        (
            datetime(2022, 3, 15, 3, 2, 1, 123000),
            datetime(2022, 3, 31, 3, 2, 1, 123000),
        ),
        (datetime(2022, 3, 15), datetime(2022, 3, 31)),
        (datetime(2022, 3, 31), datetime(2022, 3, 31)),
    ],
)
@pytest.mark.parametrize(
    ("tzinfo", "time_zone"),
    [
        (None, None),
        (ZoneInfo("Asia/Kathmandu"), "Asia/Kathmandu"),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_month_end_datetime(
    dt: datetime,
    expected: datetime,
    time_unit: TimeUnit,
    tzinfo: ZoneInfo | None,
    time_zone: str | None,
) -> None:
    ser = pl.Series([dt]).dt.replace_time_zone(time_zone).dt.cast_time_unit(time_unit)
    result = ser.dt.month_end().item()
    assert result == expected.replace(tzinfo=tzinfo)


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (date(2022, 3, 15), date(2022, 3, 31)),
        (date(2022, 3, 31), date(2022, 3, 31)),
    ],
)
def test_month_end_date(dt: date, expected: date) -> None:
    ser = pl.Series([dt])
    result = ser.dt.month_end().item()
    assert result == expected


def test_month_start_end_invalid() -> None:
    ser = pl.Series([time(1, 2, 3)])
    with pytest.raises(
        InvalidOperationError,
        match=r"`month_start` operation not supported for dtype `time` \(expected: date/datetime\)",
    ):
        ser.dt.month_start()
    with pytest.raises(
        InvalidOperationError,
        match=r"`month_end` operation not supported for dtype `time` \(expected: date/datetime\)",
    ):
        ser.dt.month_end()


def test_month_end_ambiguous_start_24646() -> None:
    dt = datetime(1987, 3, 3, 2, 45, 00, tzinfo=ZoneInfo("Pacific/Chatham"))
    with pytest.raises(ComputeError, match="is ambiguous"):
        pl.DataFrame({"a": [dt]}).select(pl.col("a").dt.month_start())
    result = pl.DataFrame({"a": [dt]}).select(pl.col("a").dt.month_end())
    expected = pl.DataFrame({"a": [datetime(1987, 3, 31, 2, 45)]}).with_columns(
        pl.col("a").dt.replace_time_zone("Pacific/Chatham")
    )
    assert_frame_equal(result, expected)


def test_month_end_non_existent_start_24646() -> None:
    # 2015-03-01 00:30 in America/Havana
    dt = datetime(1990, 4, 2, 0, 30, tzinfo=ZoneInfo("America/Havana"))
    with pytest.raises(ComputeError, match="is non-existent"):
        pl.DataFrame({"a": [dt]}).select(pl.col("a").dt.month_start())
    result = pl.DataFrame({"a": [dt]}).select(pl.col("a").dt.month_end())
    expected = pl.DataFrame(
        {"a": [datetime(1990, 4, 30, 0, 30, tzinfo=ZoneInfo("America/Havana"))]}
    )
    assert_frame_equal(result, expected)

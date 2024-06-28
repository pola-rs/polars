from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import TimeUnit
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


def test_date_datetime() -> None:
    df = pl.DataFrame(
        {
            "year": [2001, 2002, 2003],
            "month": [1, 2, 3],
            "day": [1, 2, 3],
            "hour": [23, 12, 8],
        }
    )
    out = df.select(
        pl.all(),
        pl.datetime("year", "month", "day", "hour").dt.hour().cast(int).alias("h2"),
        pl.date("year", "month", "day").dt.day().cast(int).alias("date"),
    )
    assert_series_equal(out["date"], df["day"].rename("date"))
    assert_series_equal(out["h2"], df["hour"].rename("h2"))


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_datetime_time_unit(time_unit: TimeUnit) -> None:
    result = pl.datetime(2022, 1, 2, time_unit=time_unit)

    assert pl.select(result.dt.year()).item() == 2022
    assert pl.select(result.dt.month()).item() == 1
    assert pl.select(result.dt.day()).item() == 2


@pytest.mark.parametrize("time_zone", [None, "Europe/Amsterdam", "UTC"])
def test_datetime_time_zone(time_zone: str | None) -> None:
    result = pl.datetime(2022, 1, 2, 10, time_zone=time_zone)

    assert pl.select(result.dt.year()).item() == 2022
    assert pl.select(result.dt.month()).item() == 1
    assert pl.select(result.dt.day()).item() == 2
    assert pl.select(result.dt.hour()).item() == 10


def test_datetime_ambiguous_time_zone() -> None:
    expr = pl.datetime(2018, 10, 28, 2, 30, time_zone="Europe/Brussels")

    with pytest.raises(ComputeError):
        pl.select(expr)


def test_datetime_ambiguous_time_zone_earliest() -> None:
    expr = pl.datetime(
        2018, 10, 28, 2, 30, time_zone="Europe/Brussels", ambiguous="earliest"
    )

    result = pl.select(expr).item()

    expected = datetime(2018, 10, 28, 2, 30, tzinfo=ZoneInfo("Europe/Brussels"))
    assert result == expected
    assert result.fold == 0

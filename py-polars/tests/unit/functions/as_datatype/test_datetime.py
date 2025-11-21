from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


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


@pytest.mark.parametrize(
    "components",
    [
        [2025, 13, 1],
        [2025, 1, 32],
        [2025, 2, 29],
    ],
)
def test_date_invalid_component(components: list[int]) -> None:
    y, m, d = components
    msg = rf"Invalid date components \({y}, {m}, {d}\) supplied"
    with pytest.raises(ComputeError, match=msg):
        pl.select(pl.date(*components))


@pytest.mark.parametrize(
    "components",
    [
        [2025, 13, 1, 0, 0, 0, 0],
        [2025, 1, 32, 0, 0, 0, 0],
        [2025, 2, 29, 0, 0, 0, 0],
    ],
)
def test_datetime_invalid_date_component(components: list[int]) -> None:
    y, m, d = components[0:3]
    msg = rf"Invalid date components \({y}, {m}, {d}\) supplied"
    with pytest.raises(ComputeError, match=msg):
        pl.select(pl.datetime(*components))


@pytest.mark.parametrize(
    "components",
    [
        [2025, 1, 1, 25, 0, 0, 0],
        [2025, 1, 1, 0, 60, 0, 0],
        [2025, 1, 1, 0, 0, 60, 0],
        [2025, 1, 1, 0, 0, 0, 2_000_000],
    ],
)
def test_datetime_invalid_time_component(components: list[int]) -> None:
    h, mnt, s, us = components[3:]
    ns = us * 1_000
    msg = rf"Invalid time components \({h}, {mnt}, {s}, {ns}\) supplied"
    with pytest.raises(ComputeError, match=msg):
        pl.select(pl.datetime(*components))


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


def test_datetime_wildcard_expansion() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})
    assert df.select(
        pl.datetime(year=pl.all(), month=pl.all(), day=pl.all()).name.keep()
    ).to_dict(as_series=False) == {
        "a": [datetime(1, 1, 1, 0, 0)],
        "b": [datetime(2, 2, 2, 0, 0)],
    }


def test_datetime_invalid_time_zone() -> None:
    df1 = pl.DataFrame({"year": []}, schema={"year": pl.Int32})
    df2 = pl.DataFrame({"year": [2024]})

    with pytest.raises(ComputeError, match="unable to parse time zone: 'foo'"):
        df1.select(pl.datetime("year", 1, 1, time_zone="foo"))

    with pytest.raises(ComputeError, match="unable to parse time zone: 'foo'"):
        df2.select(pl.datetime("year", 1, 1, time_zone="foo"))


def test_datetime_from_empty_column() -> None:
    df = pl.DataFrame({"year": []}, schema={"year": pl.Int32})

    assert df.select(datetime=pl.datetime("year", 1, 1)).shape == (0, 1)
    assert df.with_columns(datetime=pl.datetime("year", 1, 1)).shape == (0, 2)

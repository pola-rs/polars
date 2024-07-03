from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


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

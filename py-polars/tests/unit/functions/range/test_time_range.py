from __future__ import annotations

from datetime import time, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval


def test_time_range_schema() -> None:
    df = pl.DataFrame({"start": [time(1)], "end": [time(1, 30)]}).lazy()
    result = df.with_columns(pl.time_ranges(pl.col("start"), pl.col("end")))
    expected_schema = {"start": pl.Time, "end": pl.Time, "time_range": pl.List(pl.Time)}
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


def test_time_range_no_alias_schema_9037() -> None:
    df = pl.DataFrame({"start": [time(1)], "end": [time(1, 30)]}).lazy()
    result = df.with_columns(pl.time_ranges(pl.col("start"), pl.col("end")))
    expected_schema = {"start": pl.Time, "end": pl.Time, "time_range": pl.List(pl.Time)}
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


def test_time_ranges_eager() -> None:
    start = pl.Series([time(9, 0), time(10, 0)])
    end = pl.Series([time(12, 0), time(11, 0)])

    result = pl.time_ranges(start, end, eager=True)

    expected = pl.Series(
        "time_range",
        [
            [time(9, 0), time(10, 0), time(11, 0), time(12, 0)],
            [time(10, 0), time(11, 0)],
        ],
    )
    assert_series_equal(result, expected)


def test_time_range_eager_explode() -> None:
    start = pl.Series([time(9, 0)])
    end = pl.Series([time(11, 0)])

    result = pl.time_range(start, end, eager=True)

    expected = pl.Series("time", [time(9, 0), time(10, 0), time(11, 0)])
    assert_series_equal(result, expected)


def test_time_range_empty() -> None:
    empty = pl.Series(dtype=pl.Time)
    single = pl.Series([time(12, 0)])

    with pytest.raises(pl.ComputeError, match="`start` must contain a single value"):
        pl.time_range(empty, single, eager=True)
    with pytest.raises(pl.ComputeError, match="`end` must contain a single value"):
        pl.time_range(single, empty, eager=True)
    with pytest.raises(pl.ComputeError, match="`start` must contain a single value"):
        pl.time_range(empty, empty, eager=True)


def test_time_range_multiple_values() -> None:
    single = pl.Series([time(12, 0)])
    multiple = pl.Series([time(11, 0), time(12, 0)])

    with pytest.raises(pl.ComputeError, match="`start` must contain a single value"):
        pl.time_range(multiple, single, eager=True)
    with pytest.raises(pl.ComputeError, match="`end` must contain a single value"):
        pl.time_range(single, multiple, eager=True)
    with pytest.raises(pl.ComputeError, match="`start` must contain a single value"):
        pl.time_range(multiple, multiple, eager=True)


def test_time_range_start_equals_end() -> None:
    t = time(12, 0)

    result = pl.time_range(t, t, closed="both", eager=True)

    expected = pl.Series("time", [t])
    assert_series_equal(result, expected)


@pytest.mark.parametrize("closed", ["left", "right", "none"])
def test_time_range_start_equals_end_open(closed: ClosedInterval) -> None:
    t = time(12, 0)

    result = pl.time_range(t, t, closed=closed, eager=True)

    expected = pl.Series("time", dtype=pl.Time)
    assert_series_equal(result, expected)


def test_time_range_invalid_start_end() -> None:
    with pytest.raises(
        pl.ComputeError, match="`end` must be equal to or greater than `start`"
    ):
        pl.time_range(time(12), time(11), eager=True)


@pytest.mark.parametrize("interval", [timedelta(0), timedelta(minutes=-10)])
def test_time_range_invalid_step(interval: timedelta) -> None:
    with pytest.raises(pl.ComputeError, match="`interval` must be positive"):
        pl.time_range(time(11), time(12), interval=interval, eager=True)

from __future__ import annotations

from datetime import time

import polars as pl
from polars.testing import assert_series_equal


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


def test_time_range_only_first_entry_used() -> None:
    start = pl.Series([time(9, 0), time(10, 0)])
    end = pl.Series([time(12, 0), time(11, 0)])

    result = pl.time_range(start, end, eager=True)

    expected = pl.Series("time", [time(9, 0), time(10, 0), time(11, 0), time(12, 0)])
    assert_series_equal(result, expected)

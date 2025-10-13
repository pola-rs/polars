from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_unique_counts() -> None:
    s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
    expected = pl.Series("id", [1, 2, 3], dtype=pl.UInt32)
    assert_series_equal(s.unique_counts(), expected)


def test_unique_counts_on_dates() -> None:
    assert pl.DataFrame(
        {
            "dt_ns": pl.datetime_range(
                datetime(2020, 1, 1), datetime(2020, 3, 1), "1mo", eager=True
            ),
        }
    ).with_columns(
        pl.col("dt_ns").dt.cast_time_unit("us").alias("dt_us"),
        pl.col("dt_ns").dt.cast_time_unit("ms").alias("dt_ms"),
        pl.col("dt_ns").cast(pl.Date).alias("date"),
    ).select(pl.all().unique_counts().sum()).to_dict(as_series=False) == {
        "dt_ns": [3],
        "dt_us": [3],
        "dt_ms": [3],
        "date": [3],
    }


def test_unique_counts_null() -> None:
    s = pl.Series([])
    expected = pl.Series([], dtype=pl.UInt32)
    assert_series_equal(s.unique_counts(), expected)

    s = pl.Series([None])
    expected = pl.Series([1], dtype=pl.UInt32)
    assert_series_equal(s.unique_counts(), expected)

    s = pl.Series([None, None, None])
    expected = pl.Series([3], dtype=pl.UInt32)
    assert_series_equal(s.unique_counts(), expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ([], []),
        ([None, None, None], [3]),
        ([True, True, True], [3]),
        ([False, False, False], [3]),
        ([True, False, False, True, True], [3, 2]),
        ([False, True, False, True, True], [2, 3]),
        ([True, None, True, None, None], [2, 3]),
        ([None, True, None, True, True], [2, 3]),
        ([False, None, False, None, None], [2, 3]),
        ([None, False, None, False, False], [2, 3]),
        ([True, False, None, False, None, None], [1, 2, 3]),
        ([True, None, False, False, None, False], [1, 2, 3]),
        ([False, True, True, None, None, None], [1, 2, 3]),
        ([False, None, True, True, None, True], [1, 2, 3]),
        ([None, False, True, False, True, True], [1, 2, 3]),
        ([None, True, True, False, False, False], [1, 2, 3]),
    ],
)
def test_unique_counts_bool(input: list[bool], expected: list[int]) -> None:
    assert_series_equal(
        pl.Series("bool", input, dtype=pl.Boolean).unique_counts(),
        pl.Series("bool", expected, dtype=pl.UInt32),
    )

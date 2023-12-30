from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    "input",
    [
        [[1, 2], [3, 4, 5]],
        [1, 2, 3],
    ],
)
def test_lit_list_input(input: list[Any]) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns(pl.lit(input))
    expected = pl.DataFrame({"a": [1, 2], "literal": [input, input]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input",
    [
        ([1, 2], [3, 4, 5]),
        (1, 2, 3),
    ],
)
def test_lit_tuple_input(input: tuple[Any, ...]) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    result = df.with_columns(pl.lit(input))

    expected = pl.DataFrame({"a": [1, 2], "literal": [list(input), list(input)]})
    assert_frame_equal(result, expected)


def test_lit_numpy_array_input() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    input = np.array([3, 4])

    result = df.with_columns(pl.lit(input, dtype=pl.Int64))

    expected = pl.DataFrame({"a": [1, 2], "literal": [3, 4]})
    assert_frame_equal(result, expected)


def test_lit_ambiguous_datetimes_11379() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                datetime(2020, 10, 25),
                datetime(2020, 10, 25, 2),
                "1h",
                time_zone="Europe/London",
                eager=True,
            )
        }
    )
    for i in range(len(df)):
        result = df.filter(pl.col("ts") >= df["ts"][i])
        expected = df[i:]
        assert_frame_equal(result, expected)


def test_list_datetime_11571() -> None:
    sec_np_ns = np.timedelta64(1_000_000_000, "ns")
    sec_np_us = np.timedelta64(1_000_000, "us")
    assert pl.select(pl.lit(sec_np_ns))[0, 0] == timedelta(seconds=1)
    assert pl.select(pl.lit(sec_np_us))[0, 0] == timedelta(seconds=1)


@pytest.mark.parametrize(
    ("input", "dtype"),
    [
        pytest.param(-(2**31), pl.Int32, id="i32 min"),
        pytest.param(-(2**31) - 1, pl.Int64, id="below i32 min"),
        pytest.param(2**31 - 1, pl.Int32, id="i32 max"),
        pytest.param(2**31, pl.Int64, id="above i32 max"),
        pytest.param(2**63 - 1, pl.Int64, id="i64 max"),
        pytest.param(2**63, pl.UInt64, id="above i64 max"),
    ],
)
def test_lit_int_return_type(input: int, dtype: pl.PolarsDataType) -> None:
    assert pl.select(pl.lit(input)).to_series().dtype == dtype

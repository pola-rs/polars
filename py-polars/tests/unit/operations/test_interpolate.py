from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.dependencies import _ZONEINFO_AVAILABLE
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType, PolarsTemporalType

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (pl.Int8, pl.Float64),
        (pl.Int16, pl.Float64),
        (pl.Int32, pl.Float64),
        (pl.Int64, pl.Float64),
        (pl.UInt8, pl.Float64),
        (pl.UInt16, pl.Float64),
        (pl.UInt32, pl.Float64),
        (pl.UInt64, pl.Float64),
        (pl.Float32, pl.Float32),
        (pl.Float64, pl.Float64),
    ],
)
def test_interpolate_linear(
    input_dtype: PolarsDataType, output_dtype: PolarsDataType
) -> None:
    df = pl.LazyFrame({"a": [1, None, 2, None, 3]}, schema={"a": input_dtype})
    result = df.with_columns(pl.all().interpolate(method="linear"))
    assert result.collect_schema()["a"] == output_dtype
    expected = pl.DataFrame(
        {"a": [1.0, 1.5, 2.0, 2.5, 3.0]}, schema={"a": output_dtype}
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    ("input", "input_dtype", "output"),
    [
        (
            [date(2020, 1, 1), None, date(2020, 1, 2)],
            pl.Date,
            [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2)],
        ),
        (
            [datetime(2020, 1, 1), None, datetime(2020, 1, 2)],
            pl.Datetime("ms"),
            [datetime(2020, 1, 1), datetime(2020, 1, 1, 12), datetime(2020, 1, 2)],
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu")),
                None,
                datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu")),
            ],
            pl.Datetime("us", "Asia/Kathmandu"),
            [
                datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu")),
                datetime(2020, 1, 1, 12, tzinfo=ZoneInfo("Asia/Kathmandu")),
                datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu")),
            ],
        ),
        ([time(1), None, time(2)], pl.Time, [time(1), time(1, 30), time(2)]),
        (
            [timedelta(1), None, timedelta(2)],
            pl.Duration("ms"),
            [timedelta(1), timedelta(1, hours=12), timedelta(2)],
        ),
    ],
)
def test_interpolate_temporal_linear(
    input: list[Any], input_dtype: PolarsTemporalType, output: list[Any]
) -> None:
    df = pl.LazyFrame({"a": input}, schema={"a": input_dtype})
    result = df.with_columns(pl.all().interpolate(method="linear"))
    assert result.collect_schema()["a"] == input_dtype
    expected = pl.DataFrame({"a": output}, schema={"a": input_dtype})
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    "input_dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ],
)
def test_interpolate_nearest(input_dtype: PolarsDataType) -> None:
    df = pl.LazyFrame({"a": [1, None, 2, None, 3]}, schema={"a": input_dtype})
    result = df.with_columns(pl.all().interpolate(method="nearest"))
    assert result.collect_schema()["a"] == input_dtype
    expected = pl.DataFrame({"a": [1, 2, 2, 3, 3]}, schema={"a": input_dtype})
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    ("input", "input_dtype", "output"),
    [
        (
            [date(2020, 1, 1), None, date(2020, 1, 2)],
            pl.Date,
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
        ),
        (
            [datetime(2020, 1, 1), None, datetime(2020, 1, 2)],
            pl.Datetime("ms"),
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 2)],
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu")),
                None,
                datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu")),
            ],
            pl.Datetime("us", "Asia/Kathmandu"),
            [
                datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu")),
                datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu")),
                datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu")),
            ],
        ),
        ([time(1), None, time(2)], pl.Time, [time(1), time(2), time(2)]),
        (
            [timedelta(1), None, timedelta(2)],
            pl.Duration("ms"),
            [timedelta(1), timedelta(2), timedelta(2)],
        ),
    ],
)
def test_interpolate_temporal_nearest(
    input: list[Any], input_dtype: PolarsTemporalType, output: list[Any]
) -> None:
    df = pl.LazyFrame({"a": input}, schema={"a": input_dtype})
    result = df.with_columns(pl.all().interpolate(method="nearest"))
    assert result.collect_schema()["a"] == input_dtype
    expected = pl.DataFrame({"a": output}, schema={"a": input_dtype})
    assert_frame_equal(result.collect(), expected)

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.dependencies import _ZONEINFO_AVAILABLE
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsIntegerType, TimeUnit

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo


@pytest.mark.parametrize("sort", [True, False])
def test_ewma_by_date(sort: bool) -> None:
    df = pl.LazyFrame(
        {
            "values": [3.0, 1.0, 2.0, None, 4.0],
            "times": [
                None,
                date(2020, 1, 4),
                date(2020, 1, 11),
                date(2020, 1, 16),
                date(2020, 1, 18),
            ],
        }
    )
    if sort:
        df = df.sort("times")
    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life=timedelta(days=2)),
    )
    expected = pl.DataFrame(
        {"values": [None, 1.0, 1.9116116523516815, None, 3.815410804703363]}
    )
    assert_frame_equal(result.collect(), expected)
    assert result.collect_schema()["values"] == pl.Float64
    assert result.collect().schema["values"] == pl.Float64


def test_ewma_by_date_constant() -> None:
    df = pl.DataFrame(
        {
            "values": [1, 1, 1],
            "times": [
                date(2020, 1, 4),
                date(2020, 1, 11),
                date(2020, 1, 16),
            ],
        }
    )
    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life=timedelta(days=2)),
    )
    expected = pl.DataFrame({"values": [1.0, 1, 1]})
    assert_frame_equal(result, expected)


def test_ewma_f32() -> None:
    df = pl.LazyFrame(
        {
            "values": [3.0, 1.0, 2.0, None, 4.0],
            "times": [
                None,
                date(2020, 1, 4),
                date(2020, 1, 11),
                date(2020, 1, 16),
                date(2020, 1, 18),
            ],
        },
        schema_overrides={"values": pl.Float32},
    )
    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life=timedelta(days=2)),
    )
    expected = pl.DataFrame(
        {"values": [None, 1.0, 1.9116116523516815, None, 3.815410804703363]},
        schema_overrides={"values": pl.Float32},
    )
    assert_frame_equal(result.collect(), expected)
    assert result.collect_schema()["values"] == pl.Float32
    assert result.collect().schema["values"] == pl.Float32


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", [None, "UTC"])
def test_ewma_by_datetime(time_unit: TimeUnit, time_zone: str | None) -> None:
    df = pl.DataFrame(
        {
            "values": [3.0, 1.0, 2.0, None, 4.0],
            "times": [
                None,
                datetime(2020, 1, 4),
                datetime(2020, 1, 11),
                datetime(2020, 1, 16),
                datetime(2020, 1, 18),
            ],
        },
        schema_overrides={"times": pl.Datetime(time_unit, time_zone)},
    )
    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life=timedelta(days=2)),
    )
    expected = pl.DataFrame(
        {"values": [None, 1.0, 1.9116116523516815, None, 3.815410804703363]}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_ewma_by_datetime_tz_aware(time_unit: TimeUnit) -> None:
    tzinfo = ZoneInfo("Asia/Kathmandu")
    df = pl.DataFrame(
        {
            "values": [3.0, 1.0, 2.0, None, 4.0],
            "times": [
                None,
                datetime(2020, 1, 4, tzinfo=tzinfo),
                datetime(2020, 1, 11, tzinfo=tzinfo),
                datetime(2020, 1, 16, tzinfo=tzinfo),
                datetime(2020, 1, 18, tzinfo=tzinfo),
            ],
        },
        schema_overrides={"times": pl.Datetime(time_unit, "Asia/Kathmandu")},
    )
    msg = "expected `half_life` to be a constant duration"
    with pytest.raises(InvalidOperationError, match=msg):
        df.select(
            pl.col("values").ewm_mean_by("times", half_life="2d"),
        )

    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life="48h0ns"),
    )
    expected = pl.DataFrame(
        {"values": [None, 1.0, 1.9116116523516815, None, 3.815410804703363]}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("data_type", [pl.Int64, pl.Int32, pl.UInt64, pl.UInt32])
def test_ewma_by_index(data_type: PolarsIntegerType) -> None:
    df = pl.LazyFrame(
        {
            "values": [3.0, 1.0, 2.0, None, 4.0],
            "times": [
                None,
                4,
                11,
                16,
                18,
            ],
        },
        schema_overrides={"times": data_type},
    )
    result = df.select(
        pl.col("values").ewm_mean_by("times", half_life="2i"),
    )
    expected = pl.DataFrame(
        {"values": [None, 1.0, 1.9116116523516815, None, 3.815410804703363]}
    )
    assert_frame_equal(result.collect(), expected)
    assert result.collect_schema()["values"] == pl.Float64
    assert result.collect().schema["values"] == pl.Float64


def test_ewma_by_empty() -> None:
    df = pl.DataFrame({"values": []}, schema_overrides={"values": pl.Float64})
    result = df.with_row_index().select(
        pl.col("values").ewm_mean_by("index", half_life="2i"),
    )
    expected = pl.DataFrame({"values": []}, schema_overrides={"values": pl.Float64})
    assert_frame_equal(result, expected)


def test_ewma_by_if_unsorted() -> None:
    df = pl.DataFrame({"values": [3.0, 2.0], "by": [3, 1]})
    result = df.with_columns(
        pl.col("values").ewm_mean_by("by", half_life="2i"),
    )
    expected = pl.DataFrame({"values": [2.5, 2.0], "by": [3, 1]})
    assert_frame_equal(result, expected)

    result = df.with_columns(
        pl.col("values").ewm_mean_by("by", half_life="2i"),
    )
    assert_frame_equal(result, expected)

    result = df.sort("by").with_columns(
        pl.col("values").ewm_mean_by("by", half_life="2i"),
    )
    assert_frame_equal(result, expected.sort("by"))


def test_ewma_by_invalid() -> None:
    df = pl.DataFrame({"values": [1, 2]})
    with pytest.raises(InvalidOperationError, match="half_life cannot be negative"):
        df.with_row_index().select(
            pl.col("values").ewm_mean_by("index", half_life="-2i"),
        )
    df = pl.DataFrame({"values": [[1, 2], [3, 4]]})
    with pytest.raises(
        InvalidOperationError, match=r"expected series to be Float64, Float32, .*"
    ):
        df.with_row_index().select(
            pl.col("values").ewm_mean_by("index", half_life="2i"),
        )


def test_ewma_by_warn_two_chunks() -> None:
    df = pl.DataFrame({"values": [3.0, 2.0], "by": [3, 1]})
    df = pl.concat([df, df], rechunk=False)

    result = df.with_columns(
        pl.col("values").ewm_mean_by("by", half_life="2i"),
    )
    expected = pl.DataFrame({"values": [2.5, 2.0, 2.5, 2], "by": [3, 1, 3, 1]})
    assert_frame_equal(result, expected)
    result = df.sort("by").with_columns(
        pl.col("values").ewm_mean_by("by", half_life="2i"),
    )
    assert_frame_equal(result, expected.sort("by"))


def test_ewma_by_multiple_chunks() -> None:
    # times contains null
    times = pl.Series([1, 2]).append(pl.Series([None], dtype=pl.Int64))
    values = pl.Series([1, 2]).append(pl.Series([3]))
    result = values.ewm_mean_by(times, half_life="2i")
    expected = pl.Series([1.0, 1.292893, None])
    assert_series_equal(result, expected)

    # values contains null
    times = pl.Series([1, 2]).append(pl.Series([3]))
    values = pl.Series([1, 2]).append(pl.Series([None], dtype=pl.Int64))
    result = values.ewm_mean_by(times, half_life="2i")
    assert_series_equal(result, expected)

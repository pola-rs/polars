from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval, PolarsDataType, TimeUnit


@pytest.mark.parametrize("low", ["start", pl.col("start")])
@pytest.mark.parametrize("high", ["stop", pl.col("stop")])
def test_datetime_ranges_lazy_with_expressions(
    low: str | pl.Expr, high: str | pl.Expr
) -> None:
    df = pl.DataFrame(
        {
            "start": [datetime(2000, 1, 1), datetime(2022, 6, 1)],
            "stop": [datetime(2000, 1, 2), datetime(2022, 6, 2)],
        }
    )

    result_df = df.with_columns(
        pl.datetime_ranges(start=low, end=high, interval="1d").alias("dts")
    )

    assert result_df.to_dict(as_series=False) == {
        "start": [datetime(2000, 1, 1, 0, 0), datetime(2022, 6, 1, 0, 0)],
        "stop": [datetime(2000, 1, 2, 0, 0), datetime(2022, 6, 2, 0, 0)],
        "dts": [
            [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 2, 0, 0)],
            [datetime(2022, 6, 1, 0, 0), datetime(2022, 6, 2, 0, 0)],
        ],
    }


@pytest.mark.parametrize(
    ("values_time_zone", "input_time_zone", "output_time_zone"),
    [
        ("Asia/Kathmandu", "Asia/Kathmandu", "Asia/Kathmandu"),
        ("Asia/Kathmandu", None, "Asia/Kathmandu"),
        (None, "Asia/Kathmandu", "Asia/Kathmandu"),
        (None, None, None),
    ],
)
@pytest.mark.parametrize(
    ("values_time_unit", "input_time_unit", "output_time_unit"),
    [
        ("ms", None, "ms"),
        ("us", None, "us"),
        ("ns", None, "ns"),
        ("ms", "ms", "ms"),
        ("us", "ms", "ms"),
        ("ns", "ms", "ms"),
        ("ms", "us", "us"),
        ("us", "us", "us"),
        ("ns", "us", "us"),
        ("ms", "ns", "ns"),
        ("us", "ns", "ns"),
        ("ns", "ns", "ns"),
    ],
)
def test_datetime_ranges_schema(
    values_time_zone: str | None,
    input_time_zone: str | None,
    output_time_zone: str | None,
    values_time_unit: TimeUnit,
    input_time_unit: TimeUnit | None,
    output_time_unit: TimeUnit,
) -> None:
    df = (
        pl.DataFrame({"start": [datetime(2020, 1, 1)], "end": [datetime(2020, 1, 2)]})
        .with_columns(
            pl.col("*")
            .dt.replace_time_zone(values_time_zone)
            .dt.cast_time_unit(values_time_unit)
        )
        .lazy()
    )
    result = df.with_columns(
        datetime_range=pl.datetime_ranges(
            pl.col("start"),
            pl.col("end"),
            time_zone=input_time_zone,
            time_unit=input_time_unit,
        )
    )
    expected_schema = {
        "start": pl.Datetime(time_unit=values_time_unit, time_zone=values_time_zone),
        "end": pl.Datetime(time_unit=values_time_unit, time_zone=values_time_zone),
        "datetime_range": pl.List(
            pl.Datetime(time_unit=output_time_unit, time_zone=output_time_zone)
        ),
    }
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema

    expected = pl.DataFrame(
        {
            "start": [datetime(2020, 1, 1)],
            "end": [datetime(2020, 1, 2)],
            "datetime_range": [[datetime(2020, 1, 1), datetime(2020, 1, 2)]],
        }
    ).with_columns(
        pl.col("start")
        .dt.replace_time_zone(values_time_zone)
        .dt.cast_time_unit(values_time_unit),
        pl.col("end")
        .dt.replace_time_zone(values_time_zone)
        .dt.cast_time_unit(values_time_unit),
        pl.col("datetime_range")
        .explode()
        .dt.replace_time_zone(output_time_zone)
        .dt.cast_time_unit(output_time_unit)
        .implode(),
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    (
        "input_time_unit",
        "input_time_zone",
        "output_dtype",
        "interval",
        "expected_datetime_range",
    ),
    [
        (None, None, pl.Datetime("us"), "1s1d", ["2020-01-01", "2020-01-02 00:00:01"]),
        (None, None, pl.Datetime("us"), "1d1s", ["2020-01-01", "2020-01-02 00:00:01"]),
        (
            None,
            None,
            pl.Datetime("ns"),
            "1d1ns",
            ["2020-01-01", "2020-01-02 00:00:00.000000001"],
        ),
        ("ms", None, pl.Datetime("ms"), "1s1d", ["2020-01-01", "2020-01-02 00:00:01"]),
        ("ms", None, pl.Datetime("ms"), "1d1s", ["2020-01-01", "2020-01-02 00:00:01"]),
        (
            None,
            "Asia/Kathmandu",
            pl.Datetime("us", "Asia/Kathmandu"),
            "1s1d",
            ["2020-01-01", "2020-01-02 00:00:01"],
        ),
        (
            None,
            "Asia/Kathmandu",
            pl.Datetime("us", "Asia/Kathmandu"),
            "1d1s",
            ["2020-01-01", "2020-01-02 00:00:01"],
        ),
        (
            None,
            "Asia/Kathmandu",
            pl.Datetime("ns", "Asia/Kathmandu"),
            "1d1ns",
            ["2020-01-01", "2020-01-02 00:00:00.000000001"],
        ),
        (
            "ms",
            "Asia/Kathmandu",
            pl.Datetime("ms", "Asia/Kathmandu"),
            "1s1d",
            ["2020-01-01", "2020-01-02 00:00:01"],
        ),
        (
            "ms",
            "Asia/Kathmandu",
            pl.Datetime("ms", "Asia/Kathmandu"),
            "1d1s",
            ["2020-01-01", "2020-01-02 00:00:01"],
        ),
    ],
)
def test_datetime_ranges_schema_upcasts_to_datetime(
    input_time_unit: TimeUnit | None,
    input_time_zone: str | None,
    output_dtype: PolarsDataType,
    interval: str,
    expected_datetime_range: list[str],
) -> None:
    df = pl.DataFrame({"start": [date(2020, 1, 1)], "end": [date(2020, 1, 3)]}).lazy()
    result = df.with_columns(
        datetime_range=pl.datetime_ranges(
            start=pl.col("start"),
            end=pl.col("end"),
            interval=interval,
            time_unit=input_time_unit,
            time_zone=input_time_zone,
        )
    )
    expected_schema = {
        "start": pl.Date,
        "end": pl.Date,
        "datetime_range": pl.List(output_dtype),
    }
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema

    expected = pl.DataFrame(
        {
            "start": [date(2020, 1, 1)],
            "end": [date(2020, 1, 3)],
            "datetime_range": pl.Series(expected_datetime_range)
            .str.to_datetime(time_unit="ns")
            .implode(),
        }
    ).with_columns(
        pl.col("datetime_range")
        .explode()
        .dt.cast_time_unit(output_dtype.time_unit)  # type: ignore[union-attr]
        .dt.replace_time_zone(output_dtype.time_zone)  # type: ignore[union-attr]
        .implode(),
    )
    assert_frame_equal(result.collect(), expected)

    # check datetime_range too
    result_single = pl.datetime_range(
        date(2020, 1, 1),
        date(2020, 1, 3),
        interval=interval,
        time_unit=input_time_unit,
        time_zone=input_time_zone,
        eager=True,
    ).alias("datetime")
    assert_series_equal(
        result_single, expected["datetime_range"].explode().rename("datetime")
    )


def test_datetime_ranges_no_alias_schema_9037() -> None:
    df = pl.DataFrame(
        {"start": [datetime(2020, 1, 1)], "end": [datetime(2020, 1, 2)]}
    ).lazy()
    result = df.with_columns(pl.datetime_ranges(pl.col("start"), pl.col("end")))
    expected_schema = {
        "start": pl.List(pl.Datetime(time_unit="us", time_zone=None)),
        "end": pl.Datetime(time_unit="us", time_zone=None),
    }
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema


def test_datetime_ranges_broadcasting() -> None:
    df = pl.DataFrame(
        {
            "datetimes": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 3),
            ]
        }
    )
    result = df.select(
        pl.datetime_ranges(start="datetimes", end=datetime(2021, 1, 3)).alias("end"),
        pl.datetime_ranges(start=datetime(2021, 1, 1), end="datetimes").alias("start"),
    )
    expected = pl.DataFrame(
        {
            "end": [
                [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
                [datetime(2021, 1, 2), datetime(2021, 1, 3)],
                [datetime(2021, 1, 3)],
            ],
            "start": [
                [datetime(2021, 1, 1)],
                [datetime(2021, 1, 1), datetime(2021, 1, 2)],
                [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
            ],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 10),
                ],
                [
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 17),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 8, 12),
                ],
                [
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                ],
            ],
        ),
        (
            "right",
            [
                [
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 10),
                ],
                [
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 17),
                ],
            ],
        ),
        (
            "none",
            [
                [
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 8, 12),
                ],
                [
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                ],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("ms", time_zone="Asia/Kathmandu"),
        pl.Datetime("us", time_zone="Asia/Kathmandu"),
        pl.Datetime("ns", time_zone="Asia/Kathmandu"),
    ],
)
def test_datetime_ranges_start_end_interval_forwards(
    closed: ClosedInterval,
    expected: list[list[datetime]],
    dtype: PolarsDataType,
) -> None:
    tu = dtype.time_unit if dtype == pl.Datetime else None  # type: ignore[union-attr]
    tz = dtype.time_zone if dtype == pl.Datetime else None  # type: ignore[union-attr]
    if tz is not None:
        time_zone = ZoneInfo(tz)
        expected = [[e.replace(tzinfo=time_zone) for e in x] for x in expected]
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 8)],
            "end": [date(2025, 1, 10), date(2025, 1, 17)],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            end="end",
            interval="1d12h",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


def test_datetime_ranges_lit_combinations_start_end_interval() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
        }
    )
    result = df.select(
        start_lit=pl.datetime_ranges(start=date(2025, 1, 1), end="end", interval="1d"),
        end_lit=pl.datetime_ranges(start="start", end=date(2025, 1, 3), interval="1d"),
    )
    dt = [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
    s = pl.Series([dt, dt], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "start_lit": s,
            "end_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_null_lit_combinations_start_end_interval() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
        }
    )
    lit_dt = pl.lit(None, dtype=pl.Date)
    result = df.select(
        start_lit=pl.datetime_ranges(start=lit_dt, end="end", interval="1d"),
        end_lit=pl.datetime_ranges(start="start", end=lit_dt, interval="1d"),
        all_lit=pl.datetime_ranges(start=lit_dt, end=lit_dt, interval="1d"),
    )
    s = pl.Series([None, None], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame({"start_lit": s, "end_lit": s, "all_lit": s})
    assert_frame_equal(result, expected)

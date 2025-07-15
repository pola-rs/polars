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


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                ],
                [
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
                ],
            ],
        ),
        (
            "right",
            [
                [
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                ],
            ],
        ),
        (
            "none",
            [
                [
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                ],
                [
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
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
def test_datetime_ranges_start_end_interval_backwards(
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
            "start": [date(2025, 1, 10), date(2025, 1, 17)],
            "end": [date(2025, 1, 1), date(2025, 1, 8)],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            end="end",
            interval="-1d12h",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 3, 6),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7, 18),
                    datetime(2025, 1, 10),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 10, 4, 48),
                    datetime(2025, 1, 10, 9, 36),
                    datetime(2025, 1, 10, 14, 24),
                    datetime(2025, 1, 10, 19, 12),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "left",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 2, 19, 12),
                    datetime(2025, 1, 4, 14, 24),
                    datetime(2025, 1, 6, 9, 36),
                    datetime(2025, 1, 8, 4, 48),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 10, 4),
                    datetime(2025, 1, 10, 8),
                    datetime(2025, 1, 10, 12),
                    datetime(2025, 1, 10, 16),
                    datetime(2025, 1, 10, 20),
                ],
            ],
        ),
        (
            "right",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 2, 19, 12),
                    datetime(2025, 1, 4, 14, 24),
                    datetime(2025, 1, 6, 9, 36),
                    datetime(2025, 1, 8, 4, 48),
                    datetime(2025, 1, 10),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10, 4),
                    datetime(2025, 1, 10, 8),
                    datetime(2025, 1, 10, 12),
                    datetime(2025, 1, 10, 16),
                    datetime(2025, 1, 10, 20),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "none",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 8, 12),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10, 3, 25, 42, 857142),
                    datetime(2025, 1, 10, 6, 51, 25, 714285),
                    datetime(2025, 1, 10, 10, 17, 8, 571428),
                    datetime(2025, 1, 10, 13, 42, 51, 428571),
                    datetime(2025, 1, 10, 17, 8, 34, 285714),
                    datetime(2025, 1, 10, 20, 34, 17, 142857),
                ],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        # we disable ns for this test, as it's hard to provide the expected value
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        # pl.Datetime("ns"),
        pl.Datetime("ms", time_zone="Asia/Kathmandu"),
        pl.Datetime("us", time_zone="Asia/Kathmandu"),
        # pl.Datetime("ns", time_zone="Asia/Kathmandu"),
    ],
)
def test_date_ranges_start_end_samples_forwards(
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
            "start": [date(2025, 1, 1), date(2025, 1, 10)],
            "end": [date(2025, 1, 10), date(2025, 1, 11)],
            "samples": [5, 6],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            end="end",
            num_samples="samples",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 7, 18),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 3, 6),
                    datetime(2025, 1, 1),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 10, 19, 12),
                    datetime(2025, 1, 10, 14, 24),
                    datetime(2025, 1, 10, 9, 36),
                    datetime(2025, 1, 10, 4, 48),
                    datetime(2025, 1, 10),
                ],
            ],
        ),
        (
            "left",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 10),
                    datetime(2025, 1, 8, 4, 48),
                    datetime(2025, 1, 6, 9, 36),
                    datetime(2025, 1, 4, 14, 24),
                    datetime(2025, 1, 2, 19, 12),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 10, 20),
                    datetime(2025, 1, 10, 16),
                    datetime(2025, 1, 10, 12),
                    datetime(2025, 1, 10, 8),
                    datetime(2025, 1, 10, 4),
                ],
            ],
        ),
        (
            "right",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 8, 4, 48),
                    datetime(2025, 1, 6, 9, 36),
                    datetime(2025, 1, 4, 14, 24),
                    datetime(2025, 1, 2, 19, 12),
                    datetime(2025, 1, 1),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10, 20),
                    datetime(2025, 1, 10, 16),
                    datetime(2025, 1, 10, 12),
                    datetime(2025, 1, 10, 8),
                    datetime(2025, 1, 10, 4),
                    datetime(2025, 1, 10),
                ],
            ],
        ),
        (
            "none",
            [
                [  # (1, 10, 5)
                    datetime(2025, 1, 8, 12),
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                ],
                [  # (10, 11, 6)
                    datetime(2025, 1, 10, 20, 34, 17, 142857),
                    datetime(2025, 1, 10, 17, 8, 34, 285714),
                    datetime(2025, 1, 10, 13, 42, 51, 428571),
                    datetime(2025, 1, 10, 10, 17, 8, 571428),
                    datetime(2025, 1, 10, 6, 51, 25, 714285),
                    datetime(2025, 1, 10, 3, 25, 42, 857143),
                ],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        # we disable ns for this test, as it's hard to provide the expected value
        pl.Date,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        # pl.Datetime("ns"),
        pl.Datetime("ms", time_zone="Asia/Kathmandu"),
        pl.Datetime("us", time_zone="Asia/Kathmandu"),
        # pl.Datetime("ns", time_zone="Asia/Kathmandu"),
    ],
)
def test_date_ranges_start_end_samples_backwards(
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
            "start": [date(2025, 1, 10), date(2025, 1, 11)],
            "end": [date(2025, 1, 1), date(2025, 1, 10)],
            "samples": [5, 6],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            end="end",
            num_samples="samples",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


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
                ],
                [
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
                ],
                [
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 17),
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
                ],
                [
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 18, 12),
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
                ],
                [
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 18, 12),
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
def test_datetime_ranges_start_interval_samples_forwards(
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
            "start": [date(2025, 1, 1), date(2025, 1, 11)],
            "samples": [4, 5],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            num_samples="samples",
            interval="1d12h",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    datetime(2025, 1, 1),
                    datetime(2024, 12, 30, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 27, 12),
                ],
                [
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 5),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    datetime(2025, 1, 1),
                    datetime(2024, 12, 30, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 27, 12),
                ],
                [
                    datetime(2025, 1, 11),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 5),
                ],
            ],
        ),
        (
            "right",
            [
                [
                    datetime(2024, 12, 30, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 26),
                ],
                [
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 3, 12),
                ],
            ],
        ),
        (
            "none",
            [
                [
                    datetime(2024, 12, 30, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 26),
                ],
                [
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 3, 12),
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
def test_datetime_ranges_start_interval_samples_backwards(
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
            "start": [date(2025, 1, 1), date(2025, 1, 11)],
            "samples": [4, 5],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            start="start",
            num_samples="samples",
            interval="-1d12h",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 30, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    datetime(2024, 12, 26),
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 30, 12),
                ],
                [
                    datetime(2025, 1, 3, 12),
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
                ],
            ],
        ),
        (
            "right",
            [
                [
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 30, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "none",
            [
                [
                    datetime(2024, 12, 26),
                    datetime(2024, 12, 27, 12),
                    datetime(2024, 12, 29),
                    datetime(2024, 12, 30, 12),
                ],
                [
                    datetime(2025, 1, 3, 12),
                    datetime(2025, 1, 5),
                    datetime(2025, 1, 6, 12),
                    datetime(2025, 1, 8),
                    datetime(2025, 1, 9, 12),
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
def test_datetime_ranges_end_interval_samples_forwards(
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
            "end": [date(2025, 1, 1), date(2025, 1, 11)],
            "samples": [4, 5],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            end="end",
            num_samples="samples",
            interval="1d12h",
            closed=closed,
            time_unit=tu,
            time_zone=tz,
        )
    )
    dt_out = pl.List(pl.Datetime("us")) if dtype == pl.Date else pl.List(dtype)
    s_expected = pl.Series("dates", expected, dtype=dt_out)
    assert_frame_equal(result, s_expected.to_frame())


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                ],
                [
                    datetime(2025, 1, 18, 12),
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                ],
            ],
        ),
        (
            "right",
            [
                [
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                    datetime(2025, 1, 1),
                ],
                [
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
                    datetime(2025, 1, 11),
                ],
            ],
        ),
        (
            "none",
            [
                [
                    datetime(2025, 1, 7),
                    datetime(2025, 1, 5, 12),
                    datetime(2025, 1, 4),
                    datetime(2025, 1, 2, 12),
                ],
                [
                    datetime(2025, 1, 18, 12),
                    datetime(2025, 1, 17),
                    datetime(2025, 1, 15, 12),
                    datetime(2025, 1, 14),
                    datetime(2025, 1, 12, 12),
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
def test_datetime_ranges_end_interval_samples_backwards(
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
            "end": [date(2025, 1, 1), date(2025, 1, 11)],
            "samples": [4, 5],
        }
    )
    result = df.select(
        dates=pl.datetime_ranges(
            end="end",
            num_samples="samples",
            interval="-1d12h",
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


def test_datetime_ranges_lit_combinations_start_end_samples() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
            "samples": [3, 3],
        }
    )
    start = date(2025, 1, 1)
    end = date(2025, 1, 3)
    result = df.select(
        start_lit=pl.datetime_ranges(start=start, end="end", num_samples="samples"),
        end_lit=pl.datetime_ranges(start="start", end=end, num_samples="samples"),
        samples_lit=pl.datetime_ranges(start="start", end="end", num_samples=3),
        start_end_lit=pl.datetime_ranges(start=start, end=end, num_samples="samples"),
        start_samples_lit=pl.datetime_ranges(start=start, end="end", num_samples=3),
        end_samples_lit=pl.datetime_ranges(start="start", end=end, num_samples=3),
        all_lit=pl.datetime_ranges(start=start, end=end, num_samples=3),
    )
    dt = [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
    s = pl.Series([dt, dt], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "start_lit": s,
            "end_lit": s,
            "samples_lit": s,
            "start_end_lit": s,
            "start_samples_lit": s,
            "end_samples_lit": s,
            "all_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_null_lit_combinations_start_end_samples() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
            "samples": [3, 3],
        }
    )
    lit_dt = pl.lit(None, dtype=pl.Date)
    lit_n = pl.lit(None, dtype=pl.Int64)
    result = df.select(
        start_lit=pl.datetime_ranges(start=lit_dt, end="end", num_samples="samples"),
        end_lit=pl.datetime_ranges(start="start", end=lit_dt, num_samples="samples"),
        samples_lit=pl.datetime_ranges(start="start", end="end", num_samples=lit_n),
        start_end_lit=pl.datetime_ranges(
            start=lit_dt, end=lit_dt, num_samples="samples"
        ),
        start_samples_lit=pl.datetime_ranges(
            start=lit_dt, end="end", num_samples=lit_n
        ),
        end_samples_lit=pl.datetime_ranges(
            start="start", end=lit_dt, num_samples=lit_n
        ),
        all_lit=pl.datetime_ranges(start=lit_dt, end=lit_dt, num_samples=lit_n),
    )
    s = pl.Series([None, None], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "start_lit": s,
            "end_lit": s,
            "samples_lit": s,
            "start_end_lit": s,
            "start_samples_lit": s,
            "end_samples_lit": s,
            "all_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_lit_combinations_start_interval_samples() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "samples": [3, 3],
        }
    )
    result = df.select(
        start_lit=pl.datetime_ranges(
            start=date(2025, 1, 1), interval="1d", num_samples="samples"
        ),
        samples_lit=pl.datetime_ranges(start="start", interval="1d", num_samples=3),
    )
    dt = [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
    s = pl.Series([dt, dt], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "start_lit": s,
            "samples_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_null_lit_combinations_start_interval_samples() -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "samples": [3, 3],
        }
    )
    lit_dt = pl.lit(None, dtype=pl.Date)
    lit_n = pl.lit(None, dtype=pl.Int64)
    result = df.select(
        start_lit=pl.datetime_ranges(
            start=lit_dt, interval="1d", num_samples="samples"
        ),
        samples_lit=pl.datetime_ranges(start="start", interval="1d", num_samples=lit_n),
        all_lit=pl.datetime_ranges(start=lit_dt, interval="1d", num_samples=lit_n),
    )
    s = pl.Series([None, None], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "start_lit": s,
            "samples_lit": s,
            "all_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_lit_combinations_end_interval_samples() -> None:
    df = pl.DataFrame(
        {
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
            "samples": [3, 3],
        }
    )
    result = df.select(
        end_lit=pl.datetime_ranges(
            end=date(2025, 1, 3), num_samples="samples", interval="1d"
        ),
        samples_lit=pl.datetime_ranges(end="end", num_samples=3, interval="1d"),
    )
    dt = [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
    s = pl.Series([dt, dt], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "end_lit": s,
            "samples_lit": s,
        }
    )
    assert_frame_equal(result, expected)


def test_datetime_ranges_null_lit_combinations_end_interval_samples() -> None:
    df = pl.DataFrame(
        {
            "end": [date(2025, 1, 3), date(2025, 1, 3)],
            "samples": [3, 3],
        }
    )
    lit_dt = pl.lit(None, dtype=pl.Date)
    lit_n = pl.lit(None, dtype=pl.Int64)
    result = df.select(
        end_lit=pl.datetime_ranges(end=lit_dt, num_samples="samples", interval="1d"),
        samples_lit=pl.datetime_ranges(end="end", num_samples=lit_n, interval="1d"),
        all_lit=pl.datetime_ranges(end=lit_dt, num_samples=lit_n, interval="1d"),
    )
    s = pl.Series([None, None], dtype=pl.List(pl.Datetime("us")))
    expected = pl.DataFrame(
        {
            "end_lit": s,
            "samples_lit": s,
            "all_lit": s,
        }
    )
    assert_frame_equal(result, expected)

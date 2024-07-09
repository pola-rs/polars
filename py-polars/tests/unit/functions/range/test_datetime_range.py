from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.exceptions import ComputeError, PanicException, SchemaError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import ClosedInterval, PolarsDataType, TimeUnit
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


def test_datetime_range() -> None:
    result = pl.datetime_range(
        date(1985, 1, 1), date(2015, 7, 1), timedelta(days=1, hours=12), eager=True
    )
    assert len(result) == 7426
    assert result.dt[0] == datetime(1985, 1, 1)
    assert result.dt[1] == datetime(1985, 1, 2, 12, 0)
    assert result.dt[2] == datetime(1985, 1, 4, 0, 0)
    assert result.dt[-1] == datetime(2015, 6, 30, 12, 0)

    for time_unit in DTYPE_TEMPORAL_UNITS:
        rng = pl.datetime_range(
            datetime(2020, 1, 1),
            date(2020, 1, 2),
            "2h",
            time_unit=time_unit,
            eager=True,
        )
        assert rng.dtype.time_unit == time_unit  # type: ignore[attr-defined]
        assert rng.shape == (13,)
        assert rng.dt[0] == datetime(2020, 1, 1)
        assert rng.dt[-1] == datetime(2020, 1, 2)

    result = pl.datetime_range(date(2022, 1, 1), date(2022, 1, 2), "1h30m", eager=True)
    assert list(result) == [
        datetime(2022, 1, 1, 0, 0),
        datetime(2022, 1, 1, 1, 30),
        datetime(2022, 1, 1, 3, 0),
        datetime(2022, 1, 1, 4, 30),
        datetime(2022, 1, 1, 6, 0),
        datetime(2022, 1, 1, 7, 30),
        datetime(2022, 1, 1, 9, 0),
        datetime(2022, 1, 1, 10, 30),
        datetime(2022, 1, 1, 12, 0),
        datetime(2022, 1, 1, 13, 30),
        datetime(2022, 1, 1, 15, 0),
        datetime(2022, 1, 1, 16, 30),
        datetime(2022, 1, 1, 18, 0),
        datetime(2022, 1, 1, 19, 30),
        datetime(2022, 1, 1, 21, 0),
        datetime(2022, 1, 1, 22, 30),
        datetime(2022, 1, 2, 0, 0),
    ]

    result = pl.datetime_range(
        datetime(2022, 1, 1), datetime(2022, 1, 1, 0, 1), "987456321ns", eager=True
    )
    assert len(result) == 61
    assert result.dtype.time_unit == "ns"  # type: ignore[attr-defined]
    assert result.dt.second()[-1] == 59
    assert result.cast(pl.String)[-1] == "2022-01-01 00:00:59.247379260"


@pytest.mark.parametrize(
    ("time_unit", "expected_micros"),
    [
        ("ms", 986000),
        ("us", 986759),
        ("ns", 986759),
        (None, 986759),
    ],
)
def test_datetime_range_precision(
    time_unit: TimeUnit | None, expected_micros: int
) -> None:
    micros = 986759
    start = datetime(2000, 5, 30, 1, 53, 4, micros)
    stop = datetime(2000, 5, 31, 1, 53, 4, micros)
    result = pl.datetime_range(start, stop, time_unit=time_unit, eager=True)
    expected_start = start.replace(microsecond=expected_micros)
    expected_stop = stop.replace(microsecond=expected_micros)
    assert result[0] == expected_start
    assert result[1] == expected_stop


def test_datetime_range_invalid_time_unit() -> None:
    with pytest.raises(PanicException, match="'x' not supported"):
        pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="1X",
            eager=True,
        )


def test_datetime_range_lazy_time_zones() -> None:
    start = datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))
    stop = datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu"))
    result = (
        pl.DataFrame({"start": [start], "stop": [stop]})
        .with_columns(
            pl.datetime_range(
                start,
                stop,
                interval="678d",
                eager=False,
                time_zone="Pacific/Tarawa",
            )
        )
        .lazy()
    )
    expected = pl.DataFrame(
        {
            "start": [datetime(2019, 12, 31, 18, 15, tzinfo=ZoneInfo(key="UTC"))],
            "stop": [datetime(2020, 1, 1, 18, 15, tzinfo=ZoneInfo(key="UTC"))],
            "literal": [
                datetime(2020, 1, 1, 6, 15, tzinfo=ZoneInfo(key="Pacific/Tarawa"))
            ],
        }
    ).with_columns(pl.col("literal").dt.convert_time_zone("Pacific/Tarawa"))
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("low", ["start", pl.col("start")])
@pytest.mark.parametrize("high", ["stop", pl.col("stop")])
def test_datetime_range_lazy_with_expressions(
    low: str | pl.Expr, high: str | pl.Expr
) -> None:
    df = pl.DataFrame(
        {
            "start": [datetime(2000, 1, 1), datetime(2022, 6, 1)],
            "stop": [datetime(2000, 1, 2), datetime(2022, 6, 2)],
        }
    )

    result_df = df.with_columns(
        pl.datetime_ranges(low, high, interval="1d").alias("dts")
    )

    assert result_df.to_dict(as_series=False) == {
        "start": [datetime(2000, 1, 1, 0, 0), datetime(2022, 6, 1, 0, 0)],
        "stop": [datetime(2000, 1, 2, 0, 0), datetime(2022, 6, 2, 0, 0)],
        "dts": [
            [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 2, 0, 0)],
            [datetime(2022, 6, 1, 0, 0), datetime(2022, 6, 2, 0, 0)],
        ],
    }


def test_datetime_range_invalid_time_zone() -> None:
    with pytest.raises(ComputeError, match="unable to parse time zone: 'foo'"):
        pl.datetime_range(
            datetime(2001, 1, 1),
            datetime(2001, 1, 3),
            time_zone="foo",
            eager=True,
        )


def test_timezone_aware_datetime_range() -> None:
    low = datetime(2022, 10, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    high = datetime(2022, 11, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert pl.datetime_range(
        low, high, interval=timedelta(days=5), eager=True
    ).to_list() == [
        datetime(2022, 10, 17, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 10, 22, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 10, 27, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 11, 1, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 11, 6, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 11, 11, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2022, 11, 16, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    ]

    with pytest.raises(
        SchemaError,
        match="failed to determine supertype",
    ):
        pl.datetime_range(
            low,
            high.replace(tzinfo=None),
            interval=timedelta(days=5),
            time_zone="UTC",
            eager=True,
        )


def test_tzaware_datetime_range_crossing_dst_hourly() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 7, 2),
        "1h",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, fold=1, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 2, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_datetime_range_crossing_dst_daily() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 11),
        "2d",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 9, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 11, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_datetime_range_crossing_dst_weekly() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 20),
        "1w",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 14, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_tzaware_datetime_range_crossing_dst_monthly() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 12, 20),
        "1mo",
        time_zone="US/Central",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 12, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
    ]


def test_datetime_range_with_unsupported_datetimes() -> None:
    with pytest.raises(
        ComputeError,
        match=r"datetime '2021-11-07 01:00:00' is ambiguous in time zone 'US/Central'",
    ):
        pl.datetime_range(
            datetime(2021, 11, 7, 1),
            datetime(2021, 11, 7, 2),
            "1h",
            time_zone="US/Central",
            eager=True,
        )
    with pytest.raises(
        ComputeError,
        match=r"datetime '2021-03-28 02:30:00' is non-existent in time zone 'Europe/Vienna'",
    ):
        pl.datetime_range(
            datetime(2021, 3, 28, 2, 30),
            datetime(2021, 3, 28, 4),
            "1h",
            time_zone="Europe/Vienna",
            eager=True,
        )


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
def test_datetime_range_schema_upcasts_to_datetime(
    input_time_unit: TimeUnit | None,
    input_time_zone: str | None,
    output_dtype: PolarsDataType,
    interval: str,
    expected_datetime_range: list[str],
) -> None:
    df = pl.DataFrame({"start": [date(2020, 1, 1)], "end": [date(2020, 1, 3)]}).lazy()
    result = df.with_columns(
        datetime_range=pl.datetime_ranges(
            pl.col("start"),
            pl.col("end"),
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


@pytest.mark.parametrize("interval", [timedelta(0), timedelta(minutes=-10)])
def test_datetime_range_invalid_interval(interval: timedelta) -> None:
    with pytest.raises(ComputeError, match="`interval` must be positive"):
        pl.datetime_range(
            datetime(2000, 3, 20), datetime(2000, 3, 21), interval="-1h", eager=True
        )


@pytest.mark.parametrize(
    ("closed", "expected_values"),
    [
        ("right", [datetime(2020, 2, 29), datetime(2020, 3, 31)]),
        ("left", [datetime(2020, 1, 31), datetime(2020, 2, 29)]),
        ("none", [datetime(2020, 2, 29)]),
        ("both", [datetime(2020, 1, 31), datetime(2020, 2, 29), datetime(2020, 3, 31)]),
    ],
)
def test_datetime_range_end_of_month_5441(
    closed: ClosedInterval, expected_values: list[datetime]
) -> None:
    start = date(2020, 1, 31)
    stop = date(2020, 3, 31)
    result = pl.datetime_range(start, stop, interval="1mo", closed=closed, eager=True)
    expected = pl.Series("literal", expected_values)
    assert_series_equal(result, expected)


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


def test_datetime_range_specifying_ambiguous_11713() -> None:
    result = pl.datetime_range(
        pl.datetime(2023, 10, 29, 2, 0).dt.replace_time_zone(
            "Europe/Madrid", ambiguous="earliest"
        ),
        pl.datetime(2023, 10, 29, 3, 0).dt.replace_time_zone("Europe/Madrid"),
        "1h",
        eager=True,
    )
    expected = pl.Series(
        "datetime",
        [
            datetime(2023, 10, 29, 2),
            datetime(2023, 10, 29, 2),
            datetime(2023, 10, 29, 3),
        ],
    ).dt.replace_time_zone(
        "Europe/Madrid", ambiguous=pl.Series(["earliest", "latest", "raise"])
    )
    assert_series_equal(result, expected)
    result = pl.datetime_range(
        pl.datetime(2023, 10, 29, 2, 0).dt.replace_time_zone(
            "Europe/Madrid", ambiguous="latest"
        ),
        pl.datetime(2023, 10, 29, 3, 0).dt.replace_time_zone("Europe/Madrid"),
        "1h",
        eager=True,
    )
    expected = pl.Series(
        "datetime", [datetime(2023, 10, 29, 2), datetime(2023, 10, 29, 3)]
    ).dt.replace_time_zone("Europe/Madrid", ambiguous=pl.Series(["latest", "raise"]))
    assert_series_equal(result, expected)

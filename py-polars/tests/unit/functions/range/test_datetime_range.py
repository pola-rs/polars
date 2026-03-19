from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.exceptions import ComputeError, InvalidOperationError, SchemaError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval, PolarsDataType, TimeUnit


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
    with pytest.raises(InvalidOperationError, match="'x' not supported"):
        pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="1X",
            eager=True,
        )


def test_datetime_range_interval_too_small() -> None:
    # start/end/interval
    with pytest.raises(
        InvalidOperationError,
        match="interval 1ns is too small for time unit μs and was rounded down to zero",
    ):
        pl.datetime_range(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 5),
            interval="1ns",
            time_unit="us",
            eager=True,
        )


def test_datetime_range_output_ns_due_to_interval() -> None:
    result = pl.datetime_range(
        start=datetime(2025, 1, 1),
        end=datetime(2025, 1, 1, 0, 0, 0, 1),
        interval="1ns",
        eager=True,
    )
    assert result.len() == 1001
    assert result.dtype == pl.Datetime(time_unit="ns")


def test_datetime_range_lazy_time_zones() -> None:
    start = datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))
    stop = datetime(2020, 1, 2, tzinfo=ZoneInfo("Asia/Kathmandu"))
    result = (
        pl.DataFrame({"start": [start], "stop": [stop]})
        .with_columns(
            pl.datetime_range(
                start=start,
                end=stop,
                interval="678d",
                eager=False,
                time_zone="Pacific/Tarawa",
            )
        )
        .lazy()
    )
    expected = pl.DataFrame(
        {
            "start": [
                datetime(2020, 1, 1, 00, 00, tzinfo=ZoneInfo(key="Asia/Kathmandu"))
            ],
            "stop": [
                datetime(2020, 1, 2, 00, 00, tzinfo=ZoneInfo(key="Asia/Kathmandu"))
            ],
            "literal": [
                datetime(2020, 1, 1, 6, 15, tzinfo=ZoneInfo(key="Pacific/Tarawa"))
            ],
        }
    ).with_columns(pl.col("literal").dt.convert_time_zone("Pacific/Tarawa"))
    assert_frame_equal(result.collect(), expected)


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
        time_zone="America/Chicago",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 7, 1, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 7, 1, 0, fold=1, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 7, 2, 0, tzinfo=ZoneInfo("America/Chicago")),
    ]


def test_tzaware_datetime_range_crossing_dst_daily() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 11),
        "2d",
        time_zone="America/Chicago",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 9, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 11, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
    ]


def test_tzaware_datetime_range_crossing_dst_weekly() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 11, 20),
        "1w",
        time_zone="America/Chicago",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 11, 14, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
    ]


def test_tzaware_datetime_range_crossing_dst_monthly() -> None:
    result = pl.datetime_range(
        datetime(2021, 11, 7),
        datetime(2021, 12, 20),
        "1mo",
        time_zone="America/Chicago",
        eager=True,
    )
    assert result.to_list() == [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
        datetime(2021, 12, 7, 0, 0, tzinfo=ZoneInfo("America/Chicago")),
    ]


def test_datetime_range_with_unsupported_datetimes() -> None:
    with pytest.raises(
        ComputeError,
        match=r"datetime '2021-11-07 01:00:00' is ambiguous in time zone 'America/Chicago'",
    ):
        pl.datetime_range(
            datetime(2021, 11, 7, 1),
            datetime(2021, 11, 7, 2),
            "1h",
            time_zone="America/Chicago",
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
    result = pl.datetime_range(
        start=start,
        end=stop,
        interval="1mo",
        closed=closed,
        eager=True,
    )
    expected = pl.Series("literal", expected_values, dtype=pl.Datetime("us"))
    assert_series_equal(result, expected)


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


@given(
    closed=st.sampled_from(["none", "left", "right", "both"]),
    time_unit=st.sampled_from(["ms", "us", "ns"]),
    n=st.integers(1, 10),
    size=st.integers(8, 10),
    unit=st.sampled_from(["s", "m", "h", "d", "mo"]),
    start=st.datetimes(datetime(1965, 1, 1), datetime(2100, 1, 1)),
)
@settings(max_examples=50)
@pytest.mark.benchmark
def test_datetime_range_fast_slow_paths(
    closed: ClosedInterval,
    time_unit: TimeUnit,
    n: int,
    size: int,
    unit: str,
    start: datetime,
) -> None:
    end = pl.select(pl.lit(start).dt.offset_by(f"{n * size}{unit}")).item()
    result_slow = pl.datetime_range(
        start,
        end,
        closed=closed,
        time_unit=time_unit,
        interval=f"{n}{unit}",
        time_zone="Asia/Kathmandu",
        eager=True,
    ).dt.replace_time_zone(None)
    result_fast = pl.datetime_range(
        start,
        end,
        closed=closed,
        time_unit=time_unit,
        interval=f"{n}{unit}",
        eager=True,
    )
    assert_series_equal(result_slow, result_fast)


def test_dt_range_with_nanosecond_interval_19931() -> None:
    with pytest.raises(
        InvalidOperationError, match="interval 1ns is too small for time unit ms"
    ):
        pl.datetime_range(
            pl.date(2022, 1, 1),
            pl.date(2022, 1, 1),
            time_zone="Asia/Kathmandu",
            interval="1ns",
            time_unit="ms",
            eager=True,
        )


def test_datetime_range_with_nanoseconds_overflow_15735() -> None:
    s = pl.datetime_range(date(2000, 1, 1), date(2300, 1, 1), "24h", eager=True)
    assert s.dtype == pl.Datetime("us")
    assert s.shape == (109574,)


# Helper function to generate output Series with expected dtype.
def to_expected(
    values: list[date] | list[datetime], dtype: PolarsDataType
) -> pl.Series:
    if dtype == pl.Date:
        return pl.Series("literal", values, dtype=pl.Datetime("us"))
    else:
        if (tz := dtype.time_zone) is not None:  # type: ignore[union-attr]
            return pl.Series(
                "literal",
                values,
                dtype=pl.Datetime(dtype.time_unit),  # type: ignore[union-attr]
            ).dt.replace_time_zone(tz)
        else:
            return pl.Series("literal", values, dtype=dtype)


# start/end/interval
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
def test_datetime_range_start_end_interval_forwards(dtype: PolarsDataType) -> None:
    start = date(2025, 1, 1)
    end = date(2025, 1, 10)
    tu: TimeUnit = dtype.time_unit if dtype == pl.Datetime else None  # type: ignore[assignment, union-attr]
    tz: str = dtype.time_zone if dtype == pl.Datetime else None  # type: ignore[assignment, union-attr]

    assert_series_equal(
        pl.datetime_range(
            start=start,
            end=end,
            interval="3d",
            closed="left",
            eager=True,
            time_unit=tu,
            time_zone=tz,
        ),
        to_expected([date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 7)], dtype),
    )
    assert_series_equal(
        pl.datetime_range(
            start=start,
            end=end,
            interval="3d",
            closed="right",
            eager=True,
            time_unit=tu,
            time_zone=tz,
        ),
        to_expected([date(2025, 1, 4), date(2025, 1, 7), date(2025, 1, 10)], dtype),
    )
    assert_series_equal(
        pl.datetime_range(
            start=start,
            end=end,
            interval="3d",
            closed="none",
            eager=True,
            time_unit=tu,
            time_zone=tz,
        ),
        to_expected([date(2025, 1, 4), date(2025, 1, 7)], dtype),
    )
    assert_series_equal(
        pl.datetime_range(
            start=start,
            end=end,
            interval="3d",
            closed="both",
            eager=True,
            time_unit=tu,
            time_zone=tz,
        ),
        to_expected(
            [date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 7), date(2025, 1, 10)],
            dtype,
        ),
    )
    # test wrong direction is empty
    assert_series_equal(
        pl.datetime_range(
            start=end,
            end=start,
            interval="3d",
            eager=True,
            time_unit=tu,
            time_zone=tz,
        ),
        to_expected([], dtype),
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
def test_datetime_range_expr_scalar(dtype: PolarsDataType) -> None:
    df = pl.DataFrame(
        {
            "a": [date(2025, 1, 3), date(2025, 1, 1)],
            "interval": ["1d", "2d"],
        }
    )
    tu: TimeUnit = dtype.time_unit if dtype == pl.Datetime else None  # type: ignore[assignment, union-attr]
    tz: str = dtype.time_zone if dtype == pl.Datetime else None  # type: ignore[assignment, union-attr]
    result = df.select(
        forward_start_end_interval=pl.datetime_range(
            start=pl.col("a").min(),
            end=pl.col("a").max(),
            interval="1d",
            time_unit=tu,
            time_zone=tz,
        ),
    )
    forward = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    expected = pl.DataFrame(
        {
            "forward_start_end_interval": to_expected(forward, dtype=dtype),
        }
    )
    assert_frame_equal(result, expected)

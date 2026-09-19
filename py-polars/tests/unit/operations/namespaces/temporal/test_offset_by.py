from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import Ambiguous, EngineType, TimeUnit


@pytest.mark.parametrize(
    ("inputs", "offset", "outputs"),
    [
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "1d",
            [date(2020, 1, 2), date(2020, 1, 3)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-1d",
            [date(2019, 12, 31), date(2020, 1, 1)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "3d",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "72h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "+2d24h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-2mo",
            [date(2019, 11, 1), date(2019, 11, 2)],
        ),
    ],
)
def test_date_offset_by(inputs: list[date], offset: str, outputs: list[date]) -> None:
    result = pl.Series(inputs).dt.offset_by(offset)
    expected = pl.Series(outputs)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("inputs", "offset", "outputs"),
    [
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "1d",
            [date(2020, 1, 2), date(2020, 1, 3)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-1d",
            [date(2019, 12, 31), date(2020, 1, 1)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "3d",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "72h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "2d24h",
            [date(2020, 1, 4), date(2020, 1, 5)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "7m",
            [datetime(2020, 1, 1, 0, 7), datetime(2020, 1, 2, 0, 7)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "-3m",
            [datetime(2019, 12, 31, 23, 57), datetime(2020, 1, 1, 23, 57)],
        ),
        (
            [date(2020, 1, 1), date(2020, 1, 2)],
            "2mo",
            [datetime(2020, 3, 1), datetime(2020, 3, 2)],
        ),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", ["Europe/London", "Asia/Kathmandu", None])
def test_datetime_offset_by(
    inputs: list[date],
    offset: str,
    outputs: list[datetime],
    time_unit: TimeUnit,
    time_zone: str | None,
) -> None:
    result = (
        pl.Series(inputs, dtype=pl.Datetime(time_unit))
        .dt.replace_time_zone(time_zone)
        .dt.offset_by(offset)
    )
    expected = pl.Series(outputs, dtype=pl.Datetime(time_unit)).dt.replace_time_zone(
        time_zone
    )
    assert_series_equal(result, expected)


def test_offset_by_unique_29_feb_19608() -> None:
    df20 = pl.select(
        t=pl.datetime_range(
            pl.datetime(2020, 2, 28),
            pl.datetime(2020, 3, 1),
            closed="left",
            time_unit="ms",
            interval="8h",
            time_zone="UTC",
        ),
    ).with_columns(x=pl.int_range(pl.len()))
    df19 = df20.with_columns(pl.col("t").dt.offset_by("-1y"))
    result = df19.unique("t", keep="first").sort("t")
    expected = pl.DataFrame(
        {
            "t": [
                datetime(2019, 2, 28),
                datetime(2019, 2, 28, 8),
                datetime(2019, 2, 28, 16),
            ],
            "x": [0, 1, 2],
        },
        schema_overrides={"t": pl.Datetime("ms", "UTC")},
    )
    assert_frame_equal(result, expected)


def test_month_then_day_21283() -> None:
    series_vienna = pl.Series(
        [datetime(2024, 5, 15, 8, 0)], dtype=pl.Datetime(time_zone="Europe/Vienna")
    )
    result = series_vienna.dt.offset_by("2y1mo1q1h")[0]
    expected = datetime.strptime("2026-09-15 11:00:00+02:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    result = series_vienna.dt.offset_by("2y1mo1q1h1d")[0]
    expected = datetime.strptime("2026-09-16 11:00:00+02:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    series_utc = pl.Series(
        [datetime(2024, 5, 15, 8, 0)], dtype=pl.Datetime(time_zone="UTC")
    )
    result = series_utc.dt.offset_by("2y1mo1q1h")[0]
    expected = datetime.strptime("2026-09-15 09:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected
    result = series_utc.dt.offset_by("2y1mo1q1h1d")[0]
    expected = datetime.strptime("2026-09-16 09:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
    assert result == expected


def test_offset_by_unequal_length_22018() -> None:
    with pytest.raises(pl.exceptions.ShapeError):
        pl.Series([datetime(2088, 8, 8, 8, 8, 8, 8)] * 2).dt.offset_by(
            pl.Series([f"{h}y" for h in range(3)])
        )


@pytest.mark.parametrize(
    ("start", "by", "ambiguous"),
    [
        (datetime(2025, 10, 25, 2), "1d", "earliest"),
        (datetime(2025, 10, 27, 2), "-1d", "latest"),
        (datetime(2025, 10, 19, 2), "1w", "earliest"),
        (datetime(2025, 11, 2, 2), "-1w", "latest"),
        (datetime(2024, 10, 26, 2), "1y", "earliest"),
        (datetime(2026, 10, 26, 2), "-1y", "latest"),
        (datetime(2025, 9, 26, 2), "1mo", "earliest"),
        (datetime(2025, 11, 26, 2), "-1mo", "latest"),
    ],
)
def test_offset_by_rfc_5545_boundaries(
    start: datetime, by: str, ambiguous: Ambiguous
) -> None:
    s = pl.Series([start]).dt.replace_time_zone("Europe/Amsterdam")
    result = s.dt.offset_by(by)
    expected = pl.Series([datetime(2025, 10, 26, 2)]).dt.replace_time_zone(
        "Europe/Amsterdam", ambiguous=ambiguous
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("start", "by", "expected_dt"),
    [
        (datetime(2025, 3, 29, 2, 30), "1d", datetime(2025, 3, 30, 1, 30)),
        (datetime(2025, 3, 31, 2, 30), "-1d", datetime(2025, 3, 30, 3, 30)),
        (datetime(2025, 1, 30, 2, 30), "2mo", datetime(2025, 3, 30, 1, 30)),
        (datetime(2025, 4, 30, 2, 30), "-1mo", datetime(2025, 3, 30, 3, 30)),
    ],
)
def test_offset_by_rfc_5545_boundaries_non_existest(
    start: datetime, by: str, expected_dt: datetime
) -> None:
    s = pl.Series([start]).dt.replace_time_zone("Europe/Amsterdam")
    result = s.dt.offset_by(by)
    expected = pl.Series([expected_dt]).dt.replace_time_zone("Europe/Amsterdam")
    assert_series_equal(result, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "expression",
    [
        pl.lit(2**63 - 1, dtype=pl.Datetime("us")).dt.offset_by("1mo"),
        pl.lit(-(2**63), dtype=pl.Datetime("ms")).dt.offset_by("-1mo"),
        pl.lit(datetime(2262, 4, 1), dtype=pl.Datetime("ns")).dt.offset_by("1y"),
        pl.lit(datetime(2000, 1, 1), dtype=pl.Datetime("ns")).dt.offset_by("200000d"),
        pl.lit(date(2000, 12, 1)).dt.offset_by("25769779765mo"),
    ],
)
def test_offset_by_out_of_range_unused_expression(
    engine: EngineType, expression: pl.Expr
) -> None:
    frame = pl.DataFrame({"x": [1, 2]})
    query = frame.lazy().with_columns(expression.alias("unused")).select("x")
    assert_frame_equal(query.collect(engine=engine), frame)
    with pytest.raises(pl.exceptions.ComputeError, match=r"out of .*range"):
        frame.lazy().select(expression).collect(engine=engine)


@pytest.mark.parametrize(
    "offset", ["9223372036854775808d", "9223372036854775807y", "9223372036854775807d1d"]
)
def test_offset_by_duration_parse_overflow(offset: str) -> None:
    expression = pl.lit(date(2000, 1, 1)).dt.offset_by(offset)
    frame = pl.DataFrame({"x": [1]})
    assert_frame_equal(
        frame.lazy().with_columns(expression.alias("unused")).select("x").collect(),
        frame,
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="out of range"):
        frame.select(expression)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("timestamp", "time_zone", "offset"),
    [
        ("+262142-12-31 23:00:00", "Etc/GMT-2", "1d"),
        ("+262142-12-31 23:00:00", "Etc/GMT-2", "1mo"),
        ("+262142-12-31 01:30:00", "Etc/GMT+2", "1d"),
    ],
)
def test_offset_by_timezone_overflow_unused_expression(
    engine: EngineType, timestamp: str, time_zone: str, offset: str
) -> None:
    value = (
        pl.Series([timestamp])
        .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us")
        .cast(pl.Int64)
        .item()
    )
    expr = pl.lit(value, dtype=pl.Datetime("us", time_zone)).dt.offset_by(offset)
    frame = pl.LazyFrame({"x": [1, 2]})
    result = frame.with_columns(expr.alias("unused")).select("x").collect(engine=engine)
    assert_frame_equal(result, frame.collect())
    with pytest.raises(pl.exceptions.ComputeError, match="out of range"):
        frame.select(expr).collect(engine=engine)


@pytest.mark.parametrize("time_zone", [None, "UTC", "Etc/GMT+1", "Etc/GMT-1"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize(("offset", "days"), [("109573d", 109573), ("15654w", 109578)])
def test_offset_by_large_valid_duration(
    time_zone: str | None, vector: bool, negative: bool, offset: str, days: int
) -> None:
    start = datetime(2100, 1, 1) if negative else datetime(1800, 1, 1)
    end = start + timedelta(days=-days if negative else days)
    offset = f"-{offset}" if negative else offset
    dtype = pl.Datetime("ns", time_zone)
    series = pl.Series([start, None], dtype=dtype)
    offsets = pl.Series([offset, offset]) if vector else offset
    result = series.dt.offset_by(offsets)
    assert_series_equal(result, pl.Series([end, None], dtype=dtype))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_offset_by_large_valid_duration_literal(engine: EngineType) -> None:
    query = pl.LazyFrame().select(
        pl.lit(datetime(1800, 1, 1), dtype=pl.Datetime("ns"))
        .dt.offset_by("109573d")
        .alias("date")
    )
    expected = pl.DataFrame(
        {"date": [datetime(2100, 1, 1)]}, schema={"date": pl.Datetime("ns")}
    )
    assert_frame_equal(query.collect(engine=engine), expected)


@pytest.mark.parametrize(
    ("value", "time_zone", "sign"),
    [(-(2**63), "Etc/GMT+1", 1), (2**63 - 1, "Etc/GMT-1", -1)],
)
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("unit", "days"), [("d", 1), ("w", 7)])
def test_offset_by_timezone_ns_boundary(
    value: int,
    time_zone: str,
    sign: int,
    vector: bool,
    reverse: bool,
    unit: str,
    days: int,
) -> None:
    end = value + sign * days * 86_400_000_000_000
    if reverse:
        value, end = end, value
        sign = -sign
    offset = f"{sign}{unit}"
    offsets = pl.Series([offset, offset]) if vector else offset
    series = pl.Series([value, None], dtype=pl.Datetime("ns", time_zone))
    result = series.dt.offset_by(offsets).cast(pl.Int64)
    assert_series_equal(result, pl.Series([end, None], dtype=pl.Int64))


@pytest.mark.parametrize("time_zone", [None, "Etc/GMT+1"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize(
    "offset", ["106752d", "9223372036854775807d", "9223372036854775807w"]
)
def test_offset_by_large_duration_out_of_range(
    time_zone: str | None, vector: bool, negative: bool, offset: str
) -> None:
    offset = f"-{offset}" if negative else offset
    offsets = pl.Series([offset, offset]) if vector else offset
    series = pl.Series([0, None], dtype=pl.Datetime("ns", time_zone))
    with pytest.raises(pl.exceptions.ComputeError, match="out of range"):
        series.dt.offset_by(offsets)


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_offset_by_scalar_and_vector_overflow(
    time_unit: TimeUnit,
    vector: bool,
    negative: bool,
) -> None:
    value = -(2**63) if negative else 2**63 - 1
    offset = f"{'-' if negative else ''}1{time_unit}"
    offsets = pl.Series([offset, offset]) if vector else offset
    values = pl.Series([value, None], dtype=pl.Datetime(time_unit))
    with pytest.raises(pl.exceptions.ComputeError, match="out of range"):
        values.dt.offset_by(offsets)

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars.type_aliases import ClosedInterval, TimeUnit


def test_date_range() -> None:
    # if low/high are both date, range is also be date _iif_ the granularity is >= 1d
    result = pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    assert result.to_list() == [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]


def test_date_range_invalid_time_unit() -> None:
    with pytest.raises(pl.PolarsPanicError, match="'x' not supported"):
        pl.date_range(
            start=date(2021, 12, 16),
            end=date(2021, 12, 18),
            interval="1X",
            eager=True,
        )


def test_date_range_invalid_time() -> None:
    with pytest.raises(pl.ComputeError, match="end is an out-of-range time"):
        pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 2, 30), eager=True)


def test_date_range_lazy_with_literals() -> None:
    df = pl.DataFrame({"misc": ["x"]}).with_columns(
        pl.date_ranges(
            date(2000, 1, 1),
            date(2023, 8, 31),
            interval="987d",
            eager=False,
        ).alias("dts")
    )
    assert df.rows() == [
        (
            "x",
            [
                date(2000, 1, 1),
                date(2002, 9, 14),
                date(2005, 5, 28),
                date(2008, 2, 9),
                date(2010, 10, 23),
                date(2013, 7, 6),
                date(2016, 3, 19),
                date(2018, 12, 1),
                date(2021, 8, 14),
            ],
        )
    ]
    assert (
        df.rows()[0][1]
        == pd.date_range(
            date(2000, 1, 1), date(2023, 12, 31), freq="987d"
        ).date.tolist()
    )


@pytest.mark.parametrize("low", ["start", pl.col("start")])
@pytest.mark.parametrize("high", ["stop", pl.col("stop")])
def test_date_range_lazy_with_expressions(
    low: str | pl.Expr, high: str | pl.Expr
) -> None:
    lf = pl.LazyFrame(
        {
            "start": [date(2015, 6, 30)],
            "stop": [date(2022, 12, 31)],
        }
    )

    result = lf.with_columns(
        pl.date_ranges(low, high, interval="678d", eager=False).alias("dts")
    )

    assert result.collect().rows() == [
        (
            date(2015, 6, 30),
            date(2022, 12, 31),
            [
                date(2015, 6, 30),
                date(2017, 5, 8),
                date(2019, 3, 17),
                date(2021, 1, 23),
                date(2022, 12, 2),
            ],
        )
    ]

    df = pl.DataFrame(
        {
            "start": [date(2000, 1, 1), date(2022, 6, 1)],
            "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        }
    )

    result_df = df.with_columns(pl.date_ranges(low, high, interval="1d").alias("dts"))

    assert result_df.to_dict(as_series=False) == {
        "start": [date(2000, 1, 1), date(2022, 6, 1)],
        "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        "dts": [
            [date(2000, 1, 1), date(2000, 1, 2)],
            [date(2022, 6, 1), date(2022, 6, 2)],
        ],
    }


def test_date_ranges_single_row_lazy_7110() -> None:
    df = pl.DataFrame(
        {
            "name": ["A"],
            "from": [date(2020, 1, 1)],
            "to": [date(2020, 1, 2)],
        }
    )
    result = df.with_columns(
        pl.date_ranges(
            start=pl.col("from"),
            end=pl.col("to"),
            interval="1d",
            eager=False,
        ).alias("date_range")
    )
    expected = pl.DataFrame(
        {
            "name": ["A"],
            "from": [date(2020, 1, 1)],
            "to": [date(2020, 1, 2)],
            "date_range": [[date(2020, 1, 1), date(2020, 1, 2)]],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("closed", "expected_values"),
    [
        ("right", [date(2020, 2, 29), date(2020, 3, 31)]),
        ("left", [date(2020, 1, 31), date(2020, 2, 29)]),
        ("none", [date(2020, 2, 29)]),
        ("both", [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]),
    ],
)
def test_date_range_end_of_month_5441(
    closed: ClosedInterval, expected_values: list[date]
) -> None:
    start = date(2020, 1, 31)
    stop = date(2020, 3, 31)
    result = pl.date_range(start, stop, interval="1mo", closed=closed, eager=True)
    expected = pl.Series("date", expected_values)
    assert_series_equal(result, expected)


def test_date_range_name() -> None:
    expected_name = "date"
    result_eager = pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=True)
    assert result_eager.name == expected_name

    result_lazy = pl.select(
        pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=False)
    ).to_series()
    assert result_lazy.name == expected_name


def test_date_ranges_eager() -> None:
    start = pl.Series([date(2022, 1, 1), date(2022, 1, 2)])
    end = pl.Series([date(2022, 1, 4), date(2022, 1, 3)])

    result = pl.date_ranges(start, end, eager=True)

    expected = pl.Series(
        "date_range",
        [
            [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4)],
            [date(2022, 1, 2), date(2022, 1, 3)],
        ],
    )
    assert_series_equal(result, expected)


def test_date_range_eager() -> None:
    start = pl.Series([date(2022, 1, 1)])
    end = pl.Series([date(2022, 1, 3)])

    result = pl.date_range(start, end, eager=True)

    expected = pl.Series("date", [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)])
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    (
        "input_time_unit",
        "input_time_zone",
        "expected_date_range",
    ),
    [
        (None, None, ["2020-01-01", "2020-01-02", "2020-01-03"]),
    ],
)
def test_date_range_schema_no_upcast(
    input_time_unit: TimeUnit | None,
    input_time_zone: str | None,
    expected_date_range: list[str],
) -> None:
    output_dtype = pl.Date
    interval = "1d"

    df = pl.DataFrame({"start": [date(2020, 1, 1)], "end": [date(2020, 1, 3)]}).lazy()
    result = df.with_columns(
        pl.date_ranges(
            pl.col("start"),
            pl.col("end"),
            interval=interval,
            time_unit=input_time_unit,
            time_zone=input_time_zone,
        ).alias("date_range")
    )
    expected_schema = {
        "start": pl.Date,
        "end": pl.Date,
        "date_range": pl.List(output_dtype),
    }
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema

    expected = pl.DataFrame(
        {
            "start": [date(2020, 1, 1)],
            "end": [date(2020, 1, 3)],
            "date_range": pl.Series(expected_date_range)
            .str.to_datetime(time_unit="ns")
            .implode(),
        }
    ).with_columns(
        pl.col("date_range").explode().cast(output_dtype).implode(),
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize(
    (
        "input_time_unit",
        "input_time_zone",
        "expected_date_range",
    ),
    [
        ("ms", None, ["2020-01-01", "2020-01-02", "2020-01-03"]),
        (None, "Asia/Kathmandu", ["2020-01-01", "2020-01-02", "2020-01-03"]),
        ("ms", "Asia/Kathmandu", ["2020-01-01", "2020-01-02", "2020-01-03"]),
    ],
)
def test_date_range_schema_no_upcast2(
    input_time_unit: TimeUnit | None,
    input_time_zone: str | None,
    expected_date_range: list[str],
) -> None:
    output_dtype = pl.Date
    interval = "1d"

    df = pl.DataFrame({"start": [date(2020, 1, 1)], "end": [date(2020, 1, 3)]}).lazy()
    with pytest.deprecated_call():
        result = df.with_columns(
            pl.date_ranges(
                pl.col("start"),
                pl.col("end"),
                interval=interval,
                time_unit=input_time_unit,
                time_zone=input_time_zone,
            ).alias("date_range")
        )
    expected_schema = {
        "start": pl.Date,
        "end": pl.Date,
        "date_range": pl.List(output_dtype),
    }
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema

    expected = pl.DataFrame(
        {
            "start": [date(2020, 1, 1)],
            "end": [date(2020, 1, 3)],
            "date_range": pl.Series(expected_date_range)
            .str.to_datetime(time_unit="ns")
            .implode(),
        }
    ).with_columns(
        pl.col("date_range").explode().cast(output_dtype).implode(),
    )
    assert_frame_equal(result.collect(), expected)


def test_date_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Datetime)
    single = pl.Series([datetime(2022, 1, 2)])

    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.date_range(empty, single, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`end` must contain exactly one value, got 0 values"
    ):
        pl.date_range(single, empty, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.date_range(empty, empty, eager=True)


def test_date_range_input_shape_multiple_values() -> None:
    single = pl.Series([datetime(2022, 1, 2)])
    multiple = pl.Series([datetime(2022, 1, 3), datetime(2022, 1, 4)])

    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.date_range(multiple, single, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`end` must contain exactly one value, got 2 values"
    ):
        pl.date_range(single, multiple, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.date_range(multiple, multiple, eager=True)


def test_date_range_start_later_than_end() -> None:
    result = pl.date_range(date(2000, 3, 20), date(2000, 3, 5), eager=True)
    expected = pl.Series("date", dtype=pl.Date)
    assert_series_equal(result, expected)


def test_date_range_24h_interval_results_in_datetime() -> None:
    with pytest.deprecated_call():
        result = pl.LazyFrame().select(
            pl.date_range(date(2022, 1, 1), date(2022, 1, 3), interval="24h")
        )

    assert result.schema == {"date": pl.Datetime}
    expected = pl.Series(
        "date", [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)]
    )
    assert_series_equal(result.collect().to_series(), expected)


def test_long_date_range_12461() -> None:
    result = pl.date_range(date(1900, 1, 1), date(2300, 1, 1), "1d", eager=True)
    assert result[0] == date(1900, 1, 1)
    assert result[-1] == date(2300, 1, 1)
    assert (result.diff()[1:].dt.total_days() == 1).all()


def test_date_ranges_broadcasting() -> None:
    df = pl.DataFrame({"dates": [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]})
    result = df.select(
        pl.date_ranges(start="dates", end=date(2021, 1, 3)).alias("end"),
        pl.date_ranges(start=date(2021, 1, 1), end="dates").alias("start"),
    )
    expected = pl.DataFrame(
        {
            "end": [
                [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
                [date(2021, 1, 2), date(2021, 1, 3)],
                [date(2021, 1, 3)],
            ],
            "start": [
                [date(2021, 1, 1)],
                [date(2021, 1, 1), date(2021, 1, 2)],
                [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
            ],
        }
    )
    assert_frame_equal(result, expected)

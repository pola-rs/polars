from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import polars as pl
from polars.exceptions import ComputeError, PanicException
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval


def test_date_range() -> None:
    # if low/high are both date, range is also be date _iff_ the granularity is >= 1d
    result = pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)
    assert result.to_list() == [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]


def test_date_range_invalid_time_unit() -> None:
    with pytest.raises(PanicException, match="'x' not supported"):
        pl.date_range(
            start=date(2021, 12, 16),
            end=date(2021, 12, 18),
            interval="1X",
            eager=True,
        )


def test_date_range_invalid_time() -> None:
    with pytest.raises(ComputeError, match="end is an out-of-range time"):
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
    expected = pl.Series("literal", expected_values)
    assert_series_equal(result, expected)


def test_date_range_name() -> None:
    result_eager = pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=True)
    assert result_eager.name == "literal"

    start = pl.Series("left", [date(2020, 1, 1)])
    result_lazy = pl.select(
        pl.date_range(start, date(2020, 1, 3), eager=False)
    ).to_series()
    assert result_lazy.name == "left"


def test_date_ranges_eager() -> None:
    start = pl.Series("start", [date(2022, 1, 1), date(2022, 1, 2)])
    end = pl.Series("end", [date(2022, 1, 4), date(2022, 1, 3)])

    result = pl.date_ranges(start, end, eager=True)

    expected = pl.Series(
        "start",
        [
            [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4)],
            [date(2022, 1, 2), date(2022, 1, 3)],
        ],
    )
    assert_series_equal(result, expected)


def test_date_range_eager() -> None:
    start = pl.Series("start", [date(2022, 1, 1)])
    end = pl.Series("end", [date(2022, 1, 3)])

    result = pl.date_range(start, end, eager=True)

    expected = pl.Series(
        "start", [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    )
    assert_series_equal(result, expected)


def test_date_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Datetime)
    single = pl.Series([datetime(2022, 1, 2)])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.date_range(empty, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 0 values"
    ):
        pl.date_range(single, empty, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.date_range(empty, empty, eager=True)


def test_date_range_input_shape_multiple_values() -> None:
    single = pl.Series([datetime(2022, 1, 2)])
    multiple = pl.Series([datetime(2022, 1, 3), datetime(2022, 1, 4)])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.date_range(multiple, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 2 values"
    ):
        pl.date_range(single, multiple, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.date_range(multiple, multiple, eager=True)


def test_date_range_start_later_than_end() -> None:
    result = pl.date_range(date(2000, 3, 20), date(2000, 3, 5), eager=True)
    expected = pl.Series("literal", dtype=pl.Date)
    assert_series_equal(result, expected)


def test_date_range_24h_interval_raises() -> None:
    with pytest.raises(
        ComputeError,
        match="`interval` input for `date_range` must consist of full days",
    ):
        pl.date_range(date(2022, 1, 1), date(2022, 1, 3), interval="24h", eager=True)


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


def test_date_ranges_broadcasting_fail() -> None:
    start = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
    end = pl.Series([date(2021, 1, 2), date(2021, 1, 3)])

    with pytest.raises(
        ComputeError, match=r"lengths of `start` \(3\) and `end` \(2\) do not match"
    ):
        pl.date_ranges(start, end, eager=True)


def test_date_range_datetime_input() -> None:
    result = pl.date_range(
        datetime(2022, 1, 1, 12), datetime(2022, 1, 3), interval="1d", eager=True
    )
    expected = pl.Series(
        "literal", [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    )
    assert_series_equal(result, expected)


def test_date_ranges_datetime_input() -> None:
    result = pl.date_ranges(
        datetime(2022, 1, 1, 12), datetime(2022, 1, 3), interval="1d", eager=True
    )
    expected = pl.Series(
        "literal", [[date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]]
    )
    assert_series_equal(result, expected)

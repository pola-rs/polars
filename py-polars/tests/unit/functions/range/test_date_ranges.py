from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import ClosedInterval


def test_date_ranges_lazy_with_literals() -> None:
    df = pl.DataFrame({"misc": ["x"]}).with_columns(
        pl.date_ranges(
            start=date(2000, 1, 1),
            end=date(2023, 8, 31),
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
            date(2000, 1, 1), date(2023, 12, 31), freq="987D"
        ).date.tolist()
    )


@pytest.mark.parametrize("low", ["start", pl.col("start")])
@pytest.mark.parametrize("high", ["stop", pl.col("stop")])
def test_date_ranges_lazy_with_expressions(
    low: str | pl.Expr, high: str | pl.Expr
) -> None:
    lf = pl.LazyFrame(
        {
            "start": [date(2015, 6, 30)],
            "stop": [date(2022, 12, 31)],
        }
    )

    result = lf.with_columns(
        pl.date_ranges(start=low, end=high, interval="678d", eager=False).alias("dts")
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

    result_df = df.with_columns(
        pl.date_ranges(start=low, end=high, interval="1d").alias("dts")
    )

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


def test_date_ranges_eager() -> None:
    start = pl.Series("start", [date(2022, 1, 1), date(2022, 1, 2)])
    end = pl.Series("end", [date(2022, 1, 4), date(2022, 1, 3)])

    result = pl.date_ranges(start=start, end=end, eager=True)

    expected = pl.Series(
        "start",
        [
            [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4)],
            [date(2022, 1, 2), date(2022, 1, 3)],
        ],
    )
    assert_series_equal(result, expected)


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
        pl.date_ranges(start=start, end=end, eager=True)


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        (
            "both",
            [
                [
                    date(2025, 1, 1),
                    date(2025, 1, 3),
                    date(2025, 1, 5),
                    date(2025, 1, 7),
                ],
                [
                    date(2025, 1, 11),
                    date(2025, 1, 13),
                    date(2025, 1, 15),
                    date(2025, 1, 17),
                ],
            ],
        ),
        (
            "left",
            [
                [
                    date(2025, 1, 1),
                    date(2025, 1, 3),
                    date(2025, 1, 5),
                    date(2025, 1, 7),
                ],
                [
                    date(2025, 1, 11),
                    date(2025, 1, 13),
                    date(2025, 1, 15),
                    date(2025, 1, 17),
                ],
            ],
        ),
        (
            "right",
            [
                [date(2025, 1, 3), date(2025, 1, 5), date(2025, 1, 7)],
                [date(2025, 1, 13), date(2025, 1, 15), date(2025, 1, 17)],
            ],
        ),
        (
            "none",
            [
                [date(2025, 1, 3), date(2025, 1, 5), date(2025, 1, 7)],
                [date(2025, 1, 13), date(2025, 1, 15), date(2025, 1, 17)],
            ],
        ),
    ],
)
def test_date_ranges_start_end_interval_forwards(
    closed: ClosedInterval,
    expected: list[date],
) -> None:
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 11)],
            "end": [date(2025, 1, 8), date(2025, 1, 18)],
        }
    )
    result = df.select(
        dates=pl.date_ranges(start="start", end="end", interval="2d", closed=closed)
    )
    assert_frame_equal(result, pl.Series("dates", expected).to_frame())

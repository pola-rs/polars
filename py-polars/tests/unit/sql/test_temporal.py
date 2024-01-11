from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_date() -> None:
    df = pl.DataFrame(
        {
            "date": [
                date(2021, 3, 15),
                date(2021, 3, 28),
                date(2021, 4, 4),
            ],
            "version": ["0.0.1", "0.7.3", "0.7.4"],
        }
    )
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        result = ctx.execute("SELECT date < DATE('2021-03-20') from df")

    expected = pl.DataFrame({"date": [True, False, False]})
    assert_frame_equal(result, expected)

    result = pl.select(pl.sql_expr("""CAST(DATE('2023-03', '%Y-%m') as STRING)"""))
    expected = pl.DataFrame({"literal": ["2023-03-01"]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("part", "dtype", "expected"),
    [
        ("decade", pl.Int32, [202, 202, 200]),
        ("isoyear", pl.Int32, [2024, 2020, 2005]),
        ("year", pl.Int32, [2024, 2020, 2006]),
        ("quarter", pl.Int8, [1, 4, 1]),
        ("month", pl.Int8, [1, 12, 1]),
        ("week", pl.Int8, [1, 53, 52]),
        ("doy", pl.Int16, [7, 365, 1]),
        ("isodow", pl.Int8, [7, 3, 7]),
        ("dow", pl.Int8, [0, 3, 0]),
        ("day", pl.Int8, [7, 30, 1]),
        ("hour", pl.Int8, [1, 10, 23]),
        ("minute", pl.Int8, [2, 30, 59]),
        ("second", pl.Int8, [3, 45, 59]),
        ("millisecond", pl.Float64, [3123.456, 45987.654, 59555.555]),
        ("microsecond", pl.Float64, [3123456.0, 45987654.0, 59555555.0]),
        ("nanosecond", pl.Float64, [3123456000.0, 45987654000.0, 59555555000.0]),
        (
            "time",
            pl.Time,
            [time(1, 2, 3, 123456), time(10, 30, 45, 987654), time(23, 59, 59, 555555)],
        ),
        (
            "epoch",
            pl.Float64,
            [1704589323.123456, 1609324245.987654, 1136159999.555555],
        ),
    ],
)
def test_extract_datepart(part: str, dtype: pl.DataType, expected: list[Any]) -> None:
    df = pl.DataFrame(
        {
            "dt": [
                # note: these values test several edge-cases, such as isoyear,
                # the mon/sun wrapping of dow vs isodow, epoch rounding, etc,
                # and the results have been validated against postgresql.
                datetime(2024, 1, 7, 1, 2, 3, 123456),
                datetime(2020, 12, 30, 10, 30, 45, 987654),
                datetime(2006, 1, 1, 23, 59, 59, 555555),
            ],
        }
    )
    with pl.SQLContext(frame_data=df, eager_execution=True) as ctx:
        for func in (f"EXTRACT({part} FROM dt)", f"DATE_PART(dt,'{part}')"):
            res = ctx.execute(f"SELECT {func} AS {part} FROM frame_data").to_series()

            assert res.dtype == dtype
            assert res.to_list() == expected

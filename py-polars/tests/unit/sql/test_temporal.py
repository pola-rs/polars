from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
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
def test_extract(part: str, dtype: pl.DataType, expected: list[Any]) -> None:
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


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (date(1, 1, 1), [1, 1]),
        (date(100, 1, 1), [1, 1]),
        (date(101, 1, 1), [1, 2]),
        (date(1000, 1, 1), [1, 10]),
        (date(1001, 1, 1), [2, 11]),
        (date(1899, 12, 31), [2, 19]),
        (date(1900, 12, 31), [2, 19]),
        (date(1901, 1, 1), [2, 20]),
        (date(2000, 12, 31), [2, 20]),
        (date(2001, 1, 1), [3, 21]),
        (date(5555, 5, 5), [6, 56]),
        (date(9999, 12, 31), [10, 100]),
    ],
)
def test_extract_century_millennium(dt: date, expected: list[int]) -> None:
    with pl.SQLContext(
        frame_data=pl.DataFrame({"dt": [dt]}), eager_execution=True
    ) as ctx:
        res = ctx.execute(
            """
            SELECT
              EXTRACT(MILLENNIUM FROM dt) AS c1,
              DATE_PART(dt,'century') AS c2,
              EXTRACT(millennium FROM dt) AS c3,
              DATE_PART(dt,'CENTURY') AS c4,
            FROM frame_data
            """
        )
        assert_frame_equal(
            left=res,
            right=pl.DataFrame(
                data=[expected + expected],
                schema=["c1", "c2", "c3", "c4"],
            ).cast(pl.Int32),
        )


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("ms", [1704589323123, 1609324245987, 1136159999555]),
        ("us", [1704589323123456, 1609324245987654, 1136159999555555]),
        ("ns", [1704589323123456000, 1609324245987654000, 1136159999555555000]),
        (None, [1704589323123456, 1609324245987654, 1136159999555555]),
    ],
)
def test_timestamp_time_unit(unit: str | None, expected: list[int]) -> None:
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 7, 1, 2, 3, 123456),
                datetime(2020, 12, 30, 10, 30, 45, 987654),
                datetime(2006, 1, 1, 23, 59, 59, 555555),
            ],
        }
    )
    precision = {"ms": 3, "us": 6, "ns": 9}

    with pl.SQLContext(frame_data=df, eager_execution=True) as ctx:
        prec = f"({precision[unit]})" if unit else ""
        res = ctx.execute(f"SELECT ts::timestamp{prec} FROM frame_data").to_series()

        assert res.dtype == pl.Datetime(time_unit=unit)  # type: ignore[arg-type]
        assert res.to_physical().to_list() == expected


def test_timestamp_time_unit_errors() -> None:
    df = pl.DataFrame({"ts": [datetime(2024, 1, 7, 1, 2, 3, 123456)]})

    with pl.SQLContext(frame_data=df, eager_execution=True) as ctx:
        for prec in (0, 4, 15):
            with pytest.raises(
                ComputeError, match=f"unsupported `timestamp` precision; .* prec={prec}"
            ):
                ctx.execute(f"SELECT ts::timestamp({prec}) FROM frame_data")

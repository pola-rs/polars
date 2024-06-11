from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal

import pytest

import polars as pl
from polars.exceptions import ComputeError, SQLInterfaceError, SQLSyntaxError
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
    with pl.SQLContext(df=df, eager=True) as ctx:
        result = ctx.execute("SELECT date < DATE('2021-03-20') from df")

    expected = pl.DataFrame({"date": [True, False, False]})
    assert_frame_equal(result, expected)

    result = pl.select(pl.sql_expr("""CAST(DATE('2023-03', '%Y-%m') as STRING)"""))
    expected = pl.DataFrame({"literal": ["2023-03-01"]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_datetime_to_time(time_unit: Literal["ns", "us", "ms"]) -> None:
    df = pl.DataFrame(  # noqa: F841
        {
            "dtm": [
                datetime(2099, 12, 31, 23, 59, 59),
                datetime(1999, 12, 31, 12, 30, 30),
                datetime(1969, 12, 31, 1, 1, 1),
                datetime(1899, 12, 31, 0, 0, 0),
            ],
        },
        schema={"dtm": pl.Datetime(time_unit)},
    )

    res = pl.sql("SELECT dtm::time AS tm from df").collect()
    assert res["tm"].to_list() == [
        time(23, 59, 59),
        time(12, 30, 30),
        time(1, 1, 1),
        time(0, 0, 0),
    ]


@pytest.mark.parametrize(
    ("parts", "dtype", "expected"),
    [
        (["decade", "decades"], pl.Int32, [202, 202, 200]),
        (["isoyear"], pl.Int32, [2024, 2020, 2005]),
        (["year", "y"], pl.Int32, [2024, 2020, 2006]),
        (["quarter"], pl.Int8, [1, 4, 1]),
        (["month", "months", "mon", "mons"], pl.Int8, [1, 12, 1]),
        (["week", "weeks"], pl.Int8, [1, 53, 52]),
        (["doy"], pl.Int16, [7, 365, 1]),
        (["isodow"], pl.Int8, [7, 3, 7]),
        (["dow"], pl.Int8, [0, 3, 0]),
        (["day", "days", "d"], pl.Int8, [7, 30, 1]),
        (["hour", "hours", "h"], pl.Int8, [1, 10, 23]),
        (["minute", "min", "mins", "m"], pl.Int8, [2, 30, 59]),
        (["second", "seconds", "secs", "sec"], pl.Int8, [3, 45, 59]),
        (
            ["millisecond", "milliseconds", "ms"],
            pl.Float64,
            [3123.456, 45987.654, 59555.555],
        ),
        (
            ["microsecond", "microseconds", "us"],
            pl.Float64,
            [3123456.0, 45987654.0, 59555555.0],
        ),
        (
            ["nanosecond", "nanoseconds", "ns"],
            pl.Float64,
            [3123456000.0, 45987654000.0, 59555555000.0],
        ),
        (
            ["time"],
            pl.Time,
            [time(1, 2, 3, 123456), time(10, 30, 45, 987654), time(23, 59, 59, 555555)],
        ),
        (
            ["epoch"],
            pl.Float64,
            [1704589323.123456, 1609324245.987654, 1136159999.555555],
        ),
    ],
)
def test_extract(parts: list[str], dtype: pl.DataType, expected: list[Any]) -> None:
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
    with pl.SQLContext(frame_data=df, eager=True) as ctx:
        for part in parts:
            for fn in (
                f"EXTRACT({part} FROM dt)",
                f"DATE_PART('{part}',dt)",
            ):
                res = ctx.execute(f"SELECT {fn} AS {part} FROM frame_data").to_series()
                assert res.dtype == dtype
                assert res.to_list() == expected


def test_extract_errors() -> None:
    df = pl.DataFrame({"dt": [datetime(2024, 1, 7, 1, 2, 3, 123456)]})

    with pl.SQLContext(frame_data=df, eager=True) as ctx:
        for part in ("femtosecond", "stroopwafel"):
            with pytest.raises(
                SQLSyntaxError,
                match=f"EXTRACT/DATE_PART does not support '{part}' part",
            ):
                ctx.execute(f"SELECT EXTRACT({part} FROM dt) FROM frame_data")

        with pytest.raises(
            SQLSyntaxError,
            match=r"EXTRACT/DATE_PART does not support 'week\(tuesday\)' part",
        ):
            ctx.execute("SELECT DATE_PART('week(tuesday)', dt) FROM frame_data")


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
    with pl.SQLContext(frame_data=pl.DataFrame({"dt": [dt]}), eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              EXTRACT(MILLENNIUM FROM dt) AS c1,
              DATE_PART('century',dt) AS c2,
              EXTRACT(millennium FROM dt) AS c3,
              DATE_PART('CENTURY',dt) AS c4,
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
    ("constraint", "expected"),
    [
        ("dtm >= '2020-12-30T10:30:45.987'", [0, 2]),
        ("dtm::date > '2006-01-01'", [0, 2]),
        ("dtm > '2006-01-01'", [0, 1, 2]),  # << implies '2006-01-01 00:00:00'
        ("dtm <= '2006-01-01'", []),  # << implies '2006-01-01 00:00:00'
        ("dt != '1960-01-07'", [0, 1]),
        ("dt BETWEEN '2050-01-01' AND '2100-12-31'", [1]),
        ("dt::datetime = '1960-01-07'", [2]),
        ("dt::datetime = '1960-01-07 00:00:00'", [2]),
        ("dtm BETWEEN '2020-12-30 10:30:44' AND '2023-01-01 00:00:00'", [2]),
        ("dt IN ('1960-01-07','2077-01-01','2222-02-22')", [1, 2]),
        (
            "dtm = '2024-01-07 01:02:03.123456000' OR dtm = '2020-12-30 10:30:45.987654'",
            [0, 2],
        ),
    ],
)
def test_implicit_temporal_strings(constraint: str, expected: list[int]) -> None:
    df = pl.DataFrame(
        {
            "idx": [0, 1, 2],
            "dtm": [
                datetime(2024, 1, 7, 1, 2, 3, 123456),
                datetime(2006, 1, 1, 23, 59, 59, 555555),
                datetime(2020, 12, 30, 10, 30, 45, 987654),
            ],
            "dt": [
                date(2020, 12, 30),
                date(2077, 1, 1),
                date(1960, 1, 7),
            ],
        }
    )
    res = df.sql(f"SELECT idx FROM self WHERE {constraint}")
    actual = sorted(res["idx"])
    assert actual == expected


@pytest.mark.parametrize(
    "dtval",
    [
        "2020-12-30T10:30:45",
        "yyyy-mm-dd",
        "2222-22-22",
        "10:30:45",
    ],
)
def test_implicit_temporal_string_errors(dtval: str) -> None:
    df = pl.DataFrame({"dt": [date(2020, 12, 30)]})

    with pytest.raises(
        ComputeError,
        match="(conversion.*failed)|(cannot compare.*string.*temporal)",
    ):
        df.sql(f"SELECT * FROM self WHERE dt = '{dtval}'")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("ms", [1704589323123, 1609324245987, 1136159999555]),
        ("us", [1704589323123456, 1609324245987654, 1136159999555555]),
        ("ns", [1704589323123456000, 1609324245987654000, 1136159999555555000]),
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

    with pl.SQLContext(frame_data=df, eager=True) as ctx:
        prec = f"({precision[unit]})" if unit else ""
        res = ctx.execute(f"SELECT ts::timestamp{prec} FROM frame_data").to_series()

        assert res.dtype == pl.Datetime(time_unit=unit)  # type: ignore[arg-type]
        assert res.to_physical().to_list() == expected


def test_timestamp_time_unit_errors() -> None:
    df = pl.DataFrame({"ts": [datetime(2024, 1, 7, 1, 2, 3, 123456)]})

    with pl.SQLContext(frame_data=df, eager=True) as ctx:
        for prec in (0, 15):
            with pytest.raises(
                SQLSyntaxError,
                match=f"invalid temporal type precision; expected 1-9, found {prec}",
            ):
                ctx.execute(f"SELECT ts::timestamp({prec}) FROM frame_data")

        with pytest.raises(
            SQLInterfaceError,
            match="sql parser error: Expected literal int, found: - ",
        ):
            ctx.execute("SELECT ts::timestamp(-3) FROM frame_data")

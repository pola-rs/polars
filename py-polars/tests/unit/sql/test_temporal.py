from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


def test_date_func() -> None:
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

    result = pl.select(pl.sql_expr("CAST(DATE('2023-03', '%Y-%m') as STRING)"))
    expected = pl.DataFrame({"literal": ["2023-03-01"]})
    assert_frame_equal(result, expected)

    with pytest.raises(
        SQLSyntaxError,
        match=r"DATE expects 1-2 arguments \(found 0\)",
    ):
        df.sql("SELECT DATE() FROM self")

    with pytest.raises(InvalidOperationError):
        df.sql("SELECT DATE('2077-07-07','not_a_valid_strftime_format') FROM self")


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
                orient="row",
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
        ("tm != '22:10:30'", [0, 2]),
        ("tm >= '11:00:00' AND tm < '22:00:00'", [0]),
        ("tm BETWEEN '12:00:00' AND '23:59:58'", [0, 1]),
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
            "tm": [
                time(17, 30, 45),
                time(22, 10, 30),
                time(10, 25, 15),
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
        InvalidOperationError,
        match="(conversion.*failed)|(cannot compare.*string.*temporal)",
    ):
        df.sql(f"SELECT * FROM self WHERE dt = '{dtval}'")


def test_strftime() -> None:
    df = pl.DataFrame(
        {
            "dtm": [
                None,
                datetime(1980, 9, 30, 1, 25, 50),
                datetime(2077, 7, 17, 11, 30, 55),
            ],
            "dt": [date(1978, 7, 5), date(1969, 12, 31), date(2020, 4, 10)],
            "tm": [time(10, 10, 10), time(22, 33, 55), None],
        }
    )
    res = df.sql(
        """
        SELECT
          STRFTIME(dtm,'%m.%d.%Y/%T') AS s_dtm,
          STRFTIME(dt ,'%B %d, %Y') AS s_dt,
          STRFTIME(tm ,'%S.%M.%H') AS s_tm,
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "s_dtm": [None, "09.30.1980/01:25:50", "07.17.2077/11:30:55"],
        "s_dt": ["July 05, 1978", "December 31, 1969", "April 10, 2020"],
        "s_tm": ["10.10.10", "55.33.22", None],
    }

    with pytest.raises(
        SQLSyntaxError,
        match=r"STRFTIME expects 2 arguments \(found 4\)",
    ):
        pl.sql_expr("STRFTIME(dtm,'%Y-%m-%d','[extra]','[param]')")


def test_strptime() -> None:
    df = pl.DataFrame(
        {
            "s_dtm": [None, "09.30.1980/01:25:50", "07.17.2077/11:30:55"],
            "s_dt": ["July 5, 1978", "December 31, 1969", "April 10, 2020"],
            "s_tm": ["10.10.10", "55.33.22", None],
        }
    )
    res = df.sql(
        """
        SELECT
          STRPTIME(s_dtm,'%m.%d.%Y/%T') AS dtm,
          STRPTIME(s_dt ,'%B %d, %Y')::date AS dt,
          STRPTIME(s_tm ,'%S.%M.%H')::time AS tm
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "dtm": [
            None,
            datetime(1980, 9, 30, 1, 25, 50),
            datetime(2077, 7, 17, 11, 30, 55),
        ],
        "dt": [date(1978, 7, 5), date(1969, 12, 31), date(2020, 4, 10)],
        "tm": [time(10, 10, 10), time(22, 33, 55), None],
    }
    with pytest.raises(
        SQLSyntaxError,
        match=r"STRPTIME expects 2 arguments \(found 3\)",
    ):
        pl.sql_expr("STRPTIME(s,'%Y.%m.%d',false) AS dt")


def test_temporal_stings_to_datetime() -> None:
    df = pl.DataFrame(
        {
            "s_dt": ["2077-10-10", "1942-01-08", "2000-07-05"],
            "s_dtm1": [
                "1999-12-31 10:30:45",
                "2020-06-10",
                "2022-08-07T00:01:02.654321",
            ],
            "s_dtm2": ["31-12-1999 10:30", "10-06-2020 00:00", "07-08-2022 00:01"],
            "s_tm": ["02:04:06", "12:30:45.999", "23:59:59.123456"],
        }
    )
    res = df.sql(
        """
        SELECT
          DATE(s_dt) AS dt1,
          DATETIME(s_dt) AS dt2,
          DATETIME(s_dtm1) AS dtm1,
          DATETIME(s_dtm2,'%d-%m-%Y %H:%M') AS dtm2,
          TIME(s_tm) AS tm
        FROM self
        """
    )
    assert res.schema == {
        "dt1": pl.Date,
        "dt2": pl.Datetime("us"),
        "dtm1": pl.Datetime("us"),
        "dtm2": pl.Datetime("us"),
        "tm": pl.Time,
    }
    assert res.rows() == [
        (
            date(2077, 10, 10),
            datetime(2077, 10, 10, 0, 0),
            datetime(1999, 12, 31, 10, 30, 45),
            datetime(1999, 12, 31, 10, 30),
            time(2, 4, 6),
        ),
        (
            date(1942, 1, 8),
            datetime(1942, 1, 8, 0, 0),
            datetime(2020, 6, 10, 0, 0),
            datetime(2020, 6, 10, 0, 0),
            time(12, 30, 45, 999000),
        ),
        (
            date(2000, 7, 5),
            datetime(2000, 7, 5, 0, 0),
            datetime(2022, 8, 7, 0, 1, 2, 654321),
            datetime(2022, 8, 7, 0, 1),
            time(23, 59, 59, 123456),
        ),
    ]

    for fn in ("DATE", "TIME", "DATETIME"):
        with pytest.raises(
            SQLSyntaxError,
            match=rf"{fn} expects 1-2 arguments \(found 3\)",
        ):
            pl.sql_expr(rf"{fn}(s,fmt,misc) AS xyz")


def test_temporal_typed_literals() -> None:
    res = pl.sql(
        """
        SELECT
          DATE '2020-12-30' AS dt,
          TIME '00:01:02' AS tm1,
          TIME '23:59:59.123456' AS tm2,
          TIMESTAMP '1930-01-01 12:30:00' AS dtm1,
          TIMESTAMP '2077-04-27T23:45:30.123456' AS dtm2
        FROM
          (VALUES (0)) tbl (x)
        """,
        eager=True,
    )
    assert res.to_dict(as_series=False) == {
        "dt": [date(2020, 12, 30)],
        "tm1": [time(0, 1, 2)],
        "tm2": [time(23, 59, 59, 123456)],
        "dtm1": [datetime(1930, 1, 1, 12, 30)],
        "dtm2": [datetime(2077, 4, 27, 23, 45, 30, 123456)],
    }


@pytest.mark.parametrize("fn", ["DATE", "TIME", "TIMESTAMP"])
def test_typed_literals_errors(fn: str) -> None:
    with pytest.raises(SQLSyntaxError, match=f"invalid {fn} literal '999'"):
        pl.sql_expr(f"{fn} '999'")


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
                match=rf"invalid temporal type precision \(expected 1-9, found {prec}\)",
            ):
                ctx.execute(f"SELECT ts::timestamp({prec}) FROM frame_data")

        with pytest.raises(
            SQLInterfaceError,
            match="sql parser error: Expected: literal int, found: - ",
        ):
            ctx.execute("SELECT ts::timestamp(-3) FROM frame_data")

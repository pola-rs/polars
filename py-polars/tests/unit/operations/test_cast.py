from datetime import date, datetime

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_utf8_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-01"]}).with_columns(
        **{"x1-date": pl.col("x1").cast(pl.Date)}
    )
    expected = pl.DataFrame({"x1-date": [date(2021, 1, 1)]})
    out = df.select(pl.col("x1-date"))
    assert_frame_equal(expected, out)


def test_invalid_utf8_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-aa"]})

    with pytest.raises(ComputeError):
        df.with_columns(**{"x1-date": pl.col("x1").cast(pl.Date)})


def test_utf8_datetime() -> None:
    df = pl.DataFrame(
        {"x1": ["2021-12-19T00:39:57", "2022-12-19T16:39:57"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns")),
            "x1-datetime-ms": pl.col("x1").cast(pl.Datetime(time_unit="ms")),
            "x1-datetime-us": pl.col("x1").cast(pl.Datetime(time_unit="us")),
        }
    )
    first_row = datetime(year=2021, month=12, day=19, hour=00, minute=39, second=57)
    second_row = datetime(year=2022, month=12, day=19, hour=16, minute=39, second=57)
    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [first_row, second_row],
            "x1-datetime-ms": [first_row, second_row],
            "x1-datetime-us": [first_row, second_row],
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns"),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms"),
        pl.col("x1-datetime-us").dt.cast_time_unit("us"),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )
    assert_frame_equal(expected, out)


def test_invalid_utf8_datetime() -> None:
    df = pl.DataFrame({"x1": ["2021-12-19 00:39:57", "2022-12-19 16:39:57"]})
    with pytest.raises(ComputeError):
        df.with_columns(
            **{"x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns"))}
        )


def test_utf8_datetime_timezone() -> None:
    ccs_tz = "America/Caracas"
    stg_tz = "America/Santiago"
    utc_tz = "UTC"
    df = pl.DataFrame(
        {"x1": ["1996-12-19T16:39:57 +00:00", "2022-12-19T00:39:57 +00:00"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(
                pl.Datetime(time_unit="ns", time_zone=ccs_tz)
            ),
            "x1-datetime-ms": pl.col("x1").cast(
                pl.Datetime(time_unit="ms", time_zone=stg_tz)
            ),
            "x1-datetime-us": pl.col("x1").cast(
                pl.Datetime(time_unit="us", time_zone=utc_tz)
            ),
        }
    )

    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [
                datetime(year=1996, month=12, day=19, hour=12, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=20, minute=39, second=57),
            ],
            "x1-datetime-ms": [
                datetime(year=1996, month=12, day=19, hour=13, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=21, minute=39, second=57),
            ],
            "x1-datetime-us": [
                datetime(year=1996, month=12, day=19, hour=16, minute=39, second=57),
                datetime(year=2022, month=12, day=19, hour=00, minute=39, second=57),
            ],
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns").dt.replace_time_zone(ccs_tz),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms").dt.replace_time_zone(stg_tz),
        pl.col("x1-datetime-us").dt.cast_time_unit("us").dt.replace_time_zone(utc_tz),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )

    assert_frame_equal(expected, out)

from __future__ import annotations

import io
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    PolarsInefficientMapWarning,
)
from polars.testing import (
    assert_frame_equal,
    assert_series_equal,
    assert_series_not_equal,
)
from tests.unit.conftest import DATETIME_DTYPES, TEMPORAL_DTYPES

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import (
        Ambiguous,
        PolarsTemporalType,
        TimeUnit,
    )
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


def test_fill_null() -> None:
    dtm = datetime.strptime("2021-01-01", "%Y-%m-%d")
    s = pl.Series("A", [dtm, None])

    for fill_val_datetime in (dtm, pl.lit(dtm)):
        out = s.fill_null(fill_val_datetime)

        assert out.null_count() == 0
        assert out.dt[0] == dtm
        assert out.dt[1] == dtm

    dt1 = date(2001, 1, 1)
    dt2 = date(2001, 1, 2)
    dt3 = date(2001, 1, 3)

    s = pl.Series("a", [dt1, dt2, dt3, None])
    dt_2 = date(2001, 1, 4)

    for fill_val_date in (dt_2, pl.lit(dt_2)):
        out = s.fill_null(fill_val_date)

        assert out.null_count() == 0
        assert out.dt[0] == dt1
        assert out.dt[1] == dt2
        assert out.dt[-1] == dt_2


def test_fill_null_temporal() -> None:
    # test filling nulls with temporal literals across cols that use various timeunits
    dtm = datetime.now()
    dtm_ms = dtm.replace(microsecond=(dtm.microsecond // 1000) * 1000)
    td = timedelta(days=7, seconds=45045)
    tm = dtm.time()
    dt = dtm.date()

    df = pl.DataFrame(
        [
            [dtm, dtm_ms, dtm, dtm, dt, tm, td, td, td, td],
            [None] * 10,
        ],
        schema=[
            ("a", pl.Datetime),
            ("b", pl.Datetime("ms")),
            ("c", pl.Datetime("us")),
            ("d", pl.Datetime("ns")),
            ("e", pl.Date),
            ("f", pl.Time),
            ("g", pl.Duration),
            ("h", pl.Duration("ms")),
            ("i", pl.Duration("us")),
            ("j", pl.Duration("ns")),
        ],
        orient="row",
    )

    # fill literals
    dtm_us_fill = dtm_ns_fill = datetime(2023, 12, 31, 23, 59, 59, 999999)
    dtm_ms_fill = datetime(2023, 12, 31, 23, 59, 59, 999000)
    td_us_fill = timedelta(days=7, seconds=45045, microseconds=123456)
    td_ms_fill = timedelta(days=7, seconds=45045, microseconds=123000)
    dt_fill = date(2023, 12, 31)
    tm_fill = time(23, 59, 59)

    # apply literals via fill_null
    ldf = df.lazy()
    for temporal_literal in (dtm_ns_fill, td_us_fill, dt_fill, tm_fill):
        ldf = ldf.fill_null(temporal_literal)

    # validate
    assert ldf.collect().rows() == [
        (dtm, dtm_ms, dtm, dtm, dt, tm, td, td, td, td),  # first row (no null values)
        (  # second row (was composed entirely of nulls, now filled-in with literals)
            dtm_us_fill,
            dtm_ms_fill,
            dtm_us_fill,
            dtm_ns_fill,
            dt_fill,
            tm_fill,
            td_us_fill,
            td_ms_fill,
            td_us_fill,
            td_us_fill,
        ),
    ]


def test_filter_date() -> None:
    dtcol = pl.col("date")
    df = pl.DataFrame(
        {"date": ["2020-01-02", "2020-01-03", "2020-01-04"], "index": [1, 2, 3]}
    ).with_columns(dtcol.str.strptime(pl.Date, "%Y-%m-%d"))
    assert df.rows() == [
        (date(2020, 1, 2), 1),
        (date(2020, 1, 3), 2),
        (date(2020, 1, 4), 3),
    ]

    # filter by datetime
    assert df.filter(dtcol <= pl.lit(datetime(2019, 1, 3))).is_empty()
    assert df.filter(dtcol < pl.lit(datetime(2020, 1, 4))).rows() == df.rows()[:2]
    assert df.filter(dtcol < pl.lit(datetime(2020, 1, 5))).rows() == df.rows()

    # filter by date
    assert df.filter(dtcol <= pl.lit(date(2019, 1, 3))).is_empty()
    assert df.filter(dtcol < pl.lit(date(2020, 1, 4))).rows() == df.rows()[:2]
    assert df.filter(dtcol < pl.lit(date(2020, 1, 5))).rows() == df.rows()


def test_filter_time() -> None:
    times = [time(8, 0), time(9, 0), time(10, 0)]
    df = pl.DataFrame({"t": times})

    assert df.filter(pl.col("t") <= pl.lit(time(7, 0))).is_empty()
    assert df.filter(pl.col("t") < pl.lit(time(11, 0))).rows() == [(t,) for t in times]
    assert df.filter(pl.col("t") < pl.lit(time(10, 0))).to_series().to_list() == [
        time(8, 0),
        time(9, 0),
    ]


def test_series_add_timedelta() -> None:
    dates = pl.Series(
        [datetime(2000, 1, 1), datetime(2027, 5, 19), datetime(2054, 10, 4)]
    )
    out = pl.Series(
        [datetime(2027, 5, 19), datetime(2054, 10, 4), datetime(2082, 2, 19)]
    )
    assert_series_equal((dates + timedelta(days=10_000)), out)


def test_series_add_datetime() -> None:
    deltas = pl.Series([timedelta(10_000), timedelta(20_000), timedelta(30_000)])
    out = pl.Series(
        [datetime(2027, 5, 19), datetime(2054, 10, 4), datetime(2082, 2, 19)]
    )
    assert_series_equal(deltas + pl.Series([datetime(2000, 1, 1)]), out)


def test_diff_datetime() -> None:
    df = pl.DataFrame(
        {
            "timestamp": ["2021-02-01", "2021-03-1", "2850-04-1"],
            "guild": [1, 2, 3],
            "char": ["a", "a", "b"],
        }
    )
    out = (
        df.with_columns(
            pl.col("timestamp").str.strptime(pl.Date, format="%Y-%m-%d"),
        ).with_columns(pl.col("timestamp").diff().over("char", mapping_strategy="join"))
    )["timestamp"]
    assert_series_equal(out[0], out[1])


def test_from_pydatetime() -> None:
    datetimes = [
        datetime(2021, 1, 1),
        datetime(2021, 1, 2),
        datetime(2021, 1, 3),
        datetime(2021, 1, 4, 12, 12),
        None,
    ]
    s = pl.Series("name", datetimes)
    assert s.dtype == pl.Datetime
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == datetimes[0]

    dates = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), None]
    s = pl.Series("name", dates)
    assert s.dtype == pl.Date
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]


def test_int_to_python_datetime() -> None:
    df = pl.DataFrame({"a": [100_000_000, 200_000_000]}).with_columns(
        pl.col("a").cast(pl.Datetime).alias("b"),
        pl.col("a").cast(pl.Datetime("ms")).alias("c"),
        pl.col("a").cast(pl.Datetime("us")).alias("d"),
        pl.col("a").cast(pl.Datetime("ns")).alias("e"),
    )
    assert df.rows() == [
        (
            100000000,
            datetime(1970, 1, 1, 0, 1, 40),
            datetime(1970, 1, 2, 3, 46, 40),
            datetime(1970, 1, 1, 0, 1, 40),
            datetime(1970, 1, 1, 0, 0, 0, 100000),
        ),
        (
            200000000,
            datetime(1970, 1, 1, 0, 3, 20),
            datetime(1970, 1, 3, 7, 33, 20),
            datetime(1970, 1, 1, 0, 3, 20),
            datetime(1970, 1, 1, 0, 0, 0, 200000),
        ),
    ]

    assert df.select(pl.col(col).dt.timestamp() for col in ("c", "d", "e")).rows() == [
        (100000000000, 100000000, 100000),
        (200000000000, 200000000, 200000),
    ]

    assert df.select(
        getattr(pl.col("a").cast(pl.Duration).dt, f"total_{unit}")().alias(f"u[{unit}]")
        for unit in ("milliseconds", "microseconds", "nanoseconds")
    ).rows() == [
        (100000, 100000000, 100000000000),
        (200000, 200000000, 200000000000),
    ]


def test_int_to_python_timedelta() -> None:
    df = pl.DataFrame({"a": [100_001, 200_002]}).with_columns(
        pl.col("a").cast(pl.Duration).alias("b"),
        pl.col("a").cast(pl.Duration("ms")).alias("c"),
        pl.col("a").cast(pl.Duration("us")).alias("d"),
        pl.col("a").cast(pl.Duration("ns")).alias("e"),
    )
    assert df.rows() == [
        (
            100001,
            timedelta(microseconds=100001),
            timedelta(seconds=100, microseconds=1000),
            timedelta(microseconds=100001),
            timedelta(microseconds=100),
        ),
        (
            200002,
            timedelta(microseconds=200002),
            timedelta(seconds=200, microseconds=2000),
            timedelta(microseconds=200002),
            timedelta(microseconds=200),
        ),
    ]

    assert df.select(pl.col(col).cast(pl.Int64) for col in ("c", "d", "e")).rows() == [
        (100001, 100001, 100001),
        (200002, 200002, 200002),
    ]


def test_datetime_consistency() -> None:
    dt = datetime(2022, 7, 5, 10, 30, 45, 123455)
    df = pl.DataFrame({"date": [dt]})

    assert df["date"].dt[0] == dt

    for date_literal in (
        dt,
        np.datetime64(dt, "us"),
        np.datetime64(dt, "ns"),
    ):
        assert df.select(pl.lit(date_literal))["literal"].dt[0] == dt
        assert df.filter(pl.col("date") == date_literal).rows() == [(dt,)]

    ddf = df.select(
        pl.col("date"),
        pl.lit(dt).alias("dt"),
        pl.lit(dt).cast(pl.Datetime("ms")).alias("dt_ms"),
        pl.lit(dt).cast(pl.Datetime("us")).alias("dt_us"),
        pl.lit(dt).cast(pl.Datetime("ns")).alias("dt_ns"),
    )
    assert ddf.schema == {
        "date": pl.Datetime("us"),
        "dt": pl.Datetime("us"),
        "dt_ms": pl.Datetime("ms"),
        "dt_us": pl.Datetime("us"),
        "dt_ns": pl.Datetime("ns"),
    }
    assert ddf.select([pl.col(c).cast(int) for c in ddf.schema]).rows() == [
        (
            1657017045123455,
            1657017045123455,
            1657017045123,
            1657017045123455,
            1657017045123455000,
        )
    ]

    test_data = [
        datetime(2000, 1, 1, 1, 1, 1, 555555),
        datetime(2514, 5, 30, 1, 53, 4, 986754),
        datetime(3099, 12, 31, 23, 59, 59, 123456),
        datetime(9999, 12, 31, 23, 59, 59, 999999),
    ]
    ddf = pl.DataFrame({"dtm": test_data}).with_columns(
        pl.col("dtm").dt.nanosecond().alias("ns")
    )
    assert ddf.rows() == [
        (test_data[0], 555555000),
        (test_data[1], 986754000),
        (test_data[2], 123456000),
        (test_data[3], 999999000),
    ]
    # Same as above, but for tz-aware
    test_data = [
        datetime(2000, 1, 1, 1, 1, 1, 555555, tzinfo=ZoneInfo("Asia/Kathmandu")),
        datetime(2514, 5, 30, 1, 53, 4, 986754, tzinfo=ZoneInfo("Asia/Kathmandu")),
        datetime(3099, 12, 31, 23, 59, 59, 123456, tzinfo=ZoneInfo("Asia/Kathmandu")),
        datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=ZoneInfo("Asia/Kathmandu")),
    ]
    ddf = pl.DataFrame({"dtm": test_data}).with_columns(
        pl.col("dtm").dt.nanosecond().alias("ns")
    )
    assert ddf.rows() == [
        (test_data[0], 555555000),
        (test_data[1], 986754000),
        (test_data[2], 123456000),
        (test_data[3], 999999000),
    ]
    # Similar to above, but check for no error when crossing DST
    test_data = [
        datetime(2021, 11, 7, 0, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 1, 0, fold=1, tzinfo=ZoneInfo("US/Central")),
        datetime(2021, 11, 7, 2, 0, tzinfo=ZoneInfo("US/Central")),
    ]
    ddf = pl.DataFrame({"dtm": test_data}).select(
        pl.col("dtm").dt.convert_time_zone("US/Central")
    )
    assert ddf.rows() == [
        (test_data[0],),
        (test_data[1],),
        (test_data[2],),
        (test_data[3],),
    ]


def test_timezone() -> None:
    ts = pa.timestamp("s")
    data = pa.array([1000, 2000], type=ts)
    s = cast(pl.Series, pl.from_arrow(data))

    tz_ts = pa.timestamp("s", tz="America/New_York")
    tz_data = pa.array([1000, 2000], type=tz_ts)
    tz_s = cast(pl.Series, pl.from_arrow(tz_data))

    # different timezones are not considered equal
    # we check both `null_equal=True` and `null_equal=False`
    # https://github.com/pola-rs/polars/issues/5023
    assert s.equals(tz_s, null_equal=False) is False
    assert s.equals(tz_s, null_equal=True) is False
    assert_series_not_equal(tz_s, s)
    assert_series_equal(s.cast(int), tz_s.cast(int))


def test_to_dicts() -> None:
    now = datetime.now()
    data = {
        "a": now,
        "b": now.date(),
        "c": now.time(),
        "d": timedelta(days=1, seconds=43200),
    }
    df = pl.DataFrame(
        data, schema_overrides={"a": pl.Datetime("ns"), "d": pl.Duration("ns")}
    )
    assert len(df) == 1

    d = df.to_dicts()[0]
    for col in data:
        assert d[col] == data[col]
        assert isinstance(d[col], type(data[col]))


def test_to_list() -> None:
    s = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)

    out = s.to_list()
    assert out[0] == date(2308, 4, 2)

    s = pl.Series("datetime", [a * 1_000_000 for a in [123543, 283478, 1243]]).cast(
        pl.Datetime
    )
    out = s.to_list()
    assert out[0] == datetime(1970, 1, 2, 10, 19, 3)


def test_rows() -> None:
    s0 = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)
    with pytest.deprecated_call(
        match="`ExprDateTimeNameSpace.with_time_unit` is deprecated"
    ):
        s1 = (
            pl.Series("datetime", [a * 1_000_000 for a in [123543, 283478, 1243]])
            .cast(pl.Datetime)
            .dt.with_time_unit("ns")
        )
    df = pl.DataFrame([s0, s1])

    rows = df.rows()
    assert rows[0][0] == date(2308, 4, 2)
    assert rows[0][1] == datetime(1970, 1, 1, 0, 2, 3, 543000)


@pytest.mark.parametrize(
    ("one", "two"),
    [
        (date(2001, 1, 1), date(2001, 1, 2)),
        (datetime(2001, 1, 1), datetime(2001, 1, 2)),
        (time(20, 10, 0), time(20, 10, 1)),
        # also test if the conversion stays correct with wide date ranges
        (date(201, 1, 1), date(201, 1, 2)),
        (date(5001, 1, 1), date(5001, 1, 2)),
    ],
)
def test_date_comp(one: PolarsTemporalType, two: PolarsTemporalType) -> None:
    a = pl.Series("a", [one, two])
    assert (a == one).to_list() == [True, False]
    assert (a == two).to_list() == [False, True]
    assert (a != one).to_list() == [False, True]
    assert (a > one).to_list() == [False, True]
    assert (a >= one).to_list() == [True, True]
    assert (a < one).to_list() == [False, False]
    assert (a <= one).to_list() == [True, False]


def test_datetime_comp_tz_aware() -> None:
    a = pl.Series(
        "a", [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    ).dt.replace_time_zone("Asia/Kathmandu")
    other = datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))
    result = a > other
    expected = pl.Series("a", [False, True])
    assert_series_equal(result, expected)


def test_datetime_comp_tz_aware_invalid() -> None:
    a = pl.Series(
        "a", [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    ).dt.replace_time_zone("Asia/Kathmandu")
    other = datetime(2020, 1, 1)
    with pytest.raises(
        TypeError,
        match="Datetime time zone None does not match Series timezone 'Asia/Kathmandu'",
    ):
        _ = a > other


def test_to_arrow() -> None:
    date_series = pl.Series("dates", ["2022-01-16", "2022-01-17"]).str.strptime(
        pl.Date, "%Y-%m-%d"
    )
    arr = date_series.to_arrow()
    assert arr.type == pa.date32()


def test_explode_date() -> None:
    datetimes = [
        datetime(2021, 12, 1, 0, 0),
        datetime(2021, 12, 1, 0, 0),
        datetime(2021, 12, 1, 0, 0),
        datetime(2021, 12, 1, 0, 0),
    ]
    dates = [
        date(2021, 12, 1),
        date(2021, 12, 1),
        date(2021, 12, 1),
        date(2021, 12, 1),
    ]
    for dclass, values in ((date, dates), (datetime, datetimes)):
        df = pl.DataFrame(
            {
                "a": values,
                "b": ["a", "b", "a", "b"],
                "c": [1.0, 2.0, 1.5, 2.5],
            }
        )
        out = (
            df.group_by("b", maintain_order=True)
            .agg([pl.col("a"), pl.col("c").pct_change()])
            .explode(["a", "c"])
        )
        assert out.shape == (4, 3)
        assert out.rows() == [
            ("a", dclass(2021, 12, 1), None),
            ("a", dclass(2021, 12, 1), 0.5),
            ("b", dclass(2021, 12, 1), None),
            ("b", dclass(2021, 12, 1), 0.25),
        ]


def test_microseconds_accuracy() -> None:
    timestamps = [
        datetime(2600, 1, 1, 0, 0, 0, 123456),
        datetime(2800, 1, 1, 0, 0, 0, 456789),
    ]
    a = pa.Table.from_arrays(
        arrays=[timestamps, [128, 256]],
        schema=pa.schema(
            [
                ("timestamp", pa.timestamp("us")),
                ("value", pa.int16()),
            ]
        ),
    )
    df = cast(pl.DataFrame, pl.from_arrow(a))
    assert df["timestamp"].to_list() == timestamps


def test_read_utc_times_parquet() -> None:
    df = pd.DataFrame(
        data={
            "Timestamp": pd.date_range(
                "2022-01-01T00:00+00:00", "2022-01-01T10:00+00:00", freq="h"
            )
        }
    )
    f = io.BytesIO()
    df.to_parquet(f)
    f.seek(0)
    df_in = pl.read_parquet(f)
    tz = ZoneInfo("UTC")
    assert df_in["Timestamp"][0] == datetime(2022, 1, 1, 0, 0, tzinfo=tz)


def test_asof_join_tolerance_grouper() -> None:
    from datetime import date

    df1 = pl.DataFrame({"date": [date(2020, 1, 5), date(2020, 1, 10)], "by": [1, 1]})
    df2 = pl.DataFrame(
        {
            "date": [date(2020, 1, 5), date(2020, 1, 6)],
            "by": [1, 1],
            "values": [100, 200],
        }
    )

    out = df1.join_asof(df2, by="by", on=pl.col("date").set_sorted(), tolerance="3d")

    expected = pl.DataFrame(
        {
            "date": [date(2020, 1, 5), date(2020, 1, 10)],
            "by": [1, 1],
            "date_right": [date(2020, 1, 5), None],
            "values": [100, None],
        }
    )

    assert_frame_equal(out, expected)


def test_rolling_mean_3020() -> None:
    df = pl.DataFrame(
        {
            "Date": [
                "1998-04-12",
                "1998-04-19",
                "1998-04-26",
                "1998-05-03",
                "1998-05-10",
                "1998-05-17",
                "1998-05-24",
            ],
            "val": range(7),
        }
    ).with_columns(pl.col("Date").str.strptime(pl.Date).set_sorted())

    period: str | timedelta
    for period in ("1w", timedelta(days=7)):
        result = df.rolling(index_column="Date", period=period).agg(
            pl.col("val").mean().alias("val_mean")
        )
        expected = pl.DataFrame(
            {
                "Date": [
                    date(1998, 4, 12),
                    date(1998, 4, 19),
                    date(1998, 4, 26),
                    date(1998, 5, 3),
                    date(1998, 5, 10),
                    date(1998, 5, 17),
                    date(1998, 5, 24),
                ],
                "val_mean": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        assert_frame_equal(result, expected)


def test_asof_join() -> None:
    format = "%F %T%.3f"
    dates = [
        "2016-05-25 13:30:00.023",
        "2016-05-25 13:30:00.023",
        "2016-05-25 13:30:00.030",
        "2016-05-25 13:30:00.041",
        "2016-05-25 13:30:00.048",
        "2016-05-25 13:30:00.049",
        "2016-05-25 13:30:00.072",
        "2016-05-25 13:30:00.075",
    ]
    ticker = [
        "GOOG",
        "MSFT",
        "MSFT",
        "MSFT",
        "GOOG",
        "AAPL",
        "GOOG",
        "MSFT",
    ]
    quotes = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, format=format),
            "ticker": ticker,
            "bid": [720.5, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        }
    ).set_sorted("dates")
    dates = [
        "2016-05-25 13:30:00.023",
        "2016-05-25 13:30:00.038",
        "2016-05-25 13:30:00.048",
        "2016-05-25 13:30:00.048",
        "2016-05-25 13:30:00.048",
    ]
    ticker = [
        "MSFT",
        "MSFT",
        "GOOG",
        "GOOG",
        "AAPL",
    ]
    trades = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, format=format),
            "ticker": ticker,
            "bid": [51.95, 51.95, 720.77, 720.92, 98.0],
        }
    ).set_sorted("dates")
    assert trades.schema == {
        "dates": pl.Datetime("ms"),
        "ticker": pl.String,
        "bid": pl.Float64,
    }
    out = trades.join_asof(quotes, on="dates", strategy="backward")

    assert out.schema == {
        "bid": pl.Float64,
        "bid_right": pl.Float64,
        "dates": pl.Datetime("ms"),
        "ticker": pl.String,
        "ticker_right": pl.String,
    }
    assert out.columns == ["dates", "ticker", "bid", "ticker_right", "bid_right"]
    assert (out["dates"].cast(int)).to_list() == [
        1464183000023,
        1464183000038,
        1464183000048,
        1464183000048,
        1464183000048,
    ]
    assert trades.join_asof(quotes, on="dates", strategy="forward")[
        "bid_right"
    ].to_list() == [720.5, 51.99, 720.5, 720.5, 720.5]

    out = trades.join_asof(quotes, on="dates", by="ticker")
    assert out["bid_right"].to_list() == [51.95, 51.97, 720.5, 720.5, None]

    out = quotes.join_asof(trades, on="dates", by="ticker")
    assert out["bid_right"].to_list() == [
        None,
        51.95,
        51.95,
        51.95,
        720.92,
        98.0,
        720.92,
        51.95,
    ]
    assert quotes.join_asof(trades, on="dates", strategy="backward", tolerance="5ms")[
        "bid_right"
    ].to_list() == [51.95, 51.95, None, 51.95, 98.0, 98.0, None, None]
    assert quotes.join_asof(trades, on="dates", strategy="forward", tolerance="5ms")[
        "bid_right"
    ].to_list() == [51.95, 51.95, None, None, 720.77, None, None, None]

    assert trades.join_asof(quotes, on="dates", strategy="nearest")[
        "bid_right"
    ].to_list() == [51.95, 51.99, 720.5, 720.5, 720.5]
    assert quotes.join_asof(trades, on="dates", strategy="nearest")[
        "bid_right"
    ].to_list() == [51.95, 51.95, 51.95, 51.95, 98.0, 98.0, 98.0, 98.0]

    assert trades.sort(by=["ticker", "dates"]).join_asof(
        quotes.sort(by=["ticker", "dates"]), on="dates", by="ticker", strategy="nearest"
    )["bid_right"].to_list() == [97.99, 720.5, 720.5, 51.95, 51.99]
    assert quotes.sort(by=["ticker", "dates"]).join_asof(
        trades.sort(by=["ticker", "dates"]), on="dates", by="ticker", strategy="nearest"
    )["bid_right"].to_list() == [
        98.0,
        720.92,
        720.92,
        720.92,
        51.95,
        51.95,
        51.95,
        51.95,
    ]


@pytest.mark.parametrize(
    ("skip_nulls", "expected_value"),
    [
        (True, None),
        (False, datetime(2010, 9, 12)),
    ],
)
def test_temporal_dtypes_map_elements(
    skip_nulls: bool, expected_value: datetime | None
) -> None:
    df = pl.DataFrame(
        {"timestamp": [1284286794000, None, 1234567890000]},
        schema=[("timestamp", pl.Datetime("ms"))],
    )
    const_dtm = datetime(2010, 9, 12)

    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Replace this expression.*lambda x:",
    ):
        result = df.with_columns(
            # don't actually do this; native expressions are MUCH faster ;)
            pl.col("timestamp")
            .map_elements(
                lambda x: const_dtm,
                skip_nulls=skip_nulls,
                return_dtype=pl.Datetime,
            )
            .alias("const_dtm"),
            # note: the below now trigger a PolarsInefficientMapWarning
            pl.col("timestamp")
            .map_elements(
                lambda x: x and x.date(),
                skip_nulls=skip_nulls,
                return_dtype=pl.Date,
            )
            .alias("date"),
            pl.col("timestamp")
            .map_elements(
                lambda x: x and x.time(),
                skip_nulls=skip_nulls,
                return_dtype=pl.Time,
            )
            .alias("time"),
        )
    expected = pl.DataFrame(
        [
            (
                datetime(2010, 9, 12, 10, 19, 54),
                datetime(2010, 9, 12, 0, 0),
                date(2010, 9, 12),
                time(10, 19, 54),
            ),
            (None, expected_value, None, None),
            (
                datetime(2009, 2, 13, 23, 31, 30),
                datetime(2010, 9, 12, 0, 0),
                date(2009, 2, 13),
                time(23, 31, 30),
            ),
        ],
        schema={
            "timestamp": pl.Datetime("ms"),
            "const_dtm": pl.Datetime("us"),
            "date": pl.Date,
            "time": pl.Time,
        },
        orient="row",
    )
    assert_frame_equal(result, expected)


def test_timelike_init() -> None:
    durations = [timedelta(days=1), timedelta(days=2)]
    dates = [date(2022, 1, 1), date(2022, 1, 2)]
    datetimes = [datetime(2022, 1, 1), datetime(2022, 1, 2)]

    for ts in [durations, dates, datetimes]:
        s = pl.Series(ts)
        assert s.to_list() == ts


def test_timedelta_timeunit_init() -> None:
    td_us = timedelta(days=7, seconds=45045, microseconds=123456)
    td_ms = timedelta(days=7, seconds=45045, microseconds=123000)

    df = pl.DataFrame(
        [[td_us, td_us, td_us]],
        schema=[
            ("x", pl.Duration("ms")),
            ("y", pl.Duration("us")),
            ("z", pl.Duration("ns")),
        ],
        orient="row",
    )
    assert df.rows() == [(td_ms, td_us, td_us)]


def test_duration_filter() -> None:
    df = pl.DataFrame(
        {
            "start_date": [date(2022, 1, 1), date(2022, 1, 1), date(2022, 1, 1)],
            "end_date": [date(2022, 1, 7), date(2022, 2, 20), date(2023, 1, 1)],
        }
    ).with_columns((pl.col("end_date") - pl.col("start_date")).alias("time_passed"))

    assert df.filter(pl.col("time_passed") < timedelta(days=30)).rows() == [
        (date(2022, 1, 1), date(2022, 1, 7), timedelta(days=6))
    ]
    assert df.filter(pl.col("time_passed") >= timedelta(days=30)).rows() == [
        (date(2022, 1, 1), date(2022, 2, 20), timedelta(days=50)),
        (date(2022, 1, 1), date(2023, 1, 1), timedelta(days=365)),
    ]


def test_agg_logical() -> None:
    dates = [date(2001, 1, 1), date(2002, 1, 1)]
    s = pl.Series(dates)
    assert s.max() == dates[1]
    assert s.min() == dates[0]


def test_from_time_arrow() -> None:
    pa_times = pa.table([pa.array([10, 20, 30], type=pa.time32("s"))], names=["times"])

    result: pl.DataFrame = pl.from_arrow(pa_times)  # type: ignore[assignment]

    assert result.to_series().to_list() == [
        time(0, 0, 10),
        time(0, 0, 20),
        time(0, 0, 30),
    ]
    assert result.rows() == [
        (time(0, 0, 10),),
        (time(0, 0, 20),),
        (time(0, 0, 30),),
    ]


def test_timedelta_from() -> None:
    as_dict = {
        "A": [1, 2],
        "B": [timedelta(seconds=4633), timedelta(seconds=50)],
    }
    as_rows = [
        {
            "A": 1,
            "B": timedelta(seconds=4633),
        },
        {
            "A": 2,
            "B": timedelta(seconds=50),
        },
    ]
    assert_frame_equal(pl.DataFrame(as_dict), pl.DataFrame(as_rows))


def test_duration_aggregations() -> None:
    df = pl.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "start": [
                datetime(2022, 1, 1),
                datetime(2022, 1, 2),
                datetime(2022, 1, 3),
                datetime(2022, 1, 4),
            ],
            "end": [
                datetime(2022, 1, 2),
                datetime(2022, 1, 4),
                datetime(2022, 1, 6),
                datetime(2022, 1, 6),
            ],
        }
    )
    df = df.with_columns((pl.col("end") - pl.col("start")).alias("duration"))
    assert df.group_by("group", maintain_order=True).agg(
        [
            pl.col("duration").mean().alias("mean"),
            pl.col("duration").sum().alias("sum"),
            pl.col("duration").min().alias("min"),
            pl.col("duration").max().alias("max"),
            pl.col("duration").quantile(0.1).alias("quantile"),
            pl.col("duration").median().alias("median"),
            pl.col("duration").alias("list"),
        ]
    ).to_dict(as_series=False) == {
        "group": ["A", "B"],
        "mean": [timedelta(days=2), timedelta(days=2)],
        "sum": [timedelta(days=4), timedelta(days=4)],
        "min": [timedelta(days=1), timedelta(days=2)],
        "max": [timedelta(days=3), timedelta(days=2)],
        "quantile": [timedelta(days=1), timedelta(days=2)],
        "median": [timedelta(days=2), timedelta(days=2)],
        "list": [
            [timedelta(days=1), timedelta(days=3)],
            [timedelta(days=2), timedelta(days=2)],
        ],
    }


def test_datetime_units() -> None:
    df = pl.DataFrame(
        {
            "ns": pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2020, 5, 1),
                "1mo",
                time_unit="ns",
                eager=True,
            ),
            "us": pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2020, 5, 1),
                "1mo",
                time_unit="us",
                eager=True,
            ),
            "ms": pl.datetime_range(
                datetime(2020, 1, 1),
                datetime(2020, 5, 1),
                "1mo",
                time_unit="ms",
                eager=True,
            ),
        }
    )
    names = set(df.columns)

    for unit in DTYPE_TEMPORAL_UNITS:
        subset = names - {unit}

        assert (
            len(set(df.select([pl.all().exclude(pl.Datetime(unit))]).columns) - subset)
            == 0
        )


def test_datetime_instance_selection() -> None:
    test_data = {
        "ns": [datetime(2022, 12, 31, 1, 2, 3)],
        "us": [datetime(2022, 12, 31, 4, 5, 6)],
        "ms": [datetime(2022, 12, 31, 7, 8, 9)],
    }
    df = pl.DataFrame(
        data=test_data,
        schema=[
            ("ns", pl.Datetime("ns")),
            ("us", pl.Datetime("us")),
            ("ms", pl.Datetime("ms")),
        ],
    )
    for time_unit in DTYPE_TEMPORAL_UNITS:
        res = df.select(pl.col([pl.Datetime(time_unit)])).dtypes
        assert res == [pl.Datetime(time_unit)]
        assert len(df.filter(pl.col(time_unit) == test_data[time_unit][0])) == 1

    assert list(df.select(pl.exclude(DATETIME_DTYPES))) == []


def test_sum_duration() -> None:
    assert pl.DataFrame(
        [
            {"name": "Jen", "duration": timedelta(seconds=60)},
            {"name": "Mike", "duration": timedelta(seconds=30)},
            {"name": "Jen", "duration": timedelta(seconds=60)},
        ]
    ).select(
        pl.col("duration").sum(),
        pl.col("duration").dt.total_seconds().alias("sec").sum(),
    ).to_dict(as_series=False) == {
        "duration": [timedelta(seconds=150)],
        "sec": [150],
    }


def test_supertype_timezones_4174() -> None:
    df = pl.DataFrame(
        {
            "dt": pl.datetime_range(
                datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo", eager=True
            ),
        }
    ).with_columns(
        pl.col("dt").dt.replace_time_zone("Europe/London").name.suffix("_London")
    )

    # test if this runs without error
    date_to_fill = df["dt_London"][0]
    df.with_columns(df["dt_London"].shift(fill_value=date_to_fill))


def test_from_dict_tu_consistency() -> None:
    tz = ZoneInfo("PRC")
    dt = datetime(2020, 8, 1, 12, 0, 0, tzinfo=tz)
    from_dict = pl.from_dict({"dt": [dt]})
    from_dicts = pl.from_dicts([{"dt": dt}])

    assert from_dict.dtypes == from_dicts.dtypes


def test_date_arr_concat() -> None:
    expected = {"d": [[date(2000, 1, 1), date(2000, 1, 1)]]}

    # type date
    df = pl.DataFrame({"d": [date(2000, 1, 1)]})
    assert (
        df.select(pl.col("d").list.concat(pl.col("d"))).to_dict(as_series=False)
        == expected
    )
    # type list[date]
    df = pl.DataFrame({"d": [[date(2000, 1, 1)]]})
    assert (
        df.select(pl.col("d").list.concat(pl.col("d"))).to_dict(as_series=False)
        == expected
    )


def test_date_timedelta() -> None:
    df = pl.DataFrame(
        {"date": pl.date_range(date(2001, 1, 1), date(2001, 1, 3), "1d", eager=True)}
    )
    assert df.with_columns(
        (pl.col("date") + timedelta(days=1)).alias("date_plus_one"),
        (pl.col("date") - timedelta(days=1)).alias("date_min_one"),
    ).to_dict(as_series=False) == {
        "date": [date(2001, 1, 1), date(2001, 1, 2), date(2001, 1, 3)],
        "date_plus_one": [date(2001, 1, 2), date(2001, 1, 3), date(2001, 1, 4)],
        "date_min_one": [date(2000, 12, 31), date(2001, 1, 1), date(2001, 1, 2)],
    }


def test_datetime_hashes() -> None:
    dtypes = (
        pl.Datetime,
        pl.Datetime("us"),
        pl.Datetime("us", "UTC"),
        pl.Datetime("us", "Zulu"),
    )
    assert len({hash(tp) for tp in (dtypes)}) == 4


def test_datetime_string_casts() -> None:
    df = pl.DataFrame(
        {
            "x": [1661855445123],
            "y": [1661855445123456],
            "z": [1661855445123456789],
        },
        schema=[
            ("x", pl.Datetime("ms")),
            ("y", pl.Datetime("us")),
            ("z", pl.Datetime("ns")),
        ],
    )
    assert df.select(
        [pl.col("x").dt.to_string("%F %T").alias("w")]
        + [pl.col(d).cast(str) for d in df.columns]
    ).rows() == [
        (
            "2022-08-30 10:30:45",
            "2022-08-30 10:30:45.123",
            "2022-08-30 10:30:45.123456",
            "2022-08-30 10:30:45.123456789",
        )
    ]


def test_iso_year() -> None:
    assert pl.Series([datetime(2022, 1, 1, 7, 8, 40)]).dt.iso_year()[0] == 2021
    assert pl.Series([date(2022, 1, 1)]).dt.iso_year()[0] == 2021


def test_replace_time_zone() -> None:
    ny = ZoneInfo("America/New_York")
    assert pl.DataFrame({"a": [datetime(2022, 9, 25, 14)]}).with_columns(
        pl.col("a").dt.replace_time_zone("America/New_York").alias("b")
    ).to_dict(as_series=False) == {
        "a": [datetime(2022, 9, 25, 14, 0)],
        "b": [datetime(2022, 9, 25, 14, 0, tzinfo=ny)],
    }


def test_replace_time_zone_non_existent_null() -> None:
    result = (
        pl.Series(["2021-03-28 02:30", "2021-03-28 03:30"])
        .str.to_datetime()
        .dt.replace_time_zone("Europe/Warsaw", non_existent="null")
    )
    expected = pl.Series([None, datetime(2021, 3, 28, 3, 30)]).dt.replace_time_zone(
        "Europe/Warsaw"
    )
    assert_series_equal(result, expected)


def test_invalid_non_existent() -> None:
    with pytest.raises(
        ValueError, match="`non_existent` must be one of {'null', 'raise'}, got cabbage"
    ):
        (
            pl.Series([datetime(2020, 1, 1)]).dt.replace_time_zone(
                "Europe/Warsaw",
                non_existent="cabbage",  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize(
    ("to_tz", "tzinfo"),
    [
        ("America/Barbados", ZoneInfo("America/Barbados")),
        (None, None),
    ],
)
@pytest.mark.parametrize("from_tz", ["Asia/Seoul", None])
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_replace_time_zone_from_to(
    from_tz: str,
    to_tz: str,
    tzinfo: timezone | ZoneInfo,
    time_unit: TimeUnit,
) -> None:
    ts = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime(time_unit))
    result = ts.dt.replace_time_zone(from_tz).dt.replace_time_zone(to_tz).item()
    expected = datetime(2020, 1, 1, 0, 0, tzinfo=tzinfo)
    assert result == expected


def test_strptime_with_tz() -> None:
    result = (
        pl.Series(["2020-01-01 03:00:00"])
        .str.strptime(pl.Datetime("us", "Africa/Monrovia"))
        .item()
    )
    assert result == datetime(2020, 1, 1, 3, tzinfo=ZoneInfo("Africa/Monrovia"))


@pytest.mark.parametrize(
    ("time_unit", "time_zone"),
    [
        ("us", "Europe/London"),
        ("ms", None),
        ("ns", "Africa/Lagos"),
    ],
)
def test_strptime_empty(time_unit: TimeUnit, time_zone: str | None) -> None:
    ts = (
        pl.Series([None])
        .cast(pl.String)
        .str.strptime(pl.Datetime(time_unit, time_zone))
    )
    assert ts.dtype == pl.Datetime(time_unit, time_zone)


def test_strptime_with_invalid_tz() -> None:
    with pytest.raises(ComputeError, match="unable to parse time zone: 'foo'"):
        pl.Series(["2020-01-01 03:00:00"]).str.strptime(pl.Datetime("us", "foo"))
    with pytest.raises(
        ComputeError,
        match="unable to parse time zone: 'foo'",
    ):
        pl.Series(["2020-01-01 03:00:00+01:00"]).str.strptime(
            pl.Datetime("us", "foo"), "%Y-%m-%d %H:%M:%S%z"
        )


def test_strptime_unguessable_format() -> None:
    with pytest.raises(
        ComputeError,
        match="could not find an appropriate format to parse dates, please define a format",
    ):
        pl.Series(["foobar"]).str.strptime(pl.Datetime)


def test_convert_time_zone_invalid() -> None:
    ts = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime)
    with pytest.raises(ComputeError, match="unable to parse time zone: 'foo'"):
        ts.dt.replace_time_zone("UTC").dt.convert_time_zone("foo")


def test_convert_time_zone_lazy_schema() -> None:
    ts_us = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime("us", "UTC"))
    ts_ms = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime("ms", "UTC"))
    ldf = pl.DataFrame({"ts_us": ts_us, "ts_ms": ts_ms}).lazy()
    result = ldf.with_columns(
        pl.col("ts_us").dt.convert_time_zone("America/New_York").alias("ts_us_ny"),
        pl.col("ts_ms").dt.convert_time_zone("America/New_York").alias("ts_us_kt"),
    ).collect_schema()
    expected = {
        "ts_us": pl.Datetime("us", "UTC"),
        "ts_ms": pl.Datetime("ms", "UTC"),
        "ts_us_ny": pl.Datetime("us", "America/New_York"),
        "ts_us_kt": pl.Datetime("ms", "America/New_York"),
    }
    assert result == expected


def test_convert_time_zone_on_tz_naive() -> None:
    ts = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime)
    result = ts.dt.convert_time_zone("Asia/Kathmandu").item()
    expected = datetime(2020, 1, 1, 5, 45, tzinfo=ZoneInfo("Asia/Kathmandu"))
    assert result == expected
    result = (
        ts.dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Kathmandu").item()
    )
    assert result == expected


def test_tz_aware_get_idx_5010() -> None:
    when = int(datetime(2022, 1, 1, 12, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
    a = pa.array([when]).cast(pa.timestamp("s", tz="Asia/Shanghai"))
    assert int(pl.from_arrow(a)[0].timestamp()) == when  # type: ignore[union-attr]


def test_tz_datetime_duration_arithm_5221() -> None:
    run_datetimes = [
        datetime.fromisoformat("2022-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2022-01-02T00:00:00+00:00"),
    ]
    out = pl.DataFrame(
        data={"run_datetime": run_datetimes},
        schema=[("run_datetime", pl.Datetime(time_zone="UTC"))],
    )
    out = out.with_columns(pl.col("run_datetime") + pl.duration(days=1))
    utc = ZoneInfo("UTC")
    assert out.to_dict(as_series=False) == {
        "run_datetime": [
            datetime(2022, 1, 2, 0, 0, tzinfo=utc),
            datetime(2022, 1, 3, 0, 0, tzinfo=utc),
        ]
    }


def test_auto_infer_time_zone() -> None:
    dt = datetime(2022, 10, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    s = pl.Series([dt])
    assert s.dtype == pl.Datetime("us", "UTC")
    assert s[0] == dt


def test_logical_nested_take() -> None:
    frame = pl.DataFrame(
        {
            "ix": [2, 1],
            "dt": [[datetime(2001, 1, 1)], [datetime(2001, 1, 2)]],
            "d": [[date(2001, 1, 1)], [date(2001, 1, 1)]],
            "t": [[time(10)], [time(10)]],
            "del": [[timedelta(10)], [timedelta(10)]],
            "str": [[{"a": time(10)}], [{"a": time(10)}]],
        }
    )
    out = frame.sort(by="ix")

    assert out.dtypes[:-1] == [
        pl.Int64,
        pl.List(pl.Datetime("us")),
        pl.List(pl.Date),
        pl.List(pl.Time),
        pl.List(pl.Duration("us")),
    ]
    assert out.to_dict(as_series=False) == {
        "ix": [1, 2],
        "dt": [[datetime(2001, 1, 2, 0, 0)], [datetime(2001, 1, 1, 0, 0)]],
        "d": [[date(2001, 1, 1)], [date(2001, 1, 1)]],
        "t": [[time(10, 0)], [time(10, 0)]],
        "del": [[timedelta(days=10)], [timedelta(days=10)]],
        "str": [[{"a": time(10, 0)}], [{"a": time(10, 0)}]],
    }


def test_replace_time_zone_from_naive() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(["2022-01-01", "2022-01-02"]).str.strptime(
                pl.Date, "%Y-%m-%d"
            )
        }
    )

    assert df.select(
        pl.col("date").cast(pl.Datetime).dt.replace_time_zone("America/New_York")
    ).to_dict(as_series=False) == {
        "date": [
            datetime(2022, 1, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 1, 2, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        ]
    }


@pytest.mark.parametrize(
    ("ambiguous", "expected"),
    [
        (
            "latest",
            datetime(2018, 10, 28, 2, 30, fold=0, tzinfo=ZoneInfo("Europe/Brussels")),
        ),
        (
            "earliest",
            datetime(2018, 10, 28, 2, 30, fold=1, tzinfo=ZoneInfo("Europe/Brussels")),
        ),
    ],
)
def test_replace_time_zone_ambiguous_with_ambiguous(
    ambiguous: Ambiguous, expected: datetime
) -> None:
    ts = pl.Series(["2018-10-28 02:30:00"]).str.strptime(pl.Datetime)
    result = ts.dt.replace_time_zone("Europe/Brussels", ambiguous=ambiguous).item()
    assert result == expected


def test_replace_time_zone_ambiguous_raises() -> None:
    ts = pl.Series(["2018-10-28 02:30:00"]).str.strptime(pl.Datetime)
    with pytest.raises(
        ComputeError,
        match="Please use `ambiguous` to tell how it should be localized",
    ):
        ts.dt.replace_time_zone("Europe/Brussels")


@pytest.mark.parametrize(
    ("from_tz", "expected_sortedness", "ambiguous"),
    [
        ("Europe/London", False, "earliest"),
        ("Europe/London", False, "raise"),
        ("UTC", True, "earliest"),
        ("UTC", True, "raise"),
        (None, True, "earliest"),
        (None, True, "raise"),
    ],
)
def test_replace_time_zone_sortedness_series(
    from_tz: str | None, expected_sortedness: bool, ambiguous: Ambiguous
) -> None:
    ser = (
        pl.Series("ts", [1603584000000000, 1603587600000000])
        .cast(pl.Datetime("us", from_tz))
        .sort()
    )
    assert ser.flags["SORTED_ASC"]
    result = ser.dt.replace_time_zone("UTC", ambiguous=ambiguous)
    assert result.flags["SORTED_ASC"] == expected_sortedness


@pytest.mark.parametrize(
    ("from_tz", "expected_sortedness", "ambiguous"),
    [
        ("Europe/London", False, "earliest"),
        ("Europe/London", False, "raise"),
        ("UTC", True, "earliest"),
        ("UTC", True, "raise"),
        (None, True, "earliest"),
        (None, True, "raise"),
    ],
)
def test_replace_time_zone_sortedness_expressions(
    from_tz: str | None, expected_sortedness: bool, ambiguous: str
) -> None:
    df = (
        pl.Series("ts", [1603584000000000, 1603587600000000])
        .cast(pl.Datetime("us", from_tz))
        .sort()
        .to_frame()
    )
    df = df.with_columns(ambiguous=pl.Series([ambiguous] * 2))
    assert df["ts"].flags["SORTED_ASC"]
    result = df.select(
        pl.col("ts").dt.replace_time_zone("UTC", ambiguous=pl.col("ambiguous"))
    )
    assert result["ts"].flags["SORTED_ASC"] == expected_sortedness


def test_invalid_ambiguous_value_in_expression() -> None:
    df = pl.DataFrame(
        {"a": [datetime(2020, 10, 25, 1)] * 2, "b": ["earliest", "cabbage"]}
    )
    with pytest.raises(InvalidOperationError, match="Invalid argument cabbage"):
        df.select(
            pl.col("a").dt.replace_time_zone("Europe/London", ambiguous=pl.col("b"))
        )


def test_replace_time_zone_ambiguous_null() -> None:
    df = pl.DataFrame(
        {
            "a": [datetime(2020, 10, 25, 1)] * 3 + [None],
            "b": ["earliest", "latest", "null", "raise"],
        }
    )
    # expression containing 'null'
    result = df.select(
        pl.col("a").dt.replace_time_zone("Europe/London", ambiguous=pl.col("b"))
    )["a"]
    expected = [
        datetime(2020, 10, 25, 1, fold=0, tzinfo=ZoneInfo("Europe/London")),
        datetime(2020, 10, 25, 1, fold=1, tzinfo=ZoneInfo("Europe/London")),
        None,
        None,
    ]
    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert result[2] == expected[2]
    assert result[3] == expected[3]

    # single 'null' value
    result = df.select(
        pl.col("a").dt.replace_time_zone("Europe/London", ambiguous="null")
    )["a"]
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None
    assert result[3] is None


def test_ambiguous_expressions() -> None:
    # strptime
    df = pl.DataFrame(
        {
            "ts": ["2020-10-25 01:00"] * 2,
            "ambiguous": ["earliest", "latest"],
        }
    )
    result = df.select(
        pl.col("ts").str.strptime(
            pl.Datetime("us", "Europe/London"), ambiguous=pl.col("ambiguous")
        )
    )["ts"]
    expected = pl.Series("ts", [1603584000000000, 1603587600000000]).cast(
        pl.Datetime("us", "Europe/London")
    )
    assert_series_equal(result, expected)

    # truncate
    df = pl.DataFrame(
        {
            "ts": [datetime(2020, 10, 25, 1), datetime(2020, 10, 25, 1)],
            "ambiguous": ["earliest", "latest"],
        }
    )
    df = df.with_columns(
        pl.col("ts").dt.replace_time_zone(
            "Europe/London", ambiguous=pl.col("ambiguous")
        )
    )
    result = df.select(pl.col("ts").dt.truncate("1h"))["ts"]
    expected = pl.Series("ts", [1603584000000000, 1603587600000000]).cast(
        pl.Datetime("us", "Europe/London")
    )
    assert_series_equal(result, expected)

    # replace_time_zone
    df = pl.DataFrame(
        {
            "ts": [datetime(2020, 10, 25, 1), datetime(2020, 10, 25, 1)],
            "ambiguous": ["earliest", "latest"],
        }
    )
    result = df.select(
        pl.col("ts").dt.replace_time_zone(
            "Europe/London", ambiguous=pl.col("ambiguous")
        )
    )["ts"]
    expected = pl.Series("ts", [1603584000000000, 1603587600000000]).cast(
        pl.Datetime("us", "Europe/London")
    )
    assert_series_equal(result, expected)

    # pl.datetime
    df = pl.DataFrame(
        {
            "year": [2020] * 2,
            "month": [10] * 2,
            "day": [25] * 2,
            "hour": [1] * 2,
            "minute": [0] * 2,
            "ambiguous": ["earliest", "latest"],
        }
    )
    result = df.select(
        pl.datetime(
            "year",
            "month",
            "day",
            "hour",
            "minute",
            time_zone="Europe/London",
            ambiguous=pl.col("ambiguous"),
        )
    )["datetime"]
    expected = pl.DataFrame(
        {"datetime": [1603584000000000, 1603587600000000]},
        schema={"datetime": pl.Datetime("us", "Europe/London")},
    )["datetime"]
    assert_series_equal(result, expected)


def test_single_ambiguous_null() -> None:
    df = pl.DataFrame(
        {"ts": [datetime(2020, 10, 2, 1, 1)], "ambiguous": [None]},
        schema_overrides={"ambiguous": pl.String},
    )
    result = df.select(
        pl.col("ts").dt.replace_time_zone(
            "Europe/London", ambiguous=pl.col("ambiguous")
        )
    )["ts"].item()
    assert result is None


def test_unlocalize() -> None:
    tz_naive = pl.Series(["2020-01-01 03:00:00"]).str.strptime(pl.Datetime)
    tz_aware = tz_naive.dt.replace_time_zone("UTC").dt.convert_time_zone(
        "Europe/Brussels"
    )
    result = tz_aware.dt.replace_time_zone(None).item()
    assert result == datetime(2020, 1, 1, 4)


def test_tz_aware_truncate() -> None:
    df = pl.DataFrame(
        {
            "dt": pl.datetime_range(
                start=datetime(2022, 11, 1),
                end=datetime(2022, 11, 4),
                interval="12h",
                eager=True,
            ).dt.replace_time_zone("America/New_York")
        }
    )
    result = df.with_columns(pl.col("dt").dt.truncate("1d").alias("trunced"))
    expected = {
        "dt": [
            datetime(2022, 11, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 1, 12, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 2, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 2, 12, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 3, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 3, 12, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 4, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        ],
        "trunced": [
            datetime(2022, 11, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 2, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 2, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 3, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 3, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 4, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        ],
    }
    assert result.to_dict(as_series=False) == expected

    # 5507
    lf = pl.DataFrame(
        {
            "naive": pl.datetime_range(
                start=datetime(2021, 12, 31, 23),
                end=datetime(2022, 1, 1, 6),
                interval="1h",
                eager=True,
            )
        }
    ).lazy()
    lf = lf.with_columns(pl.col("naive").dt.replace_time_zone("UTC").alias("UTC"))
    lf = lf.with_columns(pl.col("UTC").dt.convert_time_zone("US/Central").alias("CST"))
    lf = lf.with_columns(pl.col("CST").dt.truncate("1d").alias("CST truncated"))
    assert lf.collect().to_dict(as_series=False) == {
        "naive": [
            datetime(2021, 12, 31, 23, 0),
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 1, 1, 1, 0),
            datetime(2022, 1, 1, 2, 0),
            datetime(2022, 1, 1, 3, 0),
            datetime(2022, 1, 1, 4, 0),
            datetime(2022, 1, 1, 5, 0),
            datetime(2022, 1, 1, 6, 0),
        ],
        "UTC": [
            datetime(2021, 12, 31, 23, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 0, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 1, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 2, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 3, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 4, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 5, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2022, 1, 1, 6, 0, tzinfo=ZoneInfo("UTC")),
        ],
        "CST": [
            datetime(2021, 12, 31, 17, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 18, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 19, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 20, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 21, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 22, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 23, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2022, 1, 1, 0, 0, tzinfo=ZoneInfo("US/Central")),
        ],
        "CST truncated": [
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=ZoneInfo("US/Central")),
            datetime(2022, 1, 1, 0, 0, tzinfo=ZoneInfo("US/Central")),
        ],
    }


def test_to_string_invalid_format() -> None:
    tz_naive = pl.Series(["2020-01-01"]).str.strptime(pl.Datetime)
    with pytest.raises(
        ComputeError, match="cannot format timezone-naive Datetime with format '%z'"
    ):
        tz_naive.dt.to_string("%z")
    with pytest.raises(
        ComputeError, match="cannot format timezone-aware Datetime with format '%q'"
    ):
        tz_naive.dt.replace_time_zone("UTC").dt.to_string("%q")
    with pytest.raises(ComputeError, match="cannot format Date with format '%q'"):
        tz_naive.dt.date().dt.to_string("%q")


def test_tz_aware_to_string() -> None:
    df = pl.DataFrame(
        {
            "dt": pl.datetime_range(
                start=datetime(2022, 11, 1),
                end=datetime(2022, 11, 4),
                interval="24h",
                eager=True,
            ).dt.replace_time_zone("America/New_York")
        }
    )
    result = df.with_columns(pl.col("dt").dt.to_string("%c").alias("fmt"))
    expected = {
        "dt": [
            datetime(2022, 11, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 2, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 3, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2022, 11, 4, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        ],
        "fmt": [
            "Tue Nov  1 00:00:00 2022",
            "Wed Nov  2 00:00:00 2022",
            "Thu Nov  3 00:00:00 2022",
            "Fri Nov  4 00:00:00 2022",
        ],
    }
    assert result.to_dict(as_series=False) == expected


@pytest.mark.parametrize(
    ("time_zone", "directive", "expected"),
    [
        ("Pacific/Pohnpei", "%z", "+1100"),
        ("Pacific/Pohnpei", "%Z", "+11"),
    ],
)
def test_tz_aware_with_timezone_directive(
    time_zone: str, directive: str, expected: str
) -> None:
    tz_naive = pl.Series(["2020-01-01 03:00:00"]).str.strptime(pl.Datetime)
    tz_aware = tz_naive.dt.replace_time_zone(time_zone)
    result = tz_aware.dt.to_string(directive).item()
    assert result == expected


def test_local_time_zone_name() -> None:
    ser = pl.Series(["2020-01-01 03:00ACST"]).str.strptime(
        pl.Datetime, "%Y-%m-%d %H:%M%Z"
    )
    result = ser[0]
    expected = datetime(2020, 1, 1, 3)
    assert result == expected


def test_tz_aware_filter_lit() -> None:
    start = datetime(1970, 1, 1)
    stop = datetime(1970, 1, 1, 7)
    dt = datetime(1970, 1, 1, 6, tzinfo=ZoneInfo("America/New_York"))

    assert (
        pl.DataFrame({"date": pl.datetime_range(start, stop, "1h", eager=True)})
        .with_columns(
            pl.col("date").dt.replace_time_zone("America/New_York").alias("nyc")
        )
        .filter(pl.col("nyc") < dt)
    ).to_dict(as_series=False) == {
        "date": [
            datetime(1970, 1, 1, 0, 0),
            datetime(1970, 1, 1, 1, 0),
            datetime(1970, 1, 1, 2, 0),
            datetime(1970, 1, 1, 3, 0),
            datetime(1970, 1, 1, 4, 0),
            datetime(1970, 1, 1, 5, 0),
        ],
        "nyc": [
            datetime(1970, 1, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(1970, 1, 1, 1, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(1970, 1, 1, 2, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(1970, 1, 1, 3, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(1970, 1, 1, 4, 0, tzinfo=ZoneInfo("America/New_York")),
            datetime(1970, 1, 1, 5, 0, tzinfo=ZoneInfo("America/New_York")),
        ],
    }


def test_asof_join_by_forward() -> None:
    dfa = pl.DataFrame(
        {"category": ["a", "a", "a", "a", "a"], "value_one": [1, 2, 3, 5, 12]}
    ).set_sorted("value_one")

    dfb = pl.DataFrame({"category": ["a"], "value_two": [3]}).set_sorted("value_two")

    assert dfa.join_asof(
        dfb,
        left_on="value_one",
        right_on="value_two",
        by="category",
        strategy="forward",
    ).to_dict(as_series=False) == {
        "category": ["a", "a", "a", "a", "a"],
        "value_one": [1, 2, 3, 5, 12],
        "value_two": [3, 3, 3, None, None],
    }


def test_truncate_broadcast_left() -> None:
    df = pl.DataFrame({"every": [None, "1y", "1mo", "1d", "1h"]})
    out = df.select(
        date=pl.lit(date(2024, 4, 19)).dt.truncate(pl.col("every")),
        datetime=pl.lit(datetime(2024, 4, 19, 10, 30, 20)).dt.truncate(pl.col("every")),
    )
    expected = pl.DataFrame(
        {
            "date": [
                None,
                date(2024, 1, 1),
                date(2024, 4, 1),
                date(2024, 4, 19),
                date(2024, 4, 19),
            ],
            "datetime": [
                None,
                datetime(2024, 1, 1),
                datetime(2024, 4, 1),
                datetime(2024, 4, 19),
                datetime(2024, 4, 19, 10),
            ],
        }
    )
    assert_frame_equal(out, expected)


def test_truncate_expr() -> None:
    df = pl.DataFrame(
        {
            "date": [
                datetime(2022, 11, 14),
                datetime(2023, 10, 11),
                datetime(2022, 3, 20, 5, 7, 18),
                datetime(2022, 4, 3, 13, 30, 32),
            ],
            "every": ["1y", "1mo", "1m", "1s"],
        }
    )

    every_expr = df.select(pl.col("date").dt.truncate(every=pl.col("every")))
    assert every_expr.to_dict(as_series=False) == {
        "date": [
            datetime(2022, 1, 1),
            datetime(2023, 10, 1),
            datetime(2022, 3, 20, 5, 7),
            datetime(2022, 4, 3, 13, 30, 32),
        ]
    }

    all_lit = df.select(pl.col("date").dt.truncate(every=pl.lit("1mo")))
    assert all_lit.to_dict(as_series=False) == {
        "date": [
            datetime(2022, 11, 1),
            datetime(2023, 10, 1),
            datetime(2022, 3, 1),
            datetime(2022, 4, 1),
        ]
    }

    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                date(2020, 10, 25),
                datetime(2020, 10, 25, 2),
                "30m",
                eager=True,
                time_zone="Europe/London",
            ).dt.offset_by("15m"),
            "every": ["30m", "15m", "30m", "15m", "30m", "15m", "30m"],
        }
    )

    # How to deal with ambiguousness is auto-inferred
    ambiguous_expr = df.select(pl.col("date").dt.truncate(every=pl.lit("30m")))
    assert ambiguous_expr.to_dict(as_series=False) == {
        "date": [
            datetime(2020, 10, 25, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 0, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 0, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 0, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 30, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 2, 0, tzinfo=ZoneInfo("Europe/London")),
        ]
    }

    all_expr = df.select(pl.col("date").dt.truncate(every=pl.col("every")))
    assert all_expr.to_dict(as_series=False) == {
        "date": [
            datetime(2020, 10, 25, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 0, 45, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 0, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 45, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 0, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 1, 45, tzinfo=ZoneInfo("Europe/London")),
            datetime(2020, 10, 25, 2, 0, tzinfo=ZoneInfo("Europe/London")),
        ]
    }


def test_truncate_propagate_null() -> None:
    df = pl.DataFrame(
        {
            "date": [
                None,
                datetime(2022, 11, 14),
                datetime(2022, 3, 20, 5, 7, 18),
            ],
            "every": ["1y", None, "1m"],
        }
    )
    assert df.select(pl.col("date").dt.truncate(every=pl.col("every"))).to_dict(
        as_series=False
    ) == {"date": [None, None, datetime(2022, 3, 20, 5, 7, 0)]}
    assert df.select(
        pl.col("date").dt.truncate(
            every=pl.lit(None, dtype=pl.String),
        )
    ).to_dict(as_series=False) == {"date": [None, None, None]}


def test_truncate_by_calendar_weeks() -> None:
    # 5557
    start = datetime(2022, 11, 14, 0, 0, 0)
    end = datetime(2022, 11, 20, 0, 0, 0)

    assert (
        pl.datetime_range(start, end, timedelta(days=1), eager=True)
        .alias("date")
        .to_frame()
        .select([pl.col("date").dt.truncate("1w")])
    ).to_dict(as_series=False) == {
        "date": [
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
            datetime(2022, 11, 14),
        ],
    }

    df = pl.DataFrame(
        {
            "date": pl.Series(["1768-03-01", "2023-01-01"]).str.strptime(
                pl.Date, "%Y-%m-%d"
            )
        }
    )

    assert df.select(pl.col("date").dt.truncate("1w")).to_dict(as_series=False) == {
        "date": [
            date(1768, 2, 29),
            date(2022, 12, 26),
        ],
    }


def test_truncate_by_multiple_weeks() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [
                    # Wednesday and Monday
                    "2022-04-20",
                    "2022-11-28",
                ]
            ).str.strptime(pl.Date, "%Y-%m-%d")
        }
    )

    assert (
        df.select(
            pl.col("date").dt.truncate("2w").alias("2w"),
            pl.col("date").dt.truncate("3w").alias("3w"),
            pl.col("date").dt.truncate("4w").alias("4w"),
            pl.col("date").dt.truncate("5w").alias("5w"),
            pl.col("date").dt.truncate("17w").alias("17w"),
        )
    ).to_dict(as_series=False) == {
        "2w": [date(2022, 4, 18), date(2022, 11, 28)],
        "3w": [date(2022, 4, 11), date(2022, 11, 28)],
        "4w": [date(2022, 4, 18), date(2022, 11, 28)],
        "5w": [date(2022, 3, 28), date(2022, 11, 28)],
        "17w": [date(2022, 2, 21), date(2022, 10, 17)],
    }


def test_truncate_by_multiple_weeks_diffs() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.date_range(date(2020, 1, 1), date(2020, 2, 1), eager=True),
        }
    )
    result = df.select(
        pl.col("ts").dt.truncate("1w").alias("1w"),
        pl.col("ts").dt.truncate("2w").alias("2w"),
        pl.col("ts").dt.truncate("3w").alias("3w"),
    ).select(pl.all().diff().drop_nulls().unique())
    expected = pl.DataFrame(
        {
            "1w": [timedelta(0), timedelta(days=7)],
            "2w": [timedelta(0), timedelta(days=14)],
            "3w": [timedelta(0), timedelta(days=21)],
        }
    ).select(pl.all().cast(pl.Duration("ms")))
    assert_frame_equal(result, expected)


def test_truncate_ambiguous() -> None:
    ser = (
        pl.datetime_range(
            date(2020, 10, 25),
            datetime(2020, 10, 25, 2),
            "30m",
            eager=True,
            time_zone="Europe/London",
        )
        .alias("datetime")
        .dt.offset_by("15m")
    )
    df = ser.to_frame()
    result = df.select(pl.col("datetime").dt.truncate("30m"))
    expected = (
        pl.datetime_range(
            date(2020, 10, 25),
            datetime(2020, 10, 25, 2),
            "30m",
            eager=True,
            time_zone="Europe/London",
        )
        .alias("datetime")
        .to_frame()
    )
    assert_frame_equal(result, expected)


def test_truncate_ambiguous2() -> None:
    ser = (
        pl.datetime_range(
            date(2020, 10, 25),
            datetime(2020, 10, 25, 2),
            "30m",
            eager=True,
            time_zone="Europe/London",
        )
        .alias("datetime")
        .dt.offset_by("15m")
    )
    result = ser.dt.truncate("30m")
    expected = (
        pl.Series(
            [
                "2020-10-25T00:00:00+0100",
                "2020-10-25T00:30:00+0100",
                "2020-10-25T01:00:00+0100",
                "2020-10-25T01:30:00+0100",
                "2020-10-25T01:00:00+0000",
                "2020-10-25T01:30:00+0000",
                "2020-10-25T02:00:00+0000",
            ]
        )
        .str.to_datetime()
        .dt.convert_time_zone("Europe/London")
        .rename("datetime")
    )
    assert_series_equal(result, expected)


def test_truncate_non_existent_14957() -> None:
    with pytest.raises(ComputeError, match="non-existent"):
        pl.Series([datetime(2020, 3, 29, 2, 1)]).dt.replace_time_zone(
            "Europe/London"
        ).dt.truncate("46m")


def test_cast_time_to_duration() -> None:
    assert pl.Series([time(hour=0, minute=0, second=2)]).cast(
        pl.Duration
    ).item() == timedelta(seconds=2)


def test_tz_aware_day_weekday() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 9)
    df = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start, stop, timedelta(days=3), time_zone="UTC", eager=True
            )
        }
    )

    df = df.with_columns(
        pl.col("date").dt.convert_time_zone("Asia/Tokyo").alias("tk_date"),
        pl.col("date").dt.convert_time_zone("America/New_York").alias("ny_date"),
    )

    assert df.select(
        pl.col("date").dt.day().alias("day"),
        pl.col("tk_date").dt.day().alias("tk_day"),
        pl.col("ny_date").dt.day().alias("ny_day"),
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("tk_date").dt.weekday().alias("tk_weekday"),
        pl.col("ny_date").dt.weekday().alias("ny_weekday"),
    ).to_dict(as_series=False) == {
        "day": [1, 4, 7],
        "tk_day": [1, 4, 7],
        "ny_day": [31, 3, 6],
        "weekday": [1, 4, 7],
        "tk_weekday": [1, 4, 7],
        "ny_weekday": [7, 3, 6],
    }


def test_datetime_cum_agg_schema() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
            ]
        }
    )
    # Exactly the same as above but with lazy() and collect() later
    assert (
        df.lazy()
        .with_columns(
            (pl.col("timestamp").cum_min()).alias("cum_min"),
            (pl.col("timestamp").cum_max()).alias("cum_max"),
        )
        .with_columns(
            (pl.col("cum_min") + pl.duration(hours=24)).alias("cum_min+24"),
            (pl.col("cum_max") + pl.duration(hours=24)).alias("cum_max+24"),
        )
        .collect()
    ).to_dict(as_series=False) == {
        "timestamp": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
        ],
        "cum_min": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 0, 0),
        ],
        "cum_max": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
        ],
        "cum_min+24": [
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 0, 0),
        ],
        "cum_max+24": [
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
            datetime(2023, 1, 4, 0, 0),
        ],
    }


def test_infer_iso8601_datetime(iso8601_format_datetime: str) -> None:
    # construct an example time string
    time_string = (
        iso8601_format_datetime.replace("%Y", "2134")
        .replace("%m", "12")
        .replace("%d", "13")
        .replace("%H", "01")
        .replace("%M", "12")
        .replace("%S", "34")
        .replace("%3f", "123")
        .replace("%6f", "123456")
        .replace("%9f", "123456789")
    )
    parsed = pl.Series([time_string]).str.strptime(pl.Datetime("ns"))
    assert parsed.dt.year().item() == 2134
    assert parsed.dt.month().item() == 12
    assert parsed.dt.day().item() == 13
    if "%H" in iso8601_format_datetime:
        assert parsed.dt.hour().item() == 1
    if "%M" in iso8601_format_datetime:
        assert parsed.dt.minute().item() == 12
    if "%S" in iso8601_format_datetime:
        assert parsed.dt.second().item() == 34
    if "%9f" in iso8601_format_datetime:
        assert parsed.dt.nanosecond().item() == 123456789
    if "%6f" in iso8601_format_datetime:
        assert parsed.dt.nanosecond().item() == 123456000
    if "%3f" in iso8601_format_datetime:
        assert parsed.dt.nanosecond().item() == 123000000


def test_infer_iso8601_tz_aware_datetime(iso8601_tz_aware_format_datetime: str) -> None:
    # construct an example time string
    time_string = (
        iso8601_tz_aware_format_datetime.replace("%Y", "2134")
        .replace("%m", "12")
        .replace("%d", "13")
        .replace("%H", "02")
        .replace("%M", "12")
        .replace("%S", "34")
        .replace("%3f", "123")
        .replace("%6f", "123456")
        .replace("%9f", "123456789")
        .replace("%#z", "+01:00")
    )
    parsed = pl.Series([time_string]).str.strptime(pl.Datetime("ns"))
    assert parsed.dt.year().item() == 2134
    assert parsed.dt.month().item() == 12
    assert parsed.dt.day().item() == 13
    if "%H" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.hour().item() == 1
    if "%M" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.minute().item() == 12
    if "%S" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.second().item() == 34
    if "%9f" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.nanosecond().item() == 123456789
    if "%6f" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.nanosecond().item() == 123456000
    if "%3f" in iso8601_tz_aware_format_datetime:
        assert parsed.dt.nanosecond().item() == 123000000
    assert parsed.dtype == pl.Datetime("ns", "UTC")


def test_infer_iso8601_date(iso8601_format_date: str) -> None:
    # construct an example date string
    time_string = (
        iso8601_format_date.replace("%Y", "2134")
        .replace("%m", "12")
        .replace("%d", "13")
    )
    parsed = pl.Series([time_string]).str.strptime(pl.Date)
    assert parsed.dt.year().item() == 2134
    assert parsed.dt.month().item() == 12
    assert parsed.dt.day().item() == 13


def test_year_null_backed_by_out_of_range_15313() -> None:
    # Create a Series where the null value is backed by a value which would
    # be out-of-range for Datetime('us')
    s = pl.Series([None, 2**63 - 1])
    s -= 2**63 - 1
    result = s.cast(pl.Datetime).dt.year()
    expected = pl.Series([None, 1970], dtype=pl.Int32)
    assert_series_equal(result, expected)
    result = s.cast(pl.Date).dt.year()
    expected = pl.Series([None, 1970], dtype=pl.Int32)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [*TEMPORAL_DTYPES, pl.Datetime("ms", "UTC"), pl.Datetime("ns", "Europe/Amsterdam")],
)
def test_series_is_temporal(dtype: pl.DataType) -> None:
    s = pl.Series([None], dtype=dtype)
    assert s.dtype.is_temporal() is True


@pytest.mark.parametrize(
    "time_zone",
    [
        None,
        timezone.utc,
        "America/Caracas",
        "Asia/Kathmandu",
        "Asia/Taipei",
        "Europe/Amsterdam",
        "Europe/Lisbon",
        "Indian/Maldives",
        "Pacific/Norfolk",
        "Pacific/Samoa",
        "Turkey",
        "US/Eastern",
        "UTC",
        "Zulu",
    ],
)
def test_misc_precision_any_value_conversion(time_zone: Any) -> None:
    tz = ZoneInfo(time_zone) if isinstance(time_zone, str) else time_zone
    # default precision (μs)
    dt = datetime(2514, 5, 30, 1, 53, 4, 986754, tzinfo=tz)
    assert pl.Series([dt]).to_list() == [dt]

    # ms precision
    dt = datetime(2243, 1, 1, 0, 0, 0, 1000, tzinfo=tz)
    assert pl.Series([dt]).cast(pl.Datetime("ms", time_zone)).to_list() == [dt]

    # ns precision
    dt = datetime(2256, 1, 1, 0, 0, 0, 1, tzinfo=tz)
    assert pl.Series([dt]).cast(pl.Datetime("ns", time_zone)).to_list() == [dt]


@pytest.mark.parametrize(
    "tm",
    [
        time(0, 20, 30, 1),
        time(8, 40, 15, 8888),
        time(15, 10, 20, 123456),
        time(23, 59, 59, 999999),
    ],
)
def test_pytime_conversion(tm: time) -> None:
    s = pl.Series("tm", [tm])
    assert s.to_list() == [tm]


@given(
    value=st.datetimes(min_value=datetime(1800, 1, 1), max_value=datetime(2100, 1, 1)),
    time_zone=st.sampled_from(["UTC", "Asia/Kathmandu", "Europe/Amsterdam", None]),
    time_unit=st.sampled_from(["ms", "us", "ns"]),
)
def test_weekday_vs_stdlib_datetime(
    value: datetime, time_zone: str, time_unit: TimeUnit
) -> None:
    result = (
        pl.Series([value], dtype=pl.Datetime(time_unit))
        .dt.replace_time_zone(time_zone, non_existent="null", ambiguous="null")
        .dt.weekday()
        .item()
    )
    if result is not None:
        expected = value.isoweekday()
        assert result == expected


@given(
    value=st.dates(),
)
def test_weekday_vs_stdlib_date(value: date) -> None:
    result = pl.Series([value]).dt.weekday().item()
    expected = value.isoweekday()
    assert result == expected

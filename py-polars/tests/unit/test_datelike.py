from __future__ import annotations

import io
import sys
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, cast, no_type_check

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

if sys.version_info >= (3, 9):
    import zoneinfo
else:
    from backports import zoneinfo

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS, PolarsTemporalType
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing._private import verify_series_and_expr_api

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


def test_fill_null() -> None:
    dtm = datetime.strptime("2021-01-01", "%Y-%m-%d")
    s = pl.Series("A", [dtm, None])

    for fill_val in (dtm, pl.lit(dtm)):
        out = s.fill_null(fill_val)

        assert out.null_count() == 0
        assert out.dt[0] == dtm
        assert out.dt[1] == dtm

    dt1 = date(2001, 1, 1)
    dt2 = date(2001, 1, 2)
    dt3 = date(2001, 1, 3)

    s = pl.Series("a", [dt1, dt2, dt3, None])
    dt_2 = date(2001, 1, 4)

    for fill_val in (dt_2, pl.lit(dt_2)):
        out = s.fill_null(fill_val)

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
        columns=[
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
    ).with_column(dtcol.str.strptime(pl.Date, "%Y-%m-%d"))
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
    assert (dates + timedelta(days=10_000)).series_equal(out)


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
            [
                pl.col("timestamp").str.strptime(pl.Date, fmt="%Y-%m-%d"),
            ]
        ).with_columns([pl.col("timestamp").diff().list().over("char")])
    )["timestamp"]
    assert (out[0] == out[1]).all()


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
        [
            pl.col("a").cast(pl.Datetime).alias("b"),
            pl.col("a").cast(pl.Datetime("ms")).alias("c"),
            pl.col("a").cast(pl.Datetime("us")).alias("d"),
            pl.col("a").cast(pl.Datetime("ns")).alias("e"),
        ]
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
    assert df.select(
        [pl.col(col).dt.timestamp() for col in ("c", "d", "e")]
        + [
            getattr(pl.col("b").cast(pl.Duration).dt, unit)().alias(f"u[{unit}]")
            for unit in ("milliseconds", "microseconds", "nanoseconds")
        ]
    ).rows() == [
        (100000000000, 100000000, 100000, 100000, 100000000, 100000000000),
        (200000000000, 200000000, 200000, 200000, 200000000, 200000000000),
    ]


def test_int_to_python_timedelta() -> None:
    df = pl.DataFrame({"a": [100_001, 200_002]}).with_columns(
        [
            pl.col("a").cast(pl.Duration).alias("b"),
            pl.col("a").cast(pl.Duration("ms")).alias("c"),
            pl.col("a").cast(pl.Duration("us")).alias("d"),
            pl.col("a").cast(pl.Duration("ns")).alias("e"),
        ]
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

    assert df.select(
        [pl.col(col).dt.timestamp() for col in ("c", "d", "e")]
    ).rows() == [(100001, 100001, 100001), (200002, 200002, 200002)]


def test_from_numpy() -> None:
    # note: numpy timeunit support is limited to those supported by polars.
    # as a result, datetime64[s] will be stored as object.
    x = np.asarray(range(100_000, 200_000, 10_000), dtype="datetime64[s]")
    s = pl.Series(x)
    assert s[0] == x[0]
    assert len(s) == 10


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
        [
            pl.col("date"),
            pl.lit(dt).alias("dt"),
            pl.lit(dt).cast(pl.Datetime("ms")).alias("dt_ms"),
            pl.lit(dt).cast(pl.Datetime("us")).alias("dt_us"),
            pl.lit(dt).cast(pl.Datetime("ns")).alias("dt_ns"),
        ]
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
    ddf = pl.DataFrame({"dtm": test_data}).with_column(
        pl.col("dtm").dt.nanosecond().alias("ns")
    )
    assert ddf.rows() == [
        (test_data[0], 555555000),
        (test_data[1], 986754000),
        (test_data[2], 123456000),
        (test_data[3], 999999000),
    ]


def test_timezone() -> None:
    ts = pa.timestamp("s")
    data = pa.array([1000, 2000], type=ts)
    s = cast(pl.Series, pl.from_arrow(data))

    # with timezone; we do expect a warning here
    tz_ts = pa.timestamp("s", tz="America/New_York")
    tz_data = pa.array([1000, 2000], type=tz_ts)
    # with pytest.warns(Warning):
    tz_s = cast(pl.Series, pl.from_arrow(tz_data))

    # different timezones are not considered equal
    # we check both `null_equal=True` and `null_equal=False`
    # https://github.com/pola-rs/polars/issues/5023
    assert not s.series_equal(tz_s, null_equal=False)
    assert not s.series_equal(tz_s, null_equal=True)
    assert s.cast(int).series_equal(tz_s.cast(int))


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
    s1 = (
        pl.Series("datetime", [a * 1_000_000 for a in [123543, 283478, 1243]])
        .cast(pl.Datetime)
        .dt.with_time_unit("ns")
    )
    df = pl.DataFrame([s0, s1])

    rows = df.rows()
    assert rows[0][0] == date(2308, 4, 2)
    assert rows[0][1] == datetime(1970, 1, 1, 0, 2, 3, 543000)


def test_to_numpy() -> None:
    s0 = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)
    s1 = pl.Series(
        "datetime", [datetime(2021, 1, 2, 3, 4, 5), datetime(2021, 2, 3, 4, 5, 6)]
    )
    s2 = pl.date_range(
        datetime(2021, 1, 1, 0), datetime(2021, 1, 1, 1), interval="1h", time_unit="ms"
    )
    assert str(s0.to_numpy()) == "['2308-04-02' '2746-02-20' '1973-05-28']"
    assert (
        str(s1.to_numpy()[:2])
        == "['2021-01-02T03:04:05.000000' '2021-02-03T04:05:06.000000']"
    )
    assert (
        str(s2.to_numpy()[:2])
        == "['2021-01-01T00:00:00.000' '2021-01-01T01:00:00.000']"
    )
    s3 = pl.Series([timedelta(hours=1), timedelta(hours=-2)])
    out = np.array([3_600_000_000_000, -7_200_000_000_000], dtype="timedelta64[ns]")
    assert (s3.to_numpy() == out).all()


def test_truncate() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 2)

    s1 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ms]", time_unit="ms"
    )
    s2 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[us]", time_unit="us"
    )
    s3 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ns]", time_unit="ns"
    )

    # can pass strings and timedeltas
    for out in [
        s1.dt.truncate("1h"),
        s2.dt.truncate("1h0m0s"),
        s3.dt.truncate(timedelta(hours=1)),
    ]:
        assert out.dt[0] == start
        assert out.dt[1] == start
        assert out.dt[2] == start + timedelta(hours=1)
        assert out.dt[3] == start + timedelta(hours=1)
        # ...
        assert out.dt[-3] == stop - timedelta(hours=1)
        assert out.dt[-2] == stop - timedelta(hours=1)
        assert out.dt[-1] == stop


def test_round() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 2)

    s1 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ms]", time_unit="ms"
    )
    s2 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[us]", time_unit="us"
    )
    s3 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ns]", time_unit="ns"
    )

    # can pass strings and timedeltas
    for out in [
        s1.dt.round("1h"),
        s2.dt.round("1h0m0s"),
        s3.dt.round(timedelta(hours=1)),
    ]:
        assert out.dt[0] == start
        assert out.dt[1] == start + timedelta(hours=1)
        assert out.dt[2] == start + timedelta(hours=1)
        assert out.dt[3] == start + timedelta(hours=2)
        # ...
        assert out.dt[-3] == stop - timedelta(hours=1)
        assert out.dt[-2] == stop
        assert out.dt[-1] == stop


def test_date_range() -> None:
    result = pl.date_range(
        date(1985, 1, 1), date(2015, 7, 1), timedelta(days=1, hours=12)
    )
    assert len(result) == 7426
    assert result.dt[0] == datetime(1985, 1, 1)
    assert result.dt[1] == datetime(1985, 1, 2, 12, 0)
    assert result.dt[2] == datetime(1985, 1, 4, 0, 0)
    assert result.dt[-1] == datetime(2015, 6, 30, 12, 0)

    for tu in DTYPE_TEMPORAL_UNITS:
        rng = pl.date_range(datetime(2020, 1, 1), date(2020, 1, 2), "2h", time_unit=tu)
        assert rng.time_unit == tu
        assert rng.shape == (13,)
        assert rng.dt[0] == datetime(2020, 1, 1)
        assert rng.dt[-1] == datetime(2020, 1, 2)

    # if low/high are both date, range is also be date _iif_ the granularity is >= 1d
    result = pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", name="drange")
    assert result.to_list() == [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)]
    assert result.name == "drange"

    result = pl.date_range(date(2022, 1, 1), date(2022, 1, 2), "1h30m")
    assert list(result) == [
        datetime(2022, 1, 1, 0, 0),
        datetime(2022, 1, 1, 1, 30),
        datetime(2022, 1, 1, 3, 0),
        datetime(2022, 1, 1, 4, 30),
        datetime(2022, 1, 1, 6, 0),
        datetime(2022, 1, 1, 7, 30),
        datetime(2022, 1, 1, 9, 0),
        datetime(2022, 1, 1, 10, 30),
        datetime(2022, 1, 1, 12, 0),
        datetime(2022, 1, 1, 13, 30),
        datetime(2022, 1, 1, 15, 0),
        datetime(2022, 1, 1, 16, 30),
        datetime(2022, 1, 1, 18, 0),
        datetime(2022, 1, 1, 19, 30),
        datetime(2022, 1, 1, 21, 0),
        datetime(2022, 1, 1, 22, 30),
        datetime(2022, 1, 2, 0, 0),
    ]

    result = pl.date_range(
        datetime(2022, 1, 1), datetime(2022, 1, 1, 0, 1), "987456321ns"
    )
    assert len(result) == 61
    assert result.dtype.tu == "ns"  # type: ignore[attr-defined]
    assert result.dt.second()[-1] == 59
    assert result.cast(pl.Utf8)[-1] == "2022-01-01 00:00:59.247379260"


def test_date_range_lazy_with_literals() -> None:
    df = pl.DataFrame({"misc": ["x"]}).with_columns(
        pl.date_range(
            date(2000, 1, 1),
            date(2023, 8, 31),
            interval="987d",
            lazy=True,
        )
        .list()
        .alias("dts")
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
    ldf = (
        pl.DataFrame({"start": [date(2015, 6, 30)], "stop": [date(2022, 12, 31)]})
        .with_columns(
            pl.date_range(low, high, interval="678d", lazy=True).list().alias("dts")
        )
        .lazy()
    )

    assert ldf.collect().rows() == [
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

    assert pl.DataFrame(
        {
            "start": [date(2000, 1, 1), date(2022, 6, 1)],
            "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        }
    ).with_columns(
        pl.date_range(
            low,
            high,
            interval="1d",
        ).alias("dts")
    ).to_dict(
        False
    ) == {
        "start": [date(2000, 1, 1), date(2022, 6, 1)],
        "stop": [date(2000, 1, 2), date(2022, 6, 2)],
        "dts": [
            [date(2000, 1, 1), date(2000, 1, 2)],
            [date(2022, 6, 1), date(2022, 6, 2)],
        ],
    }

    assert pl.DataFrame(
        {
            "start": [datetime(2000, 1, 1), datetime(2022, 6, 1)],
            "stop": [datetime(2000, 1, 2), datetime(2022, 6, 2)],
        }
    ).with_columns(
        pl.date_range(
            low,
            high,
            interval="1d",
        ).alias("dts")
    ).to_dict(
        False
    ) == {
        "start": [datetime(2000, 1, 1, 0, 0), datetime(2022, 6, 1, 0, 0)],
        "stop": [datetime(2000, 1, 2, 0, 0), datetime(2022, 6, 2, 0, 0)],
        "dts": [
            [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 2, 0, 0)],
            [datetime(2022, 6, 1, 0, 0), datetime(2022, 6, 2, 0, 0)],
        ],
    }


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


def test_truncate_negative_offset() -> None:
    df = pl.DataFrame(
        {
            "event_date": [
                datetime(2021, 4, 11),
                datetime(2021, 4, 29),
                datetime(2021, 5, 29),
            ],
            "adm1_code": [1, 2, 1],
        }
    )
    out = df.groupby_dynamic(
        index_column="event_date",
        every="1mo",
        period="2mo",
        offset="-1mo",
        include_boundaries=True,
    ).agg(
        [
            pl.col("adm1_code"),
        ]
    )

    assert out["event_date"].to_list() == [
        datetime(2021, 3, 1),
        datetime(2021, 4, 1),
        datetime(2021, 5, 1),
    ]
    df = pl.DataFrame(
        {
            "event_date": [
                datetime(2021, 4, 11),
                datetime(2021, 4, 29),
                datetime(2021, 5, 29),
            ],
            "adm1_code": [1, 2, 1],
            "five_type": ["a", "b", "a"],
            "actor": ["a", "a", "a"],
            "admin": ["a", "a", "a"],
            "fatalities": [10, 20, 30],
        }
    )

    out = df.groupby_dynamic(
        index_column="event_date",
        every="1mo",
        by=["admin", "five_type", "actor"],
    ).agg([pl.col("adm1_code").unique(), (pl.col("fatalities") > 0).sum()])

    assert out["event_date"].to_list() == [
        datetime(2021, 4, 1),
        datetime(2021, 5, 1),
        datetime(2021, 4, 1),
    ]

    for dt in [pl.Int32, pl.Int64]:
        df = pl.DataFrame(
            {
                "idx": np.arange(6),
                "A": ["A", "A", "B", "B", "B", "C"],
            }
        ).with_columns(pl.col("idx").cast(dt))

        out = df.groupby_dynamic(
            "idx", every="2i", period="3i", include_boundaries=True
        ).agg(pl.col("A").list())

        assert out.shape == (3, 4)
        assert out["A"].to_list() == [["A", "A", "B"], ["B", "B", "B"], ["B", "C"]]


def test_to_arrow() -> None:
    date_series = pl.Series("dates", ["2022-01-16", "2022-01-17"]).str.strptime(
        pl.Date, "%Y-%m-%d"
    )
    arr = date_series.to_arrow()
    assert arr.type == pa.date32()


def test_non_exact_strptime() -> None:
    a = pl.Series("a", ["2022-01-16", "2022-01-17", "foo2022-01-18", "b2022-01-19ar"])
    fmt = "%Y-%m-%d"

    expected = pl.Series("a", [date(2022, 1, 16), date(2022, 1, 17), None, None])
    verify_series_and_expr_api(
        a, expected, "str.strptime", pl.Date, fmt, strict=False, exact=True
    )

    expected = pl.Series(
        "a",
        [date(2022, 1, 16), date(2022, 1, 17), date(2022, 1, 18), date(2022, 1, 19)],
    )
    verify_series_and_expr_api(
        a, expected, "str.strptime", pl.Date, fmt, strict=False, exact=False
    )

    with pytest.raises(Exception):
        a.str.strptime(pl.Date, fmt, strict=True, exact=True)


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
            df.groupby("b", maintain_order=True)
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


def test_rolling() -> None:
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]

    df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_column(
        pl.col("dt").str.strptime(pl.Datetime)
    )

    period: str | timedelta
    for period in ("2d", timedelta(days=2)):  # type: ignore[assignment]
        out = df.groupby_rolling(index_column="dt", period=period).agg(
            [
                pl.sum("a").alias("sum_a"),
                pl.min("a").alias("min_a"),
                pl.max("a").alias("max_a"),
            ]
        )
        assert out["sum_a"].to_list() == [3, 10, 15, 24, 11, 1]
        assert out["max_a"].to_list() == [3, 7, 7, 9, 9, 1]
        assert out["min_a"].to_list() == [3, 3, 3, 3, 2, 1]


def test_upsample() -> None:
    df = pl.DataFrame(
        {
            "time": [
                datetime(2021, 2, 1),
                datetime(2021, 4, 1),
                datetime(2021, 5, 1),
                datetime(2021, 6, 1),
            ],
            "admin": ["Åland", "Netherlands", "Åland", "Netherlands"],
            "test2": [0, 1, 2, 3],
        }
    ).with_column(pl.col("time").dt.with_time_zone("UTC"))

    up = df.upsample(
        time_column="time", every="1mo", by="admin", maintain_order=True
    ).select(pl.all().forward_fill())
    # this print will panic if timezones feature is not activated
    # don't remove
    print(up)

    expected = pl.DataFrame(
        {
            "time": [
                datetime(2021, 2, 1, 0, 0),
                datetime(2021, 3, 1, 0, 0),
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
                datetime(2021, 4, 1, 0, 0),
                datetime(2021, 5, 1, 0, 0),
                datetime(2021, 6, 1, 0, 0),
            ],
            "admin": [
                "Åland",
                "Åland",
                "Åland",
                "Åland",
                "Netherlands",
                "Netherlands",
                "Netherlands",
            ],
            "test2": [0, 0, 0, 2, 1, 1, 3],
        }
    ).with_column(pl.col("time").dt.with_time_zone("UTC"))

    assert up.frame_equal(expected)


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


def test_cast_time_units() -> None:
    dates = pl.Series("dates", [datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])
    dates_in_ns = np.array([978307200000000000, 981022089000000000])

    assert dates.dt.cast_time_unit("ns").cast(int).to_list() == list(dates_in_ns)
    assert dates.dt.cast_time_unit("us").cast(int).to_list() == list(
        dates_in_ns // 1_000
    )
    assert dates.dt.cast_time_unit("ms").cast(int).to_list() == list(
        dates_in_ns // 1_000_000
    )


def test_read_utc_times_parquet() -> None:
    df = pd.DataFrame(
        data={
            "Timestamp": pd.date_range(
                "2022-01-01T00:00+00:00", "2022-01-01T10:00+00:00", freq="H"
            )
        }
    )
    f = io.BytesIO()
    df.to_parquet(f)
    f.seek(0)
    df_in = pl.read_parquet(f)
    tz = zoneinfo.ZoneInfo("UTC")
    assert df_in["Timestamp"][0] == datetime(2022, 1, 1, 0, 0, tzinfo=tz)


def test_epoch() -> None:
    dates = pl.Series("dates", [datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    for unit in DTYPE_TEMPORAL_UNITS:
        assert dates.dt.epoch(unit).series_equal(dates.dt.timestamp(unit))

    assert dates.dt.epoch("s").series_equal(dates.dt.timestamp("ms") // 1000)
    assert dates.dt.epoch("d").series_equal(
        (dates.dt.timestamp("ms") // (1000 * 3600 * 24)).cast(pl.Int32)
    )


def test_default_negative_every_offset_dynamic_groupby() -> None:
    # 2791
    dts = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 2, 1),
        datetime(2020, 3, 1),
    ]
    df = pl.DataFrame({"dt": dts, "idx": range(len(dts))})
    out = df.groupby_dynamic(index_column="dt", every="1mo", closed="right").agg(
        pl.col("idx")
    )

    expected = pl.DataFrame(
        {
            "dt": [
                datetime(2019, 12, 1, 0, 0),
                datetime(2020, 1, 1, 0, 0),
                datetime(2020, 2, 1, 0, 0),
            ],
            "idx": [[0], [1, 2], [3]],
        }
    )
    assert out.frame_equal(expected)


def test_strptime_dates_datetimes() -> None:
    s = pl.Series("date", ["2021-04-22", "2022-01-04 00:00:00"])
    assert s.str.strptime(pl.Datetime).to_list() == [
        datetime(2021, 4, 22, 0, 0),
        datetime(2022, 1, 4, 0, 0),
    ]


def test_strptime_precision() -> None:
    s = pl.Series(
        "date", ["2022-09-12 21:54:36.789321456", "2022-09-13 12:34:56.987456321"]
    )
    ds = s.str.strptime(pl.Datetime)
    assert ds.cast(pl.Date) != None  # noqa: E711  (note: *deliberately* testing "!=")
    assert getattr(ds.dtype, "tu", None) == "us"

    time_units: list[TimeUnit] = ["ms", "us", "ns"]
    suffixes = ["%.3f", "%.6f", "%.9f"]
    test_data = zip(
        time_units,
        suffixes,
        (
            [789000000, 987000000],
            [789321000, 987456000],
            [789321456, 987456321],
        ),
    )
    for precision, suffix, expected_values in test_data:
        ds = s.str.strptime(pl.Datetime(precision), f"%Y-%m-%d %H:%M:%S{suffix}")
        assert getattr(ds.dtype, "tu", None) == precision
        assert ds.dt.nanosecond().to_list() == expected_values


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("ms", "123000000"), ("us", "123456000"), ("ns", "123456789")],
)
@pytest.mark.parametrize("fmt", ["%Y-%m-%d %H:%M:%S.%f", None])
def test_strptime_precision_with_time_unit(
    unit: TimeUnit, expected: str, fmt: str
) -> None:
    ser = pl.Series(["2020-01-01 00:00:00.123456789"])
    result = ser.str.strptime(pl.Datetime(unit), fmt=fmt).dt.strftime("%f")[0]
    assert result == expected


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

    out = df1.join_asof(df2, by="by", on="date", tolerance="3d")

    expected = pl.DataFrame(
        {
            "date": [date(2020, 1, 5), date(2020, 1, 10)],
            "by": [1, 1],
            "values": [100, None],
        }
    )

    assert out.frame_equal(expected)


def test_datetime_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(1999, 1, 1, 7),
                datetime(2022, 1, 2, 14),
                datetime(3000, 12, 31, 21),
            ],
            "add": [1, 2, -1],
        }
    )
    out = df.select(
        [
            (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
            (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
            (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
            (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
            (pl.col("datetime") + pl.duration(microseconds=pl.col("add") * 1000)).alias(
                "add_usecs"
            ),
        ]
    )
    expected = pl.DataFrame(
        {
            "add_weeks": [
                datetime(1999, 1, 8, 7),
                datetime(2022, 1, 16, 14),
                datetime(3000, 12, 24, 21),
            ],
            "add_days": [
                datetime(1999, 1, 2, 7),
                datetime(2022, 1, 4, 14),
                datetime(3000, 12, 30, 21),
            ],
            "add_hours": [
                datetime(1999, 1, 1, hour=8),
                datetime(2022, 1, 2, hour=16),
                datetime(3000, 12, 31, hour=20),
            ],
            "add_seconds": [
                datetime(1999, 1, 1, 7, second=1),
                datetime(2022, 1, 2, 14, second=2),
                datetime(3000, 12, 31, 20, 59, 59),
            ],
            "add_usecs": [
                datetime(1999, 1, 1, 7, microsecond=1000),
                datetime(2022, 1, 2, 14, microsecond=2000),
                datetime(3000, 12, 31, 20, 59, 59, 999000),
            ],
        }
    )
    assert out.frame_equal(expected)


def test_date_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "date": [date(10, 1, 1), date(2000, 7, 5), date(9990, 12, 31)],
            "offset": [365, 7, -31],
        }
    )
    out = df.select(
        [
            (pl.col("date") + pl.duration(days="offset")).alias("add_days"),
            (pl.col("date") - pl.duration(days="offset")).alias("sub_days"),
            (pl.col("date") + pl.duration(weeks="offset")).alias("add_weeks"),
            (pl.col("date") - pl.duration(weeks="offset")).alias("sub_weeks"),
        ]
    )
    assert out.to_dict(False) == {
        "add_days": [date(11, 1, 1), date(2000, 7, 12), date(9990, 11, 30)],
        "sub_days": [date(9, 1, 1), date(2000, 6, 28), date(9991, 1, 31)],
        "add_weeks": [date(16, 12, 30), date(2000, 8, 23), date(9990, 5, 28)],
        "sub_weeks": [date(3, 1, 3), date(2000, 5, 17), date(9991, 8, 5)],
    }


def test_date_time_combine() -> None:
    # test combining datetime/date and time (as expr/col and as literal)
    df = pl.DataFrame(
        {
            "dtm": [
                datetime(2022, 12, 31, 10, 30, 45),
                datetime(2023, 7, 5, 23, 59, 59),
            ],
            "dt": [date(2022, 10, 10), date(2022, 7, 5)],
            "tm": [time(1, 2, 3, 456000), time(7, 8, 9, 101000)],
        }
    ).select(
        [
            pl.col("dtm").dt.combine(pl.col("tm")).alias("d1"),
            pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),
            pl.col("dt").dt.combine(time(4, 5, 6)).alias("d3"),
        ]
    )
    # if combining with datetime, the time component should be overwritten.
    # if combining with date, should write both parts 'as-is' into the new datetime.
    assert df.to_dict(False) == {
        "d1": [
            datetime(2022, 12, 31, 1, 2, 3, 456000),
            datetime(2023, 7, 5, 7, 8, 9, 101000),
        ],
        "d2": [
            datetime(2022, 10, 10, 1, 2, 3, 456000),
            datetime(2022, 7, 5, 7, 8, 9, 101000),
        ],
        "d3": [datetime(2022, 10, 10, 4, 5, 6), datetime(2022, 7, 5, 4, 5, 6)],
    }
    assert df.schema == {
        "d1": pl.Datetime("us"),
        "d2": pl.Datetime("us"),
        "d3": pl.Datetime("us"),
    }


def test_add_duration_3786() -> None:
    df = pl.DataFrame(
        {
            "datetime": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
            "add": [1, 2],
        }
    )
    assert df.slice(0, 1).with_columns(
        [
            (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
            (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
            (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
            (pl.col("datetime") + pl.duration(milliseconds="add")).alias(
                "add_milliseconds"
            ),
            (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
        ]
    ).to_dict(False) == {
        "datetime": [datetime(2022, 1, 1, 0, 0)],
        "add": [1],
        "add_weeks": [datetime(2022, 1, 8, 0, 0)],
        "add_days": [datetime(2022, 1, 2, 0, 0)],
        "add_seconds": [datetime(2022, 1, 1, 0, 0, 1)],
        "add_milliseconds": [datetime(2022, 1, 1, 0, 0, 0, 1000)],
        "add_hours": [datetime(2022, 1, 1, 1, 0)],
    }


def test_rolling_groupby_by_argument() -> None:
    df = pl.DataFrame({"times": range(10), "groups": [1] * 4 + [2] * 6})

    out = df.groupby_rolling("times", period="5i", by=["groups"]).agg(
        pl.col("times").list().alias("agg_list")
    )

    expected = pl.DataFrame(
        {
            "groups": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "times": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "agg_list": [
                [0],
                [0, 1],
                [0, 1, 2],
                [0, 1, 2, 3],
                [4],
                [4, 5],
                [4, 5, 6],
                [4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ],
        }
    )

    assert out.frame_equal(expected)


def test_groupby_rolling_mean_3020() -> None:
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
    ).with_column(pl.col("Date").str.strptime(pl.Date))

    period: str | timedelta
    for period in ("1w", timedelta(days=7)):  # type: ignore[assignment]
        assert (
            df.groupby_rolling(index_column="Date", period=period)
            .agg(pl.col("val").mean().alias("val_mean"))
            .frame_equal(
                pl.DataFrame(
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
            )
        )


def test_asof_join() -> None:
    fmt = "%F %T%.3f"
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
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [720.5, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        }
    )
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
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [51.95, 51.95, 720.77, 720.92, 98.0],
        }
    )
    assert trades.schema == {
        "dates": pl.Datetime("ms"),
        "ticker": pl.Utf8,
        "bid": pl.Float64,
    }
    out = trades.join_asof(quotes, on="dates", strategy="backward")

    assert out.schema == {
        "bid": pl.Float64,
        "bid_right": pl.Float64,
        "dates": pl.Datetime("ms"),
        "ticker": pl.Utf8,
        "ticker_right": pl.Utf8,
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


def test_temporal_dtypes_apply() -> None:
    df = pl.DataFrame(
        {"timestamp": [1284286794000, None, 1234567890000]},
        columns=[("timestamp", pl.Datetime("ms"))],
    )
    const_dtm = datetime(2010, 9, 12)

    assert_frame_equal(
        df.with_columns(
            [
                # don't actually do any of this; native expressions are MUCH faster ;)
                pl.col("timestamp").apply(lambda x: const_dtm).alias("const_dtm"),
                pl.col("timestamp").apply(lambda x: x and x.date()).alias("date"),
                pl.col("timestamp").apply(lambda x: x and x.time()).alias("time"),
            ]
        ),
        pl.DataFrame(
            [
                (
                    datetime(2010, 9, 12, 10, 19, 54),
                    datetime(2010, 9, 12, 0, 0),
                    date(2010, 9, 12),
                    time(10, 19, 54),
                ),
                (None, const_dtm, None, None),
                (
                    datetime(2009, 2, 13, 23, 31, 30),
                    datetime(2010, 9, 12, 0, 0),
                    date(2009, 2, 13),
                    time(23, 31, 30),
                ),
            ],
            columns={
                "timestamp": pl.Datetime("ms"),
                "const_dtm": pl.Datetime("us"),
                "date": pl.Date,
                "time": pl.Time,
            },
        ),
    )


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
        columns=[
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
    ).with_column((pl.col("end_date") - pl.col("start_date")).alias("time_passed"))

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


@no_type_check
def test_from_time_arrow() -> None:
    pa_times = pa.table([pa.array([10, 20, 30], type=pa.time32("s"))], names=["times"])

    assert pl.from_arrow(pa_times).to_series().to_list() == [
        time(0, 0, 10),
        time(0, 0, 20),
        time(0, 0, 30),
    ]
    assert pl.from_arrow(pa_times).rows() == [
        (time(0, 0, 10),),
        (time(0, 0, 20),),
        (time(0, 0, 30),),
    ]


@pytest.mark.parametrize(
    ("time_string", "expected"),
    [
        ("09-05-2019", datetime(2019, 5, 9)),
        ("2018-09-05", datetime(2018, 9, 5)),
        ("2018-09-05T04:05:01", datetime(2018, 9, 5, 4, 5, 1)),
        ("2018-09-05T04:24:01.9", datetime(2018, 9, 5, 4, 24, 1, 900000)),
        ("2018-09-05T04:24:02.11", datetime(2018, 9, 5, 4, 24, 2, 110000)),
        ("2018-09-05T14:24:02.123", datetime(2018, 9, 5, 14, 24, 2, 123000)),
        ("2018-09-05T14:24:02.123Z", datetime(2018, 9, 5, 14, 24, 2, 123000)),
        ("2019-04-18T02:45:55.555000000", datetime(2019, 4, 18, 2, 45, 55, 555000)),
        ("2019-04-18T22:45:55.555123", datetime(2019, 4, 18, 22, 45, 55, 555123)),
    ],
)
def test_datetime_strptime_patterns_single(time_string: str, expected: str) -> None:
    result = pl.Series([time_string]).str.strptime(pl.Datetime).item()
    assert result == expected


def test_datetime_strptime_patterns_consistent() -> None:
    # note that all should be year first
    df = pl.Series(
        "date",
        [
            "2018-09-05",
            "2018-09-05T04:05:01",
            "2018-09-05T04:24:01.9",
            "2018-09-05T04:24:02.11",
            "2018-09-05T14:24:02.123",
            "2018-09-05T14:24:02.123Z",
            "2019-04-18T02:45:55.555000000",
            "2019-04-18T22:45:55.555123",
        ],
    ).to_frame()
    s = df.with_columns(
        [
            pl.col("date")
            .str.strptime(pl.Datetime, fmt=None, strict=False)
            .alias("parsed"),
        ]
    )["parsed"]
    assert s.null_count() == 0


def test_datetime_strptime_patterns_inconsistent() -> None:
    # note that the pattern is inferred from the first element to
    # be DatetimeDMY, and so the others (correctly) parse as `null`.
    df = pl.Series(
        "date",
        [
            "09-05-2019",
            "2018-09-05",
            "2018-09-05T04:05:01",
            "2018-09-05T04:24:01.9",
            "2018-09-05T04:24:02.11",
            "2018-09-05T14:24:02.123",
            "2018-09-05T14:24:02.123Z",
            "2019-04-18T02:45:55.555000000",
            "2019-04-18T22:45:55.555123",
        ],
    ).to_frame()
    s = df.with_columns(
        [
            pl.col("date")
            .str.strptime(pl.Datetime, fmt=None, strict=False)
            .alias("parsed"),
        ]
    )["parsed"]
    assert s.null_count() == 8
    assert s[0] is not None


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
    assert pl.DataFrame(as_dict).frame_equal(pl.DataFrame(as_rows))


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
    df = df.with_column((pl.col("end") - pl.col("start")).alias("duration"))
    assert df.groupby("group", maintain_order=True).agg(
        [
            pl.col("duration").mean().alias("mean"),
            pl.col("duration").sum().alias("sum"),
            pl.col("duration").min().alias("min"),
            pl.col("duration").max().alias("max"),
            pl.col("duration").quantile(0.1).alias("quantile"),
            pl.col("duration").median().alias("median"),
            pl.col("duration").list().alias("list"),
        ]
    ).to_dict(False) == {
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
            "ns": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 5, 1), "1mo", time_unit="ns"
            ),
            "us": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 5, 1), "1mo", time_unit="us"
            ),
            "ms": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 5, 1), "1mo", time_unit="ms"
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
        columns=[
            ("ns", pl.Datetime("ns")),
            ("us", pl.Datetime("us")),
            ("ms", pl.Datetime("ms")),
        ],
    )
    for tu in DTYPE_TEMPORAL_UNITS:
        res = df.select(pl.col([pl.Datetime(tu)])).dtypes
        assert res == [pl.Datetime(tu)]
        assert len(df.filter(pl.col(tu) == test_data[tu][0])) == 1


def test_unique_counts_on_dates() -> None:
    assert pl.DataFrame(
        {
            "dt_ns": pl.date_range(datetime(2020, 1, 1), datetime(2020, 3, 1), "1mo"),
        }
    ).with_columns(
        [
            pl.col("dt_ns").dt.cast_time_unit("us").alias("dt_us"),
            pl.col("dt_ns").dt.cast_time_unit("ms").alias("dt_ms"),
            pl.col("dt_ns").cast(pl.Date).alias("date"),
        ]
    ).select(
        pl.all().unique_counts().sum()
    ).to_dict(
        False
    ) == {
        "dt_ns": [3],
        "dt_us": [3],
        "dt_ms": [3],
        "date": [3],
    }


def test_groupby_rolling_by_ordering() -> None:
    # we must check that the keys still match the time labels after the rolling window
    # with a `by` argument.
    df = pl.DataFrame(
        {
            "dt": [
                datetime(2022, 1, 1, 0, 1),
                datetime(2022, 1, 1, 0, 2),
                datetime(2022, 1, 1, 0, 3),
                datetime(2022, 1, 1, 0, 4),
                datetime(2022, 1, 1, 0, 5),
                datetime(2022, 1, 1, 0, 6),
                datetime(2022, 1, 1, 0, 7),
            ],
            "key": ["A", "A", "B", "B", "A", "B", "A"],
            "val": [1, 1, 1, 1, 1, 1, 1],
        }
    )

    assert df.groupby_rolling(
        index_column="dt",
        period="2m",
        closed="both",
        offset="-1m",
        by="key",
    ).agg(
        [
            pl.col("val").sum().alias("sum val"),
        ]
    ).to_dict(
        False
    ) == {
        "key": ["A", "A", "A", "A", "B", "B", "B"],
        "dt": [
            datetime(2022, 1, 1, 0, 1),
            datetime(2022, 1, 1, 0, 2),
            datetime(2022, 1, 1, 0, 5),
            datetime(2022, 1, 1, 0, 7),
            datetime(2022, 1, 1, 0, 3),
            datetime(2022, 1, 1, 0, 4),
            datetime(2022, 1, 1, 0, 6),
        ],
        "sum val": [2, 2, 1, 1, 2, 2, 1],
    }


def test_groupby_rolling_by_() -> None:
    df = pl.DataFrame({"group": pl.arange(0, 3, eager=True)}).join(
        pl.DataFrame(
            {
                "datetime": pl.date_range(
                    datetime(2020, 1, 1), datetime(2020, 1, 5), "1d"
                ),
            }
        ),
        how="cross",
    )
    out = (
        df.sort("datetime")
        .groupby_rolling(index_column="datetime", by="group", period=timedelta(days=3))
        .agg([pl.count().alias("count")])
    )

    expected = (
        df.sort(["group", "datetime"])
        .groupby_rolling(index_column="datetime", by="group", period="3d")
        .agg([pl.count().alias("count")])
    )
    assert out.sort(["group", "datetime"]).frame_equal(expected)
    assert out.to_dict(False) == {
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "datetime": [
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
            datetime(2020, 1, 1, 0, 0),
            datetime(2020, 1, 2, 0, 0),
            datetime(2020, 1, 3, 0, 0),
            datetime(2020, 1, 4, 0, 0),
            datetime(2020, 1, 5, 0, 0),
        ],
        "count": [1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3],
    }


def test_quarter() -> None:
    assert pl.date_range(
        datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo"
    ).dt.quarter().to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_date_offset() -> None:
    out = pl.DataFrame(
        {"dates": pl.date_range(datetime(2000, 1, 1), datetime(2020, 1, 1), "1y")}
    ).with_columns(
        [
            pl.col("dates").dt.offset_by("1y").alias("date_plus_1y"),
            pl.col("dates").dt.offset_by("-1y2mo").alias("date_min"),
        ]
    )

    assert (out["date_plus_1y"].dt.day() == 1).all()
    assert (out["date_min"].dt.day() == 1).all()
    assert out["date_min"].to_list() == [
        datetime(1998, 11, 1, 0, 0),
        datetime(1999, 11, 1, 0, 0),
        datetime(2000, 11, 1, 0, 0),
        datetime(2001, 11, 1, 0, 0),
        datetime(2002, 11, 1, 0, 0),
        datetime(2003, 11, 1, 0, 0),
        datetime(2004, 11, 1, 0, 0),
        datetime(2005, 11, 1, 0, 0),
        datetime(2006, 11, 1, 0, 0),
        datetime(2007, 11, 1, 0, 0),
        datetime(2008, 11, 1, 0, 0),
        datetime(2009, 11, 1, 0, 0),
        datetime(2010, 11, 1, 0, 0),
        datetime(2011, 11, 1, 0, 0),
        datetime(2012, 11, 1, 0, 0),
        datetime(2013, 11, 1, 0, 0),
        datetime(2014, 11, 1, 0, 0),
        datetime(2015, 11, 1, 0, 0),
        datetime(2016, 11, 1, 0, 0),
        datetime(2017, 11, 1, 0, 0),
        datetime(2018, 11, 1, 0, 0),
    ]


def test_sorted_unique() -> None:
    assert (
        pl.DataFrame(
            [pl.Series("dt", [date(2015, 6, 24), date(2015, 6, 23)], dtype=pl.Date)]
        )
        .sort("dt")
        .unique()
    ).to_dict(False) == {"dt": [date(2015, 6, 23), date(2015, 6, 24)]}


def test_time_zero_3828() -> None:
    assert pl.Series(values=[time(0)], dtype=pl.Time).to_list() == [time(0)]


def test_time_microseconds_3843() -> None:
    in_val = [time(0, 9, 11, 558332)]
    s = pl.Series(in_val)
    assert s.to_list() == in_val


def test_date_to_time_cast_5111() -> None:
    # check date -> time casts (fast-path: always 00:00:00)
    df = pl.DataFrame(
        {
            "xyz": [
                date(1969, 1, 1),
                date(1990, 3, 8),
                date(2000, 6, 16),
                date(2010, 9, 24),
                date(2022, 12, 31),
            ]
        }
    ).with_column(pl.col("xyz").cast(pl.Time))
    assert df["xyz"].to_list() == [time(0), time(0), time(0), time(0), time(0)]


def test_year_empty_df() -> None:
    df = pl.DataFrame(pl.Series(name="date", dtype=pl.Date))
    assert df.select(pl.col("date").dt.year()).dtypes == [pl.Int32]


def test_sum_duration() -> None:
    assert pl.DataFrame(
        [
            {"name": "Jen", "duration": timedelta(seconds=60)},
            {"name": "Mike", "duration": timedelta(seconds=30)},
            {"name": "Jen", "duration": timedelta(seconds=60)},
        ]
    ).select(
        [pl.col("duration").sum(), pl.col("duration").dt.seconds().alias("sec").sum()]
    ).to_dict(
        False
    ) == {
        "duration": [timedelta(seconds=150)],
        "sec": [150],
    }


def test_supertype_timezones_4174() -> None:
    df = pl.DataFrame(
        {
            "dt": pl.date_range(datetime(2020, 3, 1), datetime(2020, 5, 1), "1mo"),
        }
    ).with_columns(
        [
            pl.col("dt").dt.with_time_zone("Europe/London").suffix("_London"),
        ]
    )

    # test if this runs without error
    date_to_fill = df["dt_London"][0]
    df.with_column(df["dt_London"].shift_and_fill(1, date_to_fill))


def test_weekday() -> None:
    # monday
    s = pl.Series([datetime(2020, 1, 6)])

    time_units: list[TimeUnit] = ["ns", "us", "ms"]
    for tu in time_units:
        assert s.dt.cast_time_unit(tu).dt.weekday()[0] == 1

    assert s.cast(pl.Date).dt.weekday()[0] == 1


@pytest.mark.skip(reason="from_dicts cannot yet infer timezones")
def test_from_dict_tu_consistency() -> None:
    tz = zoneinfo.ZoneInfo("PRC")
    dt = datetime(2020, 8, 1, 12, 0, 0, tzinfo=tz)
    from_dict = pl.from_dict({"dt": [dt]})
    from_dicts = pl.from_dicts([{"dt": dt}])

    assert from_dict.dtypes == from_dicts.dtypes


def test_date_parse_omit_day() -> None:
    df = pl.DataFrame({"month": ["2022-01"]})
    assert df.select(pl.col("month").str.strptime(pl.Date, fmt="%Y-%m")).item() == date(
        2022, 1, 1
    )
    assert df.select(
        pl.col("month").str.strptime(pl.Datetime, fmt="%Y-%m")
    ).item() == datetime(2022, 1, 1)


@pytest.mark.parametrize(
    (
        "ts",
        "fmt",
        "exp_year",
        "exp_month",
        "exp_day",
        "exp_hour",
        "exp_minute",
        "exp_second",
    ),
    [
        ("-0031-04-24 22:13:20", "%Y-%m-%d %H:%M:%S", -31, 4, 24, 22, 13, 20),
        ("-0031-04-24", "%Y-%m-%d", -31, 4, 24, 0, 0, 0),
    ],
)
def test_parse_negative_dates(
    ts: str,
    fmt: str,
    exp_year: int,
    exp_month: int,
    exp_day: int,
    exp_hour: int,
    exp_minute: int,
    exp_second: int,
) -> None:
    ser = pl.Series([ts])
    result = ser.str.strptime(pl.Datetime("ms"), fmt=fmt)
    # Python datetime.datetime doesn't support negative dates, so comparing
    # with `result.item()` directly won't work.
    assert result.dt.year().item() == exp_year
    assert result.dt.month().item() == exp_month
    assert result.dt.day().item() == exp_day
    assert result.dt.hour().item() == exp_hour
    assert result.dt.minute().item() == exp_minute
    assert result.dt.second().item() == exp_second


def test_shift_and_fill_group_logicals() -> None:
    df = pl.from_records(
        [
            (date(2001, 1, 2), "A"),
            (date(2001, 1, 3), "A"),
            (date(2001, 1, 4), "A"),
            (date(2001, 1, 3), "B"),
            (date(2001, 1, 4), "B"),
        ],
        columns=["d", "s"],
    )
    assert df.select(
        pl.col("d").shift_and_fill(-1, pl.col("d").max()).over("s")
    ).dtypes == [pl.Date]


def test_date_arr_concat() -> None:
    expected = {"d": [[date(2000, 1, 1), date(2000, 1, 1)]]}

    # type date
    df = pl.DataFrame({"d": [date(2000, 1, 1)]})
    assert df.select(pl.col("d").arr.concat(pl.col("d"))).to_dict(False) == expected
    # type list[date]
    df = pl.DataFrame({"d": [[date(2000, 1, 1)]]})
    assert df.select(pl.col("d").arr.concat(pl.col("d"))).to_dict(False) == expected


def test_date_timedelta() -> None:
    df = pl.DataFrame({"date": pl.date_range(date(2001, 1, 1), date(2001, 1, 3), "1d")})
    assert df.with_columns(
        [
            (pl.col("date") + timedelta(days=1)).alias("date_plus_one"),
            (pl.col("date") - timedelta(days=1)).alias("date_min_one"),
        ]
    ).to_dict(False) == {
        "date": [date(2001, 1, 1), date(2001, 1, 2), date(2001, 1, 3)],
        "date_plus_one": [date(2001, 1, 2), date(2001, 1, 3), date(2001, 1, 4)],
        "date_min_one": [date(2000, 12, 31), date(2001, 1, 1), date(2001, 1, 2)],
    }


def test_datetime_string_casts() -> None:
    df = pl.DataFrame(
        {
            "x": [1661855445123],
            "y": [1661855445123456],
            "z": [1661855445123456789],
        },
        columns=[
            ("x", pl.Datetime("ms")),
            ("y", pl.Datetime("us")),
            ("z", pl.Datetime("ns")),
        ],
    )
    assert df.select(
        [pl.col("x").dt.strftime("%F %T").alias("w")]
        + [pl.col(d).cast(str) for d in df.columns]
    ).rows() == [
        (
            "2022-08-30 10:30:45",
            "2022-08-30 10:30:45.123",
            "2022-08-30 10:30:45.123456",
            "2022-08-30 10:30:45.123456789",
        )
    ]


def test_short_formats() -> None:
    s = pl.Series(["20202020", "2020"])
    assert s.str.strptime(pl.Date, "%Y", strict=False).to_list() == [
        None,
        date(2020, 1, 1),
    ]
    assert s.str.strptime(pl.Date, "%foo", strict=False).to_list() == [None, None]


@pytest.mark.parametrize(
    ("time_string", "fmt", "datatype", "expected"),
    [
        ("Jul/2020", "%b/%Y", pl.Date, date(2020, 7, 1)),
        ("Jan/2020", "%b/%Y", pl.Date, date(2020, 1, 1)),
        ("02/Apr/2020", "%d/%b/%Y", pl.Date, date(2020, 4, 2)),
        ("Dec/2020", "%b/%Y", pl.Datetime, datetime(2020, 12, 1, 0, 0)),
        ("Nov/2020", "%b/%Y", pl.Datetime, datetime(2020, 11, 1, 0, 0)),
        ("02/Feb/2020", "%d/%b/%Y", pl.Datetime, datetime(2020, 2, 2, 0, 0)),
    ],
)
def test_abbrev_month(
    time_string: str, fmt: str, datatype: PolarsTemporalType, expected: date
) -> None:
    s = pl.Series([time_string])
    result = s.str.strptime(datatype, fmt).item()
    assert result == expected


def test_iso_year() -> None:
    assert pl.Series([datetime(2022, 1, 1, 7, 8, 40)]).dt.iso_year()[0] == 2021
    assert pl.Series([date(2022, 1, 1)]).dt.iso_year()[0] == 2021


def test_invalid_date_parsing_4898() -> None:
    assert pl.Series(["2022-09-18", "2022-09-50"]).str.strptime(
        pl.Date, "%Y-%m-%d", strict=False
    ).to_list() == [date(2022, 9, 18), None]


def test_cast_timezone() -> None:
    ny = zoneinfo.ZoneInfo("America/New_York")
    assert pl.DataFrame({"a": [datetime(2022, 9, 25, 14)]}).with_column(
        pl.col("a")
        .dt.with_time_zone("UTC")
        .dt.cast_time_zone("America/New_York")
        .alias("b")
    ).to_dict(False) == {
        "a": [datetime(2022, 9, 25, 14, 0)],
        "b": [datetime(2022, 9, 25, 14, 0, tzinfo=ny)],
    }


def test_tz_aware_get_idx_5010() -> None:
    when = int(
        datetime(2022, 1, 1, 12, tzinfo=zoneinfo.ZoneInfo("Asia/Shanghai")).timestamp()
    )
    a = pa.array([when]).cast(pa.timestamp("s", tz="Asia/Shanghai"))
    assert int(pl.from_arrow(a)[0].timestamp()) == when  # type: ignore[union-attr]


def test_tz_datetime_duration_arithm_5221() -> None:
    run_datetimes = [
        datetime.fromisoformat("2022-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2022-01-02T00:00:00+00:00"),
    ]
    out = pl.DataFrame(
        data={"run_datetime": run_datetimes},
        columns=[("run_datetime", pl.Datetime(time_zone="UTC"))],
    )
    utc = zoneinfo.ZoneInfo("UTC")
    assert out.to_dict(False) == {
        "run_datetime": [
            datetime(2022, 1, 1, 0, 0, tzinfo=utc),
            datetime(2022, 1, 2, 0, 0, tzinfo=utc),
        ]
    }


def test_auto_infer_time_zone() -> None:
    dt = datetime(2022, 10, 17, 10, tzinfo=zoneinfo.ZoneInfo("Asia/Shanghai"))
    s = pl.Series([dt])
    assert s.dtype == pl.Datetime("us", "Asia/Shanghai")
    assert s[0] == dt


def test_timezone_aware_date_range() -> None:
    low = datetime(2022, 10, 17, 10, tzinfo=zoneinfo.ZoneInfo("Asia/Shanghai"))
    high = datetime(2022, 11, 17, 10, tzinfo=zoneinfo.ZoneInfo("Asia/Shanghai"))

    assert pl.date_range(low, high, interval=timedelta(days=5)).to_list() == [
        datetime(2022, 10, 17, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 10, 22, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 10, 27, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 1, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 6, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 11, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
        datetime(2022, 11, 16, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="Asia/Shanghai")),
    ]

    with pytest.raises(
        ValueError,
        match="Cannot mix different timezone aware datetimes. "
        "Got: 'Asia/Shanghai' and 'None'",
    ):
        pl.date_range(
            low, high.replace(tzinfo=None), interval=timedelta(days=5), time_zone="UTC"
        )

    with pytest.raises(
        ValueError,
        match="Given time_zone is different from that timezone aware datetimes. "
        "Given: 'UTC', got: 'Asia/Shanghai'.",
    ):
        pl.date_range(low, high, interval=timedelta(days=5), time_zone="UTC")


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
    assert out.to_dict(False) == {
        "ix": [1, 2],
        "dt": [[datetime(2001, 1, 2, 0, 0)], [datetime(2001, 1, 1, 0, 0)]],
        "d": [[date(2001, 1, 1)], [date(2001, 1, 1)]],
        "t": [[time(10, 0)], [time(10, 0)]],
        "del": [[timedelta(days=10)], [timedelta(days=10)]],
        "str": [[{"a": time(10, 0)}], [{"a": time(10, 0)}]],
    }


def test_tz_localize() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(["2022-01-01", "2022-01-02"]).str.strptime(
                pl.Date, "%Y-%m-%d"
            )
        }
    )

    assert df.select(
        pl.col("date").cast(pl.Datetime).dt.tz_localize("America/New_York")
    ).to_dict(False) == {
        "date": [
            datetime(
                2022, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 1, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
        ]
    }


def test_tz_aware_truncate() -> None:
    test = pl.DataFrame(
        {
            "dt": pl.date_range(
                low=datetime(2022, 11, 1), high=datetime(2022, 11, 4), interval="12h"
            ).dt.tz_localize("America/New_York")
        }
    )
    assert test.with_column(pl.col("dt").dt.truncate("1d").alias("trunced")).to_dict(
        False
    ) == {
        "dt": [
            datetime(
                2022, 11, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 1, 12, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 2, 12, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 3, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 3, 12, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 4, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
        ],
        "trunced": [
            datetime(
                2022, 11, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 3, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 3, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 4, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
        ],
    }

    # 5507
    lf = pl.DataFrame(
        {
            "naive": pl.date_range(
                low=datetime(2021, 12, 31, 23),
                high=datetime(2022, 1, 1, 6),
                interval="1h",
            )
        }
    ).lazy()
    lf = lf.with_column(pl.col("naive").dt.tz_localize("UTC").alias("UTC"))
    lf = lf.with_column(pl.col("UTC").dt.with_time_zone("US/Central").alias("CST"))
    lf = lf.with_column(pl.col("CST").dt.truncate("1d").alias("CST truncated"))
    assert lf.collect().to_dict(False) == {
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
            datetime(2021, 12, 31, 23, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 1, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 2, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 3, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 4, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 5, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
            datetime(2022, 1, 1, 6, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
        ],
        "CST": [
            datetime(2021, 12, 31, 17, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 18, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 19, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 21, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 22, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 23, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2022, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
        ],
        "CST truncated": [
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2021, 12, 31, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
            datetime(2022, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")),
        ],
    }


def test_tz_aware_strftime() -> None:
    df = pl.DataFrame(
        {
            "dt": pl.date_range(
                low=datetime(2022, 11, 1), high=datetime(2022, 11, 4), interval="24h"
            ).dt.tz_localize("America/New_York")
        }
    )
    assert df.with_column(pl.col("dt").dt.strftime("%c").alias("fmt")).to_dict(
        False
    ) == {
        "dt": [
            datetime(
                2022, 11, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 2, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 3, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                2022, 11, 4, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
        ],
        "fmt": [
            "Tue Nov  1 00:00:00 2022",
            "Wed Nov  2 00:00:00 2022",
            "Thu Nov  3 00:00:00 2022",
            "Fri Nov  4 00:00:00 2022",
        ],
    }


def test_tz_aware_filter_lit() -> None:
    start = datetime(1970, 1, 1)
    stop = datetime(1970, 1, 1, 7)
    dt = datetime(1970, 1, 1, 6, tzinfo=zoneinfo.ZoneInfo("America/New_York"))

    assert (
        pl.DataFrame({"date": pl.date_range(start, stop, "1h")})
        .with_column(pl.col("date").dt.tz_localize("America/New_York").alias("nyc"))
        .filter(pl.col("nyc") < dt)
    ).to_dict(False) == {
        "date": [
            datetime(1970, 1, 1, 0, 0),
            datetime(1970, 1, 1, 1, 0),
            datetime(1970, 1, 1, 2, 0),
            datetime(1970, 1, 1, 3, 0),
            datetime(1970, 1, 1, 4, 0),
            datetime(1970, 1, 1, 5, 0),
        ],
        "nyc": [
            datetime(
                1970, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                1970, 1, 1, 1, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                1970, 1, 1, 2, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                1970, 1, 1, 3, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                1970, 1, 1, 4, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
            datetime(
                1970, 1, 1, 5, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York")
            ),
        ],
    }


def test_asof_join_by_forward() -> None:
    dfa = pl.DataFrame(
        {"category": ["a", "a", "a", "a", "a"], "value_one": [1, 2, 3, 5, 12]}
    )

    dfb = pl.DataFrame({"category": ["a"], "value_two": [3]})

    assert dfa.join_asof(
        dfb,
        left_on="value_one",
        right_on="value_two",
        by="category",
        strategy="forward",
    ).to_dict(False) == {
        "category": ["a", "a", "a", "a", "a"],
        "value_one": [1, 2, 3, 5, 12],
        "value_two": [3, 3, 3, None, None],
    }


def test_truncate_by_calendar_weeks() -> None:
    # 5557
    start = datetime(2022, 11, 14, 0, 0, 0)
    end = datetime(2022, 11, 20, 0, 0, 0)

    assert (
        pl.date_range(start, end, timedelta(days=1), name="date")
        .to_frame()
        .select([pl.col("date").dt.truncate("1w")])
    ).to_dict(False) == {
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

    assert df.select(pl.col("date").dt.truncate("1w")).to_dict(False) == {
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
            [
                pl.col("date").dt.truncate("2w").alias("2w"),
                pl.col("date").dt.truncate("3w").alias("3w"),
                pl.col("date").dt.truncate("4w").alias("4w"),
                pl.col("date").dt.truncate("5w").alias("5w"),
                pl.col("date").dt.truncate("17w").alias("17w"),
            ]
        )
    ).to_dict(False) == {
        "2w": [date(2022, 4, 11), date(2022, 11, 21)],
        "3w": [date(2022, 4, 4), date(2022, 11, 14)],
        "4w": [date(2022, 3, 28), date(2022, 11, 7)],
        "5w": [date(2022, 3, 21), date(2022, 10, 31)],
        "17w": [date(2021, 12, 27), date(2022, 8, 8)],
    }


def test_round_by_week() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [
                    # Sunday and Monday
                    "1998-04-12",
                    "2022-11-28",
                ]
            ).str.strptime(pl.Date, "%Y-%m-%d")
        }
    )

    assert (
        df.select(
            [
                pl.col("date").dt.round("7d").alias("7d"),
                pl.col("date").dt.round("1w").alias("1w"),
            ]
        )
    ).to_dict(False) == {
        "7d": [date(1998, 4, 9), date(2022, 12, 1)],
        "1w": [date(1998, 4, 13), date(2022, 11, 28)],
    }


def test_cast_time_to_duration() -> None:
    assert pl.Series([time(hour=0, minute=0, second=2)]).cast(
        pl.Duration
    ).item() == timedelta(seconds=2)


def test_tz_aware_day_weekday() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 9)
    df = pl.DataFrame({"date": pl.date_range(start, stop, timedelta(days=3))})

    df = df.with_columns(
        [
            pl.col("date").dt.with_time_zone("Asia/Tokyo").alias("tyo_date"),
            pl.col("date").dt.with_time_zone("America/New_York").alias("ny_date"),
        ]
    )

    assert df.select(
        [
            pl.col("date").dt.day().alias("day"),
            pl.col("tyo_date").dt.day().alias("tyo_day"),
            pl.col("ny_date").dt.day().alias("ny_day"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("tyo_date").dt.weekday().alias("tyo_weekday"),
            pl.col("ny_date").dt.weekday().alias("ny_weekday"),
        ]
    ).to_dict(False) == {
        "day": [1, 4, 7],
        "tyo_day": [1, 4, 7],
        "ny_day": [31, 3, 6],
        "weekday": [1, 4, 7],
        "tyo_weekday": [1, 4, 7],
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
            [
                (pl.col("timestamp").cummin()).alias("cummin"),
                (pl.col("timestamp").cummax()).alias("cummax"),
            ]
        )
        .with_columns(
            [
                (pl.col("cummin") + pl.duration(hours=24)).alias("cummin+24"),
                (pl.col("cummax") + pl.duration(hours=24)).alias("cummax+24"),
            ]
        )
        .collect()
    ).to_dict(False) == {
        "timestamp": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
        ],
        "cummin": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 0, 0),
        ],
        "cummax": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
        ],
        "cummin+24": [
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 0, 0),
        ],
        "cummax+24": [
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 3, 0, 0),
            datetime(2023, 1, 4, 0, 0),
        ],
    }


def test_rolling_groupby_empty_groups_by_take_6330() -> None:
    df = pl.DataFrame({"Event": ["Rain", "Sun"]}).join(
        pl.DataFrame(
            {
                "Date": [1, 2, 3, 4],
            }
        ),
        how="cross",
    )
    assert (
        df.groupby_rolling(
            index_column="Date",
            period="2i",
            offset="-2i",
            by="Event",
            closed="left",
        ).agg([pl.count()])
    ).to_dict(False) == {
        "Event": ["Rain", "Rain", "Rain", "Rain", "Sun", "Sun", "Sun", "Sun"],
        "Date": [1, 2, 3, 4, 1, 2, 3, 4],
        "count": [0, 1, 2, 2, 0, 1, 2, 2],
    }

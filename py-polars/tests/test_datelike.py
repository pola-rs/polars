import io
import typing
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_series import verify_series_and_expr_api

import polars as pl


def test_fill_null() -> None:
    dt = datetime.strptime("2021-01-01", "%Y-%m-%d")
    s = pl.Series("A", [dt, None])

    for fill_val in (dt, pl.lit(dt)):
        out = s.fill_null(fill_val)  # type: ignore

        assert out.null_count() == 0
        assert out.dt[0] == dt
        assert out.dt[1] == dt

    dt1 = date(2001, 1, 1)
    dt2 = date(2001, 1, 2)
    dt3 = date(2001, 1, 3)
    s = pl.Series("a", [dt1, dt2, dt3, None])
    dt_2 = date(2001, 1, 4)
    for fill_val in (dt_2, pl.lit(dt_2)):
        out = s.fill_null(fill_val)  # type: ignore

        assert out.null_count() == 0
        assert out.dt[0] == dt1
        assert out.dt[1] == dt2
        assert out.dt[-1] == dt_2


def test_filter_date() -> None:
    dataset = pl.DataFrame(
        {"date": ["2020-01-02", "2020-01-03", "2020-01-04"], "index": [1, 2, 3]}
    )
    df = dataset.with_column(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    assert df.filter(pl.col("date") <= pl.lit(datetime(2019, 1, 3))).is_empty()
    assert df.filter(pl.col("date") < pl.lit(datetime(2020, 1, 4))).shape[0] == 2
    assert df.filter(pl.col("date") < pl.lit(datetime(2020, 1, 5))).shape[0] == 3
    assert df.filter(pl.col("date") <= pl.lit(datetime(2019, 1, 3))).is_empty()
    assert df.filter(pl.col("date") < pl.lit(datetime(2020, 1, 4))).shape[0] == 2
    assert df.filter(pl.col("date") < pl.lit(datetime(2020, 1, 5))).shape[0] == 3


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
    assert (deltas + pl.Series([datetime(2000, 1, 1)])) == out


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
    assert out[0] == out[1]


def test_timestamp() -> None:
    a = pl.Series("a", [a * 1000_000 for a in [10000, 20000, 30000]], dtype=pl.Datetime)
    assert a.dt.timestamp("ms") == [10000, 20000, 30000]
    out = a.dt.to_python_datetime()
    assert isinstance(out[0], datetime)
    assert a.dt.min() == out[0]
    assert a.dt.max() == out[2]

    df = pl.DataFrame([out])
    # test if rows returns objects
    assert isinstance(df.row(0)[0], datetime)


def test_from_pydatetime() -> None:
    dates = [
        datetime(2021, 1, 1),
        datetime(2021, 1, 2),
        datetime(2021, 1, 3),
        datetime(2021, 1, 4, 12, 12),
        None,
    ]
    s = pl.Series("name", dates)
    assert s.dtype == pl.Datetime
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]

    dates = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), None]  # type: ignore
    s = pl.Series("name", dates)
    assert s.dtype == pl.Date
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]


def test_to_python_datetime() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.select(pl.col("a").cast(pl.Datetime).dt.to_python_datetime())["a"].dtype
        == pl.Object
    )
    assert (
        df.select(pl.col("a").cast(pl.Datetime).dt.timestamp())["a"].dtype == pl.Int64
    )


def test_from_numpy() -> None:
    # numpy support is limited; will be stored as object
    x = np.asarray(range(100_000, 200_000, 10_000), dtype="datetime64[s]")
    s = pl.Series(x)
    assert s[0] == x[0]
    assert len(s) == 10


def test_datetime_consistency() -> None:
    # dt = datetime(2021, 1, 1, 10, 30, 45, 123456)
    dt = datetime(2021, 1, 1, 10, 30, 45, 123000)
    df = pl.DataFrame({"date": [dt]})
    assert df["date"].dt[0] == dt
    assert df.select(pl.lit(dt))["literal"].dt[0] == dt


def test_timezone() -> None:
    ts = pa.timestamp("s")
    data = pa.array([1000, 2000], type=ts)
    s: pl.Series = pl.from_arrow(data)  # type: ignore

    # with timezone; we do expect a warning here
    tz_ts = pa.timestamp("s", tz="America/New_York")
    tz_data = pa.array([1000, 2000], type=tz_ts)
    with pytest.warns(Warning):
        tz_s: pl.Series = pl.from_arrow(tz_data)  # type: ignore

    # timezones have no effect, i.e. `s` equals `tz_s`
    assert s.series_equal(tz_s)


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
        .dt.and_time_unit("ns")
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

    s1 = pl.date_range(start, stop, timedelta(minutes=30), name="dates", time_unit="ms")
    s2 = pl.date_range(start, stop, timedelta(minutes=30), name="dates", time_unit="ns")
    # we can pass strings and timedeltas
    for out in [s1.dt.truncate("1h"), s2.dt.truncate(timedelta(hours=1))]:
        assert out.dt[0] == start
        assert out.dt[1] == start
        assert out.dt[2] == start + timedelta(hours=1)
        assert out.dt[3] == start + timedelta(hours=1)
        # ...
        assert out.dt[-3] == stop - timedelta(hours=1)
        assert out.dt[-2] == stop - timedelta(hours=1)
        assert out.dt[-1] == stop


def test_date_range() -> None:
    result = pl.date_range(
        datetime(1985, 1, 1), datetime(2015, 7, 1), timedelta(days=1, hours=12)
    )
    assert len(result) == 7426
    assert result.dt[0] == datetime(1985, 1, 1)
    assert result.dt[1] == datetime(1985, 1, 2, 12, 0)
    assert result.dt[2] == datetime(1985, 1, 4, 0, 0)
    assert result.dt[-1] == datetime(2015, 6, 30, 12, 0)

    for tu in ["ns", "ms"]:
        rng = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 1, 2), "2h", time_unit=tu
        )
        assert rng.time_unit == tu
        assert rng.shape == (13,)
        assert rng.dt[0] == datetime(2020, 1, 1)
        assert rng.dt[-1] == datetime(2020, 1, 2)


def test_date_comp() -> None:
    one = datetime(2001, 1, 1)
    two = datetime(2001, 1, 2)
    a = pl.Series("a", [one, two])

    assert (a == one).to_list() == [True, False]
    assert (a != one).to_list() == [False, True]
    assert (a > one).to_list() == [False, True]
    assert (a >= one).to_list() == [True, True]
    assert (a < one).to_list() == [False, False]
    assert (a <= one).to_list() == [True, False]

    one = date(2001, 1, 1)  # type: ignore
    two = date(2001, 1, 2)  # type: ignore
    a = pl.Series("a", [one, two])
    assert (a == one).to_list() == [True, False]
    assert (a != one).to_list() == [False, True]
    assert (a > one).to_list() == [False, True]
    assert (a >= one).to_list() == [True, True]
    assert (a < one).to_list() == [False, False]
    assert (a <= one).to_list() == [True, False]

    # also test if the conversion stays correct with wide date ranges
    one = date(201, 1, 1)  # type: ignore
    two = date(201, 1, 2)  # type: ignore
    a = pl.Series("a", [one, two])
    assert (a == one).to_list() == [True, False]
    assert (a == two).to_list() == [False, True]

    one = date(5001, 1, 1)  # type: ignore
    two = date(5001, 1, 2)  # type: ignore
    a = pl.Series("a", [one, two])
    assert (a == one).to_list() == [True, False]
    assert (a == two).to_list() == [False, True]


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
        datetime(2021, 4, 1),
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
    for d in [dates, datetimes]:
        df = pl.DataFrame(
            {
                "a": d,
                "b": ["a", "b", "a", "b"],
                "c": [1.0, 2.0, 1.1, 2.2],
            }
        )
        out = (
            df.groupby("b")
            .agg([pl.col("a"), pl.col("c").pct_change()])
            .explode(["a", "c"])
        )
        assert out.shape == (4, 3)


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

    out = df.groupby_rolling(index_column="dt", period="2d").agg(
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
    )

    up = df.upsample(
        time_column="time", every="1mo", by="admin", maintain_order=True
    ).select(pl.all().forward_fill())

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
    )

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

    assert pl.from_arrow(a)["timestamp"].to_list() == timestamps  # type: ignore


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
    assert df_in["Timestamp"][0] == datetime(2022, 1, 1, 0, 0)


def test_epoch() -> None:
    dates = pl.Series("dates", [datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    for unit in ["ns", "us", "ms"]:
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
                datetime(2020, 1, 1, 0, 0),
                datetime(2020, 1, 1, 0, 0),
                datetime(2020, 3, 1, 0, 0),
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


def test_duration_function() -> None:
    df = pl.DataFrame(
        {
            "datetime": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
            "add": [1, 2],
        }
    )

    out = df.select(
        [
            (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
            (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
            (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
            (pl.col("datetime") + pl.duration(milliseconds="add")).alias(
                "add_milliseconds"
            ),
            (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
        ]
    )

    expected = pl.DataFrame(
        {
            "add_weeks": [datetime(2022, 1, 8), datetime(2022, 1, 16)],
            "add_days": [datetime(2022, 1, 2), datetime(2022, 1, 4)],
            "add_seconds": [
                datetime(2022, 1, 1, second=1),
                datetime(2022, 1, 2, second=2),
            ],
            "add_milliseconds": [
                datetime(2022, 1, 1, microsecond=1000),
                datetime(2022, 1, 2, microsecond=2000),
            ],
            "add_hours": [datetime(2022, 1, 1, hour=1), datetime(2022, 1, 2, hour=2)],
        }
    )

    assert out.frame_equal(expected)


def test_rolling_groupby_by_argument() -> None:
    df = pl.DataFrame({"times": range(10), "groups": [1] * 4 + [2] * 6})

    out = df.groupby_rolling("times", "5i", by=["groups"]).agg(
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
    assert (
        df.groupby_rolling(index_column="Date", period="1w")
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
    dates = """2016-05-25 13:30:00.023
2016-05-25 13:30:00.023
2016-05-25 13:30:00.030
2016-05-25 13:30:00.041
2016-05-25 13:30:00.048
2016-05-25 13:30:00.049
2016-05-25 13:30:00.072
2016-05-25 13:30:00.075""".split(
        "\n"
    )

    ticker = """GOOG
MSFT
MSFT
MSFT
GOOG
AAPL
GOOG
MSFT""".split(
        "\n"
    )

    quotes = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [720.5, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        }
    )

    dates = """2016-05-25 13:30:00.023
2016-05-25 13:30:00.038
2016-05-25 13:30:00.048
2016-05-25 13:30:00.048
2016-05-25 13:30:00.048""".split(
        "\n"
    )

    ticker = """MSFT
MSFT
GOOG
GOOG
AAPL""".split(
        "\n"
    )

    trades = pl.DataFrame(
        {
            "dates": pl.Series(dates).str.strptime(pl.Datetime, fmt=fmt),
            "ticker": ticker,
            "bid": [51.95, 51.95, 720.77, 720.92, 98.0],
        }
    )

    out = trades.join_asof(quotes, on="dates", strategy="backward")
    assert out.columns == ["dates", "ticker", "bid", "ticker_right", "bid_right"]
    assert (out["dates"].cast(int) / 1000).to_list() == [
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


def test_lambda_with_python_datetime_return_type() -> None:
    df = pl.DataFrame({"timestamp": [1284286794, 1234567890]})

    assert df.with_column(
        pl.col("timestamp").apply(lambda x: datetime(2010, 9, 12)).alias("my_date_time")
    )["my_date_time"].to_list() == [
        datetime(2010, 9, 12),
        datetime(2010, 9, 12),
    ]


def test_timelike_init() -> None:
    durations = [timedelta(days=1), timedelta(days=2)]
    dates = [date(2022, 1, 1), date(2022, 1, 2)]
    datetimes = [datetime(2022, 1, 1), datetime(2022, 1, 2)]

    for ts in [durations, dates, datetimes]:
        s = pl.Series(ts)
        assert s.to_list() == ts


def test_duration_filter() -> None:
    date_df = pl.DataFrame(
        {
            "start_date": [date(2022, 1, 1), date(2022, 1, 1), date(2022, 1, 1)],
            "end_date": [date(2022, 1, 7), date(2022, 2, 20), date(2023, 1, 1)],
        }
    ).with_column((pl.col("end_date") - pl.col("start_date")).alias("time_passed"))

    assert date_df.filter(pl.col("time_passed") < timedelta(days=30)).shape[0] == 1
    assert date_df.filter(pl.col("time_passed") >= timedelta(days=30)).shape[0] == 2


def test_agg_logical() -> None:
    dates = [date(2001, 1, 1), date(2002, 1, 1)]
    s = pl.Series(dates)
    assert s.max() == dates[1]
    assert s.min() == dates[0]


@typing.no_type_check
def test_from_time_arrow() -> None:
    times = pa.array([10, 20, 30], type=pa.time32("s"))
    times_table = pa.table([times], names=["times"])

    assert pl.from_arrow(times_table).to_series().to_list() == [
        time(0, 0, 10),
        time(0, 0, 20),
        time(0, 0, 30),
    ]


def test_datetime_strptime_patterns() -> None:
    # note that all should be year first
    df = pl.Series(
        "date",
        [
            "09-05-2019" "2018-09-05",
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
    assert s.null_count() == 1
    assert s[0] is None

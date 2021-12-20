from datetime import date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

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


def test_diff_datetime() -> None:

    df = pl.DataFrame(
        {
            "timestamp": ["2021-02-01", "2021-03-1", "2021-04-1"],
            "guild": [1, 2, 3],
            "char": ["a", "a", "b"],
        }
    )

    out = (
        df.with_columns(
            [
                pl.col("timestamp").str.strptime(pl.Date, fmt="%Y-%m-%d"),
            ]
        ).with_columns([pl.col("timestamp").diff().over(pl.col("char"))])
    )["timestamp"]

    assert out[0] == out[1]


def test_timestamp() -> None:
    a = pl.Series("a", [a * 1000_000 for a in [10000, 20000, 30000]], dtype=pl.Datetime)
    assert a.dt.timestamp() == [10000, 20000, 30000]
    out = a.dt.to_python_datetime()
    print(a.cast(int))
    print(out)
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
    assert out[0] == datetime(1970, 1, 1, 0, 2, 3, 543000)


def test_rows() -> None:
    s0 = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)
    s1 = pl.Series("datetime", [a * 1_000_000 for a in [123543, 283478, 1243]]).cast(
        pl.Datetime
    )
    df = pl.DataFrame([s0, s1])

    rows = df.rows()
    assert rows[0][0] == date(2308, 4, 2)
    assert rows[0][1] == datetime(1970, 1, 1, 0, 2, 3, 543000)


def test_to_numpy() -> None:
    s0 = pl.Series("date", [123543, 283478, 1243]).cast(pl.Date)
    s1 = pl.Series("datetime", [123543, 283478, 1243]).cast(pl.Datetime)
    assert str(s0.to_numpy()) == "['2308-04-02' '2746-02-20' '1973-05-28']"
    assert (
        str(s1.to_numpy()[:2])
        == "['1970-01-01T00:02:03.543' '1970-01-01T00:04:43.478']"
    )


def test_truncate() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 2)
    s = pl.date_range(start, stop, timedelta(minutes=30), name="dates")

    # we can pass strings and timedeltas
    for out in [s.dt.truncate("1h"), s.dt.truncate(timedelta(hours=1))]:
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
    assert (a == two).to_list() == [False, True]
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

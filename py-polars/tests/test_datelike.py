from datetime import date, datetime

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
    for fill_val in (dt_2, pl.lit(dt_2)):  # type: ignore
        out = s.fill_null(fill_val)  # type: ignore

        assert out.null_count() == 0
        assert out.dt[0] == dt1
        assert out.dt[1] == dt2
        assert out.dt[-1] == dt_2


def test_downsample() -> None:
    s = pl.Series(
        "datetime",
        [
            946684800000,
            946684860000,
            946684920000,
            946684980000,
            946685040000,
            946685100000,
            946685160000,
            946685220000,
            946685280000,
            946685340000,
            946685400000,
            946685460000,
            946685520000,
            946685580000,
            946685640000,
            946685700000,
            946685760000,
            946685820000,
            946685880000,
            946685940000,
        ],
    ).cast(pl.Datetime)
    s2 = s.clone()
    df = pl.DataFrame({"a": s, "b": s2})
    out = df.downsample("a", rule="minute", n=5).first()
    assert out.shape == (4, 2)

    # OLHC
    out = df.downsample("a", rule="minute", n=5).agg(
        {"b": ["first", "min", "max", "last"]}
    )
    assert out.shape == (4, 5)

    # test to_pandas as well.
    out = df.to_pandas()
    assert out["a"].dtype == "datetime64[ns]"


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
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Datetime)
    assert a.dt.timestamp() == [10000, 20000, 30000]
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
    # fmt dates and nulls
    print(s)

    dates = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), None]  # type: ignore
    s = pl.Series("name", dates)
    assert s.dtype == pl.Date
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]

    # fmt dates and nulls
    print(s)


def test_to_python_datetime() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.select(pl.col("a").cast(pl.Datetime).dt.to_python_datetime())["a"].dtype
        == pl.Object
    )
    assert (
        df.select(pl.col("a").cast(pl.Datetime).dt.timestamp())["a"].dtype == pl.Int64
    )


def test_datetime_consistency() -> None:
    dt = datetime(2021, 1, 1)
    df = pl.DataFrame({"date": [dt]})
    assert df["date"].dt[0] == dt
    assert df.select(pl.lit(dt))["literal"].dt[0] == dt

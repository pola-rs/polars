import datetime

import pandas as pd

import polars as pl


def test_from_pandas_datetime():
    ts = datetime.datetime(2021, 1, 1, 20, 20, 20, 20)
    s = pd.Series([ts, ts])
    s = pl.from_pandas(s.to_frame("a"))["a"]
    assert s.dt.hour()[0] == 20
    assert s.dt.minute()[0] == 20
    assert s.dt.second()[0] == 20

    date_times = pd.date_range(
        "2021-06-24 00:00:00", "2021-06-24 10:00:00", freq="1H", closed="left"
    )
    s = pl.from_pandas(date_times)
    assert s[0] == 1624492800000
    assert s[-1] == 1624525200000
    # checks dispatch
    s.dt.round("hour", 2)
    s.dt.round("day", 5)

    # checks lazy dispatch
    pl.DataFrame([s.rename("foo")])[pl.col("foo").dt.round("hour", 2)]

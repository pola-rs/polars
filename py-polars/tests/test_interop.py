import datetime

import numpy as np
import pandas as pd
import pyarrow as pa

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


def test_arrow_list_roundtrip():
    # https://github.com/pola-rs/polars/issues/1064
    pl.from_arrow(pa.table({"a": [1], "b": [[1, 2]]})).to_arrow()


def test_arrow_dict_to_polars():
    pa_dict = pa.DictionaryArray.from_arrays(
        indices=np.array([0, 1, 2, 3, 1, 0, 2, 3, 3, 2]),
        dictionary=np.array(["AAA", "BBB", "CCC", "DDD"]),
    ).cast(pa.large_utf8())

    s = pl.Series(
        name="s",
        values=["AAA", "BBB", "CCC", "DDD", "BBB", "AAA", "CCC", "DDD", "DDD", "CCC"],
    )

    assert s.series_equal(pl.Series("pa_dict", pa_dict))


def test_arrow_list_chunked_array():
    a = pa.array([[1, 2], [3, 4]])
    ca = pa.chunked_array([a, a, a])
    s = pl.from_arrow(ca)
    assert s.dtype == pl.List

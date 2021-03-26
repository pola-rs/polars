import pandas as pd
import datetime
import polars as pl


def test_from_pandas_datetime():
    ts = datetime.datetime(2021, 1, 1, 20, 20, 20, 20)
    s = pd.Series([ts, ts])
    s = pl.from_pandas(s.to_frame("a"))["a"]
    assert s.hour()[0] == 20
    assert s.minute()[0] == 20
    assert s.second()[0] == 20

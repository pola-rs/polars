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


def test_from_pandas_null():
    df = pd.DataFrame([{"a": None}, {"a": None}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Float64]
    assert out["a"][0] is None

    df = pd.DataFrame([{"a": None, "b": 1}, {"a": None, "b": 2}])
    out = pl.DataFrame(df)
    assert out.dtypes == [pl.Float64, pl.Int64]


def test_from_pandas_nested_list():
    # this panicked in https://github.com/pola-rs/polars/issues/1615
    pddf = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [["x", "y"], ["x", "y", "z"], ["x"], ["x", "y"]]}
    )
    pldf = pl.from_pandas(pddf)
    print(pldf)
    assert pldf.shape == (4, 2)


def test_from_pandas_categorical_none():
    s = pd.Series(["a", "b", "c", pd.NA], dtype="category")
    out = pl.from_pandas(s)
    assert out.dtype == pl.Categorical
    assert out.to_list() == ["a", "b", "c", None]


def test_from_dict():
    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.from_dict(data)
    assert df.shape == (2, 2)


def test_from_dicts():
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    df = pl.from_dicts(data)
    assert df.shape == (3, 2)


def test_from_records():
    data = [[1, 2, 3], [4, 5, 6]]
    df = pl.from_records(data, columns=["a", "b"])
    assert df.shape == (3, 2)


def test_from_arrow():
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pl.from_arrow(data)
    assert df.shape == (3, 2)


def test_from_pandas_dataframe():
    pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    df = pl.from_pandas(pd_df)
    assert df.shape == (2, 3)


def test_from_pandas_series():
    pd_series = pd.Series([1, 2, 3], name="pd")
    df = pl.from_pandas(pd_series)
    assert df.shape == (3,)

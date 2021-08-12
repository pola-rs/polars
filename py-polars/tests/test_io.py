import copy
import gzip
import io
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest

import polars as pl


def test_to_from_buffer(df):
    df = df.drop("strings_nulls")

    for to_fn, from_fn in zip(
        [df.to_parquet, df.to_csv], [df.read_parquet, df.read_csv]
    ):
        f = io.BytesIO()
        to_fn(f)
        f.seek(0)

        df_1 = from_fn(f)
        assert df.frame_equal(df_1, null_equal=True)


def test_read_web_file():
    url = "https://raw.githubusercontent.com/ritchie46/polars/master/examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv"
    df = pl.read_csv(url)
    assert df.shape == (27, 4)


def test_parquet_chunks():
    """
    This failed in https://github.com/ritchie46/polars/issues/545
    """
    cases = [
        1048576,
        1048577,
    ]

    for case in cases:
        f = io.BytesIO()
        # repeat until it has case instances
        df = pd.DataFrame(
            np.tile([1.0, pd.to_datetime("2010-10-10")], [case, 1]),
            columns=["floats", "dates"],
        )
        print(df)

        # write as parquet
        df.to_parquet(f)

        print(f"reading {case} dates with polars...", end="")
        f.seek(0)

        # read it with polars
        polars_df = pl.read_parquet(f)


def test_parquet_datetime():
    """
    This failed because parquet writers cast date64 to date32
    """
    f = io.BytesIO()
    data = {
        "datetime": [  # unix timestamp in ms
            1618354800000,
            1618354740000,
            1618354680000,
            1618354620000,
            1618354560000,
        ],
        "laf_max": [73.1999969482, 71.0999984741, 74.5, 69.5999984741, 69.6999969482],
        "laf_eq": [59.5999984741, 61.0, 62.2999992371, 56.9000015259, 60.0],
    }
    df = pl.DataFrame(data)
    df = df.with_column(df["datetime"].cast(pl.Date64))

    df.to_parquet(f, use_pyarrow=True)
    f.seek(0)
    read = pl.read_parquet(f)
    assert read.frame_equal(df)


def test_csv_null_values():
    csv = """
a,b,c
na,b,c
a,na,c"""
    f = io.StringIO(csv)

    df = pl.read_csv(f, null_values="na")
    assert df[0, "a"] is None
    assert df[1, "b"] is None

    csv = """
a,b,c
na,b,c
a,n/a,c"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values=["na", "n/a"])
    assert df[0, "a"] is None
    assert df[1, "b"] is None

    csv = """
a,b,c
na,b,c
a,n/a,c"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values={"a": "na", "b": "n/a"})
    assert df[0, "a"] is None
    assert df[1, "b"] is None


def test_partial_dtype_overwrite():
    csv = """
a,b,c
1,2,3
1,2,3
"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, dtype=[pl.Utf8])
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Int64]


def test_partial_column_rename():
    csv = """
a,b,c
1,2,3
1,2,3
"""
    f = io.StringIO(csv)
    for use in [True, False]:
        f.seek(0)
        df = pl.read_csv(f, new_columns=["foo"], use_pyarrow=use)
        assert df.columns == ["foo", "b", "c"]


def test_compressed_csv():
    # gzip compression
    csv = """
a,b,c
1,a,1.0
2,b,2.0,
3,c,3.0
"""
    fout = io.BytesIO()
    with gzip.GzipFile(fileobj=fout, mode="w") as f:
        f.write(csv.encode())

    bytes = fout.getvalue()
    out = pl.read_csv(bytes)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    assert out.frame_equal(expected)

    # now from disk
    out = pl.read_csv("tests/files/gzipped.csv")
    assert out.frame_equal(expected)

    # now with column projection
    out = pl.read_csv(bytes, columns=["a", "b"])
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    assert out.frame_equal(expected)

    # zlib compression
    bytes = zlib.compress(csv.encode())
    out = pl.read_csv(bytes)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    assert out.frame_equal(expected)

    # no compression
    f = io.BytesIO(b"a, b\n1,2\n")
    out = pl.read_csv(f)
    expected = pl.DataFrame({"a": [1], "b": [2]})
    assert out.frame_equal(expected)


def test_empty_bytes():
    b = b""
    with pytest.raises(ValueError):
        pl.read_csv(b)


def test_pickle():
    a = pl.Series("a", [1, 2])
    b = pickle.dumps(a)
    out = pickle.loads(b)
    assert a.series_equal(out)
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    b = pickle.dumps(df)
    out = pickle.loads(b)
    assert df.frame_equal(out, null_equal=True)


def test_copy():
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    copy.copy(df).frame_equal(df, True)
    copy.deepcopy(df).frame_equal(df, True)

    a = pl.Series("a", [1, 2])
    copy.copy(a).series_equal(a, True)
    copy.deepcopy(a).series_equal(a, True)

import io

import numpy as np
import pandas as pd

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

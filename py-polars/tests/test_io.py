import io

import numpy as np
import pandas as pd
from utils import get_complete_df

import polars as pl


def test_to_from_buffer():
    df = get_complete_df()
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

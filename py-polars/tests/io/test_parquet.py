# flake8: noqa: W191,E101
import io
import os
from typing import List

import numpy as np
import pandas as pd
import pytest

import polars as pl


@pytest.fixture
def compressions() -> List[str]:
    return ["uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"]


def test_to_from_buffer(df: pl.DataFrame, compressions: List[str]) -> None:
    for compression in compressions:
        if compression == "lzo":
            # lzo compression is not supported now
            with pytest.raises(pl.ArrowError):
                buf = io.BytesIO()
                df.write_parquet(buf, compression=compression)
                buf.seek(0)
                _ = pl.read_parquet(buf)

            with pytest.raises(OSError):
                buf = io.BytesIO()
                df.write_parquet(buf, compression=compression, use_pyarrow=True)
                buf.seek(0)
                _ = pl.read_parquet(buf)
        else:
            buf = io.BytesIO()
            df.write_parquet(buf, compression=compression)
            buf.seek(0)
            read_df = pl.read_parquet(buf)
            assert df.frame_equal(read_df)


def test_to_from_file(
    io_test_dir: str, df: pl.DataFrame, compressions: List[str]
) -> None:
    f = os.path.join(io_test_dir, "small.parquet")
    for compression in compressions:
        if compression == "lzo":
            # lzo compression is not supported now
            with pytest.raises(pl.ArrowError):
                df.write_parquet(f, compression=compression)
                _ = pl.read_parquet(f)

            with pytest.raises(OSError):
                df.write_parquet(f, compression=compression, use_pyarrow=True)
                _ = pl.read_parquet(f)
        else:
            df.write_parquet(f, compression=compression)
            read_df = pl.read_parquet(f)
            assert df.frame_equal(read_df)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=["b", "c"], use_pyarrow=False)
    assert expected.frame_equal(read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=[1, 2], use_pyarrow=False)
    assert expected.frame_equal(read_df)


def test_parquet_chunks() -> None:
    """
    This failed in https://github.com/pola-rs/polars/issues/545
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

        # write as parquet
        df.to_parquet(f)
        f.seek(0)

        # read it with polars
        polars_df = pl.read_parquet(f)
        assert pl.DataFrame(df).frame_equal(polars_df)


def test_parquet_datetime() -> None:
    """
    This failed because parquet writers cast datetime to Date
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
    df = df.with_column(df["datetime"].cast(pl.Datetime))

    # todo! test all compressions here
    df.write_parquet(f, use_pyarrow=True, compression="snappy")
    f.seek(0)
    read = pl.read_parquet(f)
    assert read.frame_equal(df)


def test_nested_parquet() -> None:
    f = io.BytesIO()
    data = [
        {"a": [{"b": 0}]},
        {"a": [{"b": 1}, {"b": 2}]},
    ]
    df = pd.DataFrame(data)
    df.to_parquet(f)

    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.columns == ["a"]
    assert isinstance(read.dtypes[0], pl.datatypes.List)
    assert isinstance(read.dtypes[0].inner, pl.datatypes.Struct)

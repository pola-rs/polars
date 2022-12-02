from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal_local_categoricals

if TYPE_CHECKING:
    from polars.internals.type_aliases import ParquetCompression

COMPRESSIONS: list[ParquetCompression] = [
    "lz4",
    "uncompressed",
    "snappy",
    "gzip",
    "lzo",
    "brotli",
    "zstd",
]


@pytest.fixture
def compressions() -> list[ParquetCompression]:
    return COMPRESSIONS


def test_to_from_buffer(
    df: pl.DataFrame, compressions: list[ParquetCompression]
) -> None:
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
            assert_frame_equal_local_categoricals(df, read_df)

    for use_pyarrow in [True, False]:
        buf = io.BytesIO()
        df.write_parquet(buf, use_pyarrow=use_pyarrow)
        buf.seek(0)
        read_df = pl.read_parquet(buf, use_pyarrow=use_pyarrow)
        assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_file(
    io_test_dir: str, df: pl.DataFrame, compressions: list[ParquetCompression]
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
            assert_frame_equal_local_categoricals(df, read_df)


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
    # This failed in https://github.com/pola-rs/polars/issues/545
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


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_parquet_datetime(use_pyarrow: bool, compression: ParquetCompression) -> None:
    if compression == "lzo":
        back_end = "C++" if use_pyarrow else "Rust"
        pytest.skip(
            f"LZO compression is not currently not supported by the {back_end}"
            f"implementation of Arrow."
        )

    # This failed because parquet writers cast datetime to Date
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

    df.write_parquet(f, use_pyarrow=use_pyarrow, compression=compression)
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


def test_glob_parquet(io_test_dir: str) -> None:
    path = os.path.join(io_test_dir, "small*.parquet")
    assert pl.read_parquet(path).shape == (3, 16)
    assert pl.scan_parquet(path).collect().shape == (3, 16)


def test_chunked_round_trip() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1] * 2,
            "l": [[1] for j in range(0, 2)],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [2] * 3,
            "l": [[2] for j in range(0, 3)],
        }
    )

    df = df1.vstack(df2)

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert pl.read_parquet(f).frame_equal(df)


def test_lazy_self_join_file_cache_prop_3979(io_test_dir: str) -> None:
    path = os.path.join(io_test_dir, "small.parquet")
    a = pl.scan_parquet(path)
    b = pl.DataFrame({"a": [1]}).lazy()

    assert a.join(b, how="cross").collect().shape == (3, 17)
    assert b.join(a, how="cross").collect().shape == (3, 17)


def recursive_logical_type() -> None:
    df = pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
    df = df.with_column(pl.col("str").cast(pl.Categorical))

    df_groups = df.groupby("group").agg([pl.col("str").list().alias("cat_list")])
    f = io.BytesIO()
    df_groups.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.dtypes == [pl.Int64, pl.List(pl.Categorical)]
    assert read.shape == (2, 2)


def test_nested_dictionary() -> None:
    with pl.StringCache():
        df = (
            pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
            .with_column(pl.col("str").cast(pl.Categorical))
            .groupby("group")
            .agg([pl.col("str").list().alias("cat_list")])
        )
        f = io.BytesIO()
        df.write_parquet(f)
        f.seek(0)

        read_df = pl.read_parquet(f)
        assert df.frame_equal(read_df)


def test_row_group_size_saturation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()

    # request larger chunk than rows in df
    df.write_parquet(f, row_group_size=1024)
    f.seek(0)
    assert pl.read_parquet(f).frame_equal(df)

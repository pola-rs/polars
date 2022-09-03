from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal_local_categoricals


@pytest.fixture
def compressions() -> list[str]:
    return ["uncompressed", "lz4", "zstd"]


def test_from_to_buffer(df: pl.DataFrame, compressions: list[str]) -> None:
    for compression in compressions:
        buf = io.BytesIO()
        df.write_ipc(buf, compression=compression)  # type: ignore[arg-type]
        buf.seek(0)
        read_df = pl.read_ipc(buf, use_pyarrow=False)
        assert_frame_equal_local_categoricals(df, read_df)


def test_from_to_file(
    io_test_dir: str, df: pl.DataFrame, compressions: list[str]
) -> None:
    f_ipc = os.path.join(io_test_dir, "small.ipc")

    # does not yet work on windows because we hold an mmap?
    if os.name != "nt":
        for compression in compressions:
            for f in (str(f_ipc), Path(f_ipc)):
                df.write_ipc(f, compression=compression)  # type: ignore[arg-type]
                df_read = pl.read_ipc(f, use_pyarrow=False)  # type: ignore[arg-type]
                assert_frame_equal_local_categoricals(df, df_read)


def test_columns_arg(io_test_dir: str) -> None:
    if os.name != "nt":
        f_ipc = os.path.join(io_test_dir, "small.ipc")
        assert pl.read_ipc(f_ipc, columns=["bools"]).columns == ["bools"]


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_ipc(f)
    f.seek(0)

    read_df = pl.read_ipc(f, columns=["b", "c"], use_pyarrow=False)
    assert expected.frame_equal(read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    f = io.BytesIO()
    df.write_ipc(f)
    f.seek(0)

    read_df = pl.read_ipc(f, columns=[1, 2], use_pyarrow=False)
    assert expected.frame_equal(read_df)


def test_compressed_simple() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    compressions = [None, "uncompressed", "lz4", "zstd"]

    for compression in compressions:
        f = io.BytesIO()
        df.write_ipc(f, compression)  # type: ignore[arg-type]
        f.seek(0)

        df_read = pl.read_ipc(f, use_pyarrow=False)
        assert df_read.frame_equal(df)


def test_ipc_schema(compressions: list[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})

    for compression in compressions:
        f = io.BytesIO()
        df.write_ipc(f, compression=compression)  # type: ignore[arg-type]
        f.seek(0)

        assert pl.read_ipc_schema(f) == {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}


def test_ipc_schema_from_file(
    io_test_dir: str, df_no_lists: pl.DataFrame, compressions: list[str]
) -> None:
    df = df_no_lists
    f_ipc = os.path.join(io_test_dir, "small.ipc")

    # does not yet work on windows because we hold an mmap?
    if os.name != "nt":
        for compression in compressions:
            for f in (str(f_ipc), Path(f_ipc)):
                df.write_ipc(f, compression=compression)  # type: ignore[arg-type]
                assert pl.read_ipc_schema(f) == {  # type: ignore[arg-type]
                    "bools": pl.Boolean,
                    "bools_nulls": pl.Boolean,
                    "int": pl.Int64,
                    "int_nulls": pl.Int64,
                    "floats": pl.Float64,
                    "floats_nulls": pl.Float64,
                    "strings": pl.Utf8,
                    "strings_nulls": pl.Utf8,
                    "date": pl.Date,
                    "datetime": pl.Datetime,
                    "time": pl.Time,
                    "cat": pl.Categorical,
                }


def test_ipc_column_order() -> None:
    df = pl.DataFrame(
        {
            "cola": ["x", "y", "z"],
            "colb": [1, 2, 3],
            "colc": [4.5, 5.6, 6.7],
        }
    )
    f = io.BytesIO()
    df.write_ipc(f)
    f.seek(0)

    columns = ["colc", "colb", "cola"]
    # read file into polars; the specified column order is no longer respected
    assert pl.read_ipc(f, columns=columns).columns == columns


def test_glob_ipc(io_test_dir: str) -> None:
    if os.name != "nt":
        path = os.path.join(io_test_dir, "small*.ipc")
        assert pl.scan_ipc(path).collect().shape == (3, 12)
        assert pl.read_ipc(path, use_pyarrow=False).shape == (3, 12)


def test_from_float16() -> None:
    # Create a feather file with a 16-bit floating point column
    pandas_df = pd.DataFrame({"column": [1.0]}, dtype="float16")
    f = io.BytesIO()
    pandas_df.to_feather(f)
    f.seek(0)
    assert pl.read_ipc(f, use_pyarrow=False).dtypes == [pl.Float32]

# flake8: noqa: W191,E101
import io
import os
from pathlib import Path
from typing import List, Union

import pyarrow as pa
import pytest

import polars as pl


@pytest.fixture
def compressions() -> List[str]:
    return ["uncompressed", "lz4", "zstd"]


def test_from_to_buffer(df: pl.DataFrame, compressions: List[str]) -> None:
    for compression in compressions:
        buf = io.BytesIO()
        df.write_ipc(buf, compression=compression)  # type: ignore
        buf.seek(0)
        read_df = pl.read_ipc(buf)
        assert df.frame_equal(read_df)


def test_from_to_file(
    io_test_dir: str, df: pl.DataFrame, compressions: List[str]
) -> None:
    f_ipc = os.path.join(io_test_dir, "small.ipc")

    # does not yet work on windows because we hold an mmap?
    if os.name != "nt":
        for compression in compressions:
            for f in (str(f_ipc), Path(f_ipc)):
                df.write_ipc(f, compression=compression)  # type: ignore
                df_read = pl.read_ipc(f)  # type: ignore
                assert df.frame_equal(df_read)


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
        df.write_ipc(f, compression)  # type: ignore
        f.seek(0)

        df_read = pl.read_ipc(f, use_pyarrow=False)
        assert df_read.frame_equal(df)


def test_ipc_schema(compressions: List[str]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})

    for compression in compressions:
        f = io.BytesIO()
        df.write_ipc(f, compression=compression)  # type: ignore
        f.seek(0)

        assert pl.read_ipc_schema(f) == {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}


def test_ipc_schema_from_file(
    io_test_dir: str, df_no_lists: pl.DataFrame, compressions: List[str]
) -> None:
    df = df_no_lists
    f_ipc = os.path.join(io_test_dir, "small.ipc")

    # does not yet work on windows because we hold an mmap?
    if os.name != "nt":
        for compression in compressions:
            for f in (str(f_ipc), Path(f_ipc)):
                df.write_ipc(f, compression=compression)  # type: ignore
                assert pl.read_ipc_schema(f) == {  # type: ignore
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

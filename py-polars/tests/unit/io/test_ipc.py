from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_equal_local_categoricals

if TYPE_CHECKING:
    from polars.internals.type_aliases import IpcCompression

COMPRESSIONS = ["uncompressed", "lz4", "zstd"]


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_buffer(df: pl.DataFrame, compression: IpcCompression) -> None:
    buf = io.BytesIO()
    df.write_ipc(buf, compression=compression)
    buf.seek(0)
    read_df = pl.read_ipc(buf, use_pyarrow=False)
    assert_frame_equal_local_categoricals(df, read_df)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("path_type", [str, Path])
def test_from_to_file(
    df: pl.DataFrame, compression: IpcCompression, path_type: type[str] | type[Path]
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.ipc"
        file_path_cast = path_type(file_path)
        df.write_ipc(file_path_cast, compression=compression)
        df_read = pl.read_ipc(file_path_cast, use_pyarrow=False)

    assert_frame_equal_local_categoricals(df, df_read)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_select_columns_from_file(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.ipc"
        df.write_ipc(file_path)
        df_read = pl.read_ipc(file_path, columns=["bools"])

    assert df_read.columns == ["bools"]


def test_select_columns_from_buffer() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_ipc(f)
    f.seek(0)

    read_df = pl.read_ipc(f, columns=["b", "c"], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


def test_select_columns_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_ipc(f)
    f.seek(0)

    read_df = pl.read_ipc(f, columns=[1, 2], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_compressed_simple(compression: IpcCompression) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_ipc(f, compression)
    f.seek(0)

    df_read = pl.read_ipc(f, use_pyarrow=False)
    assert_frame_equal(df_read, df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_ipc_schema(compression: IpcCompression) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})

    f = io.BytesIO()
    df.write_ipc(f, compression=compression)
    f.seek(0)

    expected = {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}
    assert pl.read_ipc_schema(f) == expected


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("path_type", [str, Path])
def test_ipc_schema_from_file(
    df_no_lists: pl.DataFrame,
    compression: IpcCompression,
    path_type: type[str] | type[Path],
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.ipc"
        file_path_cast = path_type(file_path)
        df_no_lists.write_ipc(file_path_cast, compression=compression)
        schema = pl.read_ipc_schema(file_path_cast)

    expected = {
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
    assert schema == expected


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


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_glob_ipc(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.ipc"
        df.write_ipc(file_path)

        file_path_glob = Path(temp_dir) / "small*.ipc"

        result_scan = pl.scan_ipc(file_path_glob).collect()
        result_read = pl.read_ipc(file_path_glob, use_pyarrow=False)

    for result in [result_scan, result_read]:
        assert_frame_equal_local_categoricals(result, df)


def test_from_float16() -> None:
    # Create a feather file with a 16-bit floating point column
    pandas_df = pd.DataFrame({"column": [1.0]}, dtype="float16")
    f = io.BytesIO()
    pandas_df.to_feather(f)
    f.seek(0)
    assert pl.read_ipc(f, use_pyarrow=False).dtypes == [pl.Float32]

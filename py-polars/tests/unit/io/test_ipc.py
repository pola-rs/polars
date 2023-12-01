from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import IpcCompression

COMPRESSIONS = ["uncompressed", "lz4", "zstd"]


def read_ipc(is_stream: bool, *args: Any, **kwargs: Any) -> pl.DataFrame:
    if is_stream:
        return pl.read_ipc_stream(*args, **kwargs)
    else:
        return pl.read_ipc(*args, **kwargs)


def write_ipc(df: pl.DataFrame, is_stream: bool, *args: Any, **kwargs: Any) -> Any:
    if is_stream:
        return df.write_ipc_stream(*args, **kwargs)
    else:
        return df.write_ipc(*args, **kwargs)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("stream", [True, False])
def test_from_to_buffer(
    df: pl.DataFrame, compression: IpcCompression, stream: bool
) -> None:
    # use an ad-hoc buffer (file=None)
    buf1 = write_ipc(df, stream, None, compression=compression)
    read_df = read_ipc(stream, buf1, use_pyarrow=False)
    assert_frame_equal(df, read_df, categorical_as_str=True)

    # explicitly supply an existing buffer
    buf2 = io.BytesIO()
    write_ipc(df, stream, buf2, compression=compression)
    buf2.seek(0)
    read_df = read_ipc(stream, buf2, use_pyarrow=False)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("path_as_string", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.write_disk()
def test_from_to_file(
    df: pl.DataFrame,
    compression: IpcCompression,
    path_as_string: bool,
    tmp_path: Path,
    stream: bool,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.ipc"
    if path_as_string:
        file_path = str(file_path)  # type: ignore[assignment]
    write_ipc(df, stream, file_path, compression=compression)
    df_read = read_ipc(stream, file_path, use_pyarrow=False)

    assert_frame_equal(df, df_read, categorical_as_str=True)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.write_disk()
def test_select_columns_from_file(
    df: pl.DataFrame, tmp_path: Path, stream: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.ipc"
    write_ipc(df, stream, file_path)
    df_read = read_ipc(stream, file_path, columns=["bools"])

    assert df_read.columns == ["bools"]


@pytest.mark.parametrize("stream", [True, False])
def test_select_columns_from_buffer(stream: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    write_ipc(df, stream, f)
    f.seek(0)

    read_df = read_ipc(stream, f, columns=["b", "c"], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("stream", [True, False])
def test_select_columns_projection(stream: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    write_ipc(df, stream, f)
    f.seek(0)

    read_df = read_ipc(stream, f, columns=[1, 2], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("stream", [True, False])
def test_compressed_simple(compression: IpcCompression, stream: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    write_ipc(df, stream, f, compression)
    f.seek(0)

    df_read = read_ipc(stream, f, use_pyarrow=False)
    assert_frame_equal(df_read, df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_ipc_schema(compression: IpcCompression) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})

    f = io.BytesIO()
    df.write_ipc(f, compression=compression)
    f.seek(0)

    expected = {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}
    assert pl.read_ipc_schema(f) == expected


@pytest.mark.write_disk()
@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("path_as_string", [True, False])
def test_ipc_schema_from_file(
    df_no_lists: pl.DataFrame,
    compression: IpcCompression,
    path_as_string: bool,
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.ipc"
    if path_as_string:
        file_path = str(file_path)  # type: ignore[assignment]
    df_no_lists.write_ipc(file_path, compression=compression)
    schema = pl.read_ipc_schema(file_path)

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


@pytest.mark.parametrize("stream", [True, False])
def test_ipc_column_order(stream: bool) -> None:
    df = pl.DataFrame(
        {
            "cola": ["x", "y", "z"],
            "colb": [1, 2, 3],
            "colc": [4.5, 5.6, 6.7],
        }
    )
    f = io.BytesIO()
    write_ipc(df, stream, f)
    f.seek(0)

    columns = ["colc", "colb", "cola"]
    # read file into polars; the specified column order is no longer respected
    assert read_ipc(stream, f, columns=columns).columns == columns


@pytest.mark.write_disk()
def test_glob_ipc(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.ipc"
    df.write_ipc(file_path)

    file_path_glob = tmp_path / "small*.ipc"

    result_scan = pl.scan_ipc(file_path_glob).collect()
    result_read = pl.read_ipc(file_path_glob, use_pyarrow=False)

    for result in [result_scan, result_read]:
        assert_frame_equal(result, df, categorical_as_str=True)


def test_from_float16() -> None:
    # Create a feather file with a 16-bit floating point column
    pandas_df = pd.DataFrame({"column": [1.0]}, dtype="float16")
    f = io.BytesIO()
    pandas_df.to_feather(f)
    f.seek(0)
    assert pl.read_ipc(f, use_pyarrow=False).dtypes == [pl.Float32]


@pytest.mark.write_disk()
def test_sink_categorical_ipc_6407(
    tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.ipc"

    df = pl.DataFrame({"col": ["x", "y", "z"]})
    df.lazy().with_columns(pl.col("col").cast(pl.Categorical)).sink_ipc(file_path)
    result_scan = pl.scan_ipc(file_path).collect(streaming=True)

    assert_frame_equal(result_scan, df, categorical_as_str=True)
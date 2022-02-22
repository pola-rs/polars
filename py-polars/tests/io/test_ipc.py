# flake8: noqa: W191,E101
import io
import os
from typing import List, Literal

import pytest

import polars as pl

Compression = Literal["uncompressed", "lz4", "zstd"]


@pytest.fixture
def compressions() -> List[Compression]:
    return ["uncompressed", "lz4", "zstd"]


def test_from_to_buffer(df: pl.DataFrame, compressions: List[Compression]) -> None:
    for compression in compressions:
        if compression == "uncompressed":
            buf = io.BytesIO()
            df.to_ipc(buf, compression=compression)
            buf.seek(0)
            read_df = pl.read_ipc(buf)
            assert df.frame_equal(read_df)
        else:
            # Error with ipc compression
            with pytest.raises(ValueError):
                buf = io.BytesIO()
                df.to_ipc(buf, compression=compression)
                buf.seek(0)
                _ = pl.read_ipc(buf)


def test_from_to_file(
    io_test_dir: str, df: pl.DataFrame, compressions: List[Compression]
) -> None:
    f = os.path.join(io_test_dir, "small.ipc")

    for compression in compressions:
        if compression == "uncompressed":
            df.to_ipc(f, compression=compression)
            df_read = pl.read_ipc(str(f))
            assert df.frame_equal(df_read)
        else:
            # Error with ipc compression
            with pytest.raises(ValueError):
                df.to_ipc(f, compression=compression)
                _ = pl.read_ipc(str(f))


def test_select_columns():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.to_ipc(f)  # type: ignore
    f.seek(0)

    read_df = pl.read_ipc(f, columns=["b", "c"], use_pyarrow=False)  # type: ignore
    assert expected.frame_equal(read_df)


def test_select_projection():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    f = io.BytesIO()
    df.to_ipc(f)  # type: ignore
    f.seek(0)

    read_df = pl.read_ipc(f, columns=[1, 2], use_pyarrow=False)  # type: ignore
    assert expected.frame_equal(read_df)


def test_compressed_simple() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    compressions = [None, "uncompressed", "lz4", "zstd"]

    for compression in compressions:
        f = io.BytesIO()
        df.to_ipc(f, compression)  # type: ignore
        f.seek(0)

        df_read = pl.read_ipc(f, use_pyarrow=False)
        assert df_read.frame_equal(df)


def test_ipc_schema(compressions: List[Compression]) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})

    for compression in compressions:
        f = io.BytesIO()
        df.to_ipc(f, compression=compression)
        f.seek(0)

        assert pl.read_ipc_schema(f) == {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}

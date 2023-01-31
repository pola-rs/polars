from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import AvroCompression

COMPRESSIONS = ["uncompressed", "snappy", "deflate"]


@pytest.fixture()
def example_df() -> pl.DataFrame:
    return pl.DataFrame({"i64": [1, 2], "f64": [0.1, 0.2], "utf8": ["a", "b"]})


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_buffer(example_df: pl.DataFrame, compression: AvroCompression) -> None:
    buf = io.BytesIO()
    example_df.write_avro(buf, compression=compression)
    buf.seek(0)

    read_df = pl.read_avro(buf)
    assert_frame_equal(example_df, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_file(example_df: pl.DataFrame, compression: AvroCompression) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.avro"
        example_df.write_avro(file_path, compression=compression)
        df_read = pl.read_avro(file_path)

    assert_frame_equal(example_df, df_read)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=["b", "c"])
    assert_frame_equal(expected, read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=[1, 2])
    assert_frame_equal(expected, read_df)

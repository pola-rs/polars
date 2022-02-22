# flake8: noqa: W191,E101
import os
from typing import List

import pytest

import polars as pl
from polars import io


@pytest.fixture
def example_df() -> pl.DataFrame:
    return pl.DataFrame({"i64": [1, 2], "f64": [0.1, 0.2], "utf8": ["a", "b"]})


@pytest.fixture
def compressions() -> List[str]:
    return ["uncompressed", "snappy", "deflate"]


def test_from_to_buffer(example_df: pl.DataFrame, compressions: List[str]) -> None:
    for compression in compressions:
        buf = io.BytesIO()
        example_df.to_avro(buf, compression=compression)  # type: ignore
        buf.seek(0)
        read_df = pl.read_avro(buf)
        assert example_df.frame_equal(read_df)


def test_from_to_file(
    io_test_dir: str, example_df: pl.DataFrame, compressions: List[str]
) -> None:
    f = os.path.join(io_test_dir, "small.avro")

    for compression in compressions:
        example_df.to_avro(f, compression=compression)  # type: ignore
        df_read = pl.read_avro(str(f))
        assert example_df.frame_equal(df_read)

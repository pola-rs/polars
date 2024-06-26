from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

if TYPE_CHECKING:
    from pathlib import Path


@given(
    lf=dataframes(
        lazy=True,
        excluded_dtypes=[
            pl.Float32,  # Bug, see: https://github.com/pola-rs/polars/issues/17211
            pl.Float64,  # Bug, see: https://github.com/pola-rs/polars/issues/17211
        ],
    )
)
def test_lf_serde_roundtrip(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize()
    result = pl.LazyFrame.deserialize(io.StringIO(serialized))
    assert_frame_equal(result, lf, categorical_as_str=True)


@pytest.fixture()
def lf() -> pl.LazyFrame:
    """Sample LazyFrame for testing serialization/deserialization."""
    return pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).select("a").sum()


def test_lf_serde(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize()
    assert isinstance(serialized, str)
    result = pl.LazyFrame.deserialize(io.StringIO(serialized))

    assert_frame_equal(result, lf)


@pytest.mark.parametrize("buf", [io.BytesIO(), io.StringIO()])
def test_lf_serde_to_from_buffer(lf: pl.LazyFrame, buf: io.IOBase) -> None:
    lf.serialize(buf)
    buf.seek(0)
    result = pl.LazyFrame.deserialize(buf)
    assert_frame_equal(lf, result)


@pytest.mark.write_disk()
def test_lf_serde_to_from_file(lf: pl.LazyFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.bin"
    lf.serialize(file_path)
    result = pl.LazyFrame.deserialize(file_path)

    assert_frame_equal(lf, result)


def test_lazyframe_serde_json(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize(format="json")
    assert isinstance(serialized, bytes)
    result = pl.LazyFrame.deserialize(io.BytesIO(serialized), format="json")
    assert_frame_equal(result, lf)

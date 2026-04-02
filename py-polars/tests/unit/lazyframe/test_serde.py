from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import SerializationFormat
    from tests.conftest import PlMonkeyPatch


@given(
    lf=dataframes(
        lazy=True,
        excluded_dtypes=[pl.Struct],
    )
)
@example(lf=pl.LazyFrame({"foo": ["a", "b", "a"]}, schema={"foo": pl.Enum(["b", "a"])}))
def test_lf_serde_roundtrip_binary(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize(format="binary")
    result = pl.LazyFrame.deserialize(io.BytesIO(serialized), format="binary")
    assert_frame_equal(result, lf, categorical_as_str=True)


@given(
    lf=dataframes(
        lazy=True,
        excluded_dtypes=[
            pl.Float32,  # Bug, see: https://github.com/pola-rs/polars/issues/17211
            pl.Float64,  # Bug, see: https://github.com/pola-rs/polars/issues/17211
            pl.Struct,  # Outer nullability not supported
        ],
    )
)
@pytest.mark.filterwarnings("ignore")
def test_lf_serde_roundtrip_json(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize(format="json")
    result = pl.LazyFrame.deserialize(io.StringIO(serialized), format="json")
    assert_frame_equal(result, lf, categorical_as_str=True)


@pytest.fixture
def lf() -> pl.LazyFrame:
    """Sample LazyFrame for testing serialization/deserialization."""
    return pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).select("a").sum()


@pytest.mark.filterwarnings("ignore")
def test_lf_serde_json_stringio(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize(format="json")
    assert isinstance(serialized, str)
    result = pl.LazyFrame.deserialize(io.StringIO(serialized), format="json")
    assert_frame_equal(result, lf)


def test_lf_serde(lf: pl.LazyFrame) -> None:
    serialized = lf.serialize()
    assert isinstance(serialized, bytes)
    result = pl.LazyFrame.deserialize(io.BytesIO(serialized))
    assert_frame_equal(result, lf)


@pytest.mark.parametrize(
    ("format", "buf"),
    [
        ("binary", io.BytesIO()),
        ("json", io.StringIO()),
        ("json", io.BytesIO()),
    ],
)
@pytest.mark.filterwarnings("ignore")
def test_lf_serde_to_from_buffer(
    lf: pl.LazyFrame, format: SerializationFormat, buf: io.IOBase
) -> None:
    lf.serialize(buf, format=format)
    buf.seek(0)
    result = pl.LazyFrame.deserialize(buf, format=format)
    assert_frame_equal(lf, result)


@pytest.mark.write_disk
def test_lf_serde_to_from_file(lf: pl.LazyFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.bin"
    lf.serialize(file_path)
    result = pl.LazyFrame.deserialize(file_path)

    assert_frame_equal(lf, result)


def test_lf_deserialize_validation() -> None:
    f = io.BytesIO(b"hello world!")
    with pytest.raises(ComputeError, match="expected value at line 1 column 1"):
        pl.LazyFrame.deserialize(f, format="json")


@pytest.mark.write_disk
def test_lf_serde_scan(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "dataset.parquet"

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write_parquet(path)
    lf = pl.scan_parquet(path)

    ser = lf.serialize()
    result = pl.LazyFrame.deserialize(io.BytesIO(ser))
    assert_frame_equal(result, lf)
    assert_frame_equal(result.collect(), df)


@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_lf_serde_version_specific_lambda() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).select(
        pl.col("a").map_elements(lambda x: x + 1, return_dtype=pl.Int64)
    )
    ser = lf.serialize()

    result = pl.LazyFrame.deserialize(io.BytesIO(ser))
    expected = pl.LazyFrame({"a": [2, 3, 4]})
    assert_frame_equal(result, expected)


def custom_function(x: pl.Series) -> pl.Series:
    return x + 1


@pytest.mark.may_fail_cloud  # reason: cloud does not have access to this scope
@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_lf_serde_version_specific_named_function() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).select(
        pl.col("a").map_batches(custom_function, return_dtype=pl.Int64)
    )
    ser = lf.serialize()

    result = pl.LazyFrame.deserialize(io.BytesIO(ser))
    expected = pl.LazyFrame({"a": [2, 3, 4]})
    assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_lf_serde_map_batches_on_lazyframe() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).map_batches(lambda x: x + 1)
    ser = lf.serialize()

    result = pl.LazyFrame.deserialize(io.BytesIO(ser))
    expected = pl.LazyFrame({"a": [2, 3, 4]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("max_byte_slice_len", [1, 2, 3, 100, 4294967295])
def test_lf_serde_chunked_bytes(
    plmonkeypatch: PlMonkeyPatch, max_byte_slice_len: int
) -> None:
    plmonkeypatch.setenv(
        "POLARS_SERIALIZE_LAZYFRAME_MAX_BYTE_SLICE_LEN", str(max_byte_slice_len)
    )
    lf = pl.LazyFrame({"a": range(5000)})

    b = lf.serialize()

    assert_frame_equal(pl.LazyFrame.deserialize(io.BytesIO(b)).collect(), lf.collect())


def test_lf_collect_schema_does_not_change_serialize_25719() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})

    lf = df.lazy()
    lf.collect_schema()

    assert lf.serialize() == df.lazy().serialize()
    lf_sum = lf.sum()
    lf_sum.collect_schema()
    assert lf_sum.serialize() == df.lazy().sum().serialize()

    q = pl.concat([lf_sum, lf_sum])

    assert_frame_equal(
        pl.LazyFrame.deserialize(q.serialize()).collect(),
        pl.DataFrame({"x": [6, 6]}),
    )

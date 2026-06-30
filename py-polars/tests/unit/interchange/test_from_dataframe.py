from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_from_dataframe_polars() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_categorical() -> None:
    df = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})
    df_pa = df.to_arrow()

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=True)
    expected = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})
    assert_frame_equal(result, expected)


def test_from_dataframe_empty_string_zero_copy() -> None:
    df = pl.DataFrame({"a": []}, schema={"a": pl.String})
    df_pa = df.to_arrow()
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_empty_bool_zero_copy() -> None:
    df = pl.DataFrame(schema={"a": pl.Boolean})
    df_pd = df.to_pandas()
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pd, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_empty_categories_zero_copy() -> None:
    df = pl.DataFrame(schema={"a": pl.Enum([])})
    df_pa = df.to_arrow()
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pandas_zero_copy() -> None:
    data = {"a": [1, 2], "b": [3.0, 4.0]}

    df = pd.DataFrame(data)
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df, allow_copy=False)
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)


def test_from_dataframe_pyarrow_table_zero_copy() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3.0, 4.0],
        }
    )
    df_pa = df.to_arrow()

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_empty_table() -> None:
    df = pl.Series("a", dtype=pl.Int8).to_frame()
    df_pa = df.to_arrow()

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_recordbatch_zero_copy() -> None:
    a = pa.array([1, 2])
    b = pa.array([3.0, 4.0])

    batch = pa.record_batch([a, b], names=["a", "b"])
    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(batch, allow_copy=False)

    expected = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    assert_frame_equal(result, expected)


def test_from_dataframe_invalid_type() -> None:
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)  # type: ignore[arg-type]


def test_from_dataframe_pyarrow_boolean() -> None:
    df = pl.Series("a", [True, False]).to_frame()
    df_pa = df.to_arrow()

    result = pl.from_dataframe(df_pa)
    assert_frame_equal(result, df)

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_chunked() -> None:
    df = pl.Series("a", [0, 1], dtype=pl.Int8).to_frame()
    df_chunked = pl.concat([df[:1], df[1:]], rechunk=False)

    df_pa = df_chunked.to_arrow()
    result = pl.from_dataframe(df_pa, rechunk=False)

    assert_frame_equal(result, df_chunked)
    assert result.n_chunks() == 2


@pytest.mark.may_fail_auto_streaming
@pytest.mark.may_fail_cloud  # reason: chunking
def test_from_dataframe_chunked_string() -> None:
    df = pl.Series("a", ["a", None, "bc", "d", None, "efg"]).to_frame()
    df_chunked = pl.concat([df[:1], df[1:3], df[3:]], rechunk=False)

    df_pa = df_chunked.to_arrow()
    result = pl.from_dataframe(df_pa, rechunk=False)

    assert_frame_equal(result, df_chunked)
    assert result.n_chunks() == 3


def test_from_dataframe_pandas_nan_as_null() -> None:
    df = pd.Series([1.0, float("nan"), float("inf")], name="a").to_frame()
    result = pl.from_dataframe(df)
    expected = pl.Series("a", [1.0, None, float("inf")]).to_frame()
    assert_frame_equal(result, expected)
    assert result.n_chunks() == 1


def test_from_dataframe_pandas_boolean_bytes() -> None:
    df = pd.Series([True, False], name="a").to_frame()
    result = pl.from_dataframe(df)

    expected = pl.Series("a", [True, False]).to_frame()
    assert_frame_equal(result, expected)

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df, allow_copy=False)
    expected = pl.Series("a", [True, False]).to_frame()
    assert_frame_equal(result, expected)


def test_from_dataframe_categorical_pandas() -> None:
    values = ["a", "b", None, "a"]

    df_pd = pd.Series(values, dtype="category", name="a").to_frame()

    result = pl.from_dataframe(df_pd)
    expected = pl.Series("a", values, dtype=pl.Categorical).to_frame()
    assert_frame_equal(result, expected)

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pd, allow_copy=False)
    expected = pl.Series("a", values, dtype=pl.Categorical).to_frame()
    assert_frame_equal(result, expected)


def test_from_dataframe_categorical_pyarrow() -> None:
    values = ["a", "b", None, "a"]

    dtype = pa.dictionary(pa.int32(), pa.utf8())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])

    result = pl.from_dataframe(df_pa)
    expected = pl.Series("a", values, dtype=pl.Categorical).to_frame()
    assert_frame_equal(result, expected)

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, expected)


def test_from_dataframe_categorical_non_string_keys() -> None:
    values = [1, 2, None, 1]

    dtype = pa.dictionary(pa.uint32(), pa.int32())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])
    result = pl.from_dataframe(df_pa)
    expected = pl.DataFrame({"a": [1, 2, None, 1]}, schema={"a": pl.Int32})
    assert_frame_equal(result, expected)


def test_from_dataframe_categorical_non_u32_values() -> None:
    values = [None, None]

    dtype = pa.dictionary(pa.int8(), pa.utf8())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])

    result = pl.from_dataframe(df_pa)
    expected = pl.Series("a", values, dtype=pl.Categorical).to_frame()
    assert_frame_equal(result, expected)

    with pytest.deprecated_call(match="`allow_copy` is deprecated"):
        result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, expected)


def test_to_pandas_int8_20316() -> None:
    df = pl.Series("a", [None], pl.Int8).to_frame()
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)
    result = pl.from_dataframe(df_pd)
    assert_frame_equal(result, df)

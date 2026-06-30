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


def test_from_dataframe_invalid_type() -> None:
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)  # type: ignore[arg-type]


def test_from_dataframe_categorical_non_string_keys() -> None:
    values = [1, 2, None, 1]

    dtype = pa.dictionary(pa.uint32(), pa.int32())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])
    result = pl.from_dataframe(df_pa)
    expected = pl.DataFrame({"a": [1, 2, None, 1]}, schema={"a": pl.Int32})
    assert_frame_equal(result, expected)

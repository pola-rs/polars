from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import polars as pl


@pytest.fixture(
    params=[
        ("int8", [1, 3, 2], pl.Int8, np.int8),
        ("int16", [1, 3, 2], pl.Int16, np.int16),
        ("int32", [1, 3, 2], pl.Int32, np.int32),
        ("int64", [1, 3, 2], pl.Int64, np.int64),
        ("uint8", [1, 3, 2], pl.UInt8, np.uint8),
        ("uint16", [1, 3, 2], pl.UInt16, np.uint16),
        ("uint32", [1, 3, 2], pl.UInt32, np.uint32),
        ("uint64", [1, 3, 2], pl.UInt64, np.uint64),
        ("float32", [21.7, 21.8, 21], pl.Float32, np.float32),
        ("float64", [21.7, 21.8, 21], pl.Float64, np.float64),
        ("bool", [True, False, False], pl.Boolean, np.bool_),
        ("object", [21.7, "string1", object()], pl.Object, np.object_),
        ("str", ["string1", "string2", "string3"], pl.String, np.str_),
        ("intc", [1, 3, 2], pl.Int32, np.intc),
        ("uintc", [1, 3, 2], pl.UInt32, np.uintc),
        ("str_fixed", ["string1", "string2", "string3"], pl.String, np.str_),
        (
            "bytes",
            [b"byte_string1", b"byte_string2", b"byte_string3"],
            pl.Binary,
            np.bytes_,
        ),
    ]
)
def numpy_interop_test_data(request: Any) -> Any:
    return request.param


def test_df_from_numpy(numpy_interop_test_data: Any) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    df = pl.DataFrame({name: np.array(values, dtype=np_dtype)})
    assert [pl_dtype] == df.dtypes


def test_asarray(numpy_interop_test_data: Any) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    pl_series_to_numpy_array = np.asarray(pl.Series(name, values, pl_dtype))
    numpy_array = np.asarray(values, dtype=np_dtype)
    assert_array_equal(pl_series_to_numpy_array, numpy_array)


def test_to_numpy(numpy_interop_test_data: Any) -> None:
    name, values, pl_dtype, np_dtype = numpy_interop_test_data
    pl_series_to_numpy_array = pl.Series(name, values, pl_dtype).to_numpy()
    numpy_array = np.asarray(values, dtype=np_dtype)
    assert_array_equal(pl_series_to_numpy_array, numpy_array)


def test_numpy_to_lit() -> None:
    out = pl.select(pl.lit(np.array([1, 2, 3]))).to_series().to_list()
    assert out == [1, 2, 3]
    out = pl.select(pl.lit(np.float32(0))).to_series().to_list()
    assert out == [0.0]


def test_numpy_disambiguation() -> None:
    a = np.array([1, 2])
    df = pl.DataFrame({"a": a})
    result = df.with_columns(b=a).to_dict(as_series=False)  # type: ignore[arg-type]
    expected = {
        "a": [1, 2],
        "b": [1, 2],
    }
    assert result == expected


def test_respect_dtype_with_series_from_numpy() -> None:
    assert pl.Series("foo", np.array([1, 2, 3]), dtype=pl.UInt32).dtype == pl.UInt32

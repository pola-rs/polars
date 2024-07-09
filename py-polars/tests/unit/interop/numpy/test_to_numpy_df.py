from __future__ import annotations

from datetime import datetime
from decimal import Decimal as D
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_array_equal, assert_equal

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import series

if TYPE_CHECKING:
    import numpy.typing as npt

    from polars._typing import IndexOrder, PolarsDataType


def assert_zero_copy(s: pl.Series, arr: np.ndarray[Any, Any]) -> None:
    if s.len() == 0:
        return
    s_ptr = s._get_buffers()["values"]._get_buffer_info()[0]
    arr_ptr = arr.__array_interface__["data"][0]
    assert s_ptr == arr_ptr


@given(
    s=series(
        min_size=6,
        max_size=6,
        allowed_dtypes=[pl.Datetime, pl.Duration],
        allow_null=False,
        allow_chunks=False,
    )
)
def test_df_to_numpy_zero_copy(s: pl.Series) -> None:
    df = pl.DataFrame({"a": s[:3], "b": s[3:]})

    result = df.to_numpy(allow_copy=False)

    assert_zero_copy(s, result)
    assert result.flags.writeable is False


@pytest.mark.parametrize(
    ("order", "f_contiguous", "c_contiguous"),
    [
        ("fortran", True, False),
        ("c", False, True),
    ],
)
def test_to_numpy(order: IndexOrder, f_contiguous: bool, c_contiguous: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out_array = df.to_numpy(order=order)
    expected_array = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] == f_contiguous
    assert out_array.flags["C_CONTIGUOUS"] == c_contiguous

    structured_array = df.to_numpy(structured=True, order=order)
    expected_array = np.array(
        [(1, 1.0), (2, 2.0), (3, 3.0)], dtype=[("a", "<i8"), ("b", "<f8")]
    )
    assert_array_equal(structured_array, expected_array)
    assert structured_array.flags["F_CONTIGUOUS"]

    # check string conversion; if no nulls can optimise as a fixed-width dtype
    df = pl.DataFrame({"s": ["x", "y", None]})
    assert df["s"].has_nulls()
    assert_array_equal(
        df.to_numpy(structured=True),
        np.array([("x",), ("y",), (None,)], dtype=[("s", "O")]),
    )
    assert not df["s"][:2].has_nulls()
    assert_array_equal(
        df[:2].to_numpy(structured=True),
        np.array([("x",), ("y",)], dtype=[("s", "<U1")]),
    )


def test_to_numpy_structured() -> None:
    # round-trip structured array: validate init/export
    structured_array = np.array(
        [
            ("Google Pixel 7", 521.90, True),
            ("Apple iPhone 14 Pro", 999.00, True),
            ("OnePlus 11", 699.00, True),
            ("Samsung Galaxy S23 Ultra", 1199.99, False),
        ],
        dtype=np.dtype(
            [
                ("product", "U24"),
                ("price_usd", "float64"),
                ("in_stock", "bool"),
            ]
        ),
    )
    df = pl.from_numpy(structured_array)
    assert df.schema == {
        "product": pl.String,
        "price_usd": pl.Float64,
        "in_stock": pl.Boolean,
    }
    exported_array = df.to_numpy(structured=True)
    assert exported_array["product"].dtype == np.dtype("U24")
    assert_array_equal(exported_array, structured_array)

    # none/nan values
    df = pl.DataFrame({"x": ["a", None, "b"], "y": [5.5, None, -5.5]})
    exported_array = df.to_numpy(structured=True)

    assert exported_array.dtype == np.dtype([("x", object), ("y", float)])
    for name in df.columns:
        assert_equal(
            list(exported_array[name]),
            (
                df[name].fill_null(float("nan"))
                if df.schema[name].is_float()
                else df[name]
            ).to_list(),
        )


def test_numpy_preserve_uint64_4112() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.col("a").hash())
    assert df.to_numpy().dtype == np.dtype("uint64")
    assert df.to_numpy(structured=True).dtype == np.dtype([("a", "uint64")])


def test_df_to_numpy_decimal() -> None:
    decimal_data = [D("1.234"), D("2.345"), D("-3.456")]
    df = pl.Series("n", decimal_data).to_frame()

    result = df.to_numpy()

    expected = np.array(decimal_data).reshape((-1, 1))
    assert_array_equal(result, expected)


def test_df_to_numpy_zero_copy_path() -> None:
    rows = 10
    cols = 5
    x = np.ones((rows, cols), order="F")
    x[:, 1] = 2.0
    df = pl.DataFrame(x)
    x = df.to_numpy(allow_copy=False)
    assert x.flags.f_contiguous is True
    assert x.flags.writeable is False
    assert str(x[0, :]) == "[1. 2. 1. 1. 1.]"


def test_df_to_numpy_zero_copy_path_temporal() -> None:
    values = [datetime(1970 + i, 1, 1) for i in range(12)]
    s = pl.Series(values)
    df = pl.DataFrame({"a": s[:4], "b": s[4:8], "c": s[8:]})

    result = df.to_numpy(allow_copy=False)
    assert result.flags.f_contiguous is True
    assert result.flags.writeable is False
    assert result.tolist() == [list(row) for row in df.iter_rows()]


def test_to_numpy_zero_copy_path_writable() -> None:
    rows = 10
    cols = 5
    x = np.ones((rows, cols), order="F")
    x[:, 1] = 2.0
    df = pl.DataFrame(x)
    x = df.to_numpy(writable=True)
    assert x.flags["WRITEABLE"]


def test_df_to_numpy_structured_not_zero_copy() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    msg = "cannot create structured array without copying data"
    with pytest.raises(RuntimeError, match=msg):
        df.to_numpy(structured=True, allow_copy=False)


def test_df_to_numpy_writable_not_zero_copy() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    msg = "copy not allowed: cannot create a writable array without copying data"
    with pytest.raises(RuntimeError, match=msg):
        df.to_numpy(allow_copy=False, writable=True)


def test_df_to_numpy_not_zero_copy() -> None:
    df = pl.DataFrame({"a": [1, 2, None]})
    with pytest.raises(RuntimeError):
        df.to_numpy(allow_copy=False)


@pytest.mark.parametrize(
    ("schema", "expected_dtype"),
    [
        ({"a": pl.Int8, "b": pl.Int8}, np.int8),
        ({"a": pl.Int8, "b": pl.UInt16}, np.int32),
        ({"a": pl.Int8, "b": pl.String}, np.object_),
    ],
)
def test_df_to_numpy_empty_dtype_viewable(
    schema: dict[str, PolarsDataType], expected_dtype: npt.DTypeLike
) -> None:
    df = pl.DataFrame(schema=schema)
    result = df.to_numpy(allow_copy=False)
    assert result.shape == (0, 2)
    assert result.dtype == expected_dtype
    assert result.flags.writeable is True


def test_df_to_numpy_structured_nested() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3.0, 4.0],
            "c": [{"x": "a", "y": 1.0}, {"x": "b", "y": 2.0}],
        }
    )
    result = df.to_numpy(structured=True)

    expected = np.array(
        [
            (1, 3.0, ("a", 1.0)),
            (2, 4.0, ("b", 2.0)),
        ],
        dtype=[
            ("a", "<i8"),
            ("b", "<f8"),
            ("c", [("x", "<U1"), ("y", "<f8")]),
        ],
    )
    assert_array_equal(result, expected)


def test_df_to_numpy_stacking_array() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": 1},
        schema={"a": pl.Array(pl.Int64, 2), "b": pl.Int32},
    )
    result = df.to_numpy()

    expected = np.array([[np.array([1, 2]), 1]], dtype=np.object_)

    assert result.shape == (1, 2)
    assert result[0].shape == (2,)
    assert_array_equal(result[0][0], expected[0][0])


@pytest.mark.parametrize("order", ["c", "fortran"])
def test_df_to_numpy_stacking_string(order: IndexOrder) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = df.to_numpy(order=order)

    expected = np.array([[1, "x"], [2, "y"], [3, "z"]], dtype=np.object_)

    assert_array_equal(result, expected)
    if order == "c":
        assert result.flags.c_contiguous is True
    else:
        assert result.flags.f_contiguous is True


def test_to_numpy_chunked_16375() -> None:
    assert (
        pl.concat(
            [
                pl.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}),
                pl.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}),
            ],
            rechunk=False,
        ).to_numpy()
        == np.array([[1, 2], [1, 3], [2, 4], [1, 2], [1, 3], [2, 4]])
    ).all()


def test_to_numpy_c_order_1700() -> None:
    rng = np.random.default_rng()
    df = pl.DataFrame({f"col_{i}": rng.normal(size=20) for i in range(3)})
    df_chunked = pl.concat([df.slice(i * 10, 10) for i in range(3)])
    assert_frame_equal(
        df_chunked,
        pl.from_numpy(df_chunked.to_numpy(order="c"), schema=df_chunked.schema),
    )

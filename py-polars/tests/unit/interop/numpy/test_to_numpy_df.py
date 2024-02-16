from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IndexOrder


@pytest.mark.parametrize(
    ("order", "f_contiguous", "c_contiguous"),
    [("fortran", True, False), ("c", False, True)],
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
    assert df["s"].has_validity()
    assert_array_equal(
        df.to_numpy(structured=True),
        np.array([("x",), ("y",), (None,)], dtype=[("s", "O")]),
    )
    assert not df["s"][:2].has_validity()
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


def test__array__() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out_array = np.asarray(df.to_numpy())
    expected_array = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] is True

    out_array = np.asarray(df.to_numpy(), np.uint8)
    expected_array = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint8)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] is True


def test_numpy_preserve_uint64_4112() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.col("a").hash())
    assert df.to_numpy().dtype == np.dtype("uint64")
    assert df.to_numpy(structured=True).dtype == np.dtype([("a", "uint64")])


@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_df_to_numpy_decimal(use_pyarrow: bool) -> None:
    decimal_data = [D("1.234"), D("2.345"), D("-3.456")]
    df = pl.Series("n", decimal_data).to_frame()

    result = df.to_numpy(use_pyarrow=use_pyarrow)

    expected = np.array(decimal_data).reshape((-1, 1))
    assert_array_equal(result, expected)


def test_to_numpy_zero_copy_path() -> None:
    rows = 10
    cols = 5
    x = np.ones((rows, cols), order="F")
    x[:, 1] = 2.0
    df = pl.DataFrame(x)
    x = df.to_numpy()
    assert x.flags["F_CONTIGUOUS"]
    assert not x.flags["WRITEABLE"]
    assert str(x[0, :]) == "[1. 2. 1. 1. 1.]"


def test_to_numpy_zero_copy_path_writeable() -> None:
    rows = 10
    cols = 5
    x = np.ones((rows, cols), order="F")
    x[:, 1] = 2.0
    df = pl.DataFrame(x)
    x = df.to_numpy(writable=True)
    assert x.flags["WRITEABLE"]

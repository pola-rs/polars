from __future__ import annotations

import pytest
from hypothesis import given

import polars as pl
from polars.datatypes._utils import dtype_to_init_repr
from polars.testing.parametric import dtypes


@given(dtype=dtypes())
def test_dtype_to_init_repr_parametric(dtype: pl.DataType) -> None:
    assert repr(dtype) == dtype_to_init_repr(dtype, prefix="")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Struct, "pl.Struct"),
        (pl.Array(pl.Int8, 2), "pl.Array(pl.Int8, shape=(2,))"),
        (pl.List(pl.Int32), "pl.List(pl.Int32)"),
        (pl.List(pl.List(pl.Int8)), "pl.List(pl.List(pl.Int8))"),
        (
            pl.Struct({"x": pl.String, "y": pl.List(pl.Int8)}),
            "pl.Struct({'x': pl.String, 'y': pl.List(pl.Int8)})",
        ),
    ],
)
def test_dtype_to_init_repr(dtype: pl.DataType, expected: str) -> None:
    assert dtype_to_init_repr(dtype) == expected

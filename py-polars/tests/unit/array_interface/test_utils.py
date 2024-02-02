from __future__ import annotations

import pytest

import polars as pl
from polars.array_interface.utils import dtype_to_typestr


@pytest.mark.parametrize(
    ("dtype", "typestr"),
    [
        (pl.Int8, "|i1"),
        (pl.Int16, "<i2"),
        (pl.Int32, "<i4"),
        (pl.Int64, "<i8"),
        (pl.UInt8, "|u1"),
        (pl.UInt16, "<u2"),
        (pl.UInt32, "<u4"),
        (pl.UInt64, "<u8"),
        (pl.Float32, "<f4"),
        (pl.Float64, "<f8"),
    ],
)
def test_dtype_to_typestr(dtype: pl.PolarsDataType, typestr: str) -> None:
    assert dtype_to_typestr(dtype) == typestr

import pytest

import polars as pl
from polars.internals.interchange.buffer import PolarsBuffer
from polars.internals.interchange.column import PolarsColumn
from polars.internals.interchange.dataframe_protocol import Dtype, DtypeKind
from polars.internals.interchange.utils import NATIVE_ENDIANNESS


def test_size() -> None:
    s = pl.Series([1, 2, 3])
    col = PolarsColumn(s)
    assert col.size() == 3


def test_offset() -> None:
    s = pl.Series([1, 2, 3])
    col = PolarsColumn(s)
    assert col.offset == 0


def test_dtype_int() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.Int32)
    col = PolarsColumn(s)
    assert col.dtype == (DtypeKind.INT, 32, "i", NATIVE_ENDIANNESS)


def test_dtype_categorical() -> None:
    s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    col = PolarsColumn(s)
    assert col.dtype == (DtypeKind.CATEGORICAL, 32, "I", NATIVE_ENDIANNESS)


def test_describe_categorical() -> None:
    pass


def test_describe_other_dtype() -> None:
    s = pl.Series(["a", "b", "a"], dtype=pl.Utf8)
    col = PolarsColumn(s)
    with pytest.raises(TypeError):
        col.describe_categorical


def test_get_validity_buffer() -> None:
    s = pl.Series(["a", None, "b"])
    col = PolarsColumn(s)

    result_buffer, result_dtype = col._get_validity_buffer()

    mask = pl.Series([True, False, True])
    expected = PolarsBuffer(mask)
    print(expected)

    assert result_buffer == PolarsBuffer(pl.Series([True, False, True]))
    assert result_dtype == (DtypeKind.BOOL, 8, "b", NATIVE_ENDIANNESS)


def test_get_offsets_buffer() -> None:
    pass


def test_get_offsets_buffer_other_dtype() -> None:
    s = pl.Series([1, 2, 3], dtype=pl.Int32)
    col = PolarsColumn(s)
    with pytest.raises(TypeError):
        col._get_offsets_buffer()

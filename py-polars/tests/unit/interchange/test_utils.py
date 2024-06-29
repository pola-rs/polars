from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.interchange.protocol import DtypeKind, Endianness
from polars.interchange.utils import (
    dtype_to_polars_dtype,
    get_buffer_length_in_elements,
    polars_dtype_to_data_buffer_dtype,
    polars_dtype_to_dtype,
)

if TYPE_CHECKING:
    from polars._typing import PolarsDataType
    from polars.interchange.protocol import Dtype

NE = Endianness.NATIVE


@pytest.mark.parametrize(
    ("polars_dtype", "dtype"),
    [
        (pl.Int8, (DtypeKind.INT, 8, "c", NE)),
        (pl.Int16, (DtypeKind.INT, 16, "s", NE)),
        (pl.Int32, (DtypeKind.INT, 32, "i", NE)),
        (pl.Int64, (DtypeKind.INT, 64, "l", NE)),
        (pl.UInt8, (DtypeKind.UINT, 8, "C", NE)),
        (pl.UInt16, (DtypeKind.UINT, 16, "S", NE)),
        (pl.UInt32, (DtypeKind.UINT, 32, "I", NE)),
        (pl.UInt64, (DtypeKind.UINT, 64, "L", NE)),
        (pl.Float32, (DtypeKind.FLOAT, 32, "f", NE)),
        (pl.Float64, (DtypeKind.FLOAT, 64, "g", NE)),
        (pl.Boolean, (DtypeKind.BOOL, 1, "b", NE)),
        (pl.String, (DtypeKind.STRING, 8, "U", NE)),
        (pl.Date, (DtypeKind.DATETIME, 32, "tdD", NE)),
        (pl.Time, (DtypeKind.DATETIME, 64, "ttu", NE)),
        (pl.Duration, (DtypeKind.DATETIME, 64, "tDu", NE)),
        (pl.Duration(time_unit="ns"), (DtypeKind.DATETIME, 64, "tDn", NE)),
        (pl.Datetime, (DtypeKind.DATETIME, 64, "tsu:", NE)),
        (pl.Datetime(time_unit="ms"), (DtypeKind.DATETIME, 64, "tsm:", NE)),
        (
            pl.Datetime(time_zone="Amsterdam/Europe"),
            (DtypeKind.DATETIME, 64, "tsu:Amsterdam/Europe", NE),
        ),
        (
            pl.Datetime(time_unit="ns", time_zone="Asia/Seoul"),
            (DtypeKind.DATETIME, 64, "tsn:Asia/Seoul", NE),
        ),
    ],
)
def test_dtype_conversions(polars_dtype: PolarsDataType, dtype: Dtype) -> None:
    assert polars_dtype_to_dtype(polars_dtype) == dtype
    assert dtype_to_polars_dtype(dtype) == polars_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        (DtypeKind.CATEGORICAL, 32, "I", NE),
        (DtypeKind.CATEGORICAL, 8, "C", NE),
    ],
)
def test_dtype_to_polars_dtype_categorical(dtype: Dtype) -> None:
    assert dtype_to_polars_dtype(dtype) == pl.Enum


@pytest.mark.parametrize(
    "polars_dtype",
    [
        pl.Categorical,
        pl.Categorical("lexical"),
        pl.Enum,
        pl.Enum(["a", "b"]),
    ],
)
def test_polars_dtype_to_dtype_categorical(polars_dtype: PolarsDataType) -> None:
    assert polars_dtype_to_dtype(polars_dtype) == (DtypeKind.CATEGORICAL, 32, "I", NE)


def test_polars_dtype_to_dtype_unsupported_type() -> None:
    polars_dtype = pl.List(pl.Int8)
    with pytest.raises(ValueError, match="not supported"):
        polars_dtype_to_dtype(polars_dtype)


def test_dtype_to_polars_dtype_unsupported_type() -> None:
    dtype = (DtypeKind.FLOAT, 16, "e", NE)
    with pytest.raises(
        NotImplementedError,
        match="unsupported data type: \\(<DtypeKind.FLOAT: 2>, 16, 'e', '='\\)",
    ):
        dtype_to_polars_dtype(dtype)


def test_dtype_to_polars_dtype_unsupported_temporal_type() -> None:
    dtype = (DtypeKind.DATETIME, 64, "tss:", NE)
    with pytest.raises(
        NotImplementedError,
        match="unsupported temporal data type: \\(<DtypeKind.DATETIME: 22>, 64, 'tss:', '='\\)",
    ):
        dtype_to_polars_dtype(dtype)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ((DtypeKind.INT, 64, "l", NE), 3),
        ((DtypeKind.UINT, 32, "I", NE), 6),
    ],
)
def test_get_buffer_length_in_elements(dtype: Dtype, expected: int) -> None:
    assert get_buffer_length_in_elements(24, dtype) == expected


def test_get_buffer_length_in_elements_unsupported_dtype() -> None:
    dtype = (DtypeKind.BOOL, 1, "b", NE)
    with pytest.raises(
        ValueError,
        match="cannot get buffer length for buffer with dtype \\(<DtypeKind.BOOL: 20>, 1, 'b', '='\\)",
    ):
        get_buffer_length_in_elements(24, dtype)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Int8, pl.Int8),
        (pl.Date, pl.Int32),
        (pl.Time, pl.Int64),
        (pl.String, pl.UInt8),
        (pl.Enum, pl.UInt32),
    ],
)
def test_polars_dtype_to_data_buffer_dtype(
    dtype: PolarsDataType, expected: PolarsDataType
) -> None:
    assert polars_dtype_to_data_buffer_dtype(dtype) == expected


def test_polars_dtype_to_data_buffer_dtype_unsupported_dtype() -> None:
    dtype = pl.List(pl.Int8)
    with pytest.raises(NotImplementedError):
        polars_dtype_to_data_buffer_dtype(dtype)

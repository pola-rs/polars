from __future__ import annotations

from typing import Any

import pytest

import polars as pl


def test_series_mixed_dtypes_list() -> None:
    values = [[0.1, 1]]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.List(pl.Float64)
    assert s.to_list() == [[0.1, 1.0]]


def test_series_mixed_dtypes_string() -> None:
    values = [[12], "foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.String
    assert s.to_list() == ["[12]", "foo", "9"]
    assert s[1] == "foo"


def test_series_mixed_dtypes_object() -> None:
    values = [[12], b"foo", 9]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.Object
    assert s.to_list() == values
    assert s[1] == b"foo"


# https://github.com/pola-rs/polars/issues/15139
@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), None])
def test_sequence_of_series_with_dtype(dtype: pl.PolarsDataType | None) -> None:
    values = [1, 2, 3]
    int_series = pl.Series(values)
    list_series = pl.Series([int_series], dtype=dtype)

    assert list_series.to_list() == [values]
    assert list_series.dtype == pl.List(pl.Int64)


@pytest.mark.parametrize(
    ("values", "dtype", "expected_dtype"),
    [
        ([1, 1.0, 1], None, pl.Float64),
        ([1, 1, "1.0"], None, pl.String),
        ([1, 1.0, "1.0"], None, pl.String),
        ([True, 1], None, pl.Int64),
        ([True, 1.0], None, pl.Float64),
        ([True, 1], pl.Boolean, pl.Boolean),
        ([True, 1.0], pl.Boolean, pl.Boolean),
        ([False, "1.0"], None, pl.String),
    ],
)
def test_upcast_primitive_and_strings(
    values: list[Any], dtype: pl.PolarsDataType, expected_dtype: pl.PolarsDataType
) -> None:
    assert pl.Series(values, dtype=dtype).dtype == expected_dtype


def test_preserve_decimal_precision() -> None:
    dtype = pl.Decimal(None, 1)
    s = pl.Series(dtype=dtype)
    assert s.dtype == dtype

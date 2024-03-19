from __future__ import annotations

import pytest

import polars as pl


def test_series_mixed_dtypes_list() -> None:
    values = [[0.1, 1]]

    with pytest.raises(pl.SchemaError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.List(pl.Float64)
    assert s.to_list() == [[0.1, 1.0]]


def test_series_mixed_dtypes_string() -> None:
    values = [[12], "foo", 9]

    with pytest.raises(pl.SchemaError, match="unexpected value"):
        pl.Series(values)

    s = pl.Series(values, strict=False)
    assert s.dtype == pl.String
    assert s.to_list() == ["[12]", "foo", "9"]
    assert s[1] == "foo"


def test_series_mixed_dtypes_object() -> None:
    values = [[12], b"foo", 9]

    with pytest.raises(pl.SchemaError, match="unexpected value"):
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

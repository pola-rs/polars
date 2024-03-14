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

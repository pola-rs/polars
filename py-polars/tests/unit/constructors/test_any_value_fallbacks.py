# TODO: Replace direct calls to fallback constructors with calls to the Series
# constructor once the Python-side logic has been updated
from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import polars as pl
from polars.polars import PySeries
from polars.utils._wrap import wrap_s


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Boolean, [True, False, None]),
        (pl.Binary, [b"123", b"xyz", None]),
    ],
)
def test_fallback_with_dtype_strict(
    dtype: pl.PolarsDataType, values: list[Any]
) -> None:
    result = wrap_s(
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=True)
    )
    assert result.to_list() == values


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Boolean, [0, 1]),
        (pl.Binary, ["123", "xyz"]),
    ],
)
def test_fallback_with_dtype_strict_failure(
    dtype: pl.PolarsDataType, values: list[Any]
) -> None:
    with pytest.raises(pl.ComputeError, match="unexpected value"):
        PySeries.new_from_any_values_and_dtype("", values, pl.Boolean, strict=True)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (
            pl.Boolean,
            [False, True, 0, 1, 0.0, 2.5, date(1970, 1, 1)],
            [False, True, False, True, False, True, None],
        ),
        (
            pl.Binary,
            [b"123", b"xyz", None],
            [b"123", b"xyz", None],
        ),
    ],
)
def test_fallback_with_dtype_nonstrict(
    dtype: pl.PolarsDataType, values: list[Any], expected: list[Any]
) -> None:
    result = wrap_s(
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=False)
    )
    assert result.to_list() == expected

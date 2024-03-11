# TODO: Replace direct calls to fallback constructors with calls to the Series
# constructor once the Python-side logic has been updated
from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import polars as pl
from polars._utils.wrap import wrap_s
from polars.polars import PySeries


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int64, [-1, 0, 100_000, None]),
        (pl.Float64, [-1.5, 0.0, 10.0, None]),
        (pl.Boolean, [True, False, None]),
        (pl.Binary, [b"123", b"xyz", None]),
        (pl.String, ["123", "xyz", None]),
        (pl.List(pl.Int64), [[1, 2], [3], None]),
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
        (pl.Int64, [1.0, 2.0]),
        (pl.Float64, [1, 2]),
        (pl.Boolean, [0, 1]),
        (pl.Binary, ["123", "xyz"]),
        (pl.String, [b"123", b"xyz"]),
        (pl.List(pl.Int64), [[1, 2.0], [3]]),
        (pl.List(pl.List(pl.Float32)), [[[1.0], [2.0]], [[3.0], [4]]]),
    ],
)
def test_fallback_with_dtype_strict_failure(
    dtype: pl.PolarsDataType, values: list[Any]
) -> None:
    with pytest.raises(pl.SchemaError, match="unexpected value"):
        PySeries.new_from_any_values_and_dtype("", values, dtype, strict=True)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (
            pl.Int64,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 2), "5", "xyz"],
            [0, 1, 0, -1, 0, 2, 1, 5, None],
        ),
        (
            pl.Float64,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 2), "5", "xyz"],
            [0.0, 1.0, 0.0, -1.0, 0.0, 2.5, 1.0, 5.0, None],
        ),
        (
            pl.Boolean,
            [False, True, 0, -1, 0.0, 2.5, date(1970, 1, 1), "true"],
            [False, True, False, True, False, True, None, None],
        ),
        (
            pl.Binary,
            [b"123", "xyz", 100, True, None],
            [b"123", b"xyz", None, None, None],
        ),
        (
            pl.String,
            ["xyz", 1, 2.5, date(1970, 1, 1), True, b"123", None],
            ["xyz", "1", "2.5", "1970-01-01", "true", None, None],
        ),
        (
            pl.List(pl.Int64),
            [[1, 2.0], [3], ["4"], [], None],
            [[1, 2], [3], [4], [], None],
        ),
        (
            pl.List(pl.List(pl.Float32)),
            [[[1], [2.0]], [[3.0], [4]], [["5.0", 6]], [[]], None],
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0, 6.0]], [[]], None],
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

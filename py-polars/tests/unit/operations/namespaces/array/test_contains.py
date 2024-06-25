from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import SchemaError
from polars.testing import assert_series_equal


@pytest.mark.parametrize(
    ("array", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, 4]], [1, 5], [True, False], pl.Int64),
        ([[True, False], [True, True]], [True, False], [True, False], pl.Boolean),
        ([["a", "b"], ["c", "d"]], ["a", "b"], [True, False], pl.String),
        ([[b"a", b"b"], [b"c", b"d"]], [b"a", b"b"], [True, False], pl.Binary),
        (
            [[{"a": 1}, {"a": 2}], [{"b": 1}, {"a": 3}]],
            [{"a": 1}, {"a": 2}],
            [True, False],
            pl.Struct([pl.Field("a", pl.Int64)]),
        ),
    ],
)
def test_array_contains_expr(
    array: list[list[Any]], data: list[Any], expected: list[bool], dtype: pl.DataType
) -> None:
    df = pl.DataFrame(
        {
            "array": array,
            "data": data,
        },
        schema={
            "array": pl.Array(dtype, 2),
            "data": dtype,
        },
    )
    out = df.select(contains=pl.col("array").arr.contains(pl.col("data"))).to_series()
    expected_series = pl.Series("contains", expected)
    assert_series_equal(out, expected_series)


@pytest.mark.parametrize(
    ("array", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, 4]], 1, [True, False], pl.Int64),
        ([[True, False], [True, True]], True, [True, True], pl.Boolean),
        ([["a", "b"], ["c", "d"]], "a", [True, False], pl.String),
        ([[b"a", b"b"], [b"c", b"d"]], b"a", [True, False], pl.Binary),
    ],
)
def test_array_contains_literal(
    array: list[list[Any]], data: Any, expected: list[bool], dtype: pl.DataType
) -> None:
    df = pl.DataFrame(
        {
            "array": array,
        },
        schema={
            "array": pl.Array(dtype, 2),
        },
    )
    out = df.select(contains=pl.col("array").arr.contains(data)).to_series()
    expected_series = pl.Series("contains", expected)
    assert_series_equal(out, expected_series)


def test_array_contains_invalid_datatype() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4]]}, schema={"a": pl.List(pl.Int8)})
    with pytest.raises(SchemaError, match="invalid series dtype: expected `Array`"):
        df.select(pl.col("a").arr.contains(2))

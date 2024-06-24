from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, ForwardRef, Optional, Union

import pytest

import polars as pl
from polars.datatypes._parse import (
    _parse_forward_ref_into_dtype,
    _parse_union_type_into_dtype,
    parse_into_dtype,
    parse_py_type_into_dtype,
)

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType


def assert_dtype_equal(left: PolarsDataType, right: PolarsDataType) -> None:
    assert left == right
    assert type(left) == type(right)
    assert hash(left) == hash(right)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (pl.Int8(), pl.Int8()),
        (list, pl.List),
    ],
)
def test_parse_dtype(input: Any, expected: PolarsDataType) -> None:
    result = parse_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (datetime, pl.Datetime("us")),
        (date, pl.Date()),
    ],
)
def test_parse_py_type_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = parse_py_type_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (ForwardRef("date"), pl.Date()),
    ],
)
def test_parse_forward_ref_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_forward_ref_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (Optional[int], pl.Int64()),
        (Optional[pl.String], pl.String),
        (Union[float, None], pl.Float64()),
        (date | None, pl.Date()),
    ],
)
def test_parse_union_type_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_union_type_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize("input", [Union[int, float], Optional[Union[int, str]]])
def test_parse_union_type_into_dtype_invalid(input: Any) -> None:
    with pytest.raises(TypeError):
        _parse_union_type_into_dtype(input)

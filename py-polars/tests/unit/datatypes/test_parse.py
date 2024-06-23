from __future__ import annotations

from typing import TYPE_CHECKING, Any, ForwardRef

import pytest

import polars as pl
from polars.datatypes._parse import _parse_forward_ref_into_dtype, parse_into_dtype

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (pl.Int8(), pl.Int8()),
        (list, pl.List),
    ],
)
def test_parse_dtype(input: Any, expected: PolarsDataType) -> None:
    result = parse_into_dtype(input)

    assert result == expected
    assert type(result) == type(expected)
    assert hash(result) == hash(expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (ForwardRef("date"), pl.Date()),
    ],
)
def test_parse_forward_ref_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_forward_ref_into_dtype(input)

    assert result == expected
    assert type(result) == type(expected)
    assert hash(result) == hash(expected)

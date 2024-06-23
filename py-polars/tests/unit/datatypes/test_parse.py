from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.datatypes import parse_into_dtype

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

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import polars as pl


@pytest.mark.parametrize(
    ("item", "data", "expected"),
    [
        (1, [1, 2, 3], True),
        (4, [1, 2, 3], False),
        (None, [1, None], True),
        (None, [1, 2], False),
        (date(2022, 1, 1), [date(2022, 1, 1), date(2023, 1, 1)], True),
        (4, [1, None], False),
        (2, [1, None, 2], True),
        (float("nan"), [1.0, float("nan")], True),
        ("b", ["a", None, "b"], True),
        ("c", ["a", "b"], False),
        (b"b", [b"a", b"b"], True),
    ],
)
def test_contains(item: Any, data: list[Any], expected: bool) -> None:
    s = pl.Series(data)
    result = item in s
    assert result is expected


def test_contains_none() -> None:
    s = pl.Series([1, None])
    result = None in s
    assert result is True

    s = pl.Series([1, 2])
    assert (None in s) is False

from __future__ import annotations

from typing import Sequence

import pytest

from polars.io._utils import parse_columns_arg


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["a", "b"], (None, ["a", "b"])),
        ((1, 2), ((1, 2), None)),
        ("foo", (None, ["foo"])),
        (3, ([3], None)),
        (None, (None, None)),
    ],
)
def test_parse_columns_arg(
    columns: Sequence[str] | Sequence[int] | str | int | None,
    expected: tuple[Sequence[int] | None, Sequence[str] | None],
) -> None:
    assert parse_columns_arg(columns) == expected


def test_parse_columns_arg_mixed_types() -> None:
    with pytest.raises(TypeError):
        parse_columns_arg(["a", 1])


@pytest.mark.parametrize("columns", [["a", "a"], [1, 1, 2]])
def test_parse_columns_arg_duplicates(columns: Sequence[str] | Sequence[int]) -> None:
    with pytest.raises(ValueError):
        parse_columns_arg(columns)

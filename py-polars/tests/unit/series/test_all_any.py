from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SchemaError


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], False),
        ([None], False),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], True),
    ],
)
def test_any(data: list[bool | None], expected: bool) -> None:
    assert pl.Series(data, dtype=pl.Boolean).any() is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], False),
        ([None], None),
        ([False], False),
        ([False, None], None),
        ([True], True),
        ([True, None], True),
    ],
)
def test_any_kleene(data: list[bool | None], expected: bool | None) -> None:
    assert pl.Series(data, dtype=pl.Boolean).any(ignore_nulls=False) is expected


def test_any_wrong_dtype() -> None:
    with pytest.raises(SchemaError, match="expected `Boolean`"):
        pl.Series([0, 1, 0]).any()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], True),
        ([None], True),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], True),
    ],
)
def test_all(data: list[bool | None], expected: bool) -> None:
    assert pl.Series(data, dtype=pl.Boolean).all() is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], True),
        ([None], None),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], None),
    ],
)
def test_all_kleene(data: list[bool | None], expected: bool | None) -> None:
    assert pl.Series(data, dtype=pl.Boolean).all(ignore_nulls=False) is expected


def test_all_wrong_dtype() -> None:
    with pytest.raises(SchemaError, match="expected `Boolean`"):
        pl.Series([0, 1, 0]).all()

from __future__ import annotations

import pytest

import polars as pl


def test_series_item() -> None:
    s = pl.Series("a", [1])
    assert s.item() == 1


def test_series_item_empty() -> None:
    s = pl.Series("a", [])
    with pytest.raises(ValueError):
        s.item()


def test_series_item_incorrect_shape() -> None:
    s = pl.Series("a", [1, 2])
    with pytest.raises(ValueError):
        s.item()


@pytest.fixture(scope="module")
def s() -> pl.Series:
    return pl.Series("a", [1, 2])


@pytest.mark.parametrize(("index", "expected"), [(0, 1), (1, 2), (-1, 2), (-2, 1)])
def test_series_item_with_index(index: int, expected: int, s: pl.Series) -> None:
    assert s.item(index) == expected


@pytest.mark.parametrize("index", [-10, 10])
def test_df_item_out_of_bounds(index: int, s: pl.Series) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        s.item(index)

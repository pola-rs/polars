from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing.parametric.strategies.data import categories, data


@given(cat=categories(3))
def test_categories(cat: str) -> None:
    assert cat in ("c0", "c1", "c2")


@given(cat=data(pl.Categorical, n_categories=3))
def test_data_kwargs(cat: str) -> None:
    assert cat in ("c0", "c1", "c2")


@given(categories=data(pl.List(pl.Categorical), n_categories=3))
def test_data_nested_kwargs(categories: list[str]) -> None:
    assert all(c in ("c0", "c1", "c2") for c in categories)

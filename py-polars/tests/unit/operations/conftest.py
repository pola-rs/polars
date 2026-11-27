from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(
    params=[
        pytest.param(pl.Categories.random, id="own"),
        pytest.param(lambda: pl.Categories(f"named-{uuid4()}"), id="named"),
        pytest.param(pl.Categories, id="global"),
        pytest.param(lambda: pl.Categories.random(physical=pl.UInt8), id="uint8"),
    ]
)
def category_mapping(request: pytest.FixtureRequest) -> pl.Categories:
    factory: Callable[[], pl.Categories] = request.param
    return factory()

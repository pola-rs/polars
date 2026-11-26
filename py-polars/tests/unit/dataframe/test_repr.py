from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given

from polars.testing.parametric import dataframes

if TYPE_CHECKING:
    import polars as pl


@pytest.mark.slow
@given(df=dataframes())
def test_repr(df: pl.DataFrame) -> None:
    assert isinstance(repr(df), str)

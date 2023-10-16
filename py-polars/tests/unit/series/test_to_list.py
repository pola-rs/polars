from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(s=series())
def test_to_list(s: pl.Series) -> None:
    values = s.to_list()
    result = pl.Series(values, dtype=s.dtype)
    assert_series_equal(s, result, categorical_as_str=True)

import hypothesis.strategies as st
from hypothesis import given, settings

import polars as pl
from polars.testing.parametric.strategies import dtypes, series


@given(st.data())
def test_dtype(data: st.DataObject) -> None:
    dtype = data.draw(dtypes())
    s = data.draw(series(dtype=dtype))
    assert s.dtype == dtype


@given(s=series(dtype=pl.Binary))
@settings(max_examples=5)
def test_strategy_dtype_binary(s: pl.Series) -> None:
    assert s.dtype == pl.Binary


@given(s=series(dtype=pl.Decimal))
@settings(max_examples=5)
def test_strategy_dtype_decimal(s: pl.Series) -> None:
    assert s.dtype == pl.Decimal

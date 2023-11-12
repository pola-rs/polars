from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes


@given(df=dataframes())
def test_to_dict(df: pl.DataFrame) -> None:
    d = df.to_dict(as_series=False)
    result = pl.from_dict(d, schema=df.schema)
    assert_frame_equal(df, result, categorical_as_str=True)

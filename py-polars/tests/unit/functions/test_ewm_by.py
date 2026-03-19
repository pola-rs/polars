from __future__ import annotations

from datetime import date

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import column, dataframes


@given(
    data=st.data(),
    half_life=st.integers(min_value=1, max_value=1000),
)
def test_ewm_by(data: st.DataObject, half_life: int) -> None:
    # For evenly spaced times, ewm_by and ewm should be equivalent
    df = data.draw(
        dataframes(
            [
                column(
                    "values",
                    strategy=st.floats(min_value=-100, max_value=100),
                    dtype=pl.Float64,
                ),
            ],
            min_size=1,
        )
    )
    result = df.with_row_index().select(
        pl.col("values").ewm_mean_by(by="index", half_life=f"{half_life}i")
    )
    expected = df.select(
        pl.col("values").ewm_mean(half_life=half_life, ignore_nulls=False, adjust=False)
    )
    assert_frame_equal(result, expected)
    result = (
        df.with_row_index()
        .sort("values")
        .with_columns(
            pl.col("values").ewm_mean_by(by="index", half_life=f"{half_life}i")
        )
        .sort("index")
        .select("values")
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("length", [1, 3])
def test_length_mismatch_22084(length: int) -> None:
    s = pl.Series([0, None])
    by = pl.Series([date(2020, 1, 5)] * length)
    with pytest.raises(pl.exceptions.ShapeError):
        s.ewm_mean_by(by, half_life="4d")

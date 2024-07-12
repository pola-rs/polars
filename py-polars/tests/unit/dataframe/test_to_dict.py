from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes


@given(
    df=dataframes(
        excluded_dtypes=[
            pl.Categorical,  # Bug: https://github.com/pola-rs/polars/issues/16196
            pl.Struct,
        ],
        # Roundtrip doesn't work with time zones:
        # https://github.com/pola-rs/polars/issues/16297
        allow_time_zones=False,
    )
)
def test_to_dict(df: pl.DataFrame) -> None:
    d = df.to_dict(as_series=False)
    result = pl.from_dict(d, schema=df.schema)
    assert_frame_equal(df, result, categorical_as_str=True)


@pytest.mark.parametrize(
    ("as_series", "inner_dtype"),
    [
        (True, pl.Series),
        (False, list),
    ],
)
def test_to_dict_misc(as_series: bool, inner_dtype: Any) -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
            "optional": [28, 300, None, 2, -30],
        }
    )
    s = df.to_dict(as_series=as_series)
    assert isinstance(s, dict)
    for v in s.values():
        assert isinstance(v, inner_dtype)
        assert len(v) == len(df)

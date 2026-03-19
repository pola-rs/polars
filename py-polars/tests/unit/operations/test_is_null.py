from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series


@given(s=series(allow_null=True))
def test_is_null_parametric(s: pl.Series) -> None:
    is_null = s.is_null()
    is_not_null = s.is_not_null()

    assert is_null.null_count() == 0
    assert_series_equal(is_null, ~is_not_null)


def test_is_null_struct() -> None:
    df = pl.DataFrame(
        {"x": [{"a": 1, "b": 2}, {"a": None, "b": None}, {"a": None, "b": 2}, None]}
    )

    result = df.select(
        null=pl.col("x").is_null(),
        not_null=pl.col("x").is_not_null(),
    )

    expected = pl.DataFrame(
        {
            "null": [False, False, False, True],
            "not_null": [True, True, True, False],
        }
    )
    assert_frame_equal(result, expected)


def test_is_null_null() -> None:
    s = pl.Series([None, None])

    result = s.is_null()
    expected = pl.Series([True, True])
    assert_series_equal(result, expected)

    result = s.is_not_null()
    expected = pl.Series([False, False])
    assert_series_equal(result, expected)

from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series


@given(s=series(null_probability=0.5))
def test_drop_nulls_parametric(s: pl.Series) -> None:
    result = s.drop_nulls()
    assert result.len() == s.len() - s.null_count()

    filter_result = s.filter(s.is_not_null())
    assert_series_equal(result, filter_result)


def test_df_drop_nulls_struct() -> None:
    df = pl.DataFrame(
        {
            "x": [
                {"a": 1, "b": 2},
                {"a": 1, "b": None},
                {"a": None, "b": 2},
                {"a": None, "b": None},
            ]
        }
    )

    result = df.drop_nulls()

    expected = df.head(3)
    assert_frame_equal(result, expected)

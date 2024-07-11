from __future__ import annotations

from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series


@given(
    s=series(
        allow_null=True,
        excluded_dtypes=[
            pl.Struct,  # See: https://github.com/pola-rs/polars/issues/3462
        ],
    )
)
def test_drop_nulls_parametric(s: pl.Series) -> None:
    result = s.drop_nulls()
    assert result.len() == s.len() - s.null_count()

    filter_result = s.filter(s.is_not_null())
    assert_series_equal(result, filter_result)


def test_df_drop_nulls_struct() -> None:
    df = pl.DataFrame(
        {"x": [{"a": 1, "b": 2}, {"a": 1, "b": None}, {"a": None, "b": 2}, None]}
    )

    result = df.drop_nulls()

    expected = df.head(3)
    assert_frame_equal(result, expected)

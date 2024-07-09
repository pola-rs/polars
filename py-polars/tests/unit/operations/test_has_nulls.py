from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes, series


@given(s=series(allow_null=False))
def test_has_nulls_series_no_nulls(s: pl.Series) -> None:
    assert s.has_nulls() is False


@given(df=dataframes(allow_null=False))
def test_has_nulls_expr_no_nulls(df: pl.DataFrame) -> None:
    result = df.select(pl.all().has_nulls())
    assert result.select(pl.any_horizontal(df.columns)).item() is False


@given(
    s=series(
        excluded_dtypes=[
            pl.Struct,  # https://github.com/pola-rs/polars/issues/3462
        ]
    )
)
def test_has_nulls_series_parametric(s: pl.Series) -> None:
    result = s.has_nulls()
    assert result == (s.null_count() > 0)
    assert result == s.is_null().any()


@given(
    lf=dataframes(
        excluded_dtypes=[
            pl.Struct,  # https://github.com/pola-rs/polars/issues/3462
        ],
        lazy=True,
    )
)
def test_has_nulls_expr_parametric(lf: pl.LazyFrame) -> None:
    result = lf.select(pl.all().has_nulls())

    assert_frame_equal(result, lf.select(pl.all().null_count() > 0))
    assert_frame_equal(result, lf.select(pl.all().is_null().any()))


def test_has_nulls_series() -> None:
    s = pl.Series([1, 2, None])
    assert s.has_nulls() is True
    assert s[:2].has_nulls() is False


def test_has_nulls_expr() -> None:
    lf = pl.LazyFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    result = lf.select(pl.all().has_nulls())
    expected = pl.LazyFrame({"a": [True], "b": [False]})
    assert_frame_equal(result, expected)

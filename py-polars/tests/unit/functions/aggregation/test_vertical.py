import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_all_expr() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3, 4, 5, None]})
    assert_frame_equal(df.select(pl.all()), df)


def test_any_expr(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.with_columns(pl.col("A").cast(bool)).select(pl.any("A")).item()


def test_min_deprecated_alias_for_series_min() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = pl.min(s)
    assert result == s.min()


def test_max_deprecated_alias_for_series_max() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.deprecated_call():
        result = pl.max(s)
    assert result == s.max()


@pytest.mark.parametrize("input", ["a", "^a|b$"])
def test_min_alias_for_col_min(input: str) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    expr = pl.col(input).min()
    expr_alias = pl.min(input)
    assert_frame_equal(df.select(expr), df.select(expr_alias))


@pytest.mark.parametrize("input", ["a", "^a|b$"])
def test_max_alias_for_col_max(input: str) -> None:
    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    expr = pl.col(input).max()
    expr_alias = pl.max(input)
    assert_frame_equal(df.select(expr), df.select(expr_alias))

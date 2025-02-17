from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

import polars as pl
from polars.testing.parametric import series


@given(s=series())
def test_clear_series_parametric(s: pl.Series) -> None:
    result = s.clear()

    assert result.dtype == s.dtype
    assert result.name == s.name
    assert result.is_empty()


@given(
    s=series(
        excluded_dtypes=[
            pl.Struct,  # See: https://github.com/pola-rs/polars/issues/3462
        ]
    ),
    n=st.integers(min_value=0, max_value=10),
)
def test_clear_series_n_parametric(s: pl.Series, n: int) -> None:
    result = s.clear(n)

    assert result.dtype == s.dtype
    assert result.name == s.name
    assert len(result) == n
    assert result.null_count() == n


@pytest.mark.parametrize("n", [0, 2, 5])
def test_clear_series(n: int) -> None:
    a = pl.Series(name="a", values=[1, 2, 3], dtype=pl.Int16)

    result = a.clear(n)
    assert result.dtype == a.dtype
    assert result.name == a.name
    assert len(result) == n
    assert result.null_count() == n


def test_clear_df() -> None:
    df = pl.DataFrame(
        {"a": [1, 2], "b": [True, False]}, schema={"a": pl.UInt32, "b": pl.Boolean}
    )

    result = df.clear()
    assert result.schema == df.schema
    assert result.rows() == []

    result = df.clear(3)
    assert result.schema == df.schema
    assert result.rows() == [(None, None), (None, None), (None, None)]


def test_clear_lf() -> None:
    lf = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    ldfe = lf.clear()
    assert ldfe.collect_schema() == lf.collect_schema()

    ldfe = lf.clear(2)
    assert ldfe.collect_schema() == lf.collect_schema()
    assert ldfe.collect().rows() == [(None, None, None), (None, None, None)]


def test_clear_series_object_starting_with_null() -> None:
    s = pl.Series([None, object()])

    result = s.clear()

    assert result.dtype == s.dtype
    assert result.name == s.name
    assert result.is_empty()


def test_clear_raise_negative_n() -> None:
    s = pl.Series([1, 2, 3])

    msg = "`n` should be greater than or equal to 0, got -1"
    with pytest.raises(ValueError, match=msg):
        s.clear(-1)
    with pytest.raises(ValueError, match=msg):
        s.to_frame().clear(-1)

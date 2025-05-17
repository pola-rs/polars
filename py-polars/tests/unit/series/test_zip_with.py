import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_zip_with_all_true_mask() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5, 6])
    mask = pl.Series([True, True, True])

    result = s1.zip_with(mask, s2)
    assert_series_equal(result, s1)


def test_zip_with_all_false_mask() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5, 6])
    mask = pl.Series([False, False, False])

    result = s1.zip_with(mask, s2)
    assert_series_equal(result, s2)


def test_zip_with_mixed_mask() -> None:
    s1 = pl.Series([1, 2, 3, 4, 5])
    s2 = pl.Series([5, 4, 3, 2, 1])
    mask = pl.Series([True, False, True, False, True])

    result = s1.zip_with(mask, s2)
    expected = pl.Series([1, 4, 3, 2, 5])
    assert_series_equal(result, expected)


def test_zip_with_series_comparison() -> None:
    s1 = pl.Series([1, 2, 3, 4, 5])
    s2 = pl.Series([5, 4, 3, 2, 1])

    result = s1.zip_with(s1 < s2, s2)
    expected = pl.Series([1, 2, 3, 2, 1])
    assert_series_equal(result, expected)


def test_zip_with_null_values() -> None:
    s1 = pl.Series([1, None, 3, 4])
    s2 = pl.Series([5, 6, None, 8])
    mask = pl.Series([True, True, False, False])

    result = s1.zip_with(mask, s2)
    expected = pl.Series([1, None, None, 8])
    assert_series_equal(result, expected)


def test_zip_with_length_mismatch() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5])
    mask = pl.Series([True, False, True])

    with pytest.raises(pl.exceptions.ShapeError):
        s1.zip_with(mask, s2)


def test_zip_with_bad_input_type() -> None:
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([4, 5, 6])
    mask = pl.Series([True, True, True])

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'Series'.* not 'DataFrame'",
    ):
        s1.zip_with(mask, pl.DataFrame(s2))  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'Series'.* not 'LazyFrame'",
    ):
        s1.zip_with(mask, pl.DataFrame(s2).lazy())  # type: ignore[arg-type]

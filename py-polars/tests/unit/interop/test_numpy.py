import numpy as np
import pytest

import polars as pl


def test_view() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    result = s._view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))


def test_view_nulls() -> None:
    s = pl.Series("b", [1, 2, None])
    assert s.has_validity()
    with pytest.raises(AssertionError):
        s._view()


def test_view_nulls_sliced() -> None:
    s = pl.Series("b", [1, 2, None])
    sliced = s[:2]
    assert np.all(sliced._view() == np.array([1, 2]))
    assert not sliced.has_validity()


def test_view_ub() -> None:
    # this would be UB if the series was dropped and not passed to the view
    s = pl.Series([3, 1, 5])
    result = s.sort()._view()
    assert np.sum(result) == 9


def test_view_deprecated() -> None:
    s = pl.Series("a", [1.0, 2.5, 3.0])
    with pytest.deprecated_call():
        result = s.view()
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([1.0, 2.5, 3.0]))

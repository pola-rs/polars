import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_extend() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [8, 9, None])

    result = a.extend(b)

    expected = pl.Series("a", [1, 2, 8, 9, None])
    assert_series_equal(a, expected)
    assert_series_equal(result, expected)
    assert a.n_chunks() == 1


def test_extend_self() -> None:
    a = pl.Series("a", [1, 2])

    a.extend(a)

    expected = pl.Series("a", [1, 2, 1, 2])
    assert_series_equal(a, expected)
    assert a.n_chunks() == 1


def test_extend_bad_input() -> None:
    a = pl.Series("a", [1, 2])
    b = a.to_frame()

    with pytest.raises(AttributeError):
        a.extend(b)  # type: ignore[arg-type]

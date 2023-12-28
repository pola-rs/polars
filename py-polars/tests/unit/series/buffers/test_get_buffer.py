import pytest

import polars as pl
from polars.testing import assert_series_equal
from polars.utils._wrap import wrap_s


def test_get_buffer() -> None:
    s = pl.Series(["a", "bc", None, "éâç", ""])

    data = s._s.get_buffer(0)
    expected = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    assert_series_equal(wrap_s(data), expected)

    validity = s._s.get_buffer(1)
    expected = pl.Series([True, True, False, True, True])
    assert_series_equal(wrap_s(validity), expected)

    offsets = s._s.get_buffer(2)
    expected = pl.Series([0, 1, 3, 3, 9, 9], dtype=pl.Int64)
    assert_series_equal(wrap_s(offsets), expected)


def test_get_buffer_no_validity_or_offsets() -> None:
    s = pl.Series([1, 2, 3])

    validity = s._s.get_buffer(1)
    assert validity is None

    offsets = s._s.get_buffer(2)
    assert offsets is None


def test_get_buffer_invalid_index() -> None:
    s = pl.Series([1, None, 3])
    with pytest.raises(ValueError):
        s._s.get_buffer(3)

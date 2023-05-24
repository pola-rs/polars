import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_cast_list_array() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]
    s = pl.Series(payload)

    dtype = pl.Array(width=3, inner=pl.Int64)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == payload
    assert_series_equal(out.cast(pl.List(pl.Int64)), s)

    # width is incorrect
    with pytest.raises(
        pl.ArrowError,
        match=r"incompatible offsets in source list",
    ):
        s.cast(pl.Array(width=2, inner=pl.Int64))


def test_array_construction() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]

    dtype = pl.Array(width=3, inner=pl.Int64)
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # inner type
    dtype = pl.Array(2, pl.UInt8)
    payload = [[1, 2], [3, 4]]
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

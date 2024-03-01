from typing import cast

import numpy as np
from numpy.testing import assert_array_equal

import polars as pl
from polars.testing import assert_series_equal


def test_ufunc() -> None:
    # test if output dtype is calculated correctly.
    s_float32 = pl.Series("a", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float32)
    assert_series_equal(
        cast(pl.Series, np.multiply(s_float32, 4)),
        pl.Series("a", [4.0, 8.0, 12.0, 16.0], dtype=pl.Float32),
    )

    s_float64 = pl.Series("a", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float64)
    assert_series_equal(
        cast(pl.Series, np.multiply(s_float64, 4)),
        pl.Series("a", [4.0, 8.0, 12.0, 16.0], dtype=pl.Float64),
    )

    s_uint8 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt8),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2, dtype=np.uint16)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt16),
    )

    s_int8 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int8)
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int8),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2, dtype=np.int16)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int16),
    )

    s_uint32 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt32)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint32, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt32),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint32, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_int32 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int32)
    assert_series_equal(
        cast(pl.Series, np.power(s_int32, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int32),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int32, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_uint64 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt64)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint64, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint64, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_int64 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int64)
    assert_series_equal(
        cast(pl.Series, np.power(s_int64, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int64, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    # test if null bitmask is preserved
    a1 = pl.Series("a", [1.0, None, 3.0])
    b1 = cast(pl.Series, np.exp(a1))
    assert b1.null_count() == 1

    # test if it works with chunked series.
    a2 = pl.Series("a", [1.0, None, 3.0])
    b2 = pl.Series("b", [4.0, 5.0, None])
    a2.append(b2)
    assert a2.n_chunks() == 2
    c2 = np.multiply(a2, 3)
    assert_series_equal(
        cast(pl.Series, c2),
        pl.Series("a", [3.0, None, 9.0, 12.0, 15.0, None]),
    )

    # Test if nulls propagate through ufuncs
    a3 = pl.Series("a", [None, None, 3, 3])
    b3 = pl.Series("b", [None, 3, None, 3])
    assert_series_equal(
        cast(pl.Series, np.maximum(a3, b3)), pl.Series("a", [None, None, None, 3])
    )


def test_numpy_string_array() -> None:
    s_str = pl.Series("a", ["aa", "bb", "cc", "dd"], dtype=pl.String)
    assert_array_equal(
        np.char.capitalize(s_str),
        np.array(["Aa", "Bb", "Cc", "Dd"], dtype="<U2"),
    )

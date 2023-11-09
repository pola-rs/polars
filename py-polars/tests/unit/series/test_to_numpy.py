from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_array_equal

import polars as pl
from polars.testing.parametric import series


@given(
    s=series(
        min_size=1, max_size=10, excluded_dtypes=[pl.Categorical, pl.List, pl.Struct]
    ).filter(
        lambda s: (
            getattr(s.dtype, "time_unit", None) != "ms"
            and not (s.dtype == pl.Utf8 and s.str.contains("\x00").any())
            and not (s.dtype == pl.Binary and s.bin.contains(b"\x00").any())
        )
    ),
)
@settings(max_examples=250)
def test_series_to_numpy(s: pl.Series) -> None:
    result = s.to_numpy()

    values = s.to_list()
    dtype_map = {
        pl.Datetime("ns"): "datetime64[ns]",
        pl.Datetime("us"): "datetime64[us]",
        pl.Duration("ns"): "timedelta64[ns]",
        pl.Duration("us"): "timedelta64[us]",
    }
    np_dtype = dtype_map.get(s.dtype)  # type: ignore[call-overload]
    expected = np.array(values, dtype=np_dtype)

    assert_array_equal(result, expected)

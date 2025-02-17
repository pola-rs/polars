import numpy as np
import pytest
from numpy.testing import assert_array_equal

import polars as pl


def test_series_array_method_copy_false() -> None:
    s = pl.Series([1, 2, None])
    with pytest.raises(RuntimeError, match="copy not allowed"):
        s.__array__(copy=False)

    result = s.__array__(copy=None)
    expected = np.array([1.0, 2.0, np.nan])
    assert_array_equal(result, expected)


@pytest.mark.parametrize("copy", [True, False])
def test_series_array_method_copy_zero_copy(copy: bool) -> None:
    s = pl.Series([1, 2, 3])
    result = s.__array__(copy=copy)

    assert result.flags.writeable is copy


def test_df_array_method() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out_array = np.asarray(df, order="F")
    expected_array = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] is True

    out_array = np.asarray(df, dtype=np.uint8, order="C")
    expected_array = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint8)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["C_CONTIGUOUS"] is True

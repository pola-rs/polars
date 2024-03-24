import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_df_approx_n_unique_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [2, 2, 2]})
    with pytest.deprecated_call():
        result = df.approx_n_unique()
    expected = pl.DataFrame({"a": [2], "b": [1]}).cast(pl.UInt32)
    assert_frame_equal(result, expected)


def test_lf_approx_n_unique_deprecated() -> None:
    df = pl.LazyFrame({"a": [1, 2, 2], "b": [2, 2, 2]})
    with pytest.deprecated_call():
        result = df.approx_n_unique()
    expected = pl.LazyFrame({"a": [2], "b": [1]}).cast(pl.UInt32)
    assert_frame_equal(result, expected)

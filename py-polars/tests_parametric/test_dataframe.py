# -------------------------------------------------
# Validate Series behaviour with parameteric tests
# -------------------------------------------------
from hypothesis import given

import polars as pl
from polars.testing import dataframes


@given(df=dataframes())
def test_repr(df: pl.DataFrame) -> None:
    assert isinstance(repr(df), str)
    # print(df)


@given(df=dataframes(allowed_dtypes=[pl.Boolean, pl.UInt64, pl.Utf8, pl.Time]))
def test_null_count(df: pl.DataFrame) -> None:
    null_count, ncols = df.null_count(), len(df.columns)
    if ncols == 0:
        assert null_count.shape == (0, 0)
    else:
        assert null_count.shape == (1, ncols)
        for idx, count in enumerate(null_count.rows()[0]):
            assert count == sum(v is None for v in df.select_at_idx(idx).to_list())

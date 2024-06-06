from __future__ import annotations

from hypothesis import example, given

import polars as pl
from polars.testing.parametric import dataframes


@given(
    df=dataframes(
        min_size=1,
        min_cols=1,
        allow_null=True,
        excluded_dtypes=[
            pl.String,
            pl.List,
            pl.Struct,  # See: https://github.com/pola-rs/polars/issues/3462
        ],
    )
)
@example(df=pl.DataFrame(schema=["x", "y", "z"]))
@example(df=pl.DataFrame())
def test_null_count(df: pl.DataFrame) -> None:
    # note: the zero-row and zero-col cases are always passed as explicit examples
    null_count, ncols = df.null_count(), len(df.columns)
    if ncols == 0:
        assert null_count.shape == (0, 0)
    else:
        assert null_count.shape == (1, ncols)
        for idx, count in enumerate(null_count.rows()[0]):
            assert count == sum(v is None for v in df.to_series(idx).to_list())

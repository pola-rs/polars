from __future__ import annotations

from hypothesis import example, given

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal
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
    null_count, ncols = df.null_count(), df.width
    assert null_count.shape == (1, ncols)
    for idx, count in enumerate(null_count.rows()[0]):
        assert count == sum(v is None for v in df.to_series(idx).to_list())


def test_null_count_optimization_23031() -> None:
    df = pl.DataFrame(data=[None, 2, None, 4, None, 6], schema={"col": pl.Int64})

    expected = pl.DataFrame(
        [
            pl.Series("count_all", [3], pl.UInt32()),
            pl.Series("sum_all", [12], pl.Int64()),
        ]
    )

    assert_frame_equal(
        df.select(
            count_all=pl.col("col").count(),
            sum_all=pl.when(pl.col("col").is_not_null().any()).then(
                pl.col("col").sum()
            ),
        ),
        expected,
    )

    assert_frame_equal(
        df.lazy()
        .select(
            count_all=pl.col("col").count(),
            sum_all=pl.when(pl.col("col").is_not_null().any()).then(
                pl.col("col").sum()
            ),
        )
        .collect(),
        expected,
    )

import polars as pl


def test_sqrt_neg_inf() -> None:
    out = pl.DataFrame(
        {
            "val": [float("-Inf"), -9, 0, 9, float("Inf")],
        }
    ).with_column(pl.col("val").sqrt().alias("sqrt"))
    # comparing nans and infinities by string value as they are not cmp
    assert str(out["sqrt"].to_list()) == str(
        [float("NaN"), float("NaN"), 0.0, 3.0, float("Inf")]
    )

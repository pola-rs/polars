import polars as pl
from polars.testing import assert_frame_equal


# https://github.com/pola-rs/polars/issues/5867
def test_with_context_ignore_5867() -> None:
    outer = pl.LazyFrame({"OtherCol": [1, 2, 3, 4]})
    lf = pl.LazyFrame({"Category": [1, 1, 2, 2], "Counts": [1, 2, 3, 4]}).with_context(
        outer
    )

    result = lf.group_by("Category", maintain_order=True).agg(pl.col("Counts").sum())

    expected = pl.LazyFrame({"Category": [1, 2], "Counts": [3, 7]})
    assert_frame_equal(result, expected)

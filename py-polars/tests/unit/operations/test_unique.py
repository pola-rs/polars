import polars as pl
from polars.testing import assert_frame_equal


def test_unique_predicate_pd() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["abc", "abc"],
            "y": ["xxx", "xxx"],
            "z": [True, False],
        }
    )

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="last")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame(schema={"x": pl.Utf8, "y": pl.Utf8, "z": pl.Boolean})
    assert_frame_equal(result, expected)

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="any")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame({"x": ["abc"], "y": ["xxx"], "z": [True]})
    assert_frame_equal(result, expected)

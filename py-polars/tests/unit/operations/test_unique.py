import polars as pl


def test_unique_predicate_pd() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["abc", "abc"],
            "y": ["xxx", "xxx"],
            "z": [True, False],
        }
    )

    assert lf.unique(subset=["x", "y"], maintain_order=True, keep="last").filter(
        pl.col("z")
    ).collect().to_dict(False) == {"x": [], "y": [], "z": []}
    assert lf.unique(subset=["x", "y"], maintain_order=True, keep="any").filter(
        pl.col("z")
    ).collect().to_dict(False) == {"x": ["abc"], "y": ["xxx"], "z": [True]}

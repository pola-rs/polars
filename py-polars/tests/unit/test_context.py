import polars as pl


def test_context_ignore_5867() -> None:
    outer = pl.DataFrame({"OtherCol": [1, 2, 3, 4]}).lazy()
    df = (
        pl.DataFrame({"Category": [1, 1, 2, 2], "Counts": [1, 2, 3, 4]})
        .lazy()
        .with_context(outer)
    )
    assert (
        df.groupby("Category", maintain_order=True)
        .agg([(pl.col("Counts")).sum()])
        .collect()
        .to_dict(False)
    ) == {"Category": [1, 2], "Counts": [3, 7]}

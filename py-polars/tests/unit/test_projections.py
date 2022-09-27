import polars as pl


def test_projection_on_semi_join_4789() -> None:
    lfa = pl.DataFrame({"a": [1], "p": [1]}).lazy()

    lfb = pl.DataFrame({"seq": [1], "p": [1]}).lazy()

    ab = lfa.join(lfb, on="p", how="semi").inspect()

    intermediate_agg = (ab.groupby("a").agg([pl.col("a").list().alias("seq")])).select(
        ["a", "seq"]
    )

    q = ab.join(intermediate_agg, on="a")

    assert q.collect().to_dict(False) == {"a": [1], "p": [1], "seq": [[1]]}


def test_melt_projection_pd_block_4997() -> None:
    assert (
        pl.DataFrame({"col1": ["a"], "col2": ["b"]})
        .with_row_count()
        .lazy()
        .melt(id_vars="row_nr")
        .groupby("row_nr")
        .agg(pl.col("variable").alias("result"))
        .collect()
    ).to_dict(False) == {"row_nr": [0], "result": [["col1", "col2"]]}

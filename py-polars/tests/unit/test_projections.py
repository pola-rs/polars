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

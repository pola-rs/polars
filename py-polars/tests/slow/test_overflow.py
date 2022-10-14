import polars as pl


def test_overflow_mean_partitioned_groupby_5194() -> None:
    for dtype in [pl.Int32, pl.UInt32]:
        df = pl.DataFrame(
            [
                pl.Series("data", [10_00_00_00] * 100_000, dtype=dtype),
                pl.Series("group", [1, 2] * 50_000, dtype=dtype),
            ]
        )
        assert df.groupby("group").agg(pl.col("data").mean()).sort(by="group").to_dict(
            False
        ) == {"group": [1, 2], "data": [10000000.0, 10000000.0]}

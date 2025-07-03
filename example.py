import polars as pl

df = pl.DataFrame({"a": [[1], [], [1, 2, 3]], "length": [2, 2, 4], "fill": [0, 1, 2]})
result = (
    df.lazy()
    .select(
        # length_expr=pl.col("a").list.pad_start(pl.col("a").list.len().max(), 999),
        length_col=pl.col("a").list.pad_start(pl.col("length"), pl.col("fill")),
    )
    .collect()
)
print(result)

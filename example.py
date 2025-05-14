import polars as pl

s = pl.Series(
    [
        [1, 2, 3, 4, 5],
        [1, 3, 7, 8],
        [6, 1, 4, 5],
    ]
)

df = pl.DataFrame(
    {
        "a": s,
    }
)

print(df.select(pl.col("a").list.filter(pl.element() > 3)))

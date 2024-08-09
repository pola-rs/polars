import polars as pl

df = pl.DataFrame(
    {
        "foo": [1, 2, 3, None, 5],
        "bar": [1.5, 0.9, 2.0, 0.0, None],
    }
)

# --8<-- [start:example1]
pl.col("foo").sort().head(2)
# --8<-- [end:example1]

# --8<-- [start:example2]
df.select(pl.col("foo").sort().head(2), pl.col("bar").filter(pl.col("foo") == 1).sum())
# --8<-- [end:example2]

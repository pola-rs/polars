import polars as pl

df = pl.DataFrame({"a": [[1, 2], [3, 4]]})
result = df.select(pl.col("a").list())
print(result)

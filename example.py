import polars as pl

df = pl.DataFrame({"a": [[1], [], [1, 2, 3]], "fill": [0, 999, 2]})
print(df.select(pl.col("a").list.pad_start(3, pl.col("fill"))))


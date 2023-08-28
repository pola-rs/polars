# --8<-- [start:example]
import polars as pl

q = (
    pl.scan_csv("docs/src/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .groupby("species")
    .agg(pl.all().sum())
)

df = q.collect()
# --8<-- [end:example]

import polars as pl

# --8<-- [start:streaming]
q = (
    pl.scan_csv("docs/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)

df = q.collect(streaming=True)
# --8<-- [end:streaming]

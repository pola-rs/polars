import polars as pl

# --8<-- [start:eager]

df = pl.read_csv("docs/data/iris.csv")
df_small = df.filter(pl.col("sepal_length") > 5)
df_agg = df_small.group_by("species").agg(pl.col("sepal_width").mean())
print(df_agg)
# --8<-- [end:eager]

# --8<-- [start:lazy]
q = (
    pl.scan_csv("docs/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)

df = q.collect()
# --8<-- [end:lazy]

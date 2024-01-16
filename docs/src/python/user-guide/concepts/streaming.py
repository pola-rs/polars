# --8<-- [start:import]
import polars as pl
# --8<-- [end:import]

# --8<-- [start:streaming]
q1 = (
    pl.scan_csv("docs/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)
df = q1.collect(streaming=True)
# --8<-- [end:streaming]

# --8<-- [start:example]
print(q1.explain(streaming=True))

# --8<-- [end:example]

# --8<-- [start:example2]
q2 = pl.scan_csv("docs/data/iris.csv").with_columns(
    pl.col("sepal_length").mean().over("species")
)

print(q2.explain(streaming=True))
# --8<-- [end:example2]

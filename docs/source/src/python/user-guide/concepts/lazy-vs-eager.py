# --8<-- [start:import]
import polars as pl

# --8<-- [end:import]

# --8<-- [start:eager]

df = pl.read_csv("docs/assets/data/iris.csv")
df_small = df.filter(pl.col("sepal_length") > 5)
df_agg = df_small.group_by("species").agg(pl.col("sepal_width").mean())
print(df_agg)
# --8<-- [end:eager]

# --8<-- [start:lazy]
q = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)

df = q.collect()
# --8<-- [end:lazy]

# --8<-- [start:explain]
print(q.explain())
# --8<-- [end:explain]

# --8<-- [start:explain-expression-expansion]
schema = pl.Schema(
    {
        "int_1": pl.Int16,
        "int_2": pl.Int32,
        "float_1": pl.Float64,
        "float_2": pl.Float64,
        "float_3": pl.Float64,
    }
)

print(
    pl.LazyFrame(schema=schema)
    .select((pl.col(pl.Float64) * 1.1).name.suffix("*1.1"))
    .explain()
)
# --8<-- [end:explain-expression-expansion]

# --8<-- [start:setup]
import polars as pl

df = pl.LazyFrame({"a": [1.242, 1.535]})

q = df.select(pl.col("a").round(1))
# --8<-- [end:setup]

# Avoiding requiring the GPU engine for doc build output
# --8<-- [start:simple-result]
result = q.collect()
print(result)
# --8<-- [end:simple-result]


# --8<-- [start:engine-setup]
q = df.select((pl.col("a") ** 4))

# --8<-- [end:engine-setup]

# --8<-- [start:engine-result]
result = q.collect()
print(result)
# --8<-- [end:engine-result]

# --8<-- [start:fallback-setup]
df = pl.LazyFrame(
    {
        "value": [1, 2, 3, 4, 5, 6, 7, 8],
    }
)

q = df.select(pl.col("value").map_elements(lambda x: 0 if x > 4 else x + 2))

# --8<-- [end:fallback-setup]

# --8<-- [start:fallback-result]
print(
    "PerformanceWarning: Query execution with GPU not possible: unsupported operations"
)
print("The errors were:")
print("- NotImplementedError: anonymousfunction")
print("  return wrap_df(ldf.collect(engine, callback))")
print()
print(q.collect())
# --8<- [end:fallback-result]

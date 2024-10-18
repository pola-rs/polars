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
        "key": [1, 1, 1, 2, 3, 3, 2, 2],
        "value": [1, 2, 3, 4, 5, 6, 7, 8],
    }
)

q = df.select(pl.col("value").sum().over("key"))

# --8<-- [end:fallback-setup]

# --8<-- [start:fallback-result]
print(
    "PerformanceWarning: Query execution with GPU not supported, reason: \n"
    "<class 'NotImplementedError'>: Grouped rolling window not implemented"
)
print("# some details elided")
print()
print(q.collect())
# --8<- [end:fallback-result]

# --8<-- [start:dfnum]
import polars as pl

df = pl.DataFrame(
    {
        "integers": [17, 42, None, 35],
    }
)

print(df)
# --8<-- [end:dfnum]

# --8<-- [start:index_of]
result = df.select(pl.col("integers").index_of(35))
print("First index of 35 is:", result)
result = df.select(pl.col("integers").index_of(None))
print("First index of null is:", result)
# --8<-- [end:index_of]

# --8<-- [start:index_of_not_found]
result = df.select(pl.col("integers").index_of(1233))
print("Index of unfound number is:", result)
# --8<-- [end:index_of_not_found]

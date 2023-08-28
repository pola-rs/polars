# --8<-- [start:setup]

import polars as pl

# --8<-- [end:setup]

# --8<-- [start:dataframe]
df = pl.DataFrame(
    {
        "keys": ["a", "a", "b"],
        "values": [10, 7, 1],
    }
)

out = df.groupby("keys", maintain_order=True).agg(
    pl.col("values").map(lambda s: s.shift()).alias("shift_map"),
    pl.col("values").shift().alias("shift_expression"),
)
print(df)
# --8<-- [end:dataframe]


# --8<-- [start:apply]
out = df.groupby("keys", maintain_order=True).agg(
    pl.col("values").apply(lambda s: s.shift()).alias("shift_map"),
    pl.col("values").shift().alias("shift_expression"),
)
print(out)
# --8<-- [end:apply]

# --8<-- [start:counter]
counter = 0


def add_counter(val: int) -> int:
    global counter
    counter += 1
    return counter + val


out = df.select(
    pl.col("values").apply(add_counter).alias("solution_apply"),
    (pl.col("values") + pl.arange(1, pl.count() + 1)).alias("solution_expr"),
)
print(out)
# --8<-- [end:counter]

# --8<-- [start:combine]
out = df.select(
    pl.struct(["keys", "values"])
    .apply(lambda x: len(x["keys"]) + x["values"])
    .alias("solution_apply"),
    (pl.col("keys").str.lengths() + pl.col("values")).alias("solution_expr"),
)
print(out)
# --8<-- [end:combine]

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

out = df.group_by("keys", maintain_order=True).agg(
    pl.col("values").map_batches(lambda s: s.shift()).alias("shift_map"),
    pl.col("values").shift().alias("shift_expression"),
)
print(df)
# --8<-- [end:dataframe]


# --8<-- [start:apply]
out = df.group_by("keys", maintain_order=True).agg(
    pl.col("values").map_elements(lambda s: s.shift()).alias("shift_map"),
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
    pl.col("values").map_elements(add_counter).alias("solution_apply"),
    (pl.col("values") + pl.int_range(1, pl.count() + 1)).alias("solution_expr"),
)
print(out)
# --8<-- [end:counter]

# --8<-- [start:combine]
out = df.select(
    pl.struct(["keys", "values"])
    .map_elements(lambda x: len(x["keys"]) + x["values"])
    .alias("solution_apply"),
    (pl.col("keys").str.len_bytes() + pl.col("values")).alias("solution_expr"),
)
print(out)
# --8<-- [end:combine]

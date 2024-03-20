# --8<-- [start:setup]

import polars as pl

# --8<-- [end:setup]

# --8<-- [start:dataframe]
df = pl.DataFrame(
    {
        "keys": ["a", "a", "b", "b"],
        "values": [10, 7, 1, 23],
    }
)
print(df)
# --8<-- [end:dataframe]


# --8<-- [start:diff_from_mean]
def diff_from_mean(series):
    # This will be very slow for non-trivial Series, since it's all Python
    # code:
    total = 0
    for value in series:
        total += value
    mean = total / len(series)
    return pl.Series([value - mean for value in series])


# Apply our custom function a full Series with map_batches():
out = df.select(pl.col("values").map_batches(diff_from_mean))
print("== select() with UDF ==")
print(out)

# Apply our custom function per group:
print("== group_by() with UDF ==")
out = df.group_by("keys").agg(pl.col("values").map_batches(diff_from_mean))
print(out)
# --8<-- [end:diff_from_mean]

# --8<-- [start:np_log]
import numpy as np

out = df.select(pl.col("values").map_batches(np.log))
print(out)
# --8<-- [end:np_log]

# --8<-- [start:diff_from_mean_numba]
from numba import guvectorize, int64, float64


# This will be compiled to machine code, so it will be fast. The Series is
# converted to a NumPy array before being passed to the function:
@guvectorize([(int64[:], float64[:])], "(n)->(n)")
def diff_from_mean_numba(arr, result):
    total = 0
    for value in arr:
        total += value
    mean = total / len(arr)
    for i, value in enumerate(arr):
        result[i] = value - mean


out = df.select(pl.col("values").map_batches(diff_from_mean_numba))
print("== select() with UDF ==")
# assert out["values"].item() == 18
print(out)

out = df.group_by("keys").agg(pl.col("values").map_batches(diff_from_mean_numba))
print("== group_by() with UDF ==")
print(out)
# --8<-- [end:diff_from_mean_numba]

# --8<-- [start:dataframe2]
df2 = pl.DataFrame(
    {
        "values": [1, 2, 3, None, 4],
    }
)
print(df2)
# --8<-- [end:dataframe2]


# --8<-- [start:missing_data]
# Implement equivalent of diff_from_mean_numba() using Polars APIs:
out = df2.select(pl.col("values") - pl.col("values").mean())
print("== built-in mean() knows to skip empty values ==")
# assert out["values"][0] == 2.5
print(out)

out = df2.select(pl.col("values").map_batches(diff_from_mean_numba))
print("== custom mean gets the wrong answer because of missing data ==")
# assert out["values"][0] != 2.5
print(out)

# --8<-- [end:missing_data]

# --8<-- [start:combine]
out = df.select(
    pl.struct(["keys", "values"])
    .map_elements(lambda x: len(x["keys"]) + x["values"])
    .alias("solution_map_elements"),
    (pl.col("keys").str.len_bytes() + pl.col("values")).alias("solution_expr"),
)
print(out)
# --8<-- [end:combine]

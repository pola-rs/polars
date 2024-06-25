# --8<-- [start:setup]

import warnings

import polars as pl
from polars.exceptions import PolarsInefficientMapWarning

warnings.simplefilter("ignore", PolarsInefficientMapWarning)
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

# --8<-- [start:individual_log]
import math


def my_log(value):
    return math.log(value)


out = df.select(pl.col("values").map_elements(my_log, return_dtype=pl.Float64))
print(out)
# --8<-- [end:individual_log]


# --8<-- [start:diff_from_mean]
def diff_from_mean(series):
    # This will be very slow for non-trivial Series, since it's all Python
    # code:
    total = 0
    for value in series:
        total += value
    mean = total / len(series)
    return pl.Series([value - mean for value in series])


# Apply our custom function to a full Series with map_batches():
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
from numba import float64, guvectorize, int64


# This will be compiled to machine code, so it will be fast. The Series is
# converted to a NumPy array before being passed to the function. See the
# Numba documentation for more details:
# https://numba.readthedocs.io/en/stable/user/vectorize.html
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
print(out)

out = df.group_by("keys").agg(pl.col("values").map_batches(diff_from_mean_numba))
print("== group_by() with UDF ==")
print(out)
# --8<-- [end:diff_from_mean_numba]


# --8<-- [start:combine]
# Add two arrays together:
@guvectorize([(int64[:], int64[:], float64[:])], "(n),(n)->(n)")
def add(arr, arr2, result):
    for i in range(len(arr)):
        result[i] = arr[i] + arr2[i]


df3 = pl.DataFrame({"values1": [1, 2, 3], "values2": [10, 20, 30]})

out = df3.select(
    # Create a struct that has two columns in it:
    pl.struct(["values1", "values2"])
    # Pass the struct to a lambda that then passes the individual columns to
    # the add() function:
    .map_batches(
        lambda combined: add(
            combined.struct.field("values1"), combined.struct.field("values2")
        )
    )
    .alias("add_columns")
)
print(out)
# --8<-- [end:combine]

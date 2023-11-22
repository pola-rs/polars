import polars as pl

# --8<-- [start:example]
enum_dtype = pl.Enum(["Polar", "Panda", "Brown"])
enum_series = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=enum_dtype)
cat_series = pl.Series(
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
)
# --8<-- [end:example]


# --8<-- [start:append]
cat_series = pl.Series(
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
)
cat2_series = pl.Series(
    ["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=pl.Categorical
)
cat_series.append(cat2_series)
# --8<-- [end:append]


# --8<-- [start:global_append]
with pl.StringCache():
    cat_series = pl.Series(
        ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
    )
    cat2_series = pl.Series(
        ["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=pl.Categorical
    )
    cat_series.append(cat2_series)
# --8<-- [end:global_append]


# --8<-- [start:enum_append]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_series = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=dtype)
cat2_series = pl.Series(["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=dtype)
cat_series.append(cat2_series)
# --8<-- [end:enum_append]

# --8<-- [start:enum_error]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
try:
    cat_series = pl.Series(["Polar", "Panda", "Brown", "Black"], dtype=dtype)
except Exception as e:
    print(e)
# --8<-- [end:enum_error]

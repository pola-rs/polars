# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

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
# Triggers a CategoricalRemappingWarning: Local categoricals have different encodings, expensive re-encoding is done
print(cat_series.append(cat2_series))
# --8<-- [end:append]


# --8<-- [start:global_append]
with pl.StringCache():
    cat_series = pl.Series(
        ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
    )
    cat2_series = pl.Series(
        ["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=pl.Categorical
    )
    print(cat_series.append(cat2_series))
# --8<-- [end:global_append]


# --8<-- [start:enum_append]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_series = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=dtype)
cat2_series = pl.Series(["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=dtype)
print(cat_series.append(cat2_series))
# --8<-- [end:enum_append]

# --8<-- [start:enum_error]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
try:
    cat_series = pl.Series(["Polar", "Panda", "Brown", "Black"], dtype=dtype)
except Exception as e:
    print(e)
# --8<-- [end:enum_error]

# --8<-- [start:equality]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_series = pl.Series(["Brown", "Panda", "Polar"], dtype=dtype)
cat_series2 = pl.Series(["Polar", "Panda", "Brown"], dtype=dtype)
print(cat_series == cat_series2)
# --8<-- [end:equality]

# --8<-- [start:global_equality]
with pl.StringCache():
    cat_series = pl.Series(["Brown", "Panda", "Polar"], dtype=pl.Categorical)
    cat_series2 = pl.Series(["Polar", "Panda", "Black"], dtype=pl.Categorical)
    print(cat_series == cat_series2)
# --8<-- [end:global_equality]

# --8<-- [start:equality]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_series = pl.Series(["Brown", "Panda", "Polar"], dtype=dtype)
cat_series2 = pl.Series(["Polar", "Panda", "Brown"], dtype=dtype)
print(cat_series == cat_series2)
# --8<-- [end:equality]

# --8<-- [start:str_compare_single]
cat_series = pl.Series(["Brown", "Panda", "Polar"], dtype=pl.Categorical)
print(cat_series <= "Cat")
# --8<-- [end:str_compare_single]

# --8<-- [start:str_compare]
cat_series = pl.Series(["Brown", "Panda", "Polar"], dtype=pl.Categorical)
cat_series_utf = pl.Series(["Panda", "Panda", "Polar"])
print(cat_series <= cat_series_utf)
# --8<-- [end:str_compare]

# --8<-- [start:str_enum_compare_error]
try:
    cat_series = pl.Series(
        ["Low", "Medium", "High"], dtype=pl.Enum(["Low", "Medium", "High"])
    )
    cat_series <= "Excellent"
except Exception as e:
    print(e)
# --8<-- [end:str_enum_compare_error]

# --8<-- [start:str_enum_compare_single]
dtype = pl.Enum(["Low", "Medium", "High"])
cat_series = pl.Series(["Low", "Medium", "High"], dtype=dtype)
print(cat_series <= "Medium")
# --8<-- [end:str_enum_compare_single]

# --8<-- [start:str_enum_compare]
dtype = pl.Enum(["Low", "Medium", "High"])
cat_series = pl.Series(["Low", "Medium", "High"], dtype=dtype)
cat_series2 = pl.Series(["High", "High", "Low"], dtype=dtype)
print(cat_series <= cat_series2)
# --8<-- [end:str_enum_compare]

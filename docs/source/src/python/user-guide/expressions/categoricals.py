# --8<-- [start:enum-example]
import polars as pl

bears_enum = pl.Enum(["Polar", "Panda", "Brown"])
bears = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=bears_enum)
print(bears)
# --8<-- [end:enum-example]

# --8<-- [start:enum-wrong-value]
from polars.exceptions import InvalidOperationError

try:
    bears_kind_of = pl.Series(
        ["Polar", "Panda", "Brown", "Polar", "Shark"],
        dtype=bears_enum,
    )
except InvalidOperationError as exc:
    print("InvalidOperationError:", exc)
# --8<-- [end:enum-wrong-value]

# --8<-- [start:log-levels]
log_levels = pl.Enum(["debug", "info", "warning", "error"])

logs = pl.DataFrame(
    {
        "level": ["debug", "info", "debug", "error"],
        "message": [
            "process id: 525",
            "Service started correctly",
            "startup time: 67ms",
            "Cannot connect to DB!",
        ],
    },
    schema_overrides={
        "level": log_levels,
    },
)

non_debug_logs = logs.filter(
    pl.col("level") > "debug",
)
print(non_debug_logs)
# --8<-- [end:log-levels]

# --8<-- [start:string-comparison-error-display]
logs.select(pl.col("level") > "Pretty bad")  # This is not a valid logging level
# --8<-- [end:string-comparison-error-display]

# --8<-- [start:string-comparison-error-execution]
try:
    logs.select(pl.col("level") > "Pretty bad")
except InvalidOperationError as err:
    print(err)
else:
    raise AssertionError("Expected an InvalidOperationError")
# --8<-- [end:string-comparison-error-execution]

# --8<-- [start:enum-column-comparison]
str_series = pl.Series(["info", "debug", "debug", "error"])
print(logs["level"] == str_series)
# --8<-- [end:enum-column-comparison]

# --8<-- [start:categorical-example]
bears_cat = pl.Series(
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
)
print(bears_cat)
# --8<-- [end:categorical-example]

# --8<-- [start:categories-example]
bear_categories = pl.Categories(name="bear_species", physical=pl.UInt8)
bears = pl.DataFrame(
    {"species": ["Polar", "Brown", "Panda", "Brown", "Polar"]},
    schema_overrides={"species": pl.Categorical(bear_categories)},
)
print(bears)
# --8<-- [end:categories-example]

# --8<-- [start:categorical-comparison-string]
print(
    pl.DataFrame({"categorical": bears_cat}).with_columns(
        (pl.col("categorical") < "Cat").alias('categorical < "Cat"')
    )
)
# --8<-- [end:categorical-comparison-string]

# --8<-- [start:categorical-comparison-string-column]
print(
    pl.DataFrame(
        {
            "categorical": bears_cat,
            "string": pl.Series(["Panda", "Brown", "Brown", "Polar", "Polar"]),
        }
    ).with_columns(
        (pl.col("categorical") == pl.col("string")).alias("categorical == string"),
    )
)
# --8<-- [end:categorical-comparison-string-column]

# --8<-- [start:concatenating-categoricals]
male_bears = pl.DataFrame(
    {
        "species": ["Polar", "Brown", "Panda"],
        "weight": [450, 500, 110],  # kg
    },
    schema_overrides={"species": pl.Categorical},
)
female_bears = pl.DataFrame(
    {
        "species": ["Brown", "Polar", "Panda"],
        "weight": [340, 200, 90],  # kg
    },
    schema_overrides={"species": pl.Categorical},
)
bears = pl.concat([male_bears, female_bears], how="vertical")
print(bears)
# --8<-- [end:concatenating-categoricals]


# --8<-- [start:example]
import polars as pl

bears_enum = pl.Enum(["Polar", "Panda", "Brown"])
bears = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=bears_enum)
print(bears)

cat_bears = pl.Series(
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
)
# --8<-- [end:example]


# --8<-- [start:append]
cat_bears = pl.Series(
    ["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=pl.Categorical
)
cat2_series = pl.Series(
    ["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=pl.Categorical
)

print(cat_bears.extend(cat2_series))
# --8<-- [end:append]

# --8<-- [start:enum_append]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_bears = pl.Series(["Polar", "Panda", "Brown", "Brown", "Polar"], dtype=dtype)
cat2_series = pl.Series(["Panda", "Brown", "Brown", "Polar", "Polar"], dtype=dtype)
print(cat_bears.extend(cat2_series))
# --8<-- [end:enum_append]

# --8<-- [start:enum_error]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
try:
    cat_bears = pl.Series(["Polar", "Panda", "Brown", "Black"], dtype=dtype)
except Exception as e:
    print(e)
# --8<-- [end:enum_error]

# --8<-- [start:equality]
dtype = pl.Enum(["Polar", "Panda", "Brown"])
cat_bears = pl.Series(["Brown", "Panda", "Polar"], dtype=dtype)
cat_series2 = pl.Series(["Polar", "Panda", "Brown"], dtype=dtype)
print(cat_bears == cat_series2)
# --8<-- [end:equality]

# --8<-- [start:str_compare_single]
cat_bears = pl.Series(["Brown", "Panda", "Polar"], dtype=pl.Categorical)
print(cat_bears <= "Cat")
# --8<-- [end:str_compare_single]

# --8<-- [start:str_compare]
cat_bears = pl.Series(["Brown", "Panda", "Polar"], dtype=pl.Categorical)
cat_series_utf = pl.Series(["Panda", "Panda", "Polar"])
print(cat_bears <= cat_series_utf)
# --8<-- [end:str_compare]

# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:weather_df]
weather = pl.DataFrame(
    {
        "station": ["Station " + str(x) for x in range(1, 6)],
        "temperatures": [
            "20 5 5 E1 7 13 19 9 6 20",
            "18 8 16 11 23 E2 8 E2 E2 E2 90 70 40",
            "19 24 E9 16 6 12 10 22",
            "E2 E0 15 7 8 10 E1 24 17 13 6",
            "14 8 E0 16 22 24 E1",
        ],
    }
)
print(weather)
# --8<-- [end:weather_df]

# --8<-- [start:string_to_list]
out = weather.with_columns(pl.col("temperatures").str.split(" "))
print(out)
# --8<-- [end:string_to_list]

# --8<-- [start:explode_to_atomic]
out = weather.with_columns(pl.col("temperatures").str.split(" ")).explode(
    "temperatures"
)
print(out)
# --8<-- [end:explode_to_atomic]

# --8<-- [start:list_ops]
out = weather.with_columns(pl.col("temperatures").str.split(" ")).with_columns(
    pl.col("temperatures").list.head(3).alias("top3"),
    pl.col("temperatures").list.slice(-3, 3).alias("bottom_3"),
    pl.col("temperatures").list.lengths().alias("obs"),
)
print(out)
# --8<-- [end:list_ops]


# --8<-- [start:count_errors]
out = weather.with_columns(
    pl.col("temperatures")
    .str.split(" ")
    .list.eval(pl.element().cast(pl.Int64, strict=False).is_null())
    .list.sum()
    .alias("errors")
)
print(out)
# --8<-- [end:count_errors]

# --8<-- [start:count_errors_regex]
out = weather.with_columns(
    pl.col("temperatures")
    .str.split(" ")
    .list.eval(pl.element().str.contains("(?i)[a-z]"))
    .list.sum()
    .alias("errors")
)
print(out)
# --8<-- [end:count_errors_regex]

# --8<-- [start:weather_by_day]
weather_by_day = pl.DataFrame(
    {
        "station": ["Station " + str(x) for x in range(1, 11)],
        "day_1": [17, 11, 8, 22, 9, 21, 20, 8, 8, 17],
        "day_2": [15, 11, 10, 8, 7, 14, 18, 21, 15, 13],
        "day_3": [16, 15, 24, 24, 8, 23, 19, 23, 16, 10],
    }
)
print(weather_by_day)
# --8<-- [end:weather_by_day]

# --8<-- [start:weather_by_day_rank]
rank_pct = (pl.element().rank(descending=True) / pl.col("*").count()).round(2)

out = weather_by_day.with_columns(
    # create the list of homogeneous data
    pl.concat_list(pl.all().exclude("station")).alias("all_temps")
).select(
    # select all columns except the intermediate list
    pl.all().exclude("all_temps"),
    # compute the rank by calling `list.eval`
    pl.col("all_temps").list.eval(rank_pct, parallel=True).alias("temps_rank"),
)

print(out)
# --8<-- [end:weather_by_day_rank]

# --8<-- [start:array_df]
array_df = pl.DataFrame(
    [
        pl.Series("Array_1", [[1, 3], [2, 5]]),
        pl.Series("Array_2", [[1, 7, 3], [8, 1, 0]]),
    ],
    schema={"Array_1": pl.Array(2, pl.Int64), "Array_2": pl.Array(3, pl.Int64)},
)
print(array_df)
# --8<-- [end:array_df]

# --8<-- [start:array_ops]
out = array_df.select(
    pl.col("Array_1").arr.min().suffix("_min"),
    pl.col("Array_2").arr.sum().suffix("_sum"),
)
print(out)
# --8<-- [end:array_ops]

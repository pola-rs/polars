# --8<-- [start:list-example]
from datetime import datetime
import polars as pl

df = pl.DataFrame(
    {
        "names": [
            ["Anne", "Averill", "Adams"],
            ["Brandon", "Brooke", "Borden", "Branson"],
            ["Camila", "Campbell"],
            ["Dennis", "Doyle"],
        ],
        "children_ages": [
            [5, 7],
            [],
            [],
            [8, 11, 18],
        ],
        "medical_appointments": [
            [],
            [],
            [],
            [datetime(2022, 5, 22, 16, 30)],
        ],
    }
)

print(df)
# --8<-- [end:list-example]

# --8<-- [start:array-example]
df = pl.DataFrame(
    {
        "bit_flags": [
            [True, True, True, True, False],
            [False, True, True, True, True],
        ],
        "tic_tac_toe": [
            [
                [" ", "x", "o"],
                [" ", "x", " "],
                ["o", "x", " "],
            ],
            [
                ["o", "x", "x"],
                [" ", "o", "x"],
                [" ", " ", "o"],
            ],
        ],
    },
    schema={
        "bit_flags": pl.Array(pl.Boolean, 5),
        "tic_tac_toe": pl.Array(pl.String, (3, 3)),
    },
)

print(df)
# --8<-- [end:array-example]

# --8<-- [start:numpy-array-inference]
import numpy as np

array = np.arange(0, 120).reshape((5, 2, 3, 4))  # 4D array

print(pl.Series(array).dtype)  # Column with the 3D subarrays
# --8<-- [end:numpy-array-inference]

# --8<-- [start:weather]
weather = pl.DataFrame(
    {
        "station": [f"Station {idx}" for idx in range(1, 6)],
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
# --8<-- [end:weather]

# --8<-- [start:split]
weather = weather.with_columns(
    pl.col("temperatures").str.split(" "),
)
print(weather)
# --8<-- [end:split]

# --8<-- [start:explode]
result = weather.explode("temperatures")
print(result)
# --8<-- [end:explode]

# --8<-- [start:list-slicing]
result = weather.with_columns(
    pl.col("temperatures").list.head(3).alias("head"),
    pl.col("temperatures").list.tail(3).alias("tail"),
    pl.col("temperatures").list.slice(-3, 2).alias("two_next_to_last"),
)
print(result)
# --8<-- [end:list-slicing]

# --8<-- [start:element-wise-casting]
result = weather.with_columns(
    pl.col("temperatures")
    .list.eval(pl.element().cast(pl.Int64, strict=False).is_null())
    .list.sum()
    .alias("errors"),
)
print(result)
# --8<-- [end:element-wise-casting]

# --8<-- [start:element-wise-regex]
result2 = weather.with_columns(
    pl.col("temperatures")
    .list.eval(pl.element().str.contains("(?i)[a-z]"))
    .list.sum()
    .alias("errors"),
)
print(result.equals(result2))
# --8<-- [end:element-wise-regex]

# --8<-- [start:weather_by_day]
weather_by_day = pl.DataFrame(
    {
        "station": [f"Station {idx}" for idx in range(1, 11)],
        "day_1": [17, 11, 8, 22, 9, 21, 20, 8, 8, 17],
        "day_2": [15, 11, 10, 8, 7, 14, 18, 21, 15, 13],
        "day_3": [16, 15, 24, 24, 8, 23, 19, 23, 16, 10],
    }
)
print(weather_by_day)
# --8<-- [end:weather_by_day]

# --8<-- [start:rank_pct]
rank_pct = (pl.element().rank(descending=True) / pl.all().count()).round(2)

result = weather_by_day.with_columns(
    # create the list of homogeneous data
    pl.concat_list(pl.all().exclude("station")).alias("all_temps")
).select(
    # select all columns except the intermediate list
    pl.all().exclude("all_temps"),
    # compute the rank by calling `list.eval`
    pl.col("all_temps").list.eval(rank_pct, parallel=True).alias("temps_rank"),
)

print(result)
# --8<-- [end:rank_pct]

# --8<-- [start:array-overview]
df = pl.DataFrame(
    {
        "first_last": [
            ["Anne", "Adams"],
            ["Brandon", "Branson"],
            ["Camila", "Campbell"],
            ["Dennis", "Doyle"],
        ],
        "fav_numbers": [
            [42, 0, 1],
            [2, 3, 5],
            [13, 21, 34],
            [73, 3, 7],
        ],
    },
    schema={
        "first_last": pl.Array(pl.String, 2),
        "fav_numbers": pl.Array(pl.Int32, 3),
    },
)

result = df.select(
    pl.col("first_last").arr.join(" ").alias("name"),
    pl.col("fav_numbers").arr.sort(),
    pl.col("fav_numbers").arr.max().alias("largest_fav"),
    pl.col("fav_numbers").arr.sum().alias("summed"),
    pl.col("fav_numbers").arr.contains(3).alias("likes_3"),
)
print(result)
# --8<-- [end:array-overview]

# --8<-- [start:ratings_df]
import polars as pl

ratings = pl.DataFrame(
    {
        "Movie": ["Cars", "IT", "ET", "Cars", "Up", "IT", "Cars", "ET", "Up", "Cars"],
        "Theatre": ["NE", "ME", "IL", "ND", "NE", "SD", "NE", "IL", "IL", "NE"],
        "Avg_Rating": [4.5, 4.4, 4.6, 4.3, 4.8, 4.7, 4.5, 4.9, 4.7, 4.6],
        "Count": [30, 27, 26, 29, 31, 28, 28, 26, 33, 28],
    }
)
print(ratings)
# --8<-- [end:ratings_df]

# --8<-- [start:state_value_counts]
result = ratings.select(pl.col("Theatre").value_counts(sort=True))
print(result)
# --8<-- [end:state_value_counts]

# --8<-- [start:struct_unnest]
result = ratings.select(pl.col("Theatre").value_counts(sort=True)).unnest("Theatre")
print(result)
# --8<-- [end:struct_unnest]

# --8<-- [start:series_struct]
rating_series = pl.Series(
    "ratings",
    [
        {"Movie": "Cars", "Theatre": "NE", "Avg_Rating": 4.5},
        {"Movie": "Toy Story", "Theatre": "ME", "Avg_Rating": 4.9},
    ],
)
print(rating_series)
# --8<-- [end:series_struct]

# --8<-- [start:series_struct_error]
null_rating_series = pl.Series(
    "ratings",
    [
        {"Movie": "Cars", "Theatre": "NE", "Avg_Rating": 4.5},
        {"Mov": "Toy Story", "Theatre": "ME", "Avg_Rating": 4.9},
        {"Movie": "Snow White", "Theatre": "IL", "Avg_Rating": "4.7"},
    ],
    strict=False,  # To show the final structs with `null` values.
)
print(null_rating_series)
# --8<-- [end:series_struct_error]

# --8<-- [start:series_struct_extract]
result = rating_series.struct.field("Movie")
print(result)
# --8<-- [end:series_struct_extract]

# --8<-- [start:series_struct_rename]
result = rating_series.struct.rename_fields(["Film", "State", "Value"])
print(result)
# --8<-- [end:series_struct_rename]

# --8<-- [start:struct-rename-check]
print(
    result.to_frame().unnest("ratings"),
)
# --8<-- [end:struct-rename-check]

# --8<-- [start:struct_duplicates]
result = ratings.filter(pl.struct("Movie", "Theatre").is_duplicated())
print(result)
# --8<-- [end:struct_duplicates]

# --8<-- [start:struct_ranking]
result = ratings.with_columns(
    pl.struct("Count", "Avg_Rating")
    .rank("dense", descending=True)
    .over("Movie", "Theatre")
    .alias("Rank")
).filter(pl.struct("Movie", "Theatre").is_duplicated())

print(result)
# --8<-- [end:struct_ranking]

# --8<-- [start:multi_column_apply]
df = pl.DataFrame({"keys": ["a", "a", "b"], "values": [10, 7, 1]})

result = df.select(
    pl.struct(["keys", "values"])
    .map_elements(lambda x: len(x["keys"]) + x["values"], return_dtype=pl.Int64)
    .alias("solution_map_elements"),
    (pl.col("keys").str.len_bytes() + pl.col("values")).alias("solution_expr"),
)
print(result)
# --8<-- [end:multi_column_apply]


# --8<-- [start:ack]
def ack(m, n):
    if not m:
        return n + 1
    if not n:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))


# --8<-- [end:ack]

# --8<-- [start:struct-ack]
values = pl.DataFrame(
    {
        "m": [0, 0, 0, 1, 1, 1, 2],
        "n": [2, 3, 4, 1, 2, 3, 1],
    }
)
result = values.with_columns(
    pl.struct(["m", "n"])
    .map_elements(lambda s: ack(s["m"], s["n"]), return_dtype=pl.Int64)
    .alias("ack")
)

print(result)
# --8<-- [end:struct-ack]

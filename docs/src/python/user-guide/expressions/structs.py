# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:ratings_df]
ratings = pl.DataFrame(
    {
        "Movie": ["Cars", "IT", "ET", "Cars", "Up", "IT", "Cars", "ET", "Up", "ET"],
        "Theatre": ["NE", "ME", "IL", "ND", "NE", "SD", "NE", "IL", "IL", "SD"],
        "Avg_Rating": [4.5, 4.4, 4.6, 4.3, 4.8, 4.7, 4.7, 4.9, 4.7, 4.6],
        "Count": [30, 27, 26, 29, 31, 28, 28, 26, 33, 26],
    }
)
print(ratings)
# --8<-- [end:ratings_df]

# --8<-- [start:state_value_counts]
out = ratings.select(pl.col("Theatre").value_counts(sort=True))
print(out)
# --8<-- [end:state_value_counts]

# --8<-- [start:struct_unnest]
out = ratings.select(pl.col("Theatre").value_counts(sort=True)).unnest("Theatre")
print(out)
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

# --8<-- [start:series_struct_extract]
out = rating_series.struct.field("Movie")
print(out)
# --8<-- [end:series_struct_extract]

# --8<-- [start:series_struct_rename]
out = (
    rating_series.to_frame()
    .select(pl.col("ratings").struct.rename_fields(["Film", "State", "Value"]))
    .unnest("ratings")
)
print(out)
# --8<-- [end:series_struct_rename]

# --8<-- [start:struct_duplicates]
out = ratings.filter(pl.struct("Movie", "Theatre").is_duplicated())
print(out)
# --8<-- [end:struct_duplicates]

# --8<-- [start:struct_ranking]
out = ratings.with_columns(
    pl.struct("Count", "Avg_Rating")
    .rank("dense", descending=True)
    .over("Movie", "Theatre")
    .alias("Rank")
).filter(pl.struct("Movie", "Theatre").is_duplicated())
print(out)
# --8<-- [end:struct_ranking]

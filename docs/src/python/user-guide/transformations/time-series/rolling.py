# --8<-- [start:setup]
from datetime import date, datetime

import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
df = pl.read_csv("docs/data/apple_stock.csv", try_parse_dates=True)
df = df.sort("Date")
print(df)
# --8<-- [end:df]

# --8<-- [start:group_by]
annual_average_df = df.group_by_dynamic("Date", every="1y").agg(pl.col("Close").mean())

df_with_year = annual_average_df.with_columns(pl.col("Date").dt.year().alias("year"))
print(df_with_year)
# --8<-- [end:group_by]

# --8<-- [start:group_by_dyn]
df = (
    pl.date_range(
        start=date(2021, 1, 1),
        end=date(2021, 12, 31),
        interval="1d",
        eager=True,
    )
    .alias("time")
    .to_frame()
)

out = df.group_by_dynamic("time", every="1mo", period="1mo", closed="left").agg(
    pl.col("time").cum_count().reverse().head(3).alias("day/eom"),
    ((pl.col("time") - pl.col("time").first()).last().dt.total_days() + 1).alias(
        "days_in_month"
    ),
)
print(out)
# --8<-- [end:group_by_dyn]

# --8<-- [start:group_by_roll]
df = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="30m",
            eager=True,
        ),
        "groups": ["a", "a", "a", "b", "b", "a", "a"],
    }
)
print(df)
# --8<-- [end:group_by_roll]

# --8<-- [start:group_by_dyn2]
out = df.group_by_dynamic(
    "time",
    every="1h",
    closed="both",
    group_by="groups",
    include_boundaries=True,
).agg(pl.len())
print(out)
# --8<-- [end:group_by_dyn2]

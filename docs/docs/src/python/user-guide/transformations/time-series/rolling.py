# --8<-- [start:setup]
import polars as pl
from datetime import datetime

# --8<-- [end:setup]

# --8<-- [start:df]
df = pl.read_csv("docs/src/data/appleStock.csv", try_parse_dates=True)
df = df.sort("Date")
print(df)
# --8<-- [end:df]

# --8<-- [start:groupby]
annual_average_df = df.groupby_dynamic("Date", every="1y").agg(pl.col("Close").mean())

df_with_year = annual_average_df.with_columns(pl.col("Date").dt.year().alias("year"))
print(df_with_year)
# --8<-- [end:groupby]

# --8<-- [start:groupbydyn]
df = (
    pl.date_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 12, 31),
        interval="1d",
        eager=True,
    )
    .alias("time")
    .to_frame()
)

out = (
    df.groupby_dynamic("time", every="1mo", period="1mo", closed="left")
    .agg(
        [
            pl.col("time").cumcount().reverse().head(3).alias("day/eom"),
            ((pl.col("time") - pl.col("time").first()).last().dt.days() + 1).alias(
                "days_in_month"
            ),
        ]
    )
    .explode("day/eom")
)
print(out)
# --8<-- [end:groupbydyn]

# --8<-- [start:groupbyroll]
df = pl.DataFrame(
    {
        "time": pl.date_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="30m",
            eager=True,
        ),
        "groups": ["a", "a", "a", "b", "b", "a", "a"],
    }
)
print(df)
# --8<-- [end:groupbyroll]

# --8<-- [start:groupbydyn2]
out = df.groupby_dynamic(
    "time",
    every="1h",
    closed="both",
    by="groups",
    include_boundaries=True,
).agg(
    [
        pl.count(),
    ]
)
print(out)
# --8<-- [end:groupbydyn2]

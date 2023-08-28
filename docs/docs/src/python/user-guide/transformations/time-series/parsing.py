# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
df = pl.read_csv("docs/src/data/appleStock.csv", try_parse_dates=True)
print(df)
# --8<-- [end:df]


# --8<-- [start:cast]
df = pl.read_csv("docs/src/data/appleStock.csv", try_parse_dates=False)

df = df.with_columns(pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d"))
print(df)
# --8<-- [end:cast]


# --8<-- [start:df3]
df_with_year = df.with_columns(pl.col("Date").dt.year().alias("year"))
print(df_with_year)
# --8<-- [end:df3]

# --8<-- [start:extract]
df_with_year = df.with_columns(pl.col("Date").dt.year().alias("year"))
print(df_with_year)
# --8<-- [end:extract]

# --8<-- [start:mixed]
data = [
    "2021-03-27T00:00:00+0100",
    "2021-03-28T00:00:00+0100",
    "2021-03-29T00:00:00+0200",
    "2021-03-30T00:00:00+0200",
]
mixed_parsed = (
    pl.Series(data)
    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", utc=True)
    .dt.convert_time_zone("Europe/Brussels")
)
print(mixed_parsed)
# --8<-- [end:mixed]

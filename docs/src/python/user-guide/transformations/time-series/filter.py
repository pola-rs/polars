# --8<-- [start:df]
import polars as pl
from datetime import datetime

df = pl.read_csv("docs/data/apple_stock.csv", try_parse_dates=True)
print(df)
# --8<-- [end:df]

# --8<-- [start:filter]
filtered_df = df.filter(
    pl.col("Date") == datetime(1995, 10, 16),
)
print(filtered_df)
# --8<-- [end:filter]

# --8<-- [start:range]
filtered_range_df = df.filter(
    pl.col("Date").is_between(datetime(1995, 7, 1), datetime(1995, 11, 1)),
)
print(filtered_range_df)
# --8<-- [end:range]

# --8<-- [start:negative]
ts = pl.Series(["-1300-05-23", "-1400-03-02"]).str.to_date()

negative_dates_df = pl.DataFrame({"ts": ts, "values": [3, 4]})

negative_dates_filtered_df = negative_dates_df.filter(pl.col("ts").dt.year() < -1300)
print(negative_dates_filtered_df)
# --8<-- [end:negative]

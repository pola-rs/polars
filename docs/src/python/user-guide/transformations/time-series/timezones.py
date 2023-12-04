# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:example]
ts = ["2021-03-27 03:00", "2021-03-28 03:00"]
tz = ["Africa/Kigali", "America/New_York"]
tz_naive = pl.Series("tz_naive", ts).str.to_datetime()
time_zones = pl.Series("timezone", tz)
tz_aware = tz_naive.dt.replace_time_zone("UTC").rename("tz_aware")
time_zones_df = pl.DataFrame([tz_naive, tz_aware, time_zones])
print(time_zones_df)
# --8<-- [end:example]

# --8<-- [start:example2]
time_zones_operations = time_zones_df.select(
    [
        pl.col("tz_aware")
        .dt.replace_time_zone("Europe/Brussels")
        .alias("replace time zone"),
        pl.col("tz_aware")
        .dt.convert_time_zone("Asia/Kathmandu")
        .alias("convert time zone"),
        pl.col("tz_aware").dt.replace_time_zone(None).alias("unset time zone"),
    ]
)
print(time_zones_operations)
# --8<-- [end:example2]


# --8<-- [start:example3]
local_time_zones_operations = time_zones_df.select(
    [
        pl.col("tz_aware", "timezone"),
        pl.col("tz_aware").dt.to_local_datetime(pl.col("timezone")).alias("local_dt"),
    ]
).with_columns(
    pl.col("local_dt")
    .dt.from_local_datetime(pl.col("timezone"), "UTC")
    .alias("tz_aware_again")
)
print(local_time_zones_operations)
# --8<-- [end:example3]

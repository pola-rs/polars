# --8<-- [start:dataframe]
import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "integer": [1, 2, 3],
        "date": [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
        ],
        "float": [4.0, 5.0, 6.0],
    }
)

print(df)
# --8<-- [end:dataframe]

# --8<-- [start:csv]
df.write_csv("output.csv")
df_csv = pl.read_csv("output.csv")
print(df_csv)
# --8<-- [end:csv]

# --8<-- [start:csv2]
df_csv = pl.read_csv("output.csv", try_parse_dates=True)
print(df_csv)
# --8<-- [end:csv2]

# --8<-- [start:json]
df.write_json("output.json")
df_json = pl.read_json("output.json")
print(df_json)
# --8<-- [end:json]

# --8<-- [start:parquet]
df.write_parquet("output.parquet")
df_parquet = pl.read_parquet("output.parquet")
print(df_parquet)
# --8<-- [end:parquet]

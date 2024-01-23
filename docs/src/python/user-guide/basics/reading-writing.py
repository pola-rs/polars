# --8<-- [start:dataframe]
import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "integer": [1, 2, 3],
        "date": [
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            datetime(2025, 1, 3),
        ],
        "float": [4.0, 5.0, 6.0],
        "string": ["a", "b", "c"],
    }
)

print(df)
# --8<-- [end:dataframe]

# --8<-- [start:csv]
df.write_csv("docs/data/output.csv")
df_csv = pl.read_csv("docs/data/output.csv")
print(df_csv)
# --8<-- [end:csv]

# --8<-- [start:csv2]
df_csv = pl.read_csv("docs/data/output.csv", try_parse_dates=True)
print(df_csv)
# --8<-- [end:csv2]

# --8<-- [start:json]
df.write_json("docs/data/output.json")
df_json = pl.read_json("docs/data/output.json")
print(df_json)
# --8<-- [end:json]

# --8<-- [start:parquet]
df.write_parquet("docs/data/output.parquet")
df_parquet = pl.read_parquet("docs/data/output.parquet")
print(df_parquet)
# --8<-- [end:parquet]

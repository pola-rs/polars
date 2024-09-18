# --8<-- [start:setup]
import polars as pl
from datetime import datetime

df = pl.DataFrame(
    {
        "date": [
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            datetime(2025, 1, 3),
        ],
        "float": [4.0, 5.0, 6.0],
    }
)

print(df)
# --8<-- [end:setup]

# --8<-- [start:csv]
df.write_csv("docs/assets/data/output.csv")
df_csv = pl.read_csv("docs/assets/data/output.csv")
print(df_csv)
# --8<-- [end:csv]

# --8<-- [start:setup]
from datetime import datetime

import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
df = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="30m",
            eager=True,
        ),
        "groups": ["a", "a", "a", "b", "b", "a", "a"],
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)
print(df)
# --8<-- [end:df]

# --8<-- [start:upsample]
out1 = df.upsample(time_column="time", every="15m").fill_null(strategy="forward")
print(out1)
# --8<-- [end:upsample]

# --8<-- [start:upsample2]
out2 = (
    df.upsample(time_column="time", every="15m")
    .interpolate()
    .fill_null(strategy="forward")
)
print(out2)
# --8<-- [end:upsample2]

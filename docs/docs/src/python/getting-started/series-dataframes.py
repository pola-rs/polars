# --8<-- [start:series]
import polars as pl

s = pl.Series("a", [1, 2, 3, 4, 5])
print(s)
# --8<-- [end:series]

# --8<-- [start:minmax]
s = pl.Series("a", [1, 2, 3, 4, 5])
print(s.min())
print(s.max())
# --8<-- [end:minmax]

# --8<-- [start:string]
s = pl.Series("a", ["polar", "bear", "arctic", "polar fox", "polar bear"])
s2 = s.str.replace("polar", "pola")
print(s2)
# --8<-- [end:string]

# --8<-- [start:dt]
from datetime import datetime

start = datetime(2001, 1, 1)
stop = datetime(2001, 1, 9)
s = pl.date_range(start, stop, interval="2d", eager=True)
s.dt.day()
print(s)
# --8<-- [end:dt]

# --8<-- [start:dataframe]
from datetime import datetime

df = pl.DataFrame(
    {
        "integer": [1, 2, 3, 4, 5],
        "date": [
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 3),
            datetime(2022, 1, 4),
            datetime(2022, 1, 5),
        ],
        "float": [4.0, 5.0, 6.0, 7.0, 8.0],
    }
)

print(df)
# --8<-- [end:dataframe]

# --8<-- [start:head]
print(df.head(3))
# --8<-- [end:head]

# --8<-- [start:tail]
print(df.tail(3))
# --8<-- [end:tail]

# --8<-- [start:sample]
print(df.sample(2))
# --8<-- [end:sample]

# --8<-- [start:describe]
print(df.describe())
# --8<-- [end:describe]

# --8<-- [start:series]
import polars as pl

s = pl.Series("ints", [1, 2, 3, 4, 5])
print(s)
# --8<-- [end:series]

# --8<-- [start:series-dtype]
s1 = pl.Series("ints", [1, 2, 3, 4, 5])
s2 = pl.Series("uints", [1, 2, 3, 4, 5], dtype=pl.UInt64)
print(s1.dtype, s2.dtype)
# --8<-- [end:series-dtype]

# --8<-- [start:df]
from datetime import date

df = pl.DataFrame(
    {
        "name": ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate": [
            date(1997, 1, 10),
            date(1985, 2, 15),
            date(1983, 3, 22),
            date(1981, 4, 30),
        ],
        "weight": [57.9, 72.5, 53.6, 83.1],  # (kg)
        "height": [1.56, 1.77, 1.65, 1.75],  # (m)
    }
)

print(df)
# --8<-- [end:df]

# --8<-- [start:schema]
print(df.schema)
# --8<-- [end:schema]

# --8<-- [start:head]
print(df.head(3))
# --8<-- [end:head]

# --8<-- [start:glimpse]
print(df.glimpse(return_as_string=True))
# --8<-- [end:glimpse]

# --8<-- [start:tail]
print(df.tail(3))
# --8<-- [end:tail]

# --8<-- [start:sample]
import random

random.seed(42)  # For reproducibility.

print(df.sample(2))
# --8<-- [end:sample]

# --8<-- [start:describe]
print(df.describe())
# --8<-- [end:describe]

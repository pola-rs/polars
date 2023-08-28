# --8<-- [start:setup]

import polars as pl

# --8<-- [end:setup]

# --8<-- [start:dfnum]
df = pl.DataFrame(
    {
        "integers": [1, 2, 3, 4, 5],
        "big_integers": [1, 10000002, 3, 10000004, 10000005],
        "floats": [4.0, 5.0, 6.0, 7.0, 8.0],
        "floats_with_decimal": [4.532, 5.5, 6.5, 7.5, 8.5],
    }
)

print(df)
# --8<-- [end:dfnum]

# --8<-- [start:castnum]
out = df.select(
    pl.col("integers").cast(pl.Float32).alias("integers_as_floats"),
    pl.col("floats").cast(pl.Int32).alias("floats_as_integers"),
    pl.col("floats_with_decimal")
    .cast(pl.Int32)
    .alias("floats_with_decimal_as_integers"),
)
print(out)
# --8<-- [end:castnum]


# --8<-- [start:downcast]
out = df.select(
    pl.col("integers").cast(pl.Int16).alias("integers_smallfootprint"),
    pl.col("floats").cast(pl.Float32).alias("floats_smallfootprint"),
)
print(out)
# --8<-- [end:downcast]

# --8<-- [start:overflow]
try:
    out = df.select(pl.col("big_integers").cast(pl.Int8))
    print(out)
except Exception as e:
    print(e)
# --8<-- [end:overflow]

# --8<-- [start:overflow2]
out = df.select(pl.col("big_integers").cast(pl.Int8, strict=False))
print(out)
# --8<-- [end:overflow2]


# --8<-- [start:strings]
df = pl.DataFrame(
    {
        "integers": [1, 2, 3, 4, 5],
        "float": [4.0, 5.03, 6.0, 7.0, 8.0],
        "floats_as_string": ["4.0", "5.0", "6.0", "7.0", "8.0"],
    }
)

out = df.select(
    pl.col("integers").cast(pl.Utf8),
    pl.col("float").cast(pl.Utf8),
    pl.col("floats_as_string").cast(pl.Float64),
)
print(out)
# --8<-- [end:strings]


# --8<-- [start:strings2]
df = pl.DataFrame({"strings_not_float": ["4.0", "not_a_number", "6.0", "7.0", "8.0"]})
try:
    out = df.select(pl.col("strings_not_float").cast(pl.Float64))
    print(out)
except Exception as e:
    print(e)
# --8<-- [end:strings2]

# --8<-- [start:bool]
df = pl.DataFrame(
    {
        "integers": [-1, 0, 2, 3, 4],
        "floats": [0.0, 1.0, 2.0, 3.0, 4.0],
        "bools": [True, False, True, False, True],
    }
)

out = df.select(pl.col("integers").cast(pl.Boolean), pl.col("floats").cast(pl.Boolean))
print(out)
# --8<-- [end:bool]

# --8<-- [start:dates]
from datetime import date, datetime

df = pl.DataFrame(
    {
        "date": pl.date_range(date(2022, 1, 1), date(2022, 1, 5), eager=True),
        "datetime": pl.date_range(
            datetime(2022, 1, 1), datetime(2022, 1, 5), eager=True
        ),
    }
)

out = df.select(pl.col("date").cast(pl.Int64), pl.col("datetime").cast(pl.Int64))
print(out)
# --8<-- [end:dates]

# --8<-- [start:dates2]
df = pl.DataFrame(
    {
        "date": pl.date_range(date(2022, 1, 1), date(2022, 1, 5), eager=True),
        "string": [
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
        ],
    }
)

out = df.select(
    pl.col("date").dt.strftime("%Y-%m-%d"),
    pl.col("string").str.strptime(pl.Datetime, "%Y-%m-%d"),
)
print(out)
# --8<-- [end:dates2]

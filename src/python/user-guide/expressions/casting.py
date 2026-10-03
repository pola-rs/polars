# --8<-- [start:dfnum]
import polars as pl

df = pl.DataFrame(
    {
        "integers": [1, 2, 3],
        "big_integers": [10000002, 2, 30000003],
        "floats": [4.0, 5.8, -6.3],
    }
)

print(df)
# --8<-- [end:dfnum]

# --8<-- [start:castnum]
result = df.select(
    pl.col("integers").cast(pl.Float32).alias("integers_as_floats"),
    pl.col("floats").cast(pl.Int32).alias("floats_as_integers"),
)
print(result)
# --8<-- [end:castnum]


# --8<-- [start:downcast]
print(f"Before downcasting: {df.estimated_size()} bytes")
result = df.with_columns(
    pl.col("integers").cast(pl.Int16),
    pl.col("floats").cast(pl.Float32),
)
print(f"After downcasting: {result.estimated_size()} bytes")
# --8<-- [end:downcast]

# --8<-- [start:overflow]
from polars.exceptions import InvalidOperationError

try:
    result = df.select(pl.col("big_integers").cast(pl.Int8))
    print(result)
except InvalidOperationError as err:
    print(err)
# --8<-- [end:overflow]

# --8<-- [start:overflow2]
result = df.select(pl.col("big_integers").cast(pl.Int8, strict=False))
print(result)
# --8<-- [end:overflow2]


# --8<-- [start:strings]
df = pl.DataFrame(
    {
        "integers_as_strings": ["1", "2", "3"],
        "floats_as_strings": ["4.0", "5.8", "-6.3"],
        "floats": [4.0, 5.8, -6.3],
    }
)

result = df.select(
    pl.col("integers_as_strings").cast(pl.Int32),
    pl.col("floats_as_strings").cast(pl.Float64),
    pl.col("floats").cast(pl.String),
)
print(result)
# --8<-- [end:strings]


# --8<-- [start:strings2]
df = pl.DataFrame(
    {
        "floats": ["4.0", "5.8", "- 6 . 3"],
    }
)
try:
    result = df.select(pl.col("floats").cast(pl.Float64))
except InvalidOperationError as err:
    print(err)
# --8<-- [end:strings2]

# --8<-- [start:bool]
df = pl.DataFrame(
    {
        "integers": [-1, 0, 2, 3, 4],
        "floats": [0.0, 1.0, 2.0, 3.0, 4.0],
        "bools": [True, False, True, False, True],
    }
)

result = df.select(
    pl.col("integers").cast(pl.Boolean),
    pl.col("floats").cast(pl.Boolean),
    pl.col("bools").cast(pl.Int8),
)
print(result)
# --8<-- [end:bool]

# --8<-- [start:dates]
from datetime import date, datetime, time

df = pl.DataFrame(
    {
        "date": [
            date(1970, 1, 1),  # epoch
            date(1970, 1, 10),  # 9 days later
        ],
        "datetime": [
            datetime(1970, 1, 1, 0, 0, 0),  # epoch
            datetime(1970, 1, 1, 0, 1, 0),  # 1 minute later
        ],
        "time": [
            time(0, 0, 0),  # reference time
            time(0, 0, 1),  # 1 second later
        ],
    }
)

result = df.select(
    pl.col("date").cast(pl.Int64).alias("days_since_epoch"),
    pl.col("datetime").cast(pl.Int64).alias("us_since_epoch"),
    pl.col("time").cast(pl.Int64).alias("ns_since_midnight"),
)
print(result)
# --8<-- [end:dates]

# --8<-- [start:dates2]
df = pl.DataFrame(
    {
        "date": [date(2022, 1, 1), date(2022, 1, 2)],
        "string": ["2022-01-01", "2022-01-02"],
    }
)

result = df.select(
    pl.col("date").dt.to_string("%Y-%m-%d"),
    pl.col("string").str.to_datetime("%Y-%m-%d"),
)
print(result)
# --8<-- [end:dates2]

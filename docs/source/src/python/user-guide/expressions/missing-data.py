# --8<-- [start:dataframe]
import polars as pl

df = pl.DataFrame(
    {
        "value": [1, None],
    },
)
print(df)
# --8<-- [end:dataframe]


# --8<-- [start:count]
null_count_df = df.null_count()
print(null_count_df)
# --8<-- [end:count]


# --8<-- [start:isnull]
is_null_series = df.select(
    pl.col("value").is_null(),
)
print(is_null_series)
# --8<-- [end:isnull]


# --8<-- [start:dataframe2]
df = pl.DataFrame(
    {
        "col1": [0.5, 1, 1.5, 2, 2.5],
        "col2": [1, None, 3, None, 5],
    },
)
print(df)
# --8<-- [end:dataframe2]


# --8<-- [start:fill]
fill_literal_df = df.with_columns(
    pl.col("col2").fill_null(3),
)
print(fill_literal_df)
# --8<-- [end:fill]

# --8<-- [start:fillexpr]
fill_median_df = df.with_columns(
    pl.col("col2").fill_null((2 * pl.col("col1")).cast(pl.Int64)),
)
print(fill_median_df)
# --8<-- [end:fillexpr]

# --8<-- [start:fillstrategy]
fill_forward_df = df.with_columns(
    pl.col("col2").fill_null(strategy="forward").alias("forward"),
    pl.col("col2").fill_null(strategy="backward").alias("backward"),
)
print(fill_forward_df)
# --8<-- [end:fillstrategy]

# --8<-- [start:fillinterpolate]
fill_interpolation_df = df.with_columns(
    pl.col("col2").interpolate(),
)
print(fill_interpolation_df)
# --8<-- [end:fillinterpolate]

# --8<-- [start:nan]
import numpy as np

nan_df = pl.DataFrame(
    {
        "value": [1.0, np.nan, float("nan"), 3.0],
    },
)
print(nan_df)
# --8<-- [end:nan]

# --8<-- [start:nan-computed]
df = pl.DataFrame(
    {
        "dividend": [1, 0, -1],
        "divisor": [1, 0, -1],
    }
)
result = df.select(pl.col("dividend") / pl.col("divisor"))
print(result)
# --8<-- [end:nan-computed]

# --8<-- [start:nanfill]
mean_nan_df = nan_df.with_columns(
    pl.col("value").fill_nan(None).alias("replaced"),
).select(
    pl.all().mean().name.suffix("_mean"),
    pl.all().sum().name.suffix("_sum"),
)
print(mean_nan_df)
# --8<-- [end:nanfill]

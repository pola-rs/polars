# --8<-- [start:setup]
import polars as pl
import numpy as np

# --8<-- [end:setup]

# --8<-- [start:dataframe]
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
        "col1": [1, 2, 3],
        "col2": [1, None, 3],
    },
)
print(df)
# --8<-- [end:dataframe2]


# --8<-- [start:fill]
fill_literal_df = (
    df.with_columns(
        pl.col("col2").fill_null(
            pl.lit(2),
        ),
    ),
)
print(fill_literal_df)
# --8<-- [end:fill]

# --8<-- [start:fillstrategy]
fill_forward_df = df.with_columns(
    pl.col("col2").fill_null(strategy="forward"),
)
print(fill_forward_df)
# --8<-- [end:fillstrategy]

# --8<-- [start:fillexpr]
fill_median_df = df.with_columns(
    pl.col("col2").fill_null(pl.median("col2")),
)
print(fill_median_df)
# --8<-- [end:fillexpr]

# --8<-- [start:fillinterpolate]
fill_interpolation_df = df.with_columns(
    pl.col("col2").interpolate(),
)
print(fill_interpolation_df)
# --8<-- [end:fillinterpolate]

# --8<-- [start:nan]
nan_df = pl.DataFrame(
    {
        "value": [1.0, np.NaN, float("nan"), 3.0],
    },
)
print(nan_df)
# --8<-- [end:nan]

# --8<-- [start:nanfill]
mean_nan_df = nan_df.with_columns(
    pl.col("value").fill_nan(None).alias("value"),
).mean()
print(mean_nan_df)
# --8<-- [end:nanfill]

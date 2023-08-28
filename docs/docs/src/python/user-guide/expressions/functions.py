# --8<-- [start:setup]

import polars as pl
import numpy as np

np.random.seed(12)
# --8<-- [end:setup]

# --8<-- [start:dataframe]
df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", "spam"],
        "random": np.random.rand(5),
        "groups": ["A", "A", "B", "C", "B"],
    }
)
print(df)
# --8<-- [end:dataframe]

# --8<-- [start:samename]
df_samename = df.select(pl.col("nrs") + 5)
print(df_samename)
# --8<-- [end:samename]


# --8<-- [start:samenametwice]
try:
    df_samename2 = df.select(pl.col("nrs") + 5, pl.col("nrs") - 5)
    print(df_samename2)
except Exception as e:
    print(e)
# --8<-- [end:samenametwice]

# --8<-- [start:samenamealias]
df_alias = df.select(
    (pl.col("nrs") + 5).alias("nrs + 5"),
    (pl.col("nrs") - 5).alias("nrs - 5"),
)
print(df_alias)
# --8<-- [end:samenamealias]

# --8<-- [start:countunique]
df_alias = df.select(
    pl.col("names").n_unique().alias("unique"),
    pl.approx_unique("names").alias("unique_approx"),
)
print(df_alias)
# --8<-- [end:countunique]

# --8<-- [start:conditional]
df_conditional = df.select(
    pl.col("nrs"),
    pl.when(pl.col("nrs") > 2)
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("conditional"),
)
print(df_conditional)
# --8<-- [end:conditional]

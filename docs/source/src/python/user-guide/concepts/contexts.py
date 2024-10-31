# --8<-- [start:setup]
import polars as pl
import numpy as np

np.random.seed(12)
# --8<-- [end:setup]

# --8<-- [start:dataframe]
df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", None],
        "random": np.random.rand(5),
        "groups": ["A", "A", "B", "C", "B"],
    }
)
print(df)
# --8<-- [end:dataframe]

# --8<-- [start:select]

out = df.select(
    pl.sum("nrs"),
    pl.col("names").sort(),
    pl.col("names").first().alias("first name"),
    (pl.mean("nrs") * 10).alias("10xnrs"),
)
print(out)
# --8<-- [end:select]

# --8<-- [start:filter]
out = df.filter(pl.col("nrs") > 2)
print(out)
# --8<-- [end:filter]

# --8<-- [start:with_columns]

df = df.with_columns(
    pl.sum("nrs").alias("nrs_sum"),
    pl.col("random").count().alias("count"),
)
print(df)
# --8<-- [end:with_columns]


# --8<-- [start:group_by]
out = df.group_by("groups").agg(
    pl.sum("nrs"),  # sum nrs by groups
    pl.col("random").count().alias("count"),  # count group members
    # sum random where name != null
    pl.col("random").filter(pl.col("names").is_not_null()).sum().name.suffix("_sum"),
    pl.col("names").reverse().alias("reversed names"),
)
print(out)
# --8<-- [end:group_by]

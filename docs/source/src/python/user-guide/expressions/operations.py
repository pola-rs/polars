# --8<-- [start:dataframe]
import polars as pl
import numpy as np

np.random.seed(42)  # For reproducibility.

df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", "spam"],
        "random": np.random.rand(5),
        "groups": ["A", "A", "B", "A", "B"],
    }
)
print(df)
# --8<-- [end:dataframe]

# --8<-- [start:arithmetic]
result = df.select(
    (pl.col("nrs") + 5).alias("nrs + 5"),
    (pl.col("nrs") - 5).alias("nrs - 5"),
    (pl.col("nrs") * pl.col("random")).alias("nrs * random"),
    (pl.col("nrs") / pl.col("random")).alias("nrs / random"),
    (pl.col("nrs") ** 2).alias("nrs ** 2"),
    (pl.col("nrs") % 3).alias("nrs % 3"),
)

print(result)
# --8<-- [end:arithmetic]

# --8<-- [start:operator-overloading]
# Python only:
result_named_operators = df.select(
    (pl.col("nrs").add(5)).alias("nrs + 5"),
    (pl.col("nrs").sub(5)).alias("nrs - 5"),
    (pl.col("nrs").mul(pl.col("random"))).alias("nrs * random"),
    (pl.col("nrs").truediv(pl.col("random"))).alias("nrs / random"),
    (pl.col("nrs").pow(2)).alias("nrs ** 2"),
    (pl.col("nrs").mod(3)).alias("nrs % 3"),
)

print(result.equals(result_named_operators))
# --8<-- [end:operator-overloading]

# --8<-- [start:comparison]
result = df.select(
    (pl.col("nrs") > 1).alias("nrs > 1"),  # .gt
    (pl.col("nrs") >= 3).alias("nrs >= 3"),  # ge
    (pl.col("random") < 0.2).alias("random < .2"),  # .lt
    (pl.col("random") <= 0.5).alias("random <= .5"),  # .le
    (pl.col("nrs") != 1).alias("nrs != 1"),  # .ne
    (pl.col("nrs") == 1).alias("nrs == 1"),  # .eq
)
print(result)
# --8<-- [end:comparison]

# --8<-- [start:boolean]
# Boolean operators & | ~
result = df.select(
    ((~pl.col("nrs").is_null()) & (pl.col("groups") == "A")).alias(
        "number not null and group A"
    ),
    ((pl.col("random") < 0.5) | (pl.col("groups") == "B")).alias(
        "random < 0.5 or group B"
    ),
)

print(result)

# Corresponding named functions `and_`, `or_`, and `not_`.
result2 = df.select(
    (pl.col("nrs").is_null().not_().and_(pl.col("groups") == "A")).alias(
        "number not null and group A"
    ),
    ((pl.col("random") < 0.5).or_(pl.col("groups") == "B")).alias(
        "random < 0.5 or group B"
    ),
)
print(result.equals(result2))
# --8<-- [end:boolean]

# --8<-- [start:bitwise]
result = df.select(
    pl.col("nrs"),
    (pl.col("nrs") & 6).alias("nrs & 6"),
    (pl.col("nrs") | 6).alias("nrs | 6"),
    (~pl.col("nrs")).alias("not nrs"),
    (pl.col("nrs") ^ 6).alias("nrs ^ 6"),
)

print(result)
# --8<-- [end:bitwise]

# --8<-- [start:count]
long_df = pl.DataFrame({"numbers": np.random.randint(0, 100_000, 100_000)})

result = long_df.select(
    pl.col("numbers").n_unique().alias("n_unique"),
    pl.col("numbers").approx_n_unique().alias("approx_n_unique"),
)

print(result)
# --8<-- [end:count]

# --8<-- [start:value_counts]
result = df.select(
    pl.col("names").value_counts().alias("value_counts"),
)

print(result)
# --8<-- [end:value_counts]

# --8<-- [start:unique_counts]
result = df.select(
    pl.col("names").unique(maintain_order=True).alias("unique"),
    pl.col("names").unique_counts().alias("unique_counts"),
)

print(result)
# --8<-- [end:unique_counts]

# --8<-- [start:collatz]
result = df.select(
    pl.col("nrs"),
    pl.when(pl.col("nrs") % 2 == 1)  # Is the number odd?
    .then(3 * pl.col("nrs") + 1)  # If so, multiply by 3 and add 1.
    .otherwise(pl.col("nrs") // 2)  # If not, divide by 2.
    .alias("Collatz"),
)

print(result)
# --8<-- [end:collatz]

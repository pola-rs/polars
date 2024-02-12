# --8<-- [start:setup]
from datetime import datetime

import numpy as np
import polars as pl

df = pl.DataFrame(
    {
        "a": range(5),
        "b": np.random.rand(5),
        "c": [
            datetime(2025, 12, 1),
            datetime(2025, 12, 2),
            datetime(2025, 12, 3),
            datetime(2025, 12, 4),
            datetime(2025, 12, 5),
        ],
        "d": [1, 2.0, float("nan"), -42, None],
    }
)
# --8<-- [end:setup]

# --8<-- [start:select]
df.select(pl.col("*"))
# --8<-- [end:select]

# --8<-- [start:select2]
df.select(pl.col("a", "b"))
# --8<-- [end:select2]

# --8<-- [start:select3]
df.select(pl.col("a"), pl.col("b")).limit(3)
# --8<-- [end:select3]

# --8<-- [start:exclude]
df.select(pl.exclude(["a", "c"]))
# --8<-- [end:exclude]

# --8<-- [start:filter]
df.filter(
    pl.col("c").is_between(datetime(2025, 12, 2), datetime(2025, 12, 3)),
)
# --8<-- [end:filter]

# --8<-- [start:filter2]
df.filter((pl.col("a") <= 3) & (pl.col("d").is_not_nan()))
# --8<-- [end:filter2]

# --8<-- [start:with_columns]
df.with_columns(pl.col("b").sum().alias("e"), (pl.col("b") + 42).alias("b+42"))
# --8<-- [end:with_columns]

# --8<-- [start:dataframe2]
df2 = pl.DataFrame(
    {
        "x": range(8),
        "y": ["A", "A", "A", "B", "B", "C", "X", "X"],
    }
)
# --8<-- [end:dataframe2]

# --8<-- [start:group_by]
df2.group_by("y", maintain_order=True).len()
# --8<-- [end:group_by]

# --8<-- [start:group_by2]
df2.group_by("y", maintain_order=True).agg(
    pl.col("*").count().alias("count"),
    pl.col("*").sum().alias("sum"),
)
# --8<-- [end:group_by2]

# --8<-- [start:combine]
df_x = df.with_columns((pl.col("a") * pl.col("b")).alias("a * b")).select(
    pl.all().exclude(["c", "d"])
)

print(df_x)
# --8<-- [end:combine]

# --8<-- [start:combine2]
df_y = df.with_columns((pl.col("a") * pl.col("b")).alias("a * b")).select(
    pl.all().exclude("d")
)

print(df_y)
# --8<-- [end:combine2]

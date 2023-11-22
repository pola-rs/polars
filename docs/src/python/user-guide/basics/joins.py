# --8<-- [start:setup]
import numpy as np
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:join]
df = pl.DataFrame(
    {
        "a": range(8),
        "b": np.random.rand(8),
        "d": [1, 2.0, float("nan"), float("nan"), 0, -5, -42, None],
    }
)

df2 = pl.DataFrame(
    {
        "x": range(8),
        "y": ["A", "A", "A", "B", "B", "C", "X", "X"],
    }
)
joined = df.join(df2, left_on="a", right_on="x")
print(joined)
# --8<-- [end:join]

# --8<-- [start:hstack]
stacked = df.hstack(df2)
print(stacked)
# --8<-- [end:hstack]

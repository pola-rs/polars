# --8<-- [start:setup]
import polars as pl
import numpy as np

# --8<-- [end:setup]

# --8<-- [start:join]
df = pl.DataFrame(
    {
        "a": np.arange(0, 8),
        "b": np.random.rand(8),
        "d": [1, 2.0, np.NaN, np.NaN, 0, -5, -42, None],
    }
)

df2 = pl.DataFrame(
    {
        "x": np.arange(0, 8),
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

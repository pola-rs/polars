# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

"""
# --8<-- [start:dataframe]
q1 = (
    pl.scan_csv(f"docs/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)
# --8<-- [end:dataframe]
"""

# --8<-- [start:fromdf]
q3 = pl.DataFrame({"foo": ["a", "b", "c"], "bar": [0, 1, 2]}).lazy()
# --8<-- [end:fromdf]

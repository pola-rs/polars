"""
# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
q1 = (
    pl.scan_csv("docs/assets/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)
# --8<-- [end:df]

# --8<-- [start:collect]
q4 = (
    pl.scan_csv(f"docs/assets/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect()
)
# --8<-- [end:collect]
# --8<-- [start:stream]
q5 = (
    pl.scan_csv(f"docs/assets/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect(streaming=True)
)
# --8<-- [end:stream]
# --8<-- [start:partial]
q9 = (
    pl.scan_csv(f"docs/assets/data/reddit.csv")
    .head(10)
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect()
)
# --8<-- [end:partial]
"""

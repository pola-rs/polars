# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

"""
# --8<-- [start:read]
df = pl.read_parquet("docs/data/path.parquet")
# --8<-- [end:read]
"""

# --8<-- [start:write]
df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "bak", "baz"]})
df.write_parquet("docs/data/path.parquet")
# --8<-- [end:write]

# --8<-- [start:scan]
df = pl.scan_parquet("docs/data/path.parquet")
# --8<-- [end:scan]

# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

"""
# --8<-- [start:read]
df = pl.read_excel("docs/data/path.xlsx")
# --8<-- [end:read]

# --8<-- [start:read_sheet_name]
df = pl.read_excel("docs/data/path.xlsx", sheet_name="Sales")
# --8<-- [end:read_sheet_name]
"""

# --8<-- [start:write]
df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "bak", "baz"]})
df.write_excel("docs/data/path.xlsx")
# --8<-- [end:write]

"""
# --8<-- [start:write_sheet_name]
df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "bak", "baz"]})
df.write_excel("docs/data/path.xlsx", worksheet="Sales")
# --8<-- [end:write_sheet_name]
"""

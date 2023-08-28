# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

"""
# --8<-- [start:read]
df = pl.read_json("path.json")
# --8<-- [end:read]
"""

"""
# --8<-- [start:readnd]
df = pl.read_ndjson("path.json")
# --8<-- [end:readnd]
"""

# --8<-- [start:write]
df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "bak", "baz"]})
# json
df.write_json("path.json")
# ndjson
df.write_ndjson("path.json")
# --8<-- [end:write]

# --8<-- [start:scan]
df = pl.scan_ndjson("path.json")
# --8<-- [end:scan]

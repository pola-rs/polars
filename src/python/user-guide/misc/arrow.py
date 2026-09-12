# --8<-- [start:to_arrow]
import polars as pl

df = pl.DataFrame({"foo": [1, 2, 3], "bar": ["ham", "spam", "jam"]})

arrow_table = df.to_arrow()
print(arrow_table)
# --8<-- [end:to_arrow]

# --8<-- [start:to_arrow_zero]
arrow_table_zero_copy = df.to_arrow(compat_level=pl.CompatLevel.newest())
print(arrow_table_zero_copy)
# --8<-- [end:to_arrow_zero]

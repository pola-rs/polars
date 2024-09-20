# --8<-- [start:to_arrow]
import polars as pl
import pyarrow as pa

df = pl.DataFrame({"foo": [1, 2, 3], "bar": ["ham", "spam", "jam"]})
arrow_table = pa.table(df)
print(arrow_table)
# --8<-- [end:to_arrow]

# --8<-- [start:to_polars]
polars_df = pl.DataFrame(arrow_table)
print(polars_df)
# --8<-- [end:to_polars]

# --8<-- [start:to_arrow_series]
arrow_chunked_array = pa.array(df["foo"])
print(arrow_chunked_array)
# --8<-- [end:to_arrow_series]

# --8<-- [start:to_polars_series]
polars_series = pl.Series(arrow_chunked_array)
print(polars_series)
# --8<-- [end:to_polars_series]

# --8<-- [start:to_arrow_array_rechunk]
arrow_array = pa.array(df["foo"])
print(arrow_array)
# --8<-- [end:to_arrow_array_rechunk]

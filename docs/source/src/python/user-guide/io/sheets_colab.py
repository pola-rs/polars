"""
# --8<-- [start:open]
import polars as pl
from google.colab import sheets
url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
sheet = sheets.InteractiveSheet(url=url, backend="polars", display=False)
sheet.as_df()
# --8<-- [end:open]
# --8<-- [start:create_title]
sheet = sheets.InteractiveSheet(title="Colab <3 Polars", backend="polars")
# --8<-- [end:create_title]
# --8<-- [start:create_df]
df = pl.DataFrame({"a": [1,2,3], "b": ["a", "b", "c"]})
sheet = sheets.InteractiveSheet(df=df, title="Colab <3 Polars", backend="polars")
# --8<-- [end:create_df]
# --8<-- [start:update]
sheet.update(df)
# --8<-- [end:update]
# --8<-- [start:update_loc]
sheet.update(df, clear=False)
sheet.update(df, location="D3")
sheet.update(df, location=(3, 4))
# --8<-- [end:update_loc]
# --8<-- [start:update_loop]
for i, df in dfs:
  df = pl.select(x=pl.arange(5)).with_columns(pow=pl.col("x") ** i)
  sheet.update(df, loc=(1, i * 3), clear=i == 0)
# --8<-- [end:update_loop]
"""

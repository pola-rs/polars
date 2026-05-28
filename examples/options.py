import polars as pl
df0 = pl.DataFrame({"age": [6, 4, 2], "name": ["a", "b", "c"]})
df1 = pl.DataFrame({"age": [5, 3, None], "name": ["d", "e", "f"]})
df2 = df0.merge_sorted(df1, key="age", descending=True, nulls_last=False)

print(df2)


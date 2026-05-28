import polars as pl
df0 = pl.DataFrame({"age": [1, 3, 5], "name": ["a", "b", "c"]})
df1 = pl.DataFrame({"age": [2, 4, 6], "name": ["d", "e", "f"]})
df2 = df0.merge_sorted(df1, key="age", descending=True, nulls_last=True)

print(df2)


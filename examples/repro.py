import polars as pl

s = pl.Series(["b", "a", None], dtype=pl.Categorical)

s.min()  # 'a'
print(s.fill_null(strategy="min").to_list())  # ['b', 'a', 'b']   <- filled with 'b'
print(s.fill_null(s.min()).to_list())  # ['b', 'a', 'a']   (explicit fill is correct)

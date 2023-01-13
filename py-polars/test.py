import polars as pl
a = pl.List(pl.Utf8).string_repr()
print(a)

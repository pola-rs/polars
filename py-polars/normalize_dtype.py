import polars as pl

d = {
    pl.Time: 1,
    pl.Utf8: 2,
}

result = d[pl.Time()]
print(result)

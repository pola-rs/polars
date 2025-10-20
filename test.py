
import polars as pl

print(pl.__file__)

df = pl.DataFrame({ 'a': [1,2,3], }).with_columns(
    pl.col('a').foldv(lambda s, a: s + a).alias('b')
)

print(df)

cool = 1
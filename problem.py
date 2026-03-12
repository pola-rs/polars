import polars as pl

df = pl.DataFrame({'idx': [1, 2, 3], 'a': [1, 1, 1]})

print('=== in-memory ===')
out = df.lazy().with_columns(
    sum=pl.sum('a').rolling(index_column='idx', period='1i', offset='5i')
).collect(engine='in-memory')
print(out)

print('=== streaming ===')
out = df.lazy().with_columns(
    sum=pl.sum('a').rolling(index_column='idx', period='1i', offset='5i')
).collect(engine='streaming')
print(out)

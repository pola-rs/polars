# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]


# --8<-- [start:show]
# Create some DataFrames and register them with the SQLContext
df1 = pl.LazyFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
    }
)
df2 = pl.LazyFrame(
    {
        "name": ["Ellen", "Frank", "Gina", "Henry"],
        "age": [45, 50, 55, 60],
    }
)
ctx = pl.SQLContext(mytable1=df1, mytable2=df2)

tables = ctx.execute("SHOW TABLES", eager=True)

print(tables)
# --8<-- [end:show]

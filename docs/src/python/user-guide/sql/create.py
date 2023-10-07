# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:create]
data = {"name": ["Alice", "Bob", "Charlie", "David"], "age": [25, 30, 35, 40]}
df = pl.LazyFrame(data)

ctx = pl.SQLContext(my_table=df, eager_execution=True)

result = ctx.execute(
    """
    CREATE TABLE older_people
    AS
    SELECT * FROM my_table WHERE age > 30
"""
)

print(ctx.execute("SELECT * FROM older_people"))
# --8<-- [end:create]

# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:cte]
ctx = pl.SQLContext()
df = pl.LazyFrame(
    {"name": ["Alice", "Bob", "Charlie", "David"], "age": [25, 30, 35, 40]}
)
ctx.register("my_table", df)

result = ctx.execute(
    """
    WITH older_people AS (
        SELECT * FROM my_table WHERE age > 30
    )
    SELECT * FROM older_people WHERE STARTS_WITH(name,'C')
""",
    eager=True,
)

print(result)
# --8<-- [end:cte]

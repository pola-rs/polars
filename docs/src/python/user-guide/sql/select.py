# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]


# --8<-- [start:df]
df = pl.DataFrame(
    {
        "city": [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Amsterdam",
        ],
        "country": ["USA", "USA", "USA", "USA", "USA", "Netherlands"],
        "population": [8399000, 3997000, 2705000, 2320000, 1680000, 900000],
    }
)

ctx = pl.SQLContext(population=df, eager_execution=True)

print(ctx.execute("SELECT * FROM population"))
# --8<-- [end:df]

# --8<-- [start:group_by]
result = ctx.execute(
    """
        SELECT country, AVG(population) as avg_population
        FROM population
        GROUP BY country
    """
)
print(result)
# --8<-- [end:group_by]


# --8<-- [start:orderby]
result = ctx.execute(
    """
        SELECT city, population
        FROM population
        ORDER BY population
    """
)
print(result)
# --8<-- [end:orderby]

# --8<-- [start:join]
income = pl.DataFrame(
    {
        "city": [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Amsterdam",
            "Rotterdam",
            "Utrecht",
        ],
        "country": [
            "USA",
            "USA",
            "USA",
            "USA",
            "Netherlands",
            "Netherlands",
            "Netherlands",
        ],
        "income": [55000, 62000, 48000, 52000, 42000, 38000, 41000],
    }
)
ctx.register_many(income=income)
result = ctx.execute(
    """
        SELECT country, city, income, population
        FROM population
        LEFT JOIN income on population.city = income.city
    """
)
print(result)
# --8<-- [end:join]


# --8<-- [start:functions]
result = ctx.execute(
    """
        SELECT city, population
        FROM population
        WHERE STARTS_WITH(country,'U')
    """
)
print(result)
# --8<-- [end:functions]

# --8<-- [start:tablefunctions]
result = ctx.execute(
    """
        SELECT *
        FROM read_csv('docs/data/iris.csv')
    """
)
print(result)
# --8<-- [end:tablefunctions]

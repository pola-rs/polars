# --8<-- [start:setup]
import polars as pl


def trapz(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """Integrate with the trapezoidal rule."""
    return 0.5 * ((x - x.shift()) * (y + y.shift())).sum()


# --8<-- [end:setup]

# --8<-- [start:basic]
df = pl.DataFrame(
    {
        "x": [1, 2, 3],
        "y": [5, 4, 3],
    }
)

result = df.select(trapz(pl.col("x"), pl.col("y")).alias("area"))
print(result)
# --8<-- [end:basic]

# --8<-- [start:grouped]
df = pl.DataFrame(
    {
        "group": ["a", "a", "a", "b", "b", "b"],
        "x": [1, 2, 3, 1, 2, 3],
        "y": [5, 4, 3, 0, 1, 2],
    }
)

result = df.group_by("group").agg(trapz(pl.col("x"), pl.col("y")).alias("area"))
print(result)
# --8<-- [end:grouped]

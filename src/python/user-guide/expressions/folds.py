# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:mansum]
df = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [10, 20, 30],
    }
)

out = df.select(
    pl.fold(acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=pl.all()).alias(
        "sum"
    ),
)
print(out)
# --8<-- [end:mansum]

# --8<-- [start:conditional]
df = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [0, 1, 2],
    }
)

out = df.filter(
    pl.fold(
        acc=pl.lit(True),
        function=lambda acc, x: acc & x,
        exprs=pl.col("*") > 1,
    )
)
print(out)
# --8<-- [end:conditional]

# --8<-- [start:string]
df = pl.DataFrame(
    {
        "a": ["a", "b", "c"],
        "b": [1, 2, 3],
    }
)

out = df.select(pl.concat_str(["a", "b"]))
print(out)
# --8<-- [end:string]

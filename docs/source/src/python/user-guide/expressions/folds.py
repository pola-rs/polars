# --8<-- [start:mansum]
import operator
import polars as pl

df = pl.DataFrame(
    {
        "label": ["foo", "bar", "spam"],
        "a": [1, 2, 3],
        "b": [10, 20, 30],
    }
)

result = df.select(
    pl.fold(
        acc=pl.lit(0),
        function=operator.add,
        exprs=pl.col("a", "b"),
    ).alias("sum_fold"),
    pl.sum_horizontal(pl.col("a", "b")).alias("sum_horz"),
)

print(result)
# --8<-- [end:mansum]

# --8<-- [start:mansum-explicit]
acc = pl.lit(0)
f = operator.add

result = df.select(
    f(f(acc, pl.col("a")), pl.col("b")),
    pl.fold(acc=acc, function=f, exprs=pl.col("a", "b")).alias("sum_fold"),
)

print(result)
# --8<-- [end:mansum-explicit]

# --8<-- [start:manprod]
result = df.select(
    pl.fold(
        acc=pl.lit(0),
        function=operator.mul,
        exprs=pl.col("a", "b"),
    ).alias("prod"),
)

print(result)
# --8<-- [end:manprod]

# --8<-- [start:manprod-fixed]
result = df.select(
    pl.fold(
        acc=pl.lit(1),
        function=operator.mul,
        exprs=pl.col("a", "b"),
    ).alias("prod"),
)

print(result)
# --8<-- [end:manprod-fixed]

# --8<-- [start:conditional]
df = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [0, 1, 2],
    }
)

result = df.filter(
    pl.fold(
        acc=pl.lit(True),
        function=lambda acc, x: acc & x,
        exprs=pl.all() > 1,
    )
)
print(result)
# --8<-- [end:conditional]

# --8<-- [start:string]
df = pl.DataFrame(
    {
        "a": ["a", "b", "c"],
        "b": [1, 2, 3],
    }
)

result = df.select(pl.concat_str(["a", "b"]))
print(result)
# --8<-- [end:string]

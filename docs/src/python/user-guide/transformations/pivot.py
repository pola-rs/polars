# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:df]
df = pl.DataFrame(
    {
        "foo": ["A", "A", "B", "B", "C"],
        "N": [1, 2, 2, 4, 2],
        "bar": ["k", "l", "m", "n", "o"],
    }
)
print(df)
# --8<-- [end:df]

# --8<-- [start:eager]
out = df.pivot(index="foo", columns="bar", values="N", aggregate_function="first")
print(out)
# --8<-- [end:eager]

# --8<-- [start:lazy]
q = (
    df.lazy()
    .collect()
    .pivot(index="foo", columns="bar", values="N", aggregate_function="first")
    .lazy()
)
out = q.collect()
print(out)
# --8<-- [end:lazy]

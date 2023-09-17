# --8<-- [start:df]
import polars as pl

df = pl.DataFrame(
    {
        "A": ["a", "b", "a"],
        "B": [1, 3, 5],
        "C": [10, 11, 12],
        "D": [2, 4, 6],
    }
)
print(df)
# --8<-- [end:df]

# --8<-- [start:melt]
out = df.melt(id_vars=["A", "B"], value_vars=["C", "D"])
print(out)
# --8<-- [end:melt]

import polars as pl

df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "values": [[10, 20], [30, 40, 50], [], None],
        "idx": [-1, 0, 0, 3],
    }
)

out = df.select(pl.col("values").list.remove_by_index(pl.col("idx"), null_on_oob=True))

print(out)

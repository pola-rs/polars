import polars as pl 

        # >>> import polars as pl
        # >>> df = pl.DataFrame({
        # ...     "a": [1, 8, 3],
        # ...     "b": [4, 5, 2]
        # ... })
        # >>> # keep only even values in each concatenated list
        # >>> df.with_columns(
        # ...     evens=pl.concat_list("a", "b")
        # ...             .list.filter(pl.element() % 2 == 0)
        # ... )+

df = pl.DataFrame({
    "a": [1, 8, 3],
    "b": [4, 5, 2]
})

print(df.with_columns(
    evens=pl.concat_list("a", "b")
            .list.filter(pl.element() % 2 == 0)
))

s = pl.Series("a", [[1, 4], [8, 5], [3, 2]])
# keep only even values in each list
print(s.list.filter(pl.element() % 2 == 0))

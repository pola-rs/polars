# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:schema]
lf = pl.LazyFrame({"foo": ["a", "b", "c"], "bar": [0, 1, 2]})

print(lf.collect_schema())
# --8<-- [end:schema]

# --8<-- [start:lazyround]
lf = pl.LazyFrame({"foo": ["a", "b", "c"], "bar": [0, 1, 2]}).with_columns(
    pl.col("bar").round(0)
)
# --8<-- [end:lazyround]

# --8<-- [start:typecheck]
try:
    print(lf.collect())
except Exception as e:
    print(e)
# --8<-- [end:typecheck]

# --8<-- [start:lazyeager]
lazy_eager_query = (
    pl.LazyFrame(
        {
            "id": ["a", "b", "c"],
            "month": ["jan", "feb", "mar"],
            "values": [0, 1, 2],
        }
    )
    .with_columns((2 * pl.col("values")).alias("double_values"))
    .collect()
    .pivot(index="id", on="month", values="double_values", aggregate_function="first")
    .lazy()
    .filter(pl.col("mar").is_null())
    .collect()
)
print(lazy_eager_query)
# --8<-- [end:lazyeager]

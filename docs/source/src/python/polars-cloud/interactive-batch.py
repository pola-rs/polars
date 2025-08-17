"""
# --8<-- [start:example]
import polars as pl
import polars_cloud as pc
import datetime as dt

lf = pl.LazyFrame(
    {
        "name": ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate": [
            dt.date(1997, 1, 10),
            dt.date(1985, 2, 15),
            dt.date(1983, 3, 22),
            dt.date(1981, 4, 30),
        ],
        "weight": [57.9, 72.5, 53.6, 83.1],  # (kg)
        "height": [1.56, 1.77, 1.65, 1.75],  # (m)
    }
)
# --8<-- [end:example]

# --8<-- [start:batch]
ctx = pc.ComputeContext(workspace="your-workspace", cpus=24, memory=64)

lf = lf.select(
    pl.col("name"),
    pl.col("birthdate").dt.year().alias("birth_year"),
    (pl.col("weight") / (pl.col("height") ** 2)).alias("bmi"),
).sort(by="bmi")

lf.remote(context=ctx).sink_parquet("s3://bucket/output.parquet")
# --8<-- [end:batch]

# --8<-- [start:interactive]
ctx = pc.ComputeContext(
    workspace="your-workspace", cpus=24, memory=64, interactive=True
)

lf = lf.select(
    pl.col("name"),
    pl.col("birthdate").dt.year().alias("birth_year"),
    (pl.col("weight") / (pl.col("height") ** 2)).alias("bmi"),
).sort(by="bmi")

res1 = lf.remote(context=ctx).collect()

# --8<-- [end:interactive]

# --8<-- [start:interactive-next]
res2 = (
    res1
    .filter(
        pl.col("birth_year").is_in([1983, 1985]),
    )
    .remote(context=ctx)
    .collect()
)
# --8<-- [end:interactive-next]
"""

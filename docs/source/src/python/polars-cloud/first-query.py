"""
import polars_cloud as pc

# --8<-- [start:local]
import polars as pl

query = (
    pl.scan_parquet("s3://my-bucket/data/*.parquet")
    .filter(pl.col("status") == "active")
    .group_by("category")
    .agg(pl.col("amount").sum().alias("total_amount"))
    .sort("total_amount")
)

query.collect()
# --8<-- [end:local]

# --8<-- [start:context]
import polars_cloud as pc

ctx = pc.ComputeContext(
    workspace="your-workspace",
    memory=48,
    cpus=24,
    interactive=True
)

# Your query remains the same
query.remote(context=ctx).collect()
# --8<-- [end:context]

# --8<-- [start:distributed]
ctx = pc.ComputeContext(cpus=10, memory=10, cluster_size=10)

query.remote(context=ctx).distributed().sink_parquet("s3://my-bucket/output/")
# --8<-- [end:distributed]

# --8<-- [start:sink_parquet]
query.remote(context=ctx).sink_parquet("s3://my-bucket/processed-data/")
# --8<-- [end:sink_parquet]


# --8<-- [start:interactive]
# Quick preview of results
query.remote(context=ctx).show()

# Full results for further analysis
result = query.remote(context=ctx).collect()
print(result.collect())
# --8<-- [end:interactive]

"""

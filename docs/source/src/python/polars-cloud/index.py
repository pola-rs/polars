"""
# --8<-- [start:index]
import polars as pl
import polars_cloud as pc

ctx = pc.ComputeContext(workspace="your-workspace", cpus=16, memory=64)

query = (
    pl.scan_parquet("s3://my-dataset/")
    .group_by("l_returnflag", "l_linestatus")
    .agg(
        avg_price=pl.mean("l_extendedprice"),
        avg_disc=pl.mean("l_discount"),
        count_order=pl.len(),
    )
)

query.remote(context=ctx).sink_parquet("s3://my-dst/")
# --8<-- [end:index]
"""

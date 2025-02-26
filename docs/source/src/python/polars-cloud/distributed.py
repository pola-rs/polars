import polars as pl
import polars_cloud as pc
from datetime import date

# --8<-- [start:distributed]
query = (
    pl.scan_parquet("s3://dataset/")
    .filter(pl.col("l_shipdate") <= date(1998, 9, 2))
    .group_by("l_returnflag", "l_linestatus")
    .agg(
        avg_price=pl.mean("l_extendedprice"),
        avg_disc=pl.mean("l_discount"),
        count_order=pl.len(),
    )
)

result = (
    query.remote(pc.ComputeContext(cpus=16, memory=64, cluster_size=32))
    .distributed()
    .sink_parquet("s3://output/result.parquet")
)
# --8<-- [end:distributed]

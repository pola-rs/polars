"""
# --8<-- [start:general]
import polars_cloud as pc
import polars as pl

ctx = pc.ComputeContext(memory=8, cpus=2, cluster_size=1)
lf = pl.LazyFrame(
    {
        "a": [1, 2, 3],
        "b": [4, 4, 5],
    }
).with_columns(
    pl.col("a").max().over("b").alias("c"),
)
(
    lf.remote(context=ctx)
    .sink_parquet(uri="s3://my-bucket/result.parquet")
)
# --8<-- [end:general]
"""

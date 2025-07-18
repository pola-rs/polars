"""
# --8<-- [start:general]
import polars_cloud as pc
import polars as pl

# First, we need to define the hardware the cluster will run on.
# This can be done by specifying the minimum CPU and memory or by specifying the exact instance type in AWS.
ctx = pc.ComputeContext(memory=8, cpus=2, cluster_size=1)

# Then we write a regular lazy Polars query. In this example we compute the maximum of column.
lf = pl.LazyFrame(
    {
        "a": [1, 2, 3],
        "b": [4, 4, 5],
    }
).with_columns(
    pl.col("a").max().over("b").alias("c"),
)

# At this point, the query has not been executed yet.
# We need to call `.remote()` to signal that we want to run on Polars Cloud and then `.sink_parquet()` to send
# the query and execute it.

(
    lf.remote(context=ctx)
    .sink_parquet(uri="s3://my-bucket/result.parquet")
)

# We can then wait for the result with `result = lf.await_result()`.
# This will only include a few rows of the output as the result might be very large.
# The query and compute used will also show up in the portal https://cloud.pola.rs/portal/

# --8<-- [end:general]
"""

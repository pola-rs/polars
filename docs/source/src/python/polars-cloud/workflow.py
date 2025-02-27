# --8<-- [start:dataset]
import polars as pl

lf = pl.DataFrame(
    {
        "region": [
            "Australia",
            "California",
            "Benelux",
            "Siberia",
            "Mediterranean",
            "Congo",
            "Borneo",
        ],
        "temperature": [32.1, 28.5, 30.2, 22.7, 29.3, 31.8, 33.2],
        "humidity": [40, 35, 75, 30, 45, 80, 70],
        "burn_area": [120, 85, 30, 65, 95, 25, 40],
        "vegetation_density": [0.6, 0.7, 0.9, 0.4, 0.5, 0.9, 0.8],
    }
)

# --8<-- [end:dataset]

# --8<-- [start:local-analysis]
lf.with_columns(
    [
        (
            (pl.col("temperature") / 10)
            * (1 - pl.col("humidity") / 100)
            * pl.col("vegetation_density")
        ).alias("fire_risk"),
    ]
).filter(pl.col("humidity") < 70).sort(by="fire_risk", descending=True).collect()

# --8<-- [end:local-analysis]

# --8<-- [start:cloud-query]
lf = pl.scan_parquet("s3://climate-data/global/*.parquet")

query = (
    lf.with_columns(
        [
            (
                (pl.col("temperature") / 10)
                * (1 - pl.col("humidity") / 100)
                * pl.col("vegetation_density")
            ).alias("fire_risk"),
        ]
    )
    .filter(pl.col("humidity") < 70)
    .sort(by="fire_risk", descending=True)
)

# --8<-- [end:cloud-query]

# --8<-- [start:cloud-execution-batch]
import polars_cloud as pc

ctx = pc.ComputeContext(
    workspace="environmental-analysis", memory=32, cpus=8, cluster_size=4
)

query.remote(ctx).sink_parquet("s3://bucket/result.parquet")
# --8<-- [end:cloud-execution-batch]

# --8<-- [start:cloud-execution-interactive1]
ctx = pc.ComputeContext(
    workspace="environmental-analysis",
    memory=32,
    cpus=8,
    cluster_size=4,
    interactive=True,  # set interactive to True
)

result = query.remote(ctx).collect().await_result()

print(result)

# --8<-- [end:cloud-execution-interactive1]

# --8<-- [start:cloud-execution-interactive2]
res2 = (
    result.lazy()
    .filter(pl.col("fire_risk") > 1)
    .sink_parquet("s3://bucket/output-interactive.parquet")
)

# --8<-- [end:cloud-execution-interactive2]

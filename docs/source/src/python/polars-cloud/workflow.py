"""
import polars_cloud as pc

# --8<-- [start:lazyframe]
import polars as pl

lf = pl.LazyFrame(
    {
        "region": [
            "Australia",
            "California",
            "Benelux",
            "Siberia",

        ],
        "temperature": [32.1, 28.5, 30.2, 22.7,],
        "humidity": [40, 35, 75, 65,],
        "burn_area": [120, 85, 30, 65,],
        "vegetation_density": [0.6, 0.7, 0.9, 0.4,],
    }
)
# --8<-- [end:lazyframe]

# --8<-- [start:transformation]
(
    lf.with_columns(
            (
                (pl.col("temperature") / 10)
                * (1 - pl.col("humidity") / 100)
                * pl.col("vegetation_density")
            ).alias("fire_risk"),
    ).filter(pl.col("humidity") < 60)
    .sort(by="fire_risk", descending=True)
 .collect()
)
# --8<-- [end:transformation]

# --8<-- [start:remote-query]
# Point to the large dataset in cloud storage
lf = pl.scan_parquet("s3://climate-data/global/*.parquet")

# Same transformation logic as developed locally
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
    .filter(pl.col("humidity") < 60)
    .sort(by="fire_risk", descending=True)
)
# --8<-- [end:remote-query]

# --8<-- [start:set-compute]
import polars_cloud as pc

ctx = pc.ComputeContext(
    workspace="environmental-analysis",
    memory=32,
    cpus=8,
    interactive=True  # Enable interactive mode
)
# --8<-- [end:set-compute]


# --8<-- [start:remote-collect]
result = query.remote(context=ctx).collect()
# --8<-- [end:remote-collect]


# --8<-- [start:remote-filter]
query.remote(context=ctx).filter(pl.col("region") == "Australia").show()
# --8<-- [end:remote-filter]
"""

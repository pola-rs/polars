"""
# --8<-- [start:local]
import polars as pl

customer = pl.scan_parquet("data/customer.parquet")
lineitem = pl.scan_parquet("data/lineitem.parquet")
orders = pl.scan_parquet("data/orders.parquet")


def pdsh_q3(
    customer: pl.LazyFrame, lineitem: pl.LazyFrame, orders: pl.LazyFrame
) -> pl.LazyFrame:
    return (
        customer.filter(pl.col("c_mktsegment") == "BUILDING")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .filter(pl.col("o_orderdate") < pl.date(1995, 3, 15))
        .filter(pl.col("l_shipdate") > pl.date(1995, 3, 15))
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(pl.sum("revenue"))
        .select(
            pl.col("o_orderkey").alias("l_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
    )


pdsh_q3(customer, lineitem, orders).collect()
# --8<-- [end:local]

# --8<-- [start:context]
import polars_cloud as pc

ctx = pc.ComputeContext(
    # make sure to enter your own workspace name
    workspace="your-workspace",
    memory=16,
    cpus=12,
)

# Use a larger dataset available on S3
lineitem_sf10 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/lineitem.parquet",
    storage_options={"request_payer": "true"},
)
customer_sf10 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/customer.parquet",
    storage_options={"request_payer": "true"},
)
orders_sf10 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/orders.parquet",
    storage_options={"request_payer": "true"},
)

# Your query remains the same
pdsh_q3(customer_sf10, lineitem_sf10, orders_sf10).remote(context=ctx).show()

# --8<-- [end:context]

# --8<-- [start:sink_parquet]
# Replace the S3 url with your own to run the query successfully
(
    pdsh_q3(customer_sf10, lineitem_sf10, orders_sf10)
    .remote(context=ctx)
    .sink_parquet("s3://your-bucket/processed-data/")
)
# --8<-- [end:sink_parquet]

# --8<-- [start:show]
pdsh_q3(customer_sf10, lineitem_sf10, orders_sf10).remote(context=ctx).show()
# --8<-- [end:show]

# --8<-- [start:await_scan]
result = (
    pdsh_q3(customer_sf10, lineitem_sf10, orders_sf10)
    .remote(context=ctx)
    .await_and_scan()
)

print(result.collect())
# --8<-- [end:await_scan]
"""

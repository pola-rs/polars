"""
# --8<-- [start:setup]
import polars as pl
import polars_cloud as pc

lineitem_sf100 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/lineitem/*.parquet",
    storage_options={"request_payer": "true"},
)
customer_sf100 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/customer/*.parquet",
    storage_options={"request_payer": "true"},
)
orders_sf100 = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/orders/*.parquet",
    storage_options={"request_payer": "true"},
)
# --8<-- [end:setup]


# --8<-- [start:query]
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


# --8<-- [end:query]


# --8<-- [start:context-run]
ctx = pc.ComputeContext(workspace="your-workspace", cpus=4, memory=4, cluster_size=5)

pdsh_q3(customer_sf100, lineitem_sf100, orders_sf100).remote(ctx).distributed().show()
# --8<-- [end:context-run]
"""

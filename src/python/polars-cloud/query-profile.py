"""
# --8<-- [start:single-node-query]
import polars as pl
import polars_cloud as pc
from datetime import date


pc.authenticate()
ctx = pc.ComputeContext(workspace="your-workspace", cpus=8, memory=8, cluster_size=1)

lineitem = pl.scan_parquet("s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/lineitem.parquet",
    storage_options={"request_payer": "true"}
)
var1 = date(1998, 9, 2)

(
    lineitem.filter(pl.col("l_shipdate") <= var1)
    .group_by("l_returnflag", "l_linestatus")
    .agg(
        pl.sum("l_quantity").alias("sum_qty"),
        pl.sum("l_extendedprice").alias("sum_base_price"),
        (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
        .sum()
        .alias("sum_disc_price"),
        (
            pl.col("l_extendedprice")
            * (1.0 - pl.col("l_discount"))
            * (1.0 + pl.col("l_tax"))
        )
        .sum()
        .alias("sum_charge"),
        pl.mean("l_quantity").alias("avg_qty"),
        pl.mean("l_extendedprice").alias("avg_price"),
        pl.mean("l_discount").alias("avg_disc"),
        pl.len().alias("count_order"),
    )
    .sort("l_returnflag", "l_linestatus")
).remote(ctx).single_node().execute()
# --8<-- [end:single-node-query]

# --8<-- [start:distributed-query]
import polars as pl
import polars_cloud as pc

pc.authenticate()

ctx = pc.ComputeContext(workspace="your-workspace", cpus=12, memory=12, cluster_size=4)

def pdsh_q3(customer, lineitem, orders):
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

lineitem = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/lineitem/*.parquet",
    storage_options={"request_payer": "true"},
)
customer = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/customer/*.parquet",
    storage_options={"request_payer": "true"},
)
orders = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/orders/*.parquet",
    storage_options={"request_payer": "true"},
)

pdsh_q3(customer, lineitem, orders).remote(ctx).distributed().execute()
# --8<-- [end:distributed-query]
"""

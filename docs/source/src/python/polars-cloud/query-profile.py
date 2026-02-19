"""
from typing import cast

import polars as pl
import polars_cloud as pc


def pdsh_q3(
    customer: pl.LazyFrame, lineitem: pl.LazyFrame, orders: pl.LazyFrame
) -> pl.LazyFrame:
    pass


customer = pl.LazyFrame()
lineitem = pl.LazyFrame()
orders = pl.LazyFrame()

ctx = pc.ComputeContext()

# --8<-- [start:execute]
query = pdsh_q3(customer, lineitem, orders).remote(ctx).distributed().execute()
# --8<-- [end:execute]
# --8<-- [start:query]
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

result = pdsh_q3(customer, lineitem, orders).remote(ctx).distributed().execute()
# --8<-- [end:query]

query = cast("pc.DirectQuery", query)

# --8<-- [start:await_profile]
query.await_profile().data
# --8<-- [end:await_profile]

# --8<-- [start:await_summary]
query.await_profile().summary
# --8<-- [end:await_summary]

# --8<-- [start:explain]
result.await_result().plan()
# --8<-- [end:explain]

# --8<-- [start:explain_ir]
result.await_result().plan("ir")
# --8<-- [end:explain_ir]

# --8<-- [start:graph]
result.await_result().graph()
# --8<-- [end:graph]
"""

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

# ruff: noqa
# --8<-- [start:workspace]
import polars_cloud as pc
workspace = pc.Workspace(name="my-workspace")
# --8<-- [end:workspace]

# --8<-- [start:compute]
ctx = pc.ComputeContext(instance_type = "t2.large")
# --8<-- [end:compute]

# --8<-- [start:compute2]
ctx = pc.ComputeContext(cpus = 2, memory = 8)
# --8<-- [end:compute2]

# --8<-- [start:query]
ctx = pc.ComputeContext(instance_type = "t2.large")
lf = .... # A LazyFrame
lf.remote(context=ctx).write_parquet(...)
# --8<-- [end:query]

# --8<-- [start:interactive]
lf = .... # A LazyFrame
ctx = pc.ComputeContext(instance_type = "t2.large", interactive=True)
query = lf.remote(context=ctx).write_parquet()
result = query.await_result()
lf2 = result.lazy().with_columns(...)
# --8<-- [end:interactive]

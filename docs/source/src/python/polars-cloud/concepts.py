
# --8<-- [start:workspace]
import polars_cloud as pc
workspace = pc.Workspace(name="my-workspace")
# --8<-- [end:workspace]

# --8<-- [start:compute]
ctx = pc.ComputeContext(instance_type = "t2.micro")
# --8<-- [end:compute]

# --8<-- [start:compute2]
ctx = pc.ComputeContext(cpus = 2, memory = 8)
# --8<-- [end:compute2]

# --8<-- [start:query]
ctx = pc.ComputeContext(instance_type = "t2.micro")
lf = .... # A LazyFrame
lf.remote(context=ctx).write_parquet(...)
# --8<-- [end:query]

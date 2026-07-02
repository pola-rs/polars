"""
import polars_cloud as pc

# --8<-- [start:compute]
ctx = pc.ComputeContext(
    workspace="your-workspace",
    instance_type="t2.micro",
    cluster_size=2,
    labels=["docs"],
)
# --8<-- [end:compute]

# --8<-- [start:default-compute]
ctx = pc.ComputeContext(workspace="your-workspace")
# --8<-- [end:default-compute]

# --8<-- [start:defined-compute]
ctx = pc.ComputeContext(
    workspace="your-workspace",
    memory=8,
    cpus=2,
)
# --8<-- [end:defined-compute]

# --8<-- [start:set-compute]
ctx = pc.ComputeContext(
    workspace="your-workspace",
    instance_type="t2.micro",
    cluster_size=2,
)
# --8<-- [end:set-compute]
# --8<-- [start:manifest]
ctx = pc.ComputeContext(
    workspace="your-workspace",
    instance_type="t2.micro",
    cluster_size=2,
)
ctx.register("ComputeName")
# --8<-- [end:manifest]
"""

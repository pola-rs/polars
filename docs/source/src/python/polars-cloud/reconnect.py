"""
import polars as pl
import polars_cloud as pc

# --8<-- [start:setup]
ctx = pc.ComputeContext(workspace="your-workspace", cpus=4, memory=16)

ctx.start()
# --8<-- [end:setup]

# --8<-- [start:print]
print(ctx)
# --8<-- [end:print]

# --8<-- [start:connect_id]
ctx = pc.ComputeContext.connect('0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx')

# --8<-- [end:connect_id]

# --8<-- [start:select]
# Interactive interface to select the compute cluster you want to (re)connect to
ctx = pc.ComputeContext.select()
# --8<-- [end:select]

# --8<-- [start:via_workspace]
# List all clusters in the specified workspace
pc.ComputeContext.list('your-workspace-name')
# --8<-- [end:via_workspace]
"""

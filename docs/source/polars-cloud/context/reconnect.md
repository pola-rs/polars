# Reconnect to compute cluster

Polars Cloud allows you to reconnect to active compute clusters that are running in `proxy` mode.
This lets you easily reconnect to run more queries on active clusters, without having to wait for
machines to spin up.

## Start a cluster to reconnect to

We will start a simple compute instance to demonstrate reconnection and save the cluster ID for
later use:

{{code_block('polars-cloud/reconnect','setup',['ComputeContext'])}}

You can easily find the ID of your cluster by printing the ComputeContext to your console:

{{code_block('polars-cloud/reconnect','print',[])}}

```text
ComputeContext(id=0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx, cpus=4, memory=16, instance_type=None, storage=16, big_instance_type=None, big_instance_multiplier=None, big_instance_storage=None, cluster_size=1, interactive=False, insecure=False, workspace_name='Dear Claude', labels=None, log_level=LogLevelSchema.Info
```

## Reconnect with Compute ID

If you lost connection or if you want to connect to a running cluster in your workspace, use
`.connect` on `pc.ComputeContext`. This connects directly using the `compute_id` of the running
cluster:

!!! note "Connection permissions and proxy mode requirement"

    You can only reconnect to clusters that you started yourself or clusters that were started in `proxy` mode. You cannot reconnect to a cluster in `default` mode that was started by another user in your workspace.

{{code_block('polars-cloud/reconnect','connect_id',[])}}

If you don't know your `compute_id`, use `.select()` to access an interactive interface where you
can browse available clusters:

{{code_block('polars-cloud/reconnect','select',[])}}

```text
Found 1 available clusters:
------------------------------------------------------------------------------------------
#   Workspace       Type         vCPUs    Memory     Storage    Size       Runtime
------------------------------------------------------------------------------------------
1   your-workspace  Unknown      4        16 GiB     16 GiB     1          11m
```

## Find clusters by workspace

You can find your `compute_id` by listing workspaces and then finding your cluster within a specific
workspace. First, get your workspace ID using `pc.Workspace.list()`, then list all ComputeContexts
for that workspace:

{{code_block('polars-cloud/reconnect','via_workspace',[])}}

```text
[(ComputeContext(id=0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx, cpus=4, memory=16, instance_type=None, storage=16, big_instance_type=None, big_instance_multiplier=None, big_instance_storage=None, cluster_size=1, interactive=False, insecure=False, workspace_name='your-workspace', labels=None, log_level=LogLevelSchema.Info,
  IDLE),]
```

With the cluster `id`, you can then reconnect directly to the cluster:

{{code_block('polars-cloud/reconnect','connect_id',[])}}

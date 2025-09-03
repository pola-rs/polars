# Reconnect to compute cluster

Polars Cloud allows you to reconnect to active compute clusters. This lets you reconnect to run
multiple queries in a short time span, without having to wait for machines to spin up. You can also
connect to clusters of team members, if these were started in `proxy` mode.

## Set up a cluster

We will start a simple cluster to show how you can reconnect. We will save the cluster ID so we can
connect directly to the cluster in the following examples:

{{code_block('polars-cloud/reconnect','setup',['ComputeContext'])}}

You can easily find the ID of your cluster by printing the ComputeContext to your console:

{{code_block('polars-cloud/reconnect','print',[])}}

```text
ComputeContext(id=0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx, cpus=4, memory=16, instance_type=None, storage=16, ...)
```

## Reconnect to an existing cluster

If you lose connection or want to connect to a running cluster in your workspace, use `.connect` on
`pc.ComputeContext`. This connects directly using the `compute_id` of the running cluster:

!!! note "Connection permissions and proxy mode requirement"

    You can only reconnect to clusters that you started yourself or clusters that were started in `proxy` mode. You cannot reconnect to a cluster in `direct` mode that was started by another user in your workspace.

{{code_block('polars-cloud/reconnect','connect_id',[])}}

If you don't know your `compute_id`, use `.select()` to access an interactive interface where you
can browse available clusters:

{{code_block('polars-cloud/reconnect','select',[])}}

```text
Found 1 available clusters:
-----------------------------------------------------------------------------------------------------------------------------
#   Workspace       Type         vCPUs    Memory     Storage    Size       Runtime    ID
-----------------------------------------------------------------------------------------------------------------------------
1   your-workspace     Unknown    4        16 GiB     16 GiB     1          14m        0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

## Find clusters by workspace

You can find your `compute_id` by listing workspaces and then finding your cluster within a specific
workspace. First, get your workspace ID using `pc.Workspace.list()`, then list all ComputeContexts
for that workspace:

{{code_block('polars-cloud/reconnect','via_workspace',[])}}

```text
[(ComputeContext(id=0198e107-xxxx-xxxx-xxxx-xxxxxxxxxxxx, cpus=4, memory=16, instance_type=None, storage=16, ...),]
```

With the cluster `id` from the output above, you can then establish a connection using the same
`.connect()` method shown in the previous section.

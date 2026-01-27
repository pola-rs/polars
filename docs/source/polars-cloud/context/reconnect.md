# Reconnect to compute cluster

Polars Cloud allows you to reconnect to active compute clusters. This lets you reconnect to run
multiple queries in a short time span, without having to wait for machines to spin up.

## Reconnect with a Manifest (Recommended)

The preferred approach for creating a ComputeContext is through a manifest. A manifest defines all
compute cluster properties and stores them under a unique name. This unique name lets you easily
start or reconnect to a compute cluster. Each manifest can only have one active compute cluster at a
time. Attempting to start an already active manifest will reconnect you to the existing cluster
rather than spinning up a new one. Manifests can be created either through the cloud portal's
Compute tab or by calling `.register(name="WorkspaceName")` on a `ComputeContext`.

{{code_block('polars-cloud/reconnect','manifest',['ComputeContext'])}}

### Manual

#### Starting a cluster

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

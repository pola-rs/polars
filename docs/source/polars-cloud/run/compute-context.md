# Defining a compute context

The compute context defines the hardware configuration used to execute your queries. This can be
either a single node or, for distributed execution, multiple nodes. This section explains how to set
up and manage your compute context.

{{code_block('polars-cloud/compute-context','compute',['ComputeContext'])}}

## Setting the context

You can define your compute context in three ways:

1. Use your workspace default
2. Specify CPUs and RAM requirements
3. Select a specific instance type

### Workspace default

In the Polars Cloud dashboard, you can set default requirements from your cloud service provider to
be used for all queries. You can also manually define storage and the default cluster size.

Polars Cloud will use these defaults if no other parameters are passed to the `ComputeContext`.

{{code_block('polars-cloud/compute-context','default-compute',['ComputeContext'])}}

Find out more about how to [set workspace defaults](../workspace/settings.md) in the workspace
settings section.

### Define hardware specifications

You can directly specify the `cpus` and `memory` requirements in your `ComputeContext`. When set,
Polars Cloud will select the most suitable instance type from your cloud service provider that meets
the specifications. The requirements are lower bounds, meaning the machine will have at least that
number of CPUs and memory.

{{code_block('polars-cloud/compute-context','defined-compute',['ComputeContext'])}}

### Set instance type

For more control, you can specify the exact instance type for Polars to use. This is useful when you
have specific hardware requirements in a production environment.

{{code_block('polars-cloud/compute-context','set-compute',['ComputeContext'])}}

## Applying the compute context

Once defined, you can apply your compute context to queries in three ways:

1. By directly passing the context to the remote query:

   ```python
   query.remote(context=ctx).sink_parquet(...)
   ```

2. By globally setting the compute context. This way you set it once and don't need to provide it to
   every `remote` call:

   ```python
   pc.set_compute_context(ctx)

   query.remote().sink_parquet(...)
   ```

3. When a default compute context is set via the Polars Cloud dashboard. It is no longer required to
   define a compoute context.

   ```python
   query.remote().sink_parquet(...)
   ```

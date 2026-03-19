# Compute context introduction

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

## Saving the compute context

To simplify configuration and enable cluster sharing, you can save your settings under a unique
identifier called a manifest in Polars Cloud. This eliminates the need to specify all settings in
every script and makes it easy to reconnect to or collaborate on the same compute cluster. You can
create a manifest either programmatically by calling `register` on a `ComputeContext` or through the
cloud portal interface.

{{code_block('polars-cloud/compute-context','manifest',['register', 'ComputeContext'])}}

## Applying the compute context

Once defined, you can apply your compute context to queries in three ways:

<!-- dprint-ignore -->
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
   define a compute context.

    ```python
    query.remote().sink_parquet(...)
    ```

<!-- dprint-ignore -->
!!! note "Ephemeral compute"
    Each `ComputeContext` creates a new, ephemeral compute cluster. If you re-run your script,
    a new cluster will be started instead of reusing an existing one. To reuse a running cluster,
    either [reference it by name](reconnect.md#reconnect-with-a-manifest-recommended) using a
    manifest, or [reconnect to an existing cluster](reconnect.md).

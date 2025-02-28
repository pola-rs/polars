# Setup Compute context

The compute context is the abstraction of the hardware to execute the query on. This can be either a single node or multiple nodes in case of
distributed execution. In this section we will cover how to setup your compute context. 

{{code_block('polars-cloud/compute-context','compute',['ComputeContext'])}}

## Setting the context

The compute context can be set on an individual query basis by passing it to `.remote(ctx)` or globally by calling `pc.set_compute_context`.
There are three ways to define the compute context:

1. Use your workspace default
2. Define CPUs and RAM
3. Set instance type

### Workspace default

In the Polars Cloud dashboard you can set a default requirements from your cloud service provider
to be used for all queries. Next to that you can also manually define storage and the default cluster
size to run your queries on.

Polars Cloud will use these defaults if no other parameters are passed to the `ComputeContext`. 

{{code_block('polars-cloud/compute-context','default-compute',['ComputeContext'])}}

Find out more about how to [set workspace defaults](../workspace/settings.md) in the workspace settings section.

### Define CPU and RAM

You can directly specify the `cpus` and `memory` in your ComputeContext. When set, Polars Cloud will match your requirements and pick the most suitable and
efficient instance_type from your cloud service provider. The requirements are lower bounds, the machine will have at least that number of CPUs and memory.

{{code_block('polars-cloud/compute-context','defined-compute',['ComputeContext'])}}

### Set instance type

Another option is to define the specific instance type for Polars to use. This could be helpful if
you want to use a specific instance type in your production environment.

{{code_block('polars-cloud/compute-context','set-compute',['ComputeContext'])}}

## Setting the Compute Context

Once the compute context is defined, you'll need to provide it to the query. This can be done in two ways:

1. `remote(ctx)`: By directly passing the context to the remote query.
2. `pc.set_compute_context(ctx)`: By globally setting the compute context. This way you set it once and don't need to provide it to every `remote` call.

# Define compute context

To run a query in the cloud we should call `.remote()` on our `LazyFrame`. This tells Polars to
execute the query remote, given `pc.ComputeContext`. The compute context is the abstraction of the
hardware to execute the query on. This can be either a single node or multiple nodes in case of
distributed execution.

{{code_block('polars-cloud/compute-context','compute',['ComputeContext'])}}

## Setting the context

The compute context must be set and called in `.remote(ctx)`. The context must contain at least your
Workspace name. There are three ways to define the hardware:

1. Use your workspace default
2. Define CPUs and RAM
3. Set instance type

### Workspace default

In the Polars Cloud dashboard you can set a default instance type from your cloud service provider
to be used for all queries. Next to that you can also manual define storage and the default cluster
size to run your queries on.

If no hardware is specified, Polars Cloud will use the default if set.

{{code_block('polars-cloud/compute-context','default-compute',['ComputeContext'])}}

Find out more about how to [set workspace defaults](../workspace/settings.md) in the workspace
settings section.

### Define CPU and RAM

In your compute context, you can also set the number of CPUs and RAM you need for your query
execution. When set, Polars Cloud will match your requirements and pick the most suitable and
efficient instance_type from your cloud service provider to execute your query.

{{code_block('polars-cloud/compute-context','defined-compute',['ComputeContext'])}}

### Set instance type

Another option is to define the specific instance type for Polars to use. This could be helpful if
you want to use a specific instance type in your production environment.

{{code_block('polars-cloud/compute-context','set-compute',['ComputeContext'])}}

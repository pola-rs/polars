# Concepts

This section covers the main concepts present in Polars Cloud.

## Workspaces

A workspace is a logical container in which other resources within Polars Cloud live. All resources (e.g. compute, queries, ...) are bound to a specific workspace. 

!!! Access Control
    Everyone within a workspace has the same access there is no notion of roles (e.g. admin, user) in the current version. This means users within the same workspace can view each others clusters and queries. A workspace has a single IAM role in the cloud and runs under the same permissions. A user can't send queries to a compute cluster of another user.


{{code_block('polars-cloud/concepts','workspace',['Workspace'])}}


## Compute Context

The compute context describes the underlying hardware. You can start a compute context by specifying the instance requirements in terms of cpu's and memory or by directly specifying the AWS EC2 instance type.

By instance type

{{code_block('polars-cloud/concepts','compute',['ComputeContext'])}}

By instance requirements
{{code_block('polars-cloud/concepts','compute2',['ComputeContext'])}}

When specifying with instance requirements Polars will search for the cheapest available instance type with at least the requested values. In the cloud not all options are available. The following example `pc.ComputeContext(cpus = 1, memory = 32)` there is no machine with 1 core that has 32 GB. In this case Polars Cloud will find the cheapest available machine that has at least 32 GB of RAM. 

Below are the various options which you can specify

| Parameter                        | Type |  Description                          |
|----------------------------------| ----------------| -------------------------| 
| `workspace_name`<img width=100/> | string | The name of the workspace |
| `cpus`                           | number | The minimum number of CPUs the compute cluster should have access to. |
| `memory`                         | number | The minimum amount of RAM (in GB) the compute cluster should have access to. |
| `instance_type`                  | string | The AWS instance type (e.g. `t2.micro`). This parameter can not be used together with memory or cpus |
| `storage`                        | number  | The amount of local disk space (in GB) each node in the compute cluster has access to. Defaults to `16` | 
| `cluster_size`                   | number |The number of machines to spin up in the cluster. Defaults to `1`. | 
| `interactive`                    | bool | Activate interactive mode |
| `labels`                         | List[string] | Labels fo the compute context |
| `log_level`                      | string | Override the log level of the cluster for debug purposes. One of `"info", "debug", "trace"`. | 

!!! warning "Distributed Engine"
    We are currently developing our distributed engine. This engine will run on top of the new open source streaming engine and is exclusive to Polars Cloud. It is still in an experimental phase.


### Interactive vs Batch

A compute context can either run in interactive or batch mode.

The batch mode is meant for queries that are run periodically. In this mode, clients send there queries to Polars Cloud to be queued. If ready, the compute context will poll and run each query. Metadata around the query (e.g. status, query plan, logs, metrics) are send back to Polars Cloud for reporting purposes. The actual result data is not shared for privacy reasons.

This process of queuing and polling leads to some seconds delay, although negligible when running a lot of queries interactively this can lower the developer experience. Additionally, for exploratory work it is not always necessary or valuable to save the metadata of the query. In these cases the interactive mode is better. In interactive mode you as a client directly communicate with the compute cluster. This way there is no delay and queries run immediately. An additional option is that data can be shared securely, for example to view (a part of) the result. 

### Default Context

It is recommended to explicitly specify the compute context when running a query. However to simplify manners it is possible to use a default context. Under `Settings` in your workspace you'll find the option to specify default parameters for `memory`, `cpus`, `cluster_size` etc. If you run a query without a context Polars Cloud will spin up a compute cluster with these default parameters.

## Queries

Queries represent a single `LazyFrame` being executed in Polars Cloud. This can either be in Batch or interactive mode. 

{{code_block('polars-cloud/concepts','query',[])}}

Running a query remotely is as simple as calling `remote` while passing the compute context to it. Depending on the mode of Compute this either returns a `InteractiveQuery` or a `BatchQuery`.

### Interactive

In interactive mode you can directly communicate to the compute context. The communication is securely encrypted using mTLS between your client and the compute server. Queries send to an interactive compute context return a `InteractiveQuery` which can be awaited or cancelled. Queries executed in interactive mode do not show up on the polars cloud dashboard.

{{code_block('polars-cloud/concepts','interactive',['QueryResult','InteractiveQuery'])}}

In this example we create a `LazyFrame` called `lf` and we execute it on Polars Cloud. We can continue on the result by calling `lazy()` on the result which leads to a `LazyFrame` .

!!! info "Interactive mode"
    If you want to coninue on a existing query / query result you must use `write_parquet` to S3 as an intermediate storage location. We are adding a `.execute` (or similar) to our API which allows you to skip specifying this location.


### Batch

Running a query in batch mode gives a `BatchQuery` which has the same API as its interactive counterpart. The main differences are that queries go through the control plane and metadata on the query is stored in the dashboard for reporting purposes.

{{code_block('polars-cloud/concepts','interactive',['QueryResult','BatchQuery'])}}

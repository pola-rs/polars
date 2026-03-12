# Query profiling

Monitor query execution across workers to identify bottlenecks, understand data flow, and optimize
performance.

## Types of operations in a query

To optimize a query it helps to understand where it spends its time. Each worker in a distributed
query does three things: it reads data, computes on it, and exchanges data with other workers.

**Input/Output**: Each worker reads its assigned [partitions](glossary.md#partition) from storage
and writes results to a destination. These are typically the first and last activities you see in
the profiler. I/O-heavy queries benefit from more network bandwidth, either by adding more nodes or
by choosing a higher-bandwidth instance type.

**Computation**: Workers execute the query operations (such as filters, joins, aggregations, etc.)
on their local data. CPU and memory usage are visible in the resource overview of the nodes.

**Shuffling**: Some operations, such as joins and group-bys, require all rows with a given key to be
on the same worker. To accomplish this, data is redistributed across the cluster in a
[shuffle](glossary.md#shuffle) between stages. Within a stage, the streaming engine processes
incoming shuffle data as it arrives over the network, so I/O and computation overlap. Shuffle-heavy
queries produce large volumes of inter-node traffic, visible as network bandwidth usage in the
cluster dashboard and as a high percentage of time spent shuffling in the metrics.

## Using the query profiler

The cluster dashboard and built-in query profiler are available through the Polars Cloud compute
dashboard.

The profiler shows detailed metrics, both real-time and after query completion, such as workers'
resource usage and the percentage of time spent shuffling.

![Cluster dashboard](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster-dashboard.png)

### Single Node Query

Our first example is a query that runs on a single node. If you'd like you can run this in your own
environment so you can explore the functionality yourself.

??? example "Try it: Single node query"

    Queries can be run on a single node by marking your query like so:

    ```python
    query.remote(ctx).single_node().execute()
    ```

    This will let the query run on a single worker. This simplifies query execution and you don't
    need to shuffle data between workers. Copy and paste the example below to explore the feature
    yourself. Don't forget to change the workspace name to one of your own workspaces.

    {{code_block('polars-cloud/query-profile','single-node-query',[])}}

#### Query plans

You can inspect the details of a query by going to the "Queries" tab and selecting the query you
want to inspect. You can see the timeline, which shows when the query started and ended, and how
long planning and running the query took. On top of that it consists of a single stage, because the
query runs completely on a single node.

At the bottom of the query details you can inspect the
[optimized logical plan](glossary.md#optimized-logical-plan) and the
[physical plan](glossary.md#physical-plan):

![Query details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query-details.png)

The logical plan is a graph representation that shows what your query will do, and how your query
has been optimized. Clicking nodes in the plan gives you more details about the operation that will
be performed:

<!-- dprint-ignore -->
![Logical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical-plan.png){ width="50%" style="display: block; margin: 0 auto;" }

The physical plan shows how the engine executes your query: the concrete algorithms, operator
implementations, and data flow chosen at runtime.

<!-- dprint-ignore -->
![Physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical-plan.png){ width="70%" style="display: block; margin: 0 auto;" }

While the query runs and after it has finished, there are additional metrics available, such as how
many rows and morsels flow through a node and how much time is spent in that node. In our example
you can see that the group by takes particularly long and aggregates an input of 59.1 million rows
to 4 output rows:

<!-- dprint-ignore -->
![Group By node example](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/group-by-node.png){ width="50%" style="display: block; margin: 0 auto;" }

This makes sense because this query performs a list of aggregations, as we can see in the node
details information in the logical plan:

<!-- dprint-ignore -->
![Node details example](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/node-details.png){ width="50%" style="display: block; margin: 0 auto;" }

The indication that most time is spent in the GroupBy node matches our expectations for this query.

#### Indicators

Modes in the physical plan or stages in the stage graph can show indicators to help identify
bottlenecks:

| Indicator                                                                                                                                         | Description                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![CPU time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cpu-time.png)                           | Shows which operations took the most CPU time.                                                                                                                         |
| ![I/O time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png)                            | Percentage of the stage's total I/O time spent in this node, helping identify the most I/O-heavy operations.                                                           |
| ![Memory intensive](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-memory-intensive.png) | The node is potentially memory-intensive because the operation requires keeping state (e.g. storing the intermediate groups in a `group_by`).                          |
| ![Single node](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-single-node.png)           | This stage was executed on a single node because it contains operations that require a global state (e.g. `sort`). This indicator only appears in distributed queries. |
| ![In-memory fallback](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-in-memory.png)      | This operation is currently not supported on the streaming engine and was executed on the in-memory engine.                                                            |

!!! info "I/O and CPU time don't sum to 100%"

    The I/O time and CPU time percentages shown per node do not sum to the total runtime. This is because execution is pipelined: data is processed as it arrives, so I/O (reading/writing) and CPU (computation) work happens concurrently. As a result, both indicators can be non-zero at the same time for a given node, and their combined total can exceed the total runtime.

### Distributed Query

The following section is based on a distributed query. You can follow along with this example code:

??? example "Try it: Distributed query"

    Distributed is the default execution mode in Polars Cloud. You can also set it explicitly:

    ```python
    query.remote(ctx).distributed().execute()
    ```

    For more on how distributed execution works, see [Distributed queries](distributed-engine.md).
    Copy and paste the example below to explore the feature yourself. Don't forget to change the
    workspace name to one of your own workspaces.

    {{code_block('polars-cloud/query-profile','distributed-query',[])}}

#### Stage graph

When executing distributed queries, queries are often executed in [stages](glossary.md#stage). Some
operations require [shuffles](glossary.md#shuffle) to make sure the correct
[partitions](glossary.md#partition) are available to the workers. To accomplish this, data is
shuffled between workers over the network. Each stage can be expanded to inspect the operations it
contains and understand what work is happening at each point in the pipeline.

When you execute the example query, you get the result that can be seen in the image below. In the
stage graph, one of the scan stages at the bottom stands out: its indicator shows a high percentage
of total time spent in that stage.

![Stage graph with node details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage-graph-node-details.png)

When you click on that stage (not one of the nodes in it), you open the stage details, displaying
detailed metrics. You can notice that the I/O time of this stage is roughly 55%.

![Example of heavy stage](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage-example.png)

Through the details you can open the physical plan of this stage. This will display all of the
operations in this stage, how long they took, and any indicators that might help you find
bottlenecks.

<!-- dprint-ignore -->
![Example of stage's physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage-physical-plan-example.png){ width="50%" style="display: block; margin: 0 auto;" }

One thing you should immediately notice is that the MultiScan node at the bottom takes almost 100%
of the time for I/O:

<!-- dprint-ignore -->
![I/O time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png){ style="display: block; margin: 0 auto;" }

This I/O indicator shows that I/O was active for nearly the full runtime of the stage. We can
conclude that the network I/O in this node is the bottleneck in this part of the physical plan.

In this example the data is stored in `us-east-2` while the cluster runs in `eu-west-1`. The
cross-region bandwidth causes I/O to take longer than it would if the data and cluster were in the
same region. Co-locate your cluster and data in the same region to minimize I/O latency.

## Takeaways

- The [logical plan](glossary.md#optimized-logical-plan) shows how your query has been optimized.
- The [physical plan](glossary.md#physical-plan) shows how your query is executed, and which
  operations are responsible for both CPU and I/O time spent.
- In a distributed query, the [stage graph](glossary.md#stage-graph) shows which
  [stages](glossary.md#stage) take the longest and how much data is [shuffled](glossary.md#shuffle)
  between them.
- Indicators on stages and nodes highlight potential bottlenecks: start with the slowest stage and
  drill down to individual operations.
- I/O-heavy queries benefit from more bandwidth: you can add nodes or choose a higher-bandwidth
  instance type.
- [Shuffle](glossary.md#shuffle)-heavy queries may benefit from fewer, larger nodes to reduce
  inter-node traffic.

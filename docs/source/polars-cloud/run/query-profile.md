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

![Cluster dashboard](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)

??? example "Try it: Single node query"

    Queries can be run on a single node by marking your query like so:

    ```python
    query.remote(ctx).single_node().execute()
    ```

    This will let the query run on a single worker. This simplifies query execution and you don't
    need to shuffle data between workers. Copy and paste the example below to explore the feature
    yourself. Don't forget to change the workspace name to one of your own workspaces.

    {{code_block('polars-cloud/query-profile','single-node-query',[])}}

### Query plans

You can inspect the details of a query by going to the "Queries" tab and selecting the query you
want to inspect. At the bottom of the query details you can inspect the
[optimized logical plan](glossary.md#optimized-logical-plan) and the
[physical plan](glossary.md#physical-plan):

![Query details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query-details.png)

The logical plan is a graph representation that shows what your query will do, and how your query
has been optimized. Clicking nodes in the plan gives you more details about the operation that will
be performed:

![Logical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical_plan.png)

The physical plan shows how the engine executes your query: the concrete algorithms, operator
implementations, and data flow chosen at runtime.

![Physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)

### Indicators

Each node in the physical plan can show indicators to help identify bottlenecks:

| Indicator                                                                                                                                         | Description                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| ![CPU time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cpu-time.png)                           | Shows which operations took the most CPU time.                                                                                                |
| ![I/O time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png)                            | Percentage of the stage's total I/O time spent in this node, helping identify the most I/O-heavy operations.                                  |
| ![Memory intensive](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-memory-intensive.png) | The node is potentially memory-intensive because the operation requires keeping state (e.g. storing the intermediate groups in a `group_by`). |
| ![Single node](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-single-node.png)           | This stage was executed on a single node because the operation requires global state (e.g. `sort`). Only appears in distributed queries.      |

!!! info "I/O and CPU time don't sum to 100%"

    The I/O time and CPU time percentages shown per node do not sum to the total runtime. This is because execution is pipelined: data is processed as it arrives, so I/O (reading/writing) and CPU (computation) work happens concurrently. As a result, both indicators can be non-zero at the same time for a given node, and their combined total can exceed the total runtime.

??? example "Try it: Distributed query"

    Distributed is the default execution mode in Polars Cloud. You can also set it explicitly:

    ```python
    query.remote(ctx).distributed().execute()
    ```

    For more on how distributed execution works, see [Distributed queries](distributed-engine.md).
    Copy and paste the example below to explore the feature yourself. Don't forget to change the
    workspace name to one of your own workspaces.

    {{code_block('polars-cloud/query-profile','distributed-query',[])}}

### Stage graph

When executing distributed queries, queries are often executed in [stages](glossary.md#stage). Some
operations require [shuffles](glossary.md#shuffle) to make sure the correct
[partitions](glossary.md#partition) are available to the workers. To accomplish this, data is
shuffled between workers over the network. Each stage can be expanded to inspect the operations it
contains and understand what work is happening at each point in the pipeline.

When you execute the example query, you get the result that can be seen in the image below. In the
stage graph, one of the scan stages at the bottom stands out: its indicator shows a high percentage
of total time spent in that stage.

![Stage graph with node details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)

When you click on that stage (not one of the nodes in it), you open the stage details, displaying
detailed metrics. For this query, the metrics show high shuffle bytes, indicating that this stage is
shuffle-heavy. While shuffling is expensive, it can be required.

![Example of heavy stage](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage-example.png)

Through the details you can open the physical plan of this stage, which shows at the bottom that the
scan in this stage took almost all of the time:

<!-- dprint-ignore -->
![Example of stage's physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage-physical-plan-example.png){ width="50%" style="display: block; margin: 0 auto;" }

In this example the data is stored in `us-east-2` while the cluster runs in `eu-west-1`. The
cross-region bandwidth causes I/O to take longer than it would if the data and cluster were in the
same region.

## Takeaways

- The [stage graph](glossary.md#stage-graph) shows which [stages](glossary.md#stage) take the
  longest and how much data is [shuffled](glossary.md#shuffle) between them.
- The [physical plan](glossary.md#physical-plan) shows which operations within a stage are
  responsible for the time spent.
- Indicators on stages and nodes highlight potential bottlenecks: start with the slowest stage and
  drill down to individual operations.
- I/O-heavy queries benefit from more bandwidth: add nodes or choose a higher-bandwidth instance
  type.
- [Shuffle](glossary.md#shuffle)-heavy queries require data to move between workers; co-locating
  data and cluster in the same region reduces I/O time.

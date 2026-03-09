# Query profiling

Monitor query execution across workers to identify bottlenecks, understand data flow, and optimize
performance.

## Types of operations in a query

A query spends its time in three types of operation:

**Input/Output**: Each worker reads its assigned [partitions](glossary.md#partition) from storage
and stores the results in a destination location. These are typically the first and last activities
you see in the profiler. IO-heavy queries benefit from scaling horizontally: adding more workers
reduces the per-worker download and upload volume.

**Computation**: Workers execute the query operations (filters, joins, aggregations) on their local
data. CPU and memory usage can be seen on the resource overview of the nodes. Computation-heavy
queries benefit from scaling vertically, adding more CPU and memory per node.

**Shuffling**: In distributed queries, operations such as a join or group-by can require all rows
with a given key to be on the same worker. To accomplish this, data is redistributed across the
cluster in a [shuffle](glossary.md#shuffle). Within a stage, the streaming engine processes incoming
shuffle data as it arrives over the network, so IO and computation overlap. Shuffle-heavy queries
produce large volumes of inter-node traffic, which you can observe as network bandwidth usage in the
cluster dashboard, and as a high percentage of time spent shuffling in the metrics.

## Using the query profiler

The cluster dashboard and built-in query profiler are available through the Polars Cloud compute
dashboard.

The profiler shows detailed metrics, both real-time and after query completion, such as workers'
resource usage and the percentage of time spent shuffling.

![Cluster dashboard](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)

You can follow along with this example query:

<details markdown>
<summary>Example single node query</summary>

Copy and paste the example below to explore the feature yourself. Don't forget to change the
workspace name to one of your own workspaces.

{{code_block('polars-cloud/query-profile','single-node-query',[])}}

</details>

### Single Node Query

Queries can be run on a single node by marking your query like so:

```python
query.remote(ctx).single_node().execute()
```

This will let the query run on a single worker. This simplifies query execution and you won't need
to shuffle data between workers.

#### Query plans

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

#### Problem indicators

Each node in the physical plan can show indicators to help identify bottlenecks:

| Indicator                                                                                                                                         | Description                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![CPU time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cpu-time.png)                           | Percentage of total CPU time spent in this node. If this is high for unexpected operations, it might indicate a problem.                                                                       |
| ![IO time](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png)                             | Percentage of total IO time spent in this node. Useful when reading multiple files to see which takes longest.                                                                                 |
| ![Memory intensive](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-memory-intensive.png) | The node is potentially memory-intensive because the operation requires keeping state (e.g. `group_by`, `join`).                                                                               |
| ![Single node](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-single-node.png)           | This stage was executed on a single node, either because the operation is not supported by the streaming engine yet, or because it cannot be distributed. Only appears in distributed queries. |

!!! info "IO and CPU time don't sum to 100%"

    The IO time and CPU time percentages shown per node do not sum to the total runtime. This is because execution is pipelined: data is processed as it arrives, so IO (reading/writing) and CPU (computation) work happens concurrently. As a result, both indicators can be non-zero at the same time for a given node, and their combined total can exceed the total runtime.

### Distributed Query

Distributed is the default execution mode in Polars Cloud. You can also set it explicitly:

```python
query.remote(ctx).distributed().execute()
```

For more on how distributed execution works, see [Distributed queries](distributed-engine.md).

You can follow along in your own environment with the following example query:

<details markdown>
<summary>Example distributed query</summary>

You can copy and paste the example below to explore the feature yourself. Don't forget to change the
workspace name to one of your own workspaces.

{{code_block('polars-cloud/query-profile','distributed-query',[])}}

</details>

#### Stage graph

When executing distributed queries, queries are often executed in [stages](glossary.md#stage). Some
operations require [shuffles](glossary.md#shuffle) to make sure the correct
[partitions](glossary.md#partition) are available to the workers. To accomplish this, data is
shuffled between workers over the network. Each stage can be expanded to inspect the operations it
contains and understand what work is happening at each point in the pipeline.

![Stage graph with node details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)

When you click on the stage (not one of the nodes in it), you open the stage details. The displayed
metrics tell you where the query is spending its time. High shuffle bytes indicate a shuffle-heavy
query. While shuffling is expensive, it can be required.

![Stage graph with stage details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)

Together, the stage graph and query plans give you a complete picture of where your query spends its
time: from the high-level distribution of work across stages down to the specific operations driving
the cost.

<!--
#### cluster_dashboard.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)

#### compute_dashboard.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/compute_dashboard.png)

#### cpu-time.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cpu-time.png)

#### indicator-memory-intensive.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-memory-intensive.png)

#### indicator-single-node.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-single-node.png)

#### io-time.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png)

#### logical-plan-small.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical-plan-small.png)

#### logical_plan.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical_plan.png)

#### physical-plan-small.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical-plan-small.png)

#### physical_plan.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)

#### query-details.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query-details.png)

#### query_details.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query_details.png)

#### stage_graph_node_details.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)

#### stage_graph_stage_details.png

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)
-->

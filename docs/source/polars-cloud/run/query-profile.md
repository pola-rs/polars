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

**Shuffling**: When you are running a distributed query there is another type of operation, that can
run concurrently with computation. Operations such as a join or group-by can require all rows with a
given key to be on the same worker. To accomplish this, data is redistributed across the cluster in
a [shuffle](glossary.md#shuffle). Shuffle-heavy queries produce large volumes of inter-node traffic,
which you can observe as network bandwidth usage in the cluster dashboard, and as a high percentage
of time spent shuffling in the metrics.

## Using the query profiler

<!--
TODO: First single node query, 2nd distributed TODO: 1. logical plan single node TODO: Kleinere
query voor distributed (enkele join) TODO: Distributed: 1. Shuffling 2. Planning 3. Stage graph:
welke stage valt op en waarom (pie graph)-->

The cluster dashboard and built-in query profiler are available through the Polars Cloud compute
dashboard. They show detailed metrics, both real-time and after query completion, such as workers'
resource usage and the percentage of time spent shuffling.

![Query details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)

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
shuffling data between workers.

#### Query plans

You can inspect the details of a query by going to the "Queries" tab and selecting the query you
want to inspect. At the bottom of the query details you can inspect the
[optimized logical plan](glossary.md#optimized-logical-plan) and the
[physical plan](glossary.md#physical-plan). The logical plan shows what your query will do, and how
your query has been optimized. The physical plan shows how the engine executes your query: the
concrete algorithms, operator implementations, and data flow chosen at runtime.

<!-- TODO Logical plan -->

![Physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)

Each node in the physical plan is annotated with the percentage of time spent in that node, making
it straightforward to spot computational bottlenecks. Nodes are also flagged when they are
potentially memory-intensive. This is usually the case for operations that have to keep a state in
memory, such as `group_by` or `join`. Some operators are not supported on the streaming engine yet,
and they will be marked with an indicator that the operation runs on the in-memory engine.

<!--TODO: indicator examples, use screenshots (links at the bottom).-->

<!--TODO: Info box - CPU en IO can overlap, don't sum to 100%-->

!!! info

    The IO and CPU indicators don't sum to 100%. <!-- Why? -->

### Distributed Query

You can follow along in your own environment with the following example query:

<details markdown>
<summary>Example query</summary>

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

The metrics on each stage tell you where the query is spending its time. High shuffle bytes indicate
a shuffle-heavy query.

![Stage graph with stage details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)

Together, the stage graph and query plans give you a complete picture of where your query spends its
time: from the high-level distribution of work across stages down to the specific operations driving
the cost.

<!-- TODO: Available screenshots, remove before live -->

![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/compute_dashboard.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cpu-time.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-memory-intensive.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/indicator-single-node.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/io-time.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical-plan-small.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical_plan.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical-plan-small.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query-details.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query_details.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)
![](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)

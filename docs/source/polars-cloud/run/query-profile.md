# Query profiling

Monitor query execution across workers to identify bottlenecks, understand data flow, and optimize
performance.

## Phases of a distributed query

A distributed query spends its time in three phases, each of which is visible in the query profiler:

**Input/Output**: Each worker reads its assigned [partitions](glossary.md#partition) from storage
and stores the results in the destination location. These are typically the first and last
activities you see in the profiler. IO-heavy queries benefit from scaling horizontally: adding more
workers reduces the per-worker download and upload volume.

**Computation**: Workers execute the query operations (filters, joins, aggregations) on their local
data. Computation-heavy queries benefit from scaling vertically, adding more CPU and memory per
node.

**Shuffling**: When an operation such as a join or group-by requires all rows with a given key to be
on the same worker, data is redistributed across the cluster in a [shuffle](glossary.md#shuffle). A
shuffle separates two [stages](glossary.md#stage). Shuffle-heavy queries produce large volumes of
inter-node traffic, which you can observe as network bandwidth usage in the cluster dashboard.

## Using the query profiler

You can follow along in your own environment with the following example query:

<details markdown>
<summary>Example query</summary>

You can copy and paste the example below to explore the feature yourself. Don't forget to change the
workspace name to one of your own workspaces.

{{code_block('polars-cloud/query-profile','query',[])}}

</details>

The cluster dashboard and built-in query profiler are available through the Polars Cloud compute
dashboard. They show real-time status during execution and detailed metrics per stage after
completion, such as the percentage of time spent shuffling and total bytes shuffled.

![Query details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query_details.png)

### Stage graph

The stage graph shows how the query is divided into [stages](glossary.md#stage) separated by
[shuffles](glossary.md#shuffle). Each node in the stage graph represents an operation. Clicking a
node shows its details, such as the join type and keys, grouping columns, sort order, and so on,
which lets you connect the performance metrics you observe back to the specific operations in your
query.

![Stage graph with node details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)

The metrics on each stage tell you where the query is spending its time. High shuffle bytes indicate
a shuffle-heavy query; a stage with a large output row count followed by a significant reduction in
the next stage points to a computation-heavy filtering or aggregation step.

![Stage graph with stage details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)

### Query plans

The profiler also lets you inspect the [logical plan](glossary.md#logical-plan) and
[physical plan](glossary.md#physical-plan) for the query. The logical plan shows how your query has
been optimized. The physical plan shows how the engine executes them: the concrete algorithms,
operator implementations, and data flow chosen at runtime.

![Physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)

Each node in the physical plan is annotated with the percentage of total execution time it consumed,
making it straightforward to spot computational bottlenecks. Nodes are also flagged when they are
memory-intensive or execute on a single worker rather than in parallel across the cluster.

Together, the stage graph and query plans give you a complete picture of where your query spends its
time: from the high-level distribution of work across stages down to the specific operations driving
the cost.

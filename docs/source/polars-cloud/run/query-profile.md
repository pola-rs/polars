# Query profiling

Monitor query execution across workers to identify bottlenecks, understand data flow, and optimize
performance. You can see which stages are running, how data moves between workers, and where time is
spent during execution.

This visibility helps you optimize complex queries and better understand the distributed execution
of queries.

<details>
<summary>Example query and dataset</summary>

You can copy and paste the example below to explore the feature yourself. Don't forget to change the
workspace name to one of your own workspaces.

```python
import polars as pl
import polars_cloud as pc

pc.authenticate()

ctx = pc.ComputeContext(workspace="your-workspace", cpus=12, memory=12, cluster_size=4)

def pdsh_q3(customer, lineitem, orders):
    return (
        customer.filter(pl.col("c_mktsegment") == "BUILDING")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .filter(pl.col("o_orderdate") < pl.date(1995, 3, 15))
        .filter(pl.col("l_shipdate") > pl.date(1995, 3, 15))
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(pl.sum("revenue"))
        .select(
            pl.col("o_orderkey").alias("l_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
    )

lineitem = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/lineitem/*.parquet",
    storage_options={"request_payer": "true"},
)
customer = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/customer/*.parquet",
    storage_options={"request_payer": "true"},
)
orders = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/orders/*.parquet",
    storage_options={"request_payer": "true"},
)
```

{{code_block('polars-cloud/query-profile','execute',[])}}

</details>

<!-- Execute query -->

## Polars Cloud Query Profiler

Polars Cloud has a built-in query profiler.
It shows realtime status of the query during and after execution, and gives you detailed metrics to the node level.
This can help you find and analyze bottlenecks, helping you to run your queries optimally.

It can be accessed from the Cluster Dashboard.

### Cluster Dashboard

The cluster dashboard gives you insights into:

* system metrics (CPU, memory, and network) of all nodes on your cluster.
* an overview of the queries that are related to this cluster, scheduled, running, and finished.

![Cluster Dashboard](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/cluster_dashboard.png)

You can get into the cluster dashboard through the pop-ups on the Polars Cloud dashboard after starting a compute cluster, 
or by going to the details page of your compute.

![Compute Details page](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/compute_dashboard.png)

This dashboard runs from the compute that you're running your queries on.
It becomes available the moment your compute has started and is no longer available after your cluster shuts down.

The system resources allow you to find bottlenecks and tweak your cluster configuration accordingly.

* In case the CPU resources max out, you can add CPUs.
* In case your memory maxes out, you can add memory.
* In case your network bandwith maxes out, you can add more nodes.


### Query Details

When you select a query from the cluster dashboard you open the details.
An overview opens that displays the general metrics of that query.

![Query Details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/query_details.png)

From here you can dive deeper into different aspects of the query.
The first one we'll explore is the logical plan.


### Logical Plan

In Polars, a logical plan is the intermediate representation (IR) of a query that describes what operations to perform, before physical execution details are decided.
This shows the graph that is a representation of the query you sent to Polars Cloud.

![Logical Plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/logical_plan.png)

<!--What can you do with this?-->

### Stage Graph

The stage graph represents the different phases in which the plan is executed on the distributed cluster.

From the overview with the stage graph you can click the stage itself, opening the stage graph details.

![Stage graph details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_stage_details.png)

Alternatively, you can click one of the nodes in any stage to open up its details.

![Stage graph node details](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/stage_graph_node_details.png)

<!-- what is the exact definition of  a stage?-->

When executing a query single node, this is not available.


### Physical Plan

The physical plan shows the strategy that was used to execute the query.

![Physical plan](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/query-profiler/physical_plan.png)

In it you can find the time spent per node, identifying choke points.
Additionally some nodes are marked with warnings that they're memory intensive.
In the details pane you can find specific metrics on how many rows went in and out, what the morsel sizes were and how many went through, and more.


## Profile with the Polars Cloud SDK

Besides the query profiler in the cluster dashboard, you can also get diagnostic information through the Polars Cloud SDK.

### `QueryProfile` and `QueryResult`

The `await_profile` method can be used to monitor an in-progress query. It returns a QueryProfile
object containing a DataFrame with information about which stages are being processed across
workers, which can be analyzed in the same way as any Polars query.

{{code_block('polars-cloud/query-profile','await_profile',[])}}

Each row represents one worker processing a span. A span represents a chunk of work done by a
worker, for example generating the query plan, reading data from another worker, or executing the
query on that data. Some spans may output data, which is recorded in the output_rows column.

```text
shape: (53, 6)
┌──────────────┬──────────────┬───────────┬─────────────────────┬────────────────────┬─────────────┬───────────────────────┬────────────────────┐
│ stage_number ┆ span_name    ┆ worker_id ┆ start_time          ┆ end_time           ┆ output_rows ┆ shuffle_bytes_written ┆ shuffle_bytes_read │
│ ---          ┆ ---          ┆ ---       ┆ ---                 ┆ ---                ┆ ---         ┆ ---                   ┆                    │
│ u32          ┆ str          ┆ str       ┆ datetime[ns]        ┆ datetime[ns]       ┆ u64         ┆ u64                   ┆ u64                │
╞══════════════╪══════════════╪═══════════╪═════════════════════╪════════════════════╪═════════════╪═══════════════════════╪════════════════════╡
│ 6            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 282794      ┆ 72395264              ┆ null               │
│              ┆              ┆           ┆ 08:08:52.820228585  ┆ 08:08:52.878229914 ┆             ┆                       ┆                    │
│ 3            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 3643370     ┆ 932702720             ┆ null               │
│              ┆              ┆           ┆ 08:08:45.421053731  ┆ 08:08:45.600081475 ┆             ┆                       ┆                    │
│ 5            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 282044      ┆ 723203264             ┆ null               │
│              ┆              ┆           ┆ 08:08:52.667547917  ┆ 08:08:52.718114297 ┆             ┆                       ┆                    │
│ 5            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        ┆ null                  ┆ 932702720          │
│              ┆              ┆           ┆ 08:08:52.694917167  ┆ 08:08:52.720657155 ┆             ┆                       ┆                    │
│ 7            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 145179      ┆ 37165824              ┆ null               │
│              ┆              ┆           ┆ 08:08:53.039771274  ┆ 08:08:53.166535930 ┆             ┆                       ┆                    │
│ …            ┆ …            ┆ …         ┆ …                   ┆ …                  ┆ …           ┆ …                     ┆ …                  │
│ 5            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        ┆ null                  ┆ 72503808           │
│              ┆              ┆           ┆ 08:08:52.649434841  ┆ 08:08:52.667065947 ┆             ┆                       ┆                    │
│ 6            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 283218      ┆ 72503808              ┆ null               │
│              ┆              ┆           ┆ 08:08:52.818787714  ┆ 08:08:52.880324797 ┆             ┆                       ┆                    │
│ 4            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        ┆ null                  ┆ 3979787264         │
│              ┆              ┆           ┆ 08:08:46.188322234  ┆ 08:08:50.871792346 ┆             ┆                       ┆                    │
│ 1            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 15546044    ┆ 3979787264            ┆ null               │
│              ┆              ┆           ┆ 08:08:40.325404872  ┆ 08:08:44.030028095 ┆             ┆                       ┆                    │
│ 7            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        ┆ null                  ┆ 37165824           │
│              ┆              ┆           ┆ 08:08:52.925442390  ┆ 08:08:52.962600065 ┆             ┆                       ┆                    │
└──────────────┴──────────────┴───────────┴─────────────────────┴────────────────────┴─────────────┴───────────────────────┴────────────────────┘
```

As each worker starts and completes each stage of the query, it notifies the lead worker. The
`await_profile` method will poll the lead worker until there is an update from any worker, and then
return the full profile data of the query.

The `QueryProfile` object also has a `summary` property to return an aggregated view of each stage.

{{code_block('polars-cloud/query-profile','await_summary',[])}}

```text
shape: (13, 6)
┌──────────────┬──────────────┬───────────┬────────────┬──────────────┬─────────────┬───────────────────────┬────────────────────┐
│ stage_number ┆ span_name    ┆ completed ┆ worker_ids ┆ duration     ┆ output_rows ┆ shuffle_bytes_written ┆ shuffle_bytes_read │
│ ---          ┆ ---          ┆ ---       ┆ ---        ┆ ---          ┆ ---         ┆ ---                   ┆ ---                │
│ u32          ┆ str          ┆ bool      ┆ str        ┆ duration[μs] ┆ u64         ┆ u64                   ┆ u64                │
╞══════════════╪══════════════╪═══════════╪════════════╪══════════════╪═════════════╪═══════════════════════╪════════════════════╡
│ 6            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 1228µs       ┆ 0           ┆ 0                     ┆ 289546496          │
│ 5            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 140759µs     ┆ 0           ┆ 0                     ┆ 289546496          │
│ 4            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 1s 73534µs   ┆ 1131041     ┆ 289546496             ┆ 0                  │
│ 2            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 6s 944740µs  ┆ 3000188     ┆ 768048128             ┆ 0                  │
│ 5            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 167483µs     ┆ 1131041     ┆ 289546496             ┆ 0                  │
│ …            ┆ …            ┆ …         ┆ …          ┆ …            ┆ …           ┆ …                     ┆ …                  │
│ 4            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 4s 952005µs  ┆ 0           ┆ 0                     ┆ 255627121          │
│ 1            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 7s 738907µs  ┆ 72874383    ┆ 18655842048           ┆ 0                  │
│ 3            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 812807µs     ┆ 0           ┆ 0                     ┆ 768048128          │
│ 0            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 15s 2883µs   ┆ 323494519   ┆ 82814596864           ┆ 0                  │
│ 7            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 356662µs     ┆ 1131041     ┆ 289546496             ┆ 0                  │
└──────────────┴──────────────┴───────────┴────────────┴──────────────┴─────────────┴───────────────────────┴────────────────────┘
```

### Plan

`QueryProfile` also exposes `.plan()` to retrieve the physical plan as a string, and `.graph()` to
render it as a visual diagram. See [Explain](#explain) below for details.

Use `.plan()` to retrieve the executed query plan as a string. This is useful for understanding
exactly how Polars executed your query, including the physical stages and operations performed
across the cluster.

{{code_block('polars-cloud/query-profile','explain',['QueryResult'])}}

```text
# TODO: add example output
```

You can also retrieve the optimized intermediate representation (IR) of the query before execution
by passing `"ir"` as the plan type.

{{code_block('polars-cloud/query-profile','explain_ir',['QueryResult'])}}

```text
# TODO: add example output
```

``` Graph

Both `plan()` and `graph()` are available on `QueryResult` (with `plan_type` set to `"physical"` or
`"ir"`) and on `QueryProfile` (physical plan only). These methods are only available in direct mode.

Use `.graph()` to render the plan as a visual dot diagram using matplotlib.

{{code_block('polars-cloud/query-profile','graph',['QueryResult'])}}

<!-- TODO: Image of graph output -->

# Query progress monitoring

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

</details>

{{code_block('polars-cloud/query-progress','execute',[])}}

The `await_progress` method can be used to monitor an in-progress query. It returns a QueryProgress
object containing a DataFrame with information about which stages are being processed across
workers, which can be analyzed in the same way as any Polars query.

{{code_block('polars-cloud/query-progress','await_progress',[])}}

Each row represents one worker processing a span. A span represents a chunk of work done by a
worker, for example generating the query plan, reading data from another worker, or executing the
query on that data. Some spans may output data, which is recorded in the output_rows column.

```text
shape: (53, 6)
┌──────────────┬──────────────┬───────────┬─────────────────────┬────────────────────┬─────────────┐
│ stage_number ┆ span_name    ┆ worker_id ┆ start_time          ┆ end_time           ┆ output_rows │
│ ---          ┆ ---          ┆ ---       ┆ ---                 ┆ ---                ┆ ---         │
│ u32          ┆ str          ┆ str       ┆ datetime[ns]        ┆ datetime[ns]       ┆ u64         │
╞══════════════╪══════════════╪═══════════╪═════════════════════╪════════════════════╪═════════════╡
│ 6            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 282794      │
│              ┆              ┆           ┆ 08:08:52.820228585  ┆ 08:08:52.878229914 ┆             │
│ 3            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 3643370     │
│              ┆              ┆           ┆ 08:08:45.421053731  ┆ 08:08:45.600081475 ┆             │
│ 5            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 282044      │
│              ┆              ┆           ┆ 08:08:52.667547917  ┆ 08:08:52.718114297 ┆             │
│ 5            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        │
│              ┆              ┆           ┆ 08:08:52.694917167  ┆ 08:08:52.720657155 ┆             │
│ 7            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 145179      │
│              ┆              ┆           ┆ 08:08:53.039771274  ┆ 08:08:53.166535930 ┆             │
│ …            ┆ …            ┆ …         ┆ …                   ┆ …                  ┆ …           │
│ 5            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        │
│              ┆              ┆           ┆ 08:08:52.649434841  ┆ 08:08:52.667065947 ┆             │
│ 6            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 283218      │
│              ┆              ┆           ┆ 08:08:52.818787714  ┆ 08:08:52.880324797 ┆             │
│ 4            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        │
│              ┆              ┆           ┆ 08:08:46.188322234  ┆ 08:08:50.871792346 ┆             │
│ 1            ┆ Execute IR   ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ 15546044    │
│              ┆              ┆           ┆ 08:08:40.325404872  ┆ 08:08:44.030028095 ┆             │
│ 7            ┆ Shuffle read ┆ i-xxx     ┆ 2025-xx-xx          ┆ 2025-xx-xx         ┆ null        │
│              ┆              ┆           ┆ 08:08:52.925442390  ┆ 08:08:52.962600065 ┆             │
└──────────────┴──────────────┴───────────┴─────────────────────┴────────────────────┴─────────────┘
```

As each worker starts and completes each stage of the query, it notifies the lead worker. The
`await_progress` method will poll the lead worker until there is an update from any worker, and then
return the full progress data of the query.

The QueryProgress object also has a summary property to return an aggregated view of each stage.

{{code_block('polars-cloud/query-progress','await_summary',[])}}

```text
shape: (13, 6)
┌──────────────┬──────────────┬───────────┬────────────┬──────────────┬─────────────┐
│ stage_number ┆ span_name    ┆ completed ┆ worker_ids ┆ duration     ┆ output_rows │
│ ---          ┆ ---          ┆ ---       ┆ ---        ┆ ---          ┆ ---         │
│ u32          ┆ str          ┆ bool      ┆ str        ┆ duration[μs] ┆ u64         │
╞══════════════╪══════════════╪═══════════╪════════════╪══════════════╪═════════════╡
│ 6            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 1228µs       ┆ 0           │
│ 5            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 140759µs     ┆ 0           │
│ 4            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 1s 73534µs   ┆ 1131041     │
│ 2            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 6s 944740µs  ┆ 3000188     │
│ 5            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 167483µs     ┆ 1131041     │
│ …            ┆ …            ┆ …         ┆ …          ┆ …            ┆ …           │
│ 4            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 4s 952005µs  ┆ 0           │
│ 1            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 7s 738907µs  ┆ 72874383    │
│ 3            ┆ Shuffle read ┆ true      ┆ i-xxx      ┆ 812807µs     ┆ 0           │
│ 0            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 15s 2883µs   ┆ 323494519   │
│ 7            ┆ Execute IR   ┆ true      ┆ i-xxx      ┆ 356662µs     ┆ 1131041     │
└──────────────┴──────────────┴───────────┴────────────┴──────────────┴─────────────┘
```

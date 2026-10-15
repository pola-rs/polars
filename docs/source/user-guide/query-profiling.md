# Query profiling

Query profiling shows what happens inside a lazy query while it runs on your machine. Polars records
the optimized query plan and, for each operation in it, how many rows went in and out, how much data
was read, and how much time was spent. You can follow this live while the query runs, or look back
at it after the query has finished.

The query runs locally, exactly as it would without profiling. Polars sends the query plan and the
runtime metrics to [Polars Cloud](https://cloud.pola.rs/), where each run appears as a query
profile. The data that your query reads and produces stays on your machine.

## When to use query profiling

Query profiling is useful when you want to understand how a query behaves at runtime, rather than
how it was planned. For example, to:

- find the operation that takes the most time in a slow query.
- see whether a query is limited by I/O or by computation.
- check how many rows each filter and join removes or produces.
- follow the progress of a long-running query.

To see how Polars plans a query before running it, use
[`explain` or `show_graph`](lazy/query-plan.md) instead.

## Example

Install the `polars-cloud` package in the same environment as Polars:

```bash
pip install 'polars-cloud>=0.13.0'
```

The example below computes the total value of each order. After `pl.Config.enable_monitoring()`,
every query you collect is profiled. If you are not signed in to Polars Cloud yet,
`enable_monitoring()` asks you to sign in:

```python
import polars as pl

pl.Config.enable_monitoring()

print(
    pl.LazyFrame(
        {
            "quantity": [1, 2, 3, 4, 3],
            "price": [34.2, 43.2, 123.1, 23.3, 42],
            "order_id": [1, 1, 2, 2, 2],
        }
    )
    .group_by("order_id")
    .agg((pl.col("quantity") * pl.col("price")).sum())
    .collect()
)
```

Open the [Polars Cloud dashboard](https://cloud.pola.rs/portal) and select the query under
**Queries**. The profile shows the logical plan after optimization, and the physical plan with the
metrics of each node. Select a node to see its metrics, such as the number of rows in and out and
the time spent:

![A query profile. Left: the optimized logical plan. Middle: the physical plan with the number of rows between nodes. Right: the metrics of the selected node.](https://pola.rs/_astro/profiler.BbQEqRZP_Z2lnGOb.webp)

### Profiling a single query

`pl.Config.enable_monitoring()` profiles every query in the session. To profile one query only, use
`Config` as a context manager:

```python
with pl.Config(enable_monitoring=True):
    lf.collect()
```

Or pass an engine with `monitoring=True` to `collect`:

```python
lf.collect(engine=pl.StreamingEngine(monitoring=True))
```

`monitoring=False` excludes a single query while profiling is enabled for the session. To disable
profiling for the session, call `pl.Config.enable_monitoring(False)`.

### Choosing a workspace

Profiles go to the default workspace of your account. Pass `workspace` to use another workspace, and
`organization` if the workspace name exists in more than one organization:

```python
pl.Config.enable_monitoring(workspace="My workspace", organization="My organization")
```

## How it works

Profiling hooks into the execution of a lazy query:

1. After Polars optimizes the query, it sends the optimized logical and physical plan.
1. While the query runs, it sends the metrics of each physical node at a regular interval.
1. When the query finishes, it sends the final metrics. If the query fails, it sends the error
   message.

The query plan contains the operations of the query, the column names, the paths of the files that
it reads and writes, and any literal values that you write in the query, such as the `100` in
`pl.col("price") > 100`. Data in memory, such as the `LazyFrame` in the example above, appears in
the plan only as its number of rows and its column names.

The error message of a failed query is sent as is. It can contain values from your data, such as the
value that failed to convert in a cast.

## Limitations

- **Lazy queries only.** Polars profiles queries that you run with `collect`, `collect_all`,
  `collect_batches`, their async variants, and the `sink_*` methods.
- **Per-node metrics need the streaming engine.** The streaming engine is the default engine, and
  `enable_monitoring` sets the engine affinity to `"streaming"`. Queries that run on the in-memory
  engine are profiled with their logical plan and status, but without the physical plan and node
  metrics.
- **No GPU support.** The GPU engine does not support profiling.
- **Python only.** Query profiling is not available in the Rust API.

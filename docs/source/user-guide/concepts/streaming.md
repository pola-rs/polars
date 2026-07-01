# Streaming

One additional benefit of the lazy API is that it allows queries to be executed in a streaming
manner. Instead of processing all the data at once, Polars can execute the query in batches allowing
you to process datasets that do not fit in memory. Besides memory pressure, the streaming engine
also is more performant than Polars' in-memory engine.

The table below summarises the streaming APIs available in Polars:

| API | Description |
| ---- | ----------- |
| `collect(engine="streaming")` | Execute a lazy query in streaming mode and return a `DataFrame` |
| `sink_parquet(path)` | Stream query output directly to a Parquet file |
| `sink_csv(path)` | Stream query output directly to a CSV file |
| `sink_ndjson(path)` | Stream query output to a newline-delimited JSON file |
| `sink_batches(fn, chunk_size)` | Call a Python function for each output batch |
| `collect_async()` | Execute a lazy query asynchronously and return an `Awaitable[DataFrame]` |

## Basic streaming: `collect(engine="streaming")`

To tell Polars we want to execute a query in streaming mode we pass the `engine="streaming"`
argument to `collect`:

{{code_block('user-guide/concepts/streaming','streaming',['collect'])}}

The streaming engine processes data in **chunks**, keeping peak memory bounded regardless of
dataset size:

```
scan_parquet  -->  filter (pushdown)  -->  select  -->  group_by
   [chunk 1]            |                    |              |
   [chunk 2]        predicate             project         agg
   [chunk N]                                            merge results
                                                              |
                                                         DataFrame
```

Here is a more complete example -- grouping and aggregating across a full file scan where only
the matching rows need to reside in memory at any time:

{{code_block('user-guide/concepts/streaming','larger_than_ram',['collect'])}}

## Inspecting a streaming query

Polars can run many operations in a streaming manner. Some operations are inherently non-streaming,
or are not implemented in a streaming manner (yet). In the latter case, Polars will fall back to the
in-memory engine for those operations. A user doesn't have to know about this, but it can be
interesting for debugging memory or performance issues.

To inspect the physical plan of streaming query, you can plot the physical graph. The legend shows
how memory intensive the operation can be.

```python
--8<-- "python/user-guide/concepts/streaming.py:createplan_query"
```

```python exec="on" session="user-guide/concepts/streaming"
--8<-- "python/user-guide/concepts/streaming.py:createplan"
```

## Writing in streaming mode: `sink_*`

Polars can also **write** results in streaming mode. The `sink_*` family of methods flush each
batch to disk as it is produced -- the full query result never needs to fit in memory. This is
essential for large ETL pipelines that transform and re-partition data.

### `sink_parquet`

{{code_block('user-guide/concepts/streaming','sink_parquet',['sink_parquet'])}}

### `sink_csv`

{{code_block('user-guide/concepts/streaming','sink_csv',['sink_csv'])}}

## Per-batch callbacks: `sink_batches`

`sink_batches(callback, chunk_size=N)` calls your Python function once per output batch.
This enables patterns such as:

- **Real-time alerting** -- trigger on threshold violations as each chunk arrives
- **Incremental forwarding** -- push each batch onward to a WebSocket, database, or queue
- **Stateful accumulation** -- maintain running summaries without holding the full dataset in memory

{{code_block('user-guide/concepts/streaming','sink_batches',['sink_batches'])}}

## Concurrent async execution: `collect_async`

`collect_async()` submits a lazy query to Polars' thread pool and returns an
`Awaitable[DataFrame]`. Combine it with `asyncio.gather` to run multiple independent queries
concurrently and overlap their execution:

{{code_block('user-guide/concepts/streaming','collect_async',['collect_async'])}}

## Multi-file scans and partition pruning

When data is stored across many Parquet files (for example in hourly or daily partitions),
Polars' lazy scanner evaluates predicates on file statistics **before opening files**. Only files
that could satisfy the filter are opened, dramatically reducing I/O for selective queries.

{{code_block('user-guide/concepts/streaming','partition_pruning',['scan_parquet'])}}

!!! note
    Predicate pushdown works best when partition columns are encoded in Hive-style directory
    paths (`year=2024/month=01/...`) and when Parquet row-group statistics are present.
    Write partitions with `write_parquet(statistics=True)` (the default) for maximum pruning
    efficiency.

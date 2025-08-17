# Sources and sinks

## Scan

When using the `LazyFrame` API, it is important to favor `scan_*` (`scan_parquet`, `scan_csv`, etc.)
over `read_*`. A Polars `scan` is lazy and will delay execution until the query is collected. The
benefit of this, is that the Polars optimizer can push optimization into the readers. They can skip
reading columns and rows that aren't required. Another benefit is that, during streaming execution,
the engine already can process batches before the file is completely read.

## Sink

Sinks can execute a query and stream the results to storage (being disk or cloud). The benefit of
sinking data to storage is that you don't necessarily have to store all data in RAM, but can process
data in batches.

If we would want to convert many csv files to parquet, whilst dropping the missing data, we could do
something like the query below. We use a partitioning strategy that defines how many rows may be in
a single parquet file, before we generate a new file

```python
lf = scan_csv("my_dataset/*.csv").filter(pl.all().is_not_null())
lf.sink_parquet(
    pl.PartitionMaxSize(
        "my_table_{part}.parquet"
        max_size=512_000
    )
)
```

This will create the following files on disk:

```text
my_table_0.parquet
my_table_1.parquet
...
my_table_n.parquet
```

## Multiplexing sinks

Sinks can also multiplex. Meaning that we write to different sinks in a single query. In the code
snippet below, we take a `LazyFrame` and sink it into 2 sinks at the same time.

```python
# Some expensive computation
lf: LazyFrame 

q1 = lf.sink_parquet(.., lazy=True)
q2 = lf.sink_ipc(.., lazy=True)

lf.collect_all([q1, q2])
```

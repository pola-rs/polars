# Distributed queries

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process data at scale.

!!! info "Distributed engine is in open beta"

    The distributed engine is in open beta and currently supports most common operations. Check the specific operations that are [currently supported in the distributed engine](https://github.com/pola-rs/polars/issues/21487).

    We are open to feedback and learning from your use cases. Reach the team at [support@polars.tech](mailto:support@polars.tech).

## Using distributed engine

To execute queries using the distributed engine, you can call the `distributed()` method.

```python
lf: LazyFrame

result = (
      lf.remote()
      .distributed()
      .sink_parquet()
)
```

### Example

{{code_block('polars-cloud/distributed','example',[])}}

## Working with large datasets in the distributed engine

The distributed engine can only read sources partitioned with direct scan\_<file> methods such as
`scan_parquet` and `scan_csv`. Open table formats like `scan_iceberg` are not yet supported in a
distributed fashion and will run on a single node when utilized.

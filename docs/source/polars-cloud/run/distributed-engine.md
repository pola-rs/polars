# Execute distributed query

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process data at scale.

<!-- dprint-ignore-start -->

!!! info "Distributed engine is early stage"
    The distributed engine is still in the very early stages of development. Major performance improvements are planned for the near future. When an operation is not yet available in a distributed manner, Polars Cloud will execute it on a single node.
    
    Find out which operations are [currently supported in the distributed engine](https://github.com/pola-rs/polars/issues/21487).

<!-- dprint-ignore-end-->

## Using distributed engine

To execute queries using the distributed engine, you can call the `distributed()` method.

```python
lf: LazyFrame

result = (
      lf.remote()
      .distributed()
      .collect()
)
```

### Example

```python
import polars as pl
import polars_cloud as pc
from datetime import date

query = (
    pl.scan_parquet("s3://dataset/")
    .filter(pl.col("l_shipdate") <= date(1998, 9, 2))
    .group_by("l_returnflag", "l_linestatus")
    .agg(
        avg_price=pl.mean("l_extendedprice"),
        avg_disc=pl.mean("l_discount"),
        count_order=pl.len(),
    )
)

result = (
    query.remote(pc.ComputeContext(cpus=16, memory=64, cluster_size=32))
    .distributed()
    .sink_parquet("s3://output/result.parquet")
)
```

## Working with large datasets in the distributed engine

The distributed engine can only read sources partitioned with direct scan_<file> methods such as
`scan_parquet` and `scan_csv`. Open table formats like `scan_iceberg` are not yet supported in a
distributed fashion and will run on a single node when utilized.

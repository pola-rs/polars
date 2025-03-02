# Distributed query execution

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process at scale.

<!-- dprint-ignore-start -->

!!! info "Distributed engine is in early stage"
    The distributed engine is in its very early development. It currently runs all [PDS-H benchmarks](https://github.com/pola-rs/polars-benchmark). Major performance improvements will be introduced in the near future. When a operation is not available in a distributed manner, Polars Cloud will run that operation on single node.

<!-- dprint-ignore-end-->

## Using distributed engine

To execute queries using the distributed engine, you can call `distributed()`.

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

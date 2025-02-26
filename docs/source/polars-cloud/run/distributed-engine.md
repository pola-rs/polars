# Distributed query execution

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process at scale.

<!-- dprint-ignore-start -->

!!! info "Distributed engine is in early stage"
    The distributed engine is actively being developed and is labeled unstable. It currently runs all [PDS-H benchmarks](https://github.com/pola-rs/polars-benchmark). Many operations are either unsupported or not yet optimized. Major performance improvements will be introduced in the near future.

<!-- dprint-ignore-end-->

## Using distributed engine

To execute queries using the distributed engine, users can call `distributed()`. It is important to
set `cluster_size` to a value larger than 1 in the `ComputeContext`.

```python
lf: LazyFrame

result = (
      lf.remote()
      .distributed()
      .collect()
)
```

### Example

{{code_block('polars-cloud/distributed','distributed',[])}}

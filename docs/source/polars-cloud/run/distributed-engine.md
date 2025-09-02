# Distributed queries

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process data at scale.

## Using distributed engine

To execute queries using the distributed engine, you can call the `distributed()` method.

```python
lf: LazyFrame

result = (
      lf.remote()
      .distributed()
      .execute()
)
```

### Example

{{code_block('polars-cloud/distributed','example',[])}}

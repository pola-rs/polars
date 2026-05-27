# Distributed queries

Polars has always been optimized for fast and efficient performance on a single machine. The
distributed engine extends this to datasets that are too large to fit on a single node, spreading
both computation and memory across a cluster so you can query at any scale.

## Using distributed engine

To execute queries using the distributed engine, you can call the `distributed()` method. This is
the default mode of execution for remote queries.

```python
lf: LazyFrame

result = (
      lf.remote()
      .distributed()
      .execute()
)
```

### Example

This example demonstrates running query 3 of the PDS-H benchmarkon scale factor 100 (approx. 100GB
of data) using Polars Cloud distributed engine.

First import the required packages and point to the S3 bucket. In this example, we take one of the
PDS-H benchmarks queries for demonstration purposes.

{{code_block('polars-cloud/distributed','setup_on_prem',[])}}

After that we define the query. Note that this query will also run on your local machine if you have
the data available. You can generate the data with the
[Polars Benchmark repository](https://www.github.com/pola-rs/polars-benchmark).

{{code_block('polars-cloud/distributed','query',[])}}

The final step is to set the compute context and run the query. Here we're using 5 nodes with 10
CPUs and 10GB memory each. `Show()` will return the first 10 rows back to your environment. The
query takes around xx seconds to execute.

{{code_block('polars-cloud/distributed','context-run_on_prem',[])}}

!!! tip "Try on SF1000 (approx. 1TB of data)"

    You can also run this example on a higher scale factor. The data is available on the same bucket. You can change the URL from `sf100` to `sf1000`.

## How it works

When you call `.execute()` on a distributed query, it passes through the following pipeline:

![Flow graph](https://raw.githubusercontent.com/pola-rs/polars-static/master/docs/distributed-query-flow.png)

1. You write a query using the Polars [DSL](../../polars-cloud/run/glossary.md#dsl), building up a
   [LazyFrame](../../polars-cloud/run/glossary.md#query).
2. The LazyFrame is translated into a
   [logical plan](../../polars-cloud/run/glossary.md#logical-plan): a tree of operations capturing
   _what_ to compute. You can inspect this logical plan by running `lf.explain(optimized=False)`.
3. The query optimizer rewrites the logical plan into an equivalent but more efficient
   [optimized logical plan](../../polars-cloud/run/glossary.md#optimized-logical-plan). You can
   inspect the optimized logical plan with `lf.explain()`.
4. The distributed query planner walks the optimized logical plan and produces a
   [stage graph](../../polars-cloud/run/glossary.md#stage-graph): a DAG of
   [stages](../../polars-cloud/run/glossary.md#stage) separated by
   [shuffles](../../polars-cloud/run/glossary.md#shuffle) at each point where a data needs to be
   redistributed across workers.
5. The [scheduler](../../polars-cloud/run/glossary.md#scheduler) executes stages and assigns
   [partitions](../../polars-cloud/run/glossary.md#partition) to
   [workers](../../polars-cloud/run/glossary.md#worker) in dependency order, waiting for all workers
   to finish before starting the next stage.
6. Each worker receives the optimized logical plan together with its assigned partitions, derives
   its own [physical plan](../../polars-cloud/run/glossary.md#physical-plan), and executes it. After
   finishing the stage, intermediate results are written to a local or network-shared disk.
7. After the final stage, results are written to the destination location, or sent back to the user,
   depending on the query.

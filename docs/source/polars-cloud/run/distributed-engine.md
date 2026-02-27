# Distributed queries

With the introduction of Polars Cloud, we also introduced the distributed engine. This engine
enables users to horizontally scale workloads across multiple machines.

Polars has always been optimized for fast and efficient performance on a single machine. However,
when querying large datasets from cloud storage, performance is often constrained by the I/O
limitations of a single node. By scaling horizontally, these download limitations can be
significantly reduced, allowing users to process data at scale.

!!! info "Distributed engine is in open beta"

    The distributed engine currently supports most of Polars API and datatypes. Follow [the tracking issue](https://github.com/pola-rs/polars/issues/21487) to stay up to date.

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

!!! note "Run the example yourself"

    Copy and paste the code to you environment and run it. The data is hosted in S3 buckets that use [AWS Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html), meaning you pay only for pays the cost of the request and the data download from the bucket. The storage costs are covered.

First import the required packages and point to the S3 bucket. In this example, we take one of the
PDS-H benchmarks queries for demonstration purposes.

{{code_block('polars-cloud/distributed','setup',[])}}

After that we define the query. Note that this query will also run on your local machine if you have
the data available. You can generate the data with the
[Polars Benchmark repository](https://www.github.com/pola-rs/polars-benchmark).

{{code_block('polars-cloud/distributed','query',[])}}

The final step is to set the compute context and run the query. Here we're using 5 nodes with 10
CPUs and 10GB memory each. `Show()` will return the first 10 rows back to your environment. The
query takes around xx seconds to execute.

{{code_block('polars-cloud/distributed','context-run',[])}}

!!! tip "Try on SF1000 (approx. 1TB of data)"

    You can also run this example on a higher scale factor. The data is available on the same bucket. You can change the URL from `sf100` to `sf1000`.

## How it works

When you call `.execute()` on a distributed query, it passes through the following pipeline:

![Flow graph](https://raw.githubusercontent.com/pola-rs/polars-static/master/docs/distributed-query-flow.png)

1. You write a query using the Polars [DSL](glossary.md#dsl), building up a
   [LazyFrame](glossary.md#query).
2. The LazyFrame is translated into a [logical plan](glossary.md#logical-plan): a tree of operations
   capturing _what_ to compute. You can inspect this logical plan by running
   `lf.explain(optimized=False)`.
3. The query optimizer rewrites the logical plan into an equivalent but more efficient
   [optimized logical plan](glossary.md#optimized-logical-plan). You can inspect the optimized
   logical plan with `lf.explain()`.
4. The distributed query planner walks the optimized logical plan and produces a
   [stage graph](glossary.md#stage-graph): a DAG of [stages](glossary.md#stage) separated by
   [shuffles](glossary.md#shuffle) at each point where a data needs to be redistributed across
   workers.
5. The [scheduler](glossary.md#scheduler) executes stages and assigns
   [partitions](glossary.md#partition) to [workers](glossary.md#worker) in dependency order, waiting
   for all workers to finish before starting the next stage.
6. Each worker receives the optimized logical plan together with its assigned partitions, derives
   its own [physical plan](glossary.md#physical-plan), and executes it. After finishing the stage,
   intermediate results are written to a local or network-shared disk.
7. After the final stage, results are written to the destination location, or sent back to the user,
   depending on the query.

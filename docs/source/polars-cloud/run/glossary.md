# Glossary

This page defines key terms used throughout the Polars Cloud documentation.

## DSL

A **Domain Specific Language** (DSL) is a language designed for a particular problem domain. Polars
has its own DSL for expressing data transformations: the chained expressions and contexts you write
when building a query. The DSL is designed to be expressive and human-readable while giving the
query optimizer enough information to reorder and optimize operations before execution.

## Query

A **query** is a `LazyFrame` that has not yet been executed. You build a query by chaining
operations on a `LazyFrame`, such as `filter`, `join`, `group_by`, and `select`, using the Polars
DSL. No data is read or processed until you trigger execution explicitly, for example by calling
`.collect()` locally or `.execute()` on Polars Cloud.

This lazy evaluation model means Polars can inspect the full query before running it, giving the
optimizer the opportunity to reorder and simplify operations before any work is done.

A `DataFrame`, by contrast, is always the result of an already-completed computation, it holds data
in memory. Operations on a `DataFrame` execute immediately and return a new `DataFrame`, so there is
no deferred plan to optimize or distribute.

## Logical plan

The **logical plan** is the representation of your query as a tree of operations, exactly as you
expressed it in the DSL. It captures _what_ you want to compute, but not _how_ to compute it. For
example, calling `.filter(...)` after `.join(...)` produces a logical plan that describes filtering
after joining, regardless of whether that ordering is optimal.

You can inspect the logical plan with `LazyFrame.explain(optimized=False)`.

## Optimized logical plan

The **optimized logical plan** is produced by running the logical plan through the query optimizer.
The optimizer applies a set of rule-based transformations, such as predicate pushdown, projection
pushdown, and common subplan elimination, to produce an equivalent but more efficient plan.

For example, a filter that appears after a join in the logical plan may be pushed down to run before
the join in the optimized plan, reducing the amount of data processed.

You can inspect the optimized logical plan with `LazyFrame.explain()`.

## Physical plan

The **physical plan** is produced by translating the optimized logical plan into a concrete
execution strategy. Where the logical plan describes _what_ to compute, the physical plan describes
_how_ to compute it: which algorithms to use, in what order to execute operations, and how data
flows through the engine at runtime.

## Scheduler

The **scheduler** is the central bookkeeper of a distributed query execution. It is responsible for
dispatching tasks to workers in dependency order (a stage is only dispatched once all required
stages and shuffles have completed). The scheduler tracks worker progress, records where shuffle
data is stored and which workers need it, and determines when the query is finished.

On top of that it runs auxiliary services, such as managing the query queue.

## Worker

A **worker** is a machine that executes tasks on behalf of the scheduler. Each worker is assigned a
set of partitions to process and executes a stage using the Polars streaming engine independently,
reading input data or consuming shuffle data produced by a previous stage. Workers report task
completion back to the scheduler and write shuffle output for downstream stages to consume.

## Stage graph

The **stage graph** is produced by the distributed query planner from the optimized logical plan.
The planner walks the logical plan and identifies **stage boundaries**: points where a data shuffle
is required to optimize stages to maximize parallelism, minimize data shuffle, and keep peak memory
usage under control. Joins and group-bys are typical examples, a worker cannot produce its final
result without first receiving the relevant keys or partial aggregates from other workers.

At each stage boundary, the planner inserts a shuffle and starts a new stage. The result is a
directed acyclic graph (DAG) in which each node is a stage and each edge is a shuffle. All workers
can execute a stage in parallel, but a stage cannot begin until all incoming shuffles have
completed.

## Stage

A **stage** is a node in the stage graph. It represents a series of operations that every worker can
execute independently, without needing to exchange data with other workers mid-execution. A stage
begins either at the start of the query or immediately after an incoming shuffle, and ends at the
next stage boundary where a data shuffle is necessary, or at the end of the query where the final
stage sinks its results to the output.

When a stage is dispatched, each worker receives the optimized logical plan together with its
assigned inputs (which partitions to read, which shuffle data to consume) and derives its own
physical execution plan from that.

## Partition

A **partition** is a subset of the data that can be processed independently by a worker. When a
query runs on the distributed engine, the input data is split into partitions that are distributed
across workers. All worker process their own partitions in parallel, reducing the total time to
complete the query.

## Shuffle

A **shuffle** is the transfer of data between workers between two stages. Some operations, such as
joins, group-bys, and sorts, require a partitioning invariant to hold: all rows that share a key
must reside on the same worker before the operation can be computed. For example, a group-by on
`country` cannot produce a correct aggregate if rows for `"Germany"` are split across multiple
workers. A shuffle establishes this invariant by redistributing data according to a partitioning
key, so that each worker receives exactly the rows it needs for the next stage.

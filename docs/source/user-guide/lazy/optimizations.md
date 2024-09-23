# Optimizations

If you use Polars' lazy API, Polars will run several optimizations on your query. Some of them are executed up front,
others are determined just in time as the materialized data comes in.

Here is a non-complete overview of optimizations done by polars, what they do and how often they run.

| Optimization               | Explanation                                                                                                  | runs                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| Predicate pushdown         | Applies filters as early as possible/ at scan level.                                                         | 1 time                        |
| Projection pushdown        | Select only the columns that are needed at the scan level.                                                   | 1 time                        |
| Slice pushdown             | Only load the required slice from the scan level. Don't materialize sliced outputs (e.g. join.head(10)).     | 1 time                        |
| Common subplan elimination | Cache subtrees/file scans that are used by multiple subtrees in the query plan.                              | 1 time                        |
| Simplify expressions       | Various optimizations, such as constant folding and replacing expensive operations with faster alternatives. | until fixed point             |
| Join ordering              | Estimates the branches of joins that should be executed first in order to reduce memory pressure.            | 1 time                        |
| Type coercion              | Coerce types such that operations succeed and run on minimal required memory.                                | until fixed point             |
| Cardinality estimation     | Estimates cardinality in order to determine optimal group by strategy.                                       | 0/n times; dependent on query |

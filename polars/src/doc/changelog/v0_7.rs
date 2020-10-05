//! # Changelog v0.7
//!
//! * More group by aggregations:
//!     - n_unique
//!     - quantile
//!     - median
//!     - last
//!     - group indexes
//!     - agg (combined aggregations)
//! * explode operation
//! * melt operation
//! * df! macro
//! * Rem trait implemented for Series and ChunkedArrays
//! * laziness api initiated.
//!     - PredicatePushdown Optimizer
//!     - ProjectionPushdown Optimizer
//!     - Selection (filter, where clause)
//!     - Projection (select foo from bar)
//!     - Aggregation (groupby)
//!         - all eager aggregations supported
//!     - Joins
//!     - DSL (col, lit, lt, lt_eq, alias, etc.)

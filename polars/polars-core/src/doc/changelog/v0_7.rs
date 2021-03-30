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
//! * ChunkedArrays broadcasting arithmetic
//! * ChunkedArray/Series `zip_with` operation
//! * ChunkedArray/Series `expand_at_index` operation
//! * laziness api initiated.
//!     - Predicate pushdown optimizer
//!     - Projection pushdown optimizer
//!     - Type coercion optimizer
//!     - Selection (filter, where clause)
//!     - Projection (select foo from bar)
//!     - Aggregation (groupby)
//!         - all eager aggregations supported
//!     - Joins
//!     - WithColumn operation
//!     - DSL
//!         * (col, lit, lt, lt_eq, alias, etc.)
//!         * arithmetic
//!         * when / then /otherwise
//! * 1.3-1.7 performance increase of filter
//! * ChunkedArray/ Series creation speedup: No nulls: 10X speedup, Nulls: 1.1-2.2x speedup.

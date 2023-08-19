//! # Changelog v0.10 / v0.11
//!
//! * CSV Read IO
//!     - Parallel csv reader
//! * Sample DataFrames/ Series
//! * Performance increase in take kernel
//! * Performance increase in ChunkedArray builders
//! * Join operation on multiple columns.
//! * ~3.5 x performance increase in groupby operations (measured on db-benchmark),
//!   due to embarrassingly parallel grouping and better branch prediction (tight loops).
//! * Performance increase on join operation due to better branch prediction.
//! * Categorical datatype and global string cache (BETA).
//!
//! * Lazy
//!     - Lot's of bug fixes in optimizer.
//!     - Parallel execution of Physical plan
//!     - Partition window function
//!     - More simplify expression optimizations.
//!     - Caching
//!     - Alpha release of Aggregate pushdown optimization.
//! * Start of general Object type in ChunkedArray/DataFrames/Series

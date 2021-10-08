//! # Changelog v0.8
//!
//! * Upgrade to Arrow 2.0
//! * Add quantile aggregation to `ChunkedArray`
//! * Option to stop reading CSV after n rows.
//! * Read parquet file in a single batch reducing reading time.
//! * Faster kernel for zip_with and set_with operation
//! * String utilities
//!     - Utf8Chunked::str_lengths method
//!     - Utf8Chunked::contains method
//!     - Utf8Chunked::replace method
//!     - Utf8Chunked::replace_all method
//! * Temporal utilities
//!     - Utf8Chunked to dat32 / datetime
//! * Lazy
//!     - fill_null expression
//!     - shift expression
//!     - Series aggregations
//!     - aggregations on DataFrame level
//!     - aggregate to largelist
//!     - a lot of bugs fixed in optimizers
//!     - UDF's / closures in lazy dsl
//!     - DataFrame reverse operation
//!

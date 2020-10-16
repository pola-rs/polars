//! # Changelog v0.8
//!
//! * Add quantile aggregation to `ChunkedArray`
//! * Option to stop reading CSV after n rows.
//! * Read parquet file in a single batch reducing reading time.
//! * String utilities
//!     - Utf8Chunked::str_lengths method
//!     - Utf8Chunked::contains method
//!     - Utf8Chunked::replace method
//!     - Utf8Chunked::replace_all method
//! * Lazy
//!     - fill_none expression
//!     - shift expression
//!     - Series aggregations

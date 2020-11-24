//! # Changelog v0.9
//!
//! * CSV Read IO
//!     - large performance increase
//!     - skip_rows
//!     - ignore parser errors
//! * Overall performance increase by using aHash in favor of FNV.
//! * Groupby floating point keys
//! * DataFrame operations
//!     - drop_nulls
//!     - drop duplicate rows
//! * Temporal handling
//! * Lazy
//!     - a lot of bug fixes in the optimizer
//!     - start of optimizer framework
//!     - start of simplify expression optimizer
//!     - csv scan
//!     - various operations
//! * Start of general Object type in ChunkedArray/DataFrames/Series

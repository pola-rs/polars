//! Lazy database scans (`scan_database`).
//!
//! The pieces are kept separate on purpose:
//! - a [`dialect::DialectName`] decides how SQL is written and how types map,
//! - a [`backend`] decides how a SQL string becomes Arrow batches,
//! - a [`partition`] decides how the work is split.
//!
//! SQL is built as a dialect-neutral [`sql::SqlQuery`] and turned into text by a
//! [`sql::SqlRenderer`], once, on the planning machine.

pub mod backend;
pub mod catalog;
pub mod config;
pub mod connection;
pub mod dialect;
pub mod options;
pub mod partition;
pub mod planner;
pub mod schema;
pub mod sql;

pub use config::{ColumnMeta, CompiledPartition, CompiledScan, DatabaseScanConfig, TableStats};
pub use connection::{ConnectionSpec, ConnectionTarget};
pub use dialect::DialectName;
pub use options::{
    Consistency, DatabaseRelation, DatabaseScanOptions, FetchOptions, PartitionSpec,
    PartitionStrategy,
};
pub use planner::{PushedFilter, ScanPushdown, compile_scan, resolve_scan};

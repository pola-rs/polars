//! Native database scan plugin for Polars (`scan_database`), loaded through the generic I/O
//! plugin contracts of `polars_io_ext_ffi`:
//! * `polars.io_plugin.lazyframe_resolver.v1` ([`resolver`]): planning. Connects, finds the
//!   schema, translates the pushed filters, splits the scan into partitions and renders their SQL.
//! * `polars.io_plugin.file_reader.v1` ([`reader`]): reads one partition per source.
//!
//! [`database`] holds the database logic (dialects, SQL, partitioning, types, drivers). The Python
//! extension module (feature `python`) exposes the contracts as PyCapsules, the `scan_database`
//! entry point and the Python driver backends.
pub mod database;
pub mod ffi;
pub mod filter;
pub mod reader;
pub mod resolver;

#[cfg(feature = "python")]
pub mod python;

/// Python package name, as recorded in query plans and used in error messages.
pub const PACKAGE_NAME: &str = "polars_database";

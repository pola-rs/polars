#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(ambiguous_glob_reexports)]
extern crate core;

#[cfg(feature = "avro")]
pub mod avro;
pub mod cloud;
#[cfg(any(feature = "csv", feature = "json"))]
pub mod csv;
#[cfg(feature = "file_cache")]
pub mod file_cache;
#[cfg(any(feature = "ipc", feature = "ipc_streaming"))]
pub mod ipc;
#[cfg(feature = "json")]
pub mod json;
pub mod mmap;
#[cfg(feature = "json")]
pub mod ndjson;
mod options;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "parquet")]
pub mod partition;
pub mod path_utils;
#[cfg(feature = "async")]
pub mod pl_async;
pub mod predicates;
pub mod prelude;
mod shared;
pub mod utils;

#[cfg(feature = "cloud")]
pub use cloud::glob as async_glob;
pub use options::*;
pub use path_utils::*;
pub use shared::*;

pub mod hive;

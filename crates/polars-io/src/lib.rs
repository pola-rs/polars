#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
#![allow(ambiguous_glob_reexports)]
extern crate core;

#[cfg(feature = "avro")]
pub mod avro;
#[cfg(feature = "catalog")]
pub mod catalog;
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
pub mod path_utils;
#[cfg(feature = "async")]
pub mod pl_async;
pub mod predicates;
pub mod prelude;
#[cfg(feature = "scan_lines")]
pub mod scan_lines;
mod shared;
pub mod utils;

#[cfg(feature = "cloud")]
pub use cloud::glob as async_glob;
pub use options::*;
pub use path_utils::*;
pub use shared::*;

pub mod hive;

pub fn get_upload_chunk_size() -> usize {
    use std::sync::LazyLock;

    return *UPLOAD_CHUNK_SIZE;

    static UPLOAD_CHUNK_SIZE: LazyLock<usize> = LazyLock::new(|| {
        let v = std::env::var("POLARS_UPLOAD_CHUNK_SIZE")
            .map(|x| {
                x.parse::<usize>()
                    .ok()
                    .filter(|x| *x > 0)
                    .unwrap_or_else(|| panic!("invalid value for POLARS_UPLOAD_CHUNK_SIZE: {x}"))
            })
            .unwrap_or(64 * 1024 * 1024);

        if polars_core::config::verbose() {
            eprintln!("async upload_chunk_size: {v}")
        }

        v
    });
}

pub fn get_upload_concurrency() -> usize {
    use std::sync::LazyLock;

    return *UPLOAD_CONCURRENCY;

    static UPLOAD_CONCURRENCY: LazyLock<usize> = LazyLock::new(|| {
        // Max number of parts concurrently uploaded per Writer.
        // @NOTE. The object_store::BufWriter uses 8 as default.
        let v = std::env::var("POLARS_UPLOAD_CONCURRENCY")
            .map(|x| {
                x.parse::<usize>()
                    .ok()
                    .filter(|x| *x > 0)
                    .unwrap_or_else(|| panic!("invalid value for POLARS_UPLOAD_CONCURRENCY: {x}"))
            })
            .unwrap_or(8);

        if polars_core::config::verbose() {
            eprintln!("async upload_concurrency: {v}")
        }

        v
    });
}

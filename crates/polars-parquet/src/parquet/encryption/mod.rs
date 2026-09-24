//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

// TODO: Remove once encryption is used by the reader.
#![allow(dead_code)]

mod ciphers;
pub mod decrypt;
pub mod encrypt;
mod modules;

/// Path to one of the encrypted Parquet files from the parquet-testing repository.
#[cfg(test)]
pub(crate) fn test_file_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../py-polars/tests/unit/io/files/parquet-encryption")
        .join(name)
}

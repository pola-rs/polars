//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

mod ciphers;
pub mod decrypt;
// TODO: Remove once encrypted writing is implemented.
#[allow(dead_code)]
pub mod encrypt;
mod modules;

/// Path to one of the encrypted Parquet files from the parquet-testing repository.
#[cfg(test)]
pub(crate) fn test_file_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../py-polars/tests/unit/io/files/parquet-encryption")
        .join(name)
}

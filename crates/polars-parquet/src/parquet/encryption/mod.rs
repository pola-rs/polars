//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

mod ciphers;
pub mod decrypt;
pub mod encrypt;
pub mod iceberg;
pub(crate) mod modules;
#[cfg(test)]
mod write_tests;

pub use ciphers::AesGcmBlockDecryptor as AesGcmDecryptor;

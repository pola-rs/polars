//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

mod ciphers;
pub mod decrypt;
// TODO: Remove once encrypted writing is implemented.
#[allow(dead_code)]
pub mod encrypt;
mod modules;

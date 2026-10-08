//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

mod ciphers;
pub mod decrypt;
pub mod encrypt;
mod modules;
#[cfg(test)]
mod write_tests;

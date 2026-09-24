//! Implements Parquet Modular Encryption.
//! See <https://github.com/apache/parquet-format/blob/master/Encryption.md> for the specification.

// TODO: Remove once encryption is used by the reader.
#![allow(dead_code)]

mod ciphers;
pub mod decrypt;
pub mod encrypt;
mod modules;

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(clippy::len_without_is_empty)]
pub mod arrow;
pub use crate::arrow::{read, write};
pub mod parquet;

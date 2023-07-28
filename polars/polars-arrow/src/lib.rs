#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(
    feature = "nightly",
    allow(clippy::incorrect_partial_ord_impl_on_ord_type)
)] // remove once stable
pub mod array;
pub mod bit_util;
pub mod bitmap;
pub mod compute;
pub mod conversion;
pub mod data_types;
pub mod error;
pub mod export;
pub mod floats;
pub mod index;
pub mod is_valid;
pub mod kernels;
pub mod prelude;
pub mod slice;
pub mod time_zone;
pub mod trusted_len;
pub mod utils;

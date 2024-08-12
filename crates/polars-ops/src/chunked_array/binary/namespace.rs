#[cfg(feature = "binary_encoding")]
use std::borrow::Cow;

#[cfg(feature = "binary_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "binary_encoding")]
use base64::Engine as _;
use memchr::memmem::find;
use polars_compute::size::binary_size_bytes;
use polars_core::prelude::arity::{broadcast_binary_elementwise_values, unary_elementwise_values};

use super::*;

pub trait BinaryNameSpaceImpl: AsBinary {
    /// Check if binary contains given literal
    fn contains(&self, lit: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| find(s, lit).is_some();
        unary_elementwise_values(ca, f)
    }

    fn contains_chunked(&self, lit: &BinaryChunked) -> BooleanChunked {
        let ca = self.as_binary();
        match lit.len() {
            1 => match lit.get(0) {
                Some(lit) => ca.contains(lit),
                None => BooleanChunked::full_null(ca.name(), ca.len()),
            },
            _ => broadcast_binary_elementwise_values(ca, lit, |src, lit| find(src, lit).is_some()),
        }
    }

    /// Check if strings ends with a substring
    fn ends_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.ends_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }

    /// Check if strings starts with a substring
    fn starts_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.starts_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }

    fn starts_with_chunked(&self, prefix: &BinaryChunked) -> BooleanChunked {
        let ca = self.as_binary();
        match prefix.len() {
            1 => match prefix.get(0) {
                Some(s) => self.starts_with(s),
                None => BooleanChunked::full_null(ca.name(), ca.len()),
            },
            _ => broadcast_binary_elementwise_values(ca, prefix, |s, sub| s.starts_with(sub)),
        }
    }

    fn ends_with_chunked(&self, suffix: &BinaryChunked) -> BooleanChunked {
        let ca = self.as_binary();
        match suffix.len() {
            1 => match suffix.get(0) {
                Some(s) => self.ends_with(s),
                None => BooleanChunked::full_null(ca.name(), ca.len()),
            },
            _ => broadcast_binary_elementwise_values(ca, suffix, |s, sub| s.ends_with(sub)),
        }
    }

    /// Get the size of the binary values in bytes.
    fn size_bytes(&self) -> UInt32Chunked {
        let ca = self.as_binary();
        ca.apply_kernel_cast(&binary_size_bytes)
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply_nonnull_values_generic(|s| {
                hex::decode(s).map_err(|_| {
                    polars_err!(
                        ComputeError:
                        "invalid `hex` encoding found; try setting `strict=false` to ignore"
                    )
                })
            })
        } else {
            Ok(ca.apply(|opt_s| opt_s.and_then(|s| hex::decode(s).ok().map(Cow::Owned))))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_encode(&self) -> Series {
        let ca = self.as_binary();
        unsafe {
            ca.apply_values(|s| hex::encode(s).into_bytes().into())
                .cast_unchecked(&DataType::String)
                .unwrap()
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply_nonnull_values_generic(|s| {
                general_purpose::STANDARD.decode(s).map_err(|_e| {
                    polars_err!(
                        ComputeError:
                        "invalid `base64` encoding found; try setting `strict=false` to ignore"
                    )
                })
            })
        } else {
            Ok(ca.apply(|opt_s| {
                opt_s.and_then(|s| general_purpose::STANDARD.decode(s).ok().map(Cow::Owned))
            }))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_encode(&self) -> Series {
        let ca = self.as_binary();
        unsafe {
            ca.apply_values(|s| general_purpose::STANDARD.encode(s).into_bytes().into())
                .cast_unchecked(&DataType::String)
                .unwrap()
        }
    }
}

impl BinaryNameSpaceImpl for BinaryChunked {}

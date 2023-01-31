#[cfg(feature = "binary_encoding")]
use std::borrow::Cow;

#[cfg(feature = "binary_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "binary_encoding")]
use base64::Engine as _;
use memchr::memmem::find;

use super::*;

pub trait BinaryNameSpaceImpl: AsBinary {
    /// Check if binary contains given literal
    fn contains(&self, lit: &[u8]) -> PolarsResult<BooleanChunked> {
        let ca = self.as_binary();
        let f = |s: &[u8]| find(s, lit).is_some();
        let mut out: BooleanChunked = if !ca.has_validity() {
            ca.into_no_null_iter().map(f).collect()
        } else {
            ca.into_iter().map(|opt_s| opt_s.map(f)).collect()
        };
        out.rename(ca.name());
        Ok(out)
    }

    /// Check if strings contain a given literal
    fn contains_literal(&self, lit: &[u8]) -> PolarsResult<BooleanChunked> {
        self.contains(lit)
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

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply(|s| {
                let bytes = hex::decode(s).map_err(|_e| {
                    PolarsError::ComputeError(
                        "Invalid 'hex' encoding found. Try setting 'strict' to false to ignore."
                            .into(),
                    )
                })?;
                Ok(bytes.into())
            })
        } else {
            Ok(ca.apply_on_opt(|opt_s| opt_s.and_then(|s| hex::decode(s).ok().map(Cow::Owned))))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_encode(&self) -> Series {
        let ca = self.as_binary();
        ca.apply(|s| hex::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply(|s| {
                let bytes = general_purpose::STANDARD.decode(s).map_err(|_e| {
                    PolarsError::ComputeError(
                        "Invalid 'base64' encoding found. Try setting 'strict' to false to ignore."
                            .into(),
                    )
                })?;
                Ok(bytes.into())
            })
        } else {
            Ok(ca.apply_on_opt(|opt_s| {
                opt_s.and_then(|s| general_purpose::STANDARD.decode(s).ok().map(Cow::Owned))
            }))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_encode(&self) -> Series {
        let ca = self.as_binary();
        ca.apply(|s| general_purpose::STANDARD.encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }
}

impl BinaryNameSpaceImpl for BinaryChunked {}

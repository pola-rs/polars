use base64::engine::general_purpose;
use base64::Engine as _;
use hex;

use crate::prelude::*;

impl Utf8Chunked {
    #[cfg(not(feature = "binary_encoding"))]
    pub fn hex_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'dtype-binary' feature")
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .hex_decode(strict)
    }

    #[must_use]
    pub fn hex_encode(&self) -> Utf8Chunked {
        self.apply(|s| hex::encode(s).into())
    }

    #[cfg(not(feature = "binary_encoding"))]
    pub fn base64_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'dtype-binary' feature")
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .base64_decode(strict)
    }

    #[must_use]
    pub fn base64_encode(&self) -> Utf8Chunked {
        self.apply(|s| general_purpose::STANDARD.encode(s).into())
    }
}

use {base64, hex};

use crate::prelude::*;

impl Utf8Chunked {
    #[cfg(not(feature = "binary_encoding"))]
    pub fn hex_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'dtype-binary' feature")
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_decode(&self) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .hex_decode()
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
    pub fn base64_decode(&self) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .base64_decode()
    }

    #[must_use]
    pub fn base64_encode(&self) -> Utf8Chunked {
        self.apply(|s| base64::encode(s).into())
    }
}

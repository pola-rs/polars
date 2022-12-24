use {base64, hex};

use crate::prelude::*;

impl Utf8Chunked {
    #[cfg(feature = "string_encoding")]
    pub fn hex_decode(&self, strict: Option<bool>) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .hex_decode(strict)
    }
    #[cfg(feature = "string_encoding")]
    #[must_use]
    pub fn hex_encode(&self) -> Utf8Chunked {
        self.apply(|s| hex::encode(s).into())
    }

    #[cfg(feature = "string_encoding")]
    pub fn base64_decode(&self, strict: Option<bool>) -> PolarsResult<BinaryChunked> {
        self.cast_unchecked(&DataType::Binary)?
            .binary()?
            .base64_decode(strict)
    }

    #[cfg(feature = "string_encoding")]
    #[must_use]
    pub fn base64_encode(&self) -> Utf8Chunked {
        self.apply(|s| base64::encode(s).into())
    }
}

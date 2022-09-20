use {base64, hex};

use crate::prelude::*;

impl Utf8Chunked {
    #[cfg(feature = "string_encoding")]
    pub fn hex_decode(&self, strict: Option<bool>) -> PolarsResult<Utf8Chunked> {
        let ca = self.apply_on_opt(|e| {
            e.and_then(|s| {
                hex::decode(s)
                    // Safety
                    // We already know that it is a valid utf8.
                    .map(|bytes| Some(unsafe { String::from_utf8_unchecked(bytes) }.into()))
                    .unwrap_or(None)
            })
        });

        if strict.unwrap_or(false) && (ca.null_count() != self.null_count()) {
            Err(PolarsError::ComputeError("Unable to decode inputs".into()))
        } else {
            Ok(ca)
        }
    }
    #[cfg(feature = "string_encoding")]
    #[must_use]
    pub fn hex_encode(&self) -> Utf8Chunked {
        self.apply(|s| hex::encode(s).into())
    }

    #[cfg(feature = "string_encoding")]
    pub fn base64_decode(&self, strict: Option<bool>) -> PolarsResult<Utf8Chunked> {
        let ca = self.apply_on_opt(|e| {
            e.and_then(|s| {
                base64::decode(s)
                    // Safety
                    // We already know that it is a valid utf8.
                    .map(|bytes| Some(unsafe { String::from_utf8_unchecked(bytes) }.into()))
                    .unwrap_or(None)
            })
        });

        if strict.unwrap_or(false) && (ca.null_count() != self.null_count()) {
            Err(PolarsError::ComputeError("Unable to decode inputs".into()))
        } else {
            Ok(ca)
        }
    }

    #[cfg(feature = "string_encoding")]
    #[must_use]
    pub fn base64_encode(&self) -> Utf8Chunked {
        self.apply(|s| base64::encode(s).into())
    }
}

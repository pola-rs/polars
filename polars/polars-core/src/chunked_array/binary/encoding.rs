use {base64, hex};

use crate::prelude::*;

impl BinaryChunked {
    pub fn hex_decode(&self, strict: Option<bool>) -> PolarsResult<BinaryChunked> {
        let ca = self.apply_on_opt(|e| e.and_then(|s| hex::decode(s).map(Into::into).ok()));

        if strict.unwrap_or(false) && (ca.null_count() != self.null_count()) {
            Err(PolarsError::ComputeError("Unable to decode inputs".into()))
        } else {
            Ok(ca)
        }
    }

    pub fn hex_encode(&self) -> Series {
        self.apply(|s| hex::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }

    pub fn base64_decode(&self, strict: Option<bool>) -> PolarsResult<BinaryChunked> {
        let ca = self.apply_on_opt(|e| e.and_then(|s| base64::decode(s).map(Into::into).ok()));

        if strict.unwrap_or(false) && (ca.null_count() != self.null_count()) {
            Err(PolarsError::ComputeError("Unable to decode inputs".into()))
        } else {
            Ok(ca)
        }
    }

    pub fn base64_encode(&self) -> Series {
        self.apply(|s| base64::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }
}

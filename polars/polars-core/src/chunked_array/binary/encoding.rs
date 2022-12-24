use {base64, hex};

use crate::prelude::*;

impl BinaryChunked {
    pub fn hex_decode(&self) -> PolarsResult<BinaryChunked> {
        self.try_apply(|s| {
            let bytes =
                hex::decode(s).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            Ok(bytes.into())
        })
    }

    pub fn hex_encode(&self) -> Series {
        self.apply(|s| hex::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }

    pub fn base64_decode(&self) -> PolarsResult<BinaryChunked> {
        self.try_apply(|s| {
            let bytes =
                base64::decode(s).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            Ok(bytes.into())
        })
    }

    pub fn base64_encode(&self) -> Series {
        self.apply(|s| base64::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }
}

use std::borrow::Cow;

use base64::engine::general_purpose;
use base64::Engine as _;
use hex;

use crate::prelude::*;

impl BinaryChunked {
    pub fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        if strict {
            self.try_apply(|s| {
                let bytes = hex::decode(s).map_err(|_e| {
                    PolarsError::ComputeError(
                        "Invalid 'hex' encoding found. Try setting 'strict' to false to ignore."
                            .into(),
                    )
                })?;
                Ok(bytes.into())
            })
        } else {
            Ok(self.apply_on_opt(|opt_s| opt_s.and_then(|s| hex::decode(s).ok().map(Cow::Owned))))
        }
    }

    pub fn hex_encode(&self) -> Series {
        self.apply(|s| hex::encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }

    pub fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        if strict {
            self.try_apply(|s| {
                let bytes = general_purpose::STANDARD.decode(s).map_err(|_e| {
                    PolarsError::ComputeError(
                        "Invalid 'base64' encoding found. Try setting 'strict' to false to ignore."
                            .into(),
                    )
                })?;
                Ok(bytes.into())
            })
        } else {
            Ok(self.apply_on_opt(|opt_s| {
                opt_s.and_then(|s| general_purpose::STANDARD.decode(s).ok().map(Cow::Owned))
            }))
        }
    }

    pub fn base64_encode(&self) -> Series {
        self.apply(|s| general_purpose::STANDARD.encode(s).into_bytes().into())
            .cast_unchecked(&DataType::Utf8)
            .unwrap()
    }
}

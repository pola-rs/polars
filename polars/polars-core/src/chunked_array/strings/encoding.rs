use crate::prelude::*;
use std::borrow::Cow;

use hex;
use base64;

impl Utf8Chunked {
  #[cfg(feature = "string_encoding")]
  pub fn hex_decode(&self, strict: Option<bool>) -> Result<Utf8Chunked> {
    self.try_apply(|s: &str| {
      let decoded = hex::decode(s);

      match decoded {
        Ok(v) => {
          let utf8 = String::from_utf8(v).unwrap();
          Ok(Cow::Owned(utf8))
        }
        Err(_) => {
          if strict.unwrap_or(false) {
            Err(PolarsError::ValueError("unable to decode input".into()))
          } else {
            Ok(Cow::Borrowed(s))
          }
        },
      }
    })
  }
  #[cfg(feature = "string_encoding")]
  pub fn hex_encode(&self) -> Utf8Chunked {
    self.apply(|s| hex::encode(s).into())
  }

  #[cfg(feature = "string_encoding")]
  pub fn base64_decode(&self, strict: Option<bool>) -> Result<Utf8Chunked> {
    self.try_apply(|s: &str| {
      let decoded = base64::decode(s);

      match decoded {
        Ok(v) => {
          let utf8 = String::from_utf8(v).unwrap();
          Ok(Cow::Owned(utf8))
        }
        Err(_) => {
          if strict.unwrap_or(false) {
            Err(PolarsError::ValueError("unable to decode input".into()))
          } else {
            Ok(Cow::Borrowed(s))
          }

        },
      }
    })
  }

  #[cfg(feature = "string_encoding")]
  pub fn base64_encode(&self) -> Utf8Chunked {
    self.apply(|s| base64::encode(s).into())
  }

}

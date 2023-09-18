//! APIs to read from [ORC format](https://orc.apache.org).
pub mod read;

pub use orc_format as format;

use crate::error::Error;

impl From<format::error::Error> for Error {
    fn from(error: format::error::Error) -> Self {
        Error::ExternalFormat(format!("{error:?}"))
    }
}

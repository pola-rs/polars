//! Convert data between the Arrow and CSV (comma-separated values).

use crate::error::Error;

#[cfg(any(feature = "io_csv_read_async", feature = "io_csv_read"))]
mod read_utils;
#[cfg(any(feature = "io_csv_read_async", feature = "io_csv_read"))]
mod utils;

#[cfg(feature = "io_csv_read")]
impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        Error::External("".to_string(), Box::new(error))
    }
}

impl From<chrono::ParseError> for Error {
    fn from(error: chrono::ParseError) -> Self {
        Error::External("".to_string(), Box::new(error))
    }
}

#[cfg(feature = "io_csv_read")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_csv_read")))]
pub mod read;
#[cfg(feature = "io_csv_write")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_csv_write")))]
pub mod write;

#[cfg(feature = "io_csv_read_async")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_csv_read_async")))]
pub mod read_async;

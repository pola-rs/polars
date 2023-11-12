//! API to serialize and deserialize data from and to ODBC
pub use odbc_api as api;

pub mod read;
pub mod write;

impl From<api::Error> for crate::error::Error {
    fn from(error: api::Error) -> Self {
        crate::error::Error::External("".to_string(), Box::new(error))
    }
}

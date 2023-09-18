//! APIs to read and deserialize from JSON
mod deserialize;
mod infer_schema;

pub(crate) use deserialize::_deserialize;
pub use deserialize::{deserialize, deserialize_records};
pub(crate) use infer_schema::coerce_data_type;
pub use infer_schema::{infer, infer_records_schema};

pub use json_deserializer;

use crate::error::Error;

impl From<json_deserializer::Error> for Error {
    fn from(error: json_deserializer::Error) -> Self {
        Error::ExternalFormat(error.to_string())
    }
}

pub mod deserialize;
pub(crate) mod infer_schema;

pub use deserialize::deserialize;
pub use infer_schema::infer;
use polars_error::*;
use polars_utils::aliases::*;
pub mod write;

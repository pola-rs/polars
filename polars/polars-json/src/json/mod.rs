pub mod deserialize;
pub mod infer_schema;

use polars_error::*;
use polars_utils::aliases::*;
pub use infer_schema::{
    infer, infer_records_schema
};

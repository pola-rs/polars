use arrow::datatypes::*;
use polars_arrow::prelude::*;
use polars_error::*;
pub mod deserialize;
mod file;

pub use file::{infer, infer_iter};

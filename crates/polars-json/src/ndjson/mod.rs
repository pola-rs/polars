use arrow::datatypes::*;
use arrow::array::ArrayRef;
use arrow::legacy::prelude::*;
use polars_error::*;
pub mod deserialize;
mod file;
pub mod write;

pub use file::{infer, infer_iter};

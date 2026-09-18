use polars_arrow::array::ArrayRef;
use polars_arrow::datatypes::*;
use polars_error::*;
pub mod deserialize;
mod file;
pub mod write;
pub use file::{infer_iter, iter_unique_dtypes};

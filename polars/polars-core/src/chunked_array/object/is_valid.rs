use super::{ObjectArray, PolarsObject};
use polars_arrow::is_valid::ArrowArray;

impl<T: PolarsObject> ArrowArray for ObjectArray<T> {}

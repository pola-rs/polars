use arrow::legacy::is_valid::ArrowArray;

use super::{ObjectArray, PolarsObject};

impl<T: PolarsObject> ArrowArray for ObjectArray<T> {}

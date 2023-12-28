use arrow::array::Array;
use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::bitmap::Bitmap;

use crate::chunked_array::object::{ObjectArray, ObjectValueIter};
use crate::prelude::*;

impl<T: PolarsObject> StaticArray for ObjectArray<T> {
    type ValueT<'a> = &'a T;
    type ZeroableValueT<'a> = Option<&'a T>;
    type ValueIterT<'a> = ObjectValueIter<'a, T>;

    #[inline]
    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        self.value_unchecked(idx)
    }

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    fn full_null(_length: usize, _dtype: ArrowDataType) -> Self {
        panic!("ObjectArray does not support full_null");
    }
}

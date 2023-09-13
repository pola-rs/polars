use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::bitmap::Bitmap;

#[cfg(feature = "object")]
use crate::chunked_array::object::{ObjectArray, ObjectValueIter};
use crate::datatypes::static_array_collect::ArrayFromIterDtype;
use crate::prelude::*;

pub trait StaticArray:
    Array
    + for<'a> ArrayFromIterDtype<Self::ValueT<'a>>
    + for<'a> ArrayFromIterDtype<Option<Self::ValueT<'a>>>
{
    type ValueT<'a>
    where
        Self: 'a;
    type ValueIterT<'a>: Iterator<Item = Self::ValueT<'a>>
        + TrustedLen
        + arrow::trusted_len::TrustedLen
    where
        Self: 'a;

    #[inline]
    fn get(&self, idx: usize) -> Option<Self::ValueT<'_>> {
        if idx >= self.len() {
            None
        } else {
            unsafe { self.get_unchecked(idx) }
        }
    }

    /// # Safety
    /// It is the callers responsibility that the `idx < self.len()`.
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<Self::ValueT<'_>> {
        if self.is_null_unchecked(idx) {
            None
        } else {
            Some(self.value_unchecked(idx))
        }
    }

    #[inline]
    fn last(&self) -> Option<Self::ValueT<'_>> {
        unsafe { self.get_unchecked(self.len().checked_sub(1)?) }
    }

    #[inline]
    fn value(&self, idx: usize) -> Self::ValueT<'_> {
        assert!(idx < self.len());
        unsafe { self.value_unchecked(idx) }
    }

    /// # Safety
    /// It is the callers responsibility that the `idx < self.len()`.
    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_>;

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter>;
    fn values_iter(&self) -> Self::ValueIterT<'_>;
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self;
}

pub trait ParameterFreeDtypeStaticArray: StaticArray {
    fn get_dtype() -> DataType;
}

impl<T: NumericNative> StaticArray for PrimitiveArray<T> {
    type ValueT<'a> = T;
    type ValueIterT<'a> = std::iter::Copied<std::slice::Iter<'a, T>>;

    #[inline]
    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        self.value_unchecked(idx)
    }

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter().copied()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        ZipValidity::new_with_validity(self.values().iter().copied(), self.validity())
    }

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }
}

impl<T: NumericNative> ParameterFreeDtypeStaticArray for PrimitiveArray<T> {
    fn get_dtype() -> DataType {
        T::PolarsType::get_dtype()
    }
}

impl StaticArray for BooleanArray {
    type ValueT<'a> = bool;
    type ValueIterT<'a> = BitmapIter<'a>;

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
}

impl ParameterFreeDtypeStaticArray for BooleanArray {
    fn get_dtype() -> DataType {
        DataType::Boolean
    }
}

impl StaticArray for Utf8Array<i64> {
    type ValueT<'a> = &'a str;
    type ValueIterT<'a> = Utf8ValuesIter<'a, i64>;

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
}

impl ParameterFreeDtypeStaticArray for Utf8Array<i64> {
    fn get_dtype() -> DataType {
        DataType::Utf8
    }
}

impl StaticArray for BinaryArray<i64> {
    type ValueT<'a> = &'a [u8];
    type ValueIterT<'a> = BinaryValueIter<'a, i64>;

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
}

impl ParameterFreeDtypeStaticArray for BinaryArray<i64> {
    fn get_dtype() -> DataType {
        DataType::Binary
    }
}

impl StaticArray for ListArray<i64> {
    type ValueT<'a> = Box<dyn Array>;
    type ValueIterT<'a> = ListValuesIter<'a, i64>;

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
}

#[cfg(feature = "dtype-array")]
impl StaticArray for FixedSizeListArray {
    type ValueT<'a> = Box<dyn Array>;
    type ValueIterT<'a> = ArrayValuesIter<'a, FixedSizeListArray>;

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
}

#[cfg(feature = "object")]
impl<T: PolarsObject> StaticArray for ObjectArray<T> {
    type ValueT<'a> = &'a T;
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
}

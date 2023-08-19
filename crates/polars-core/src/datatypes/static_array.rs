use arrow::bitmap::utils::{BitmapIter, ZipValidity};

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;

pub trait StaticArray: Array {
    type ValueT<'a>
    where
        Self: 'a;
    type ValueIterT<'a>: Iterator<Item = Self::ValueT<'a>>
        + TrustedLen
        + arrow::trusted_len::TrustedLen
    where
        Self: 'a;

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter>;
    fn values_iter(&self) -> Self::ValueIterT<'_>;
}

impl<T: NumericNative> StaticArray for PrimitiveArray<T> {
    type ValueT<'a> = &'a T;
    type ValueIterT<'a> = std::slice::Iter<'a, T>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

impl StaticArray for BooleanArray {
    type ValueT<'a> = bool;
    type ValueIterT<'a> = BitmapIter<'a>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

impl StaticArray for Utf8Array<i64> {
    type ValueT<'a> = &'a str;
    type ValueIterT<'a> = Utf8ValuesIter<'a, i64>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

impl StaticArray for BinaryArray<i64> {
    type ValueT<'a> = &'a [u8];
    type ValueIterT<'a> = BinaryValueIter<'a, i64>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

impl StaticArray for ListArray<i64> {
    type ValueT<'a> = Box<dyn Array>;
    type ValueIterT<'a> = ListValuesIter<'a, i64>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

#[cfg(feature = "dtype-array")]
impl StaticArray for FixedSizeListArray {
    type ValueT<'a> = Box<dyn Array>;
    type ValueIterT<'a> = ArrayValuesIter<'a, FixedSizeListArray>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> StaticArray for ObjectArray<T> {
    type ValueT<'a> = &'a ();
    type ValueIterT<'a> = std::slice::Iter<'a, ()>;

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        todo!()
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        todo!()
    }
}

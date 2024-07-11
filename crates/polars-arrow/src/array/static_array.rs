use bytemuck::Zeroable;
use polars_utils::no_call_const;

use crate::array::binview::BinaryViewValueIter;
use crate::array::growable::{Growable, GrowableFixedSizeList};
use crate::array::static_array_collect::ArrayFromIterDtype;
use crate::array::{
    Array, ArrayValuesIter, BinaryArray, BinaryValueIter, BinaryViewArray, BooleanArray,
    FixedSizeListArray, ListArray, ListValuesIter, MutableBinaryViewArray, PrimitiveArray,
    StructArray, Utf8Array, Utf8ValuesIter, Utf8ViewArray,
};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::bitmap::Bitmap;
use crate::datatypes::ArrowDataType;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

pub trait StaticArray:
    Array
    + for<'a> ArrayFromIterDtype<Self::ValueT<'a>>
    + for<'a> ArrayFromIterDtype<Self::ZeroableValueT<'a>>
    + for<'a> ArrayFromIterDtype<Option<Self::ValueT<'a>>>
    + Clone
{
    type ValueT<'a>: Clone
    where
        Self: 'a;
    type ZeroableValueT<'a>: Zeroable + From<Self::ValueT<'a>>
    where
        Self: 'a;
    type ValueIterT<'a>: DoubleEndedIterator<Item = Self::ValueT<'a>> + TrustedLen + Send + Sync
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
    #[allow(unused_variables)]
    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        no_call_const!()
    }

    #[inline(always)]
    fn as_slice(&self) -> Option<&[Self::ValueT<'_>]> {
        None
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        no_call_const!()
    }
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        no_call_const!()
    }
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self;

    fn from_vec(v: Vec<Self::ValueT<'_>>, dtype: ArrowDataType) -> Self {
        Self::arr_from_iter_with_dtype(dtype, v)
    }

    fn from_zeroable_vec(v: Vec<Self::ZeroableValueT<'_>>, dtype: ArrowDataType) -> Self {
        Self::arr_from_iter_with_dtype(dtype, v)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self;

    fn full(length: usize, value: Self::ValueT<'_>, dtype: ArrowDataType) -> Self {
        Self::arr_from_iter_with_dtype(dtype, std::iter::repeat(value).take(length))
    }
}

pub trait ParameterFreeDtypeStaticArray: StaticArray {
    fn get_dtype() -> ArrowDataType;
}

impl<T: NativeType> StaticArray for PrimitiveArray<T> {
    type ValueT<'a> = T;
    type ZeroableValueT<'a> = T;
    type ValueIterT<'a> = std::iter::Copied<std::slice::Iter<'a, T>>;

    #[inline]
    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        self.value_unchecked(idx)
    }

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter().copied()
    }

    #[inline(always)]
    fn as_slice(&self) -> Option<&[Self::ValueT<'_>]> {
        Some(self.values().as_slice())
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        ZipValidity::new_with_validity(self.values().iter().copied(), self.validity())
    }

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    fn from_vec(v: Vec<Self::ValueT<'_>>, _dtype: ArrowDataType) -> Self {
        PrimitiveArray::from_vec(v)
    }

    fn from_zeroable_vec(v: Vec<Self::ZeroableValueT<'_>>, _dtype: ArrowDataType) -> Self {
        PrimitiveArray::from_vec(v)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }

    fn full(length: usize, value: Self::ValueT<'_>, _dtype: ArrowDataType) -> Self {
        PrimitiveArray::from_vec(vec![value; length])
    }
}

impl<T: NativeType> ParameterFreeDtypeStaticArray for PrimitiveArray<T> {
    fn get_dtype() -> ArrowDataType {
        T::PRIMITIVE.into()
    }
}

impl StaticArray for BooleanArray {
    type ValueT<'a> = bool;
    type ZeroableValueT<'a> = bool;
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

    fn from_vec(v: Vec<Self::ValueT<'_>>, _dtype: ArrowDataType) -> Self {
        BooleanArray::from_slice(v)
    }

    fn from_zeroable_vec(v: Vec<Self::ValueT<'_>>, _dtype: ArrowDataType) -> Self {
        BooleanArray::from_slice(v)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }

    fn full(length: usize, value: Self::ValueT<'_>, _dtype: ArrowDataType) -> Self {
        Bitmap::new_with_value(value, length).into()
    }
}

impl ParameterFreeDtypeStaticArray for BooleanArray {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::Boolean
    }
}

impl StaticArray for Utf8Array<i64> {
    type ValueT<'a> = &'a str;
    type ZeroableValueT<'a> = Option<&'a str>;
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

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }
}

impl ParameterFreeDtypeStaticArray for Utf8Array<i64> {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::LargeUtf8
    }
}

impl StaticArray for BinaryArray<i64> {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
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

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }
}

impl ParameterFreeDtypeStaticArray for BinaryArray<i64> {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::LargeBinary
    }
}

impl StaticArray for BinaryViewArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = BinaryViewValueIter<'a, [u8]>;

    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        self.value_unchecked(idx)
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }

    fn full(length: usize, value: Self::ValueT<'_>, _dtype: ArrowDataType) -> Self {
        let mut builder = MutableBinaryViewArray::with_capacity(length);
        builder.extend_constant(length, Some(value));
        builder.into()
    }
}

impl ParameterFreeDtypeStaticArray for BinaryViewArray {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::BinaryView
    }
}

impl StaticArray for Utf8ViewArray {
    type ValueT<'a> = &'a str;
    type ZeroableValueT<'a> = Option<&'a str>;
    type ValueIterT<'a> = BinaryViewValueIter<'a, str>;

    unsafe fn value_unchecked(&self, idx: usize) -> Self::ValueT<'_> {
        self.value_unchecked(idx)
    }

    fn iter(&self) -> ZipValidity<Self::ValueT<'_>, Self::ValueIterT<'_>, BitmapIter> {
        self.iter()
    }

    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }

    fn full(length: usize, value: Self::ValueT<'_>, _dtype: ArrowDataType) -> Self {
        unsafe {
            BinaryViewArray::full(length, value.as_bytes(), ArrowDataType::BinaryView)
                .to_utf8view_unchecked()
        }
    }
}

impl ParameterFreeDtypeStaticArray for Utf8ViewArray {
    fn get_dtype() -> ArrowDataType {
        ArrowDataType::Utf8View
    }
}

impl StaticArray for ListArray<i64> {
    type ValueT<'a> = Box<dyn Array>;
    type ZeroableValueT<'a> = Option<Box<dyn Array>>;
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

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }
}

impl StaticArray for FixedSizeListArray {
    type ValueT<'a> = Box<dyn Array>;
    type ZeroableValueT<'a> = Option<Box<dyn Array>>;
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

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }

    fn full(length: usize, value: Self::ValueT<'_>, dtype: ArrowDataType) -> Self {
        let singular_arr = FixedSizeListArray::new(dtype, value, None);
        let mut arr = GrowableFixedSizeList::new(vec![&singular_arr], false, length);
        unsafe { arr.extend_copies(0, 0, 1, length) }
        arr.into()
    }
}

impl StaticArray for StructArray {
    type ValueT<'a> = ();
    type ZeroableValueT<'a> = ();
    type ValueIterT<'a> = std::iter::Repeat<()>;

    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    fn full_null(length: usize, dtype: ArrowDataType) -> Self {
        Self::new_null(dtype, length)
    }
}

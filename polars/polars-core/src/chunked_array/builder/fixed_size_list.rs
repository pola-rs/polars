use arrow::array::{
    Array, MutableArray, MutableFixedSizeListArray, MutablePrimitiveArray, PrimitiveArray,
    PushUnchecked,
};
use arrow::types::NativeType;
use polars_utils::unwrap::UnwrapUncheckedRelease;
use smartstring::alias::String as SmartString;

use crate::prelude::*;

pub(crate) struct FixedSizeListNumericBuilder<T: NativeType> {
    inner: Option<MutableFixedSizeListArray<MutablePrimitiveArray<T>>>,
    width: usize,
    name: SmartString,
}

impl<T: NativeType> FixedSizeListNumericBuilder<T> {
    pub(crate) fn new(name: &str, width: usize, capacity: usize) -> Self {
        let mp = MutablePrimitiveArray::<T>::with_capacity(capacity * width);
        let inner = Some(MutableFixedSizeListArray::new(mp, width));
        Self {
            inner,
            width,
            name: name.into(),
        }
    }
}

pub(crate) trait FixedSizeListBuilder {
    unsafe fn push_unchecked(&mut self, arr: &dyn Array, offset: usize);
    unsafe fn push_null(&mut self);
    fn finish(&mut self) -> ArrayChunked;
}

impl<T: NativeType> FixedSizeListBuilder for FixedSizeListNumericBuilder<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, arr: &dyn Array, offset: usize) {
        let start = offset * self.width;
        let end = start + self.width;
        let arr = arr
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap_unchecked_release();
        let inner = self.inner.as_mut().unwrap_unchecked_release();

        let values = arr.values().as_slice();
        let validity = arr.validity();
        if let Some(validity) = validity {
            let iter = (start..end).map(|i| {
                if validity.get_bit_unchecked(i) {
                    Some(*values.get_unchecked(i))
                } else {
                    None
                }
            });
            inner.push_unchecked(Some(iter))
        } else {
            let iter = (start..end).map(|i| Some(*values.get_unchecked(i)));
            inner.push_unchecked(Some(iter))
        }
    }

    #[inline]
    unsafe fn push_null(&mut self) {
        let inner = self.inner.as_mut().unwrap_unchecked_release();
        inner.push_null()
    }

    fn finish(&mut self) -> ArrayChunked {
        let arr: FixedSizeListArray = self.inner.take().unwrap().into();
        unsafe { ChunkedArray::from_chunks(self.name.as_str(), vec![Box::new(arr) as ArrayRef]) }
    }
}

pub(crate) struct AnonymousOwnedFixedSizeListBuilder {
    inner: fixed_size_list::AnonymousBuilder,
    name: SmartString,
    inner_dtype: Option<DataType>,
}

impl AnonymousOwnedFixedSizeListBuilder {
    pub(crate) fn new(
        name: &str,
        width: usize,
        capacity: usize,
        inner_dtype: Option<DataType>,
    ) -> Self {
        let inner = fixed_size_list::AnonymousBuilder::new(capacity, width);
        Self {
            inner,
            name: name.into(),
            inner_dtype,
        }
    }
}

impl FixedSizeListBuilder for AnonymousOwnedFixedSizeListBuilder {
    #[inline]
    unsafe fn push_unchecked(&mut self, arr: &dyn Array, offset: usize) {
        let arr = arr.sliced_unchecked(offset * self.inner.width, self.inner.width);
        self.inner.push(arr)
    }

    #[inline]
    unsafe fn push_null(&mut self) {
        self.inner.push_null()
    }

    fn finish(&mut self) -> ArrayChunked {
        let arr = std::mem::take(&mut self.inner)
            .finish(self.inner_dtype.as_ref().map(|dt| dt.to_arrow()).as_ref())
            .unwrap();
        unsafe { ChunkedArray::from_chunks(self.name.as_str(), vec![Box::new(arr) as ArrayRef]) }
    }
}

pub(crate) fn get_fixed_size_list_builder(
    inner_type_logical: &DataType,
    capacity: usize,
    width: usize,
    name: &str,
) -> PolarsResult<Box<dyn FixedSizeListBuilder>> {
    let phys_dtype = inner_type_logical.to_physical();

    let builder = if phys_dtype.is_numeric() {
        with_match_physical_numeric_type!(phys_dtype, |$T| {
            Box::new(FixedSizeListNumericBuilder::<$T>::new(name, width, capacity)) as Box<dyn FixedSizeListBuilder>
        })
    } else {
        Box::new(AnonymousOwnedFixedSizeListBuilder::new(
            name,
            width,
            capacity,
            Some(inner_type_logical.clone()),
        ))
    };
    Ok(builder)
}

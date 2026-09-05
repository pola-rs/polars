use arrow::types::NativeType;
use polars_array::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
use polars_array::{PlFixedSizeListArrayBuilder, PlPrimitiveArrayBuilder};
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::new_empty_chunk;
use crate::prelude::*;

pub(crate) struct FixedSizeListNumericBuilder<T: NativeType> {
    inner: Option<PlFixedSizeListArrayBuilder<PlPrimitiveArrayBuilder<T>>>,
    width: usize,
    name: PlSmallStr,
    logical_dtype: DataType,
}

impl<T: NativeType> FixedSizeListNumericBuilder<T> {
    /// # Safety
    ///
    /// The caller must ensure that the physical numerical type match logical type.
    pub(crate) unsafe fn new(
        name: PlSmallStr,
        width: usize,
        capacity: usize,
        logical_dtype: DataType,
    ) -> Self {
        let values = PlPrimitiveArrayBuilder::<T>::with_capacity(capacity * width);
        let inner = Some(PlFixedSizeListArrayBuilder::new(values, width));
        Self {
            inner,
            width,
            name,
            logical_dtype,
        }
    }
}

pub trait FixedSizeListBuilder {
    /// # Safety
    ///
    /// `arr` must have at least `(offset + 1) * width` valid elements of the
    /// builder's expected inner type
    unsafe fn push_unchecked(&mut self, arr: &dyn PlArray, offset: usize);
    /// # Safety
    ///
    /// The builder must have been properly initialized
    unsafe fn push_null(&mut self);
    fn finish(&mut self) -> ArrayChunked;
}

impl<T: NativeType> FixedSizeListBuilder for FixedSizeListNumericBuilder<T> {
    #[inline]
    unsafe fn push_unchecked(&mut self, arr: &dyn PlArray, offset: usize) {
        let width = self.width;
        let arr = arr
            .as_any()
            .downcast_ref::<PlPrimitiveArray<T>>()
            .unwrap_unchecked();
        let inner = self.inner.as_mut().unwrap_unchecked();

        // The element is appended as a subslice rather than a value at a time, which leaves the
        // chunk it is read out of in whatever representation it is in.
        inner
            .values_mut()
            .subslice_extend(arr, offset * width, width, ShareStrategy::Always);
        inner.finish_row();
    }

    #[inline]
    unsafe fn push_null(&mut self) {
        let inner = self.inner.as_mut().unwrap_unchecked();
        inner.extend_nulls(1)
    }

    fn finish(&mut self) -> ArrayChunked {
        let arr = self.inner.take().unwrap().freeze();
        // SAFETY: physical type matches the logical
        unsafe {
            ChunkedArray::from_chunks_and_dtype(
                self.name.clone(),
                vec![Box::new(arr)],
                DataType::Array(Box::new(self.logical_dtype.clone()), self.width),
            )
        }
    }
}

pub(crate) struct AnonymousOwnedFixedSizeListBuilder {
    inner: PlFixedSizeListArrayBuilder,
    width: usize,
    name: PlSmallStr,
    inner_dtype: DataType,
}

impl AnonymousOwnedFixedSizeListBuilder {
    pub(crate) fn new(
        name: PlSmallStr,
        width: usize,
        capacity: usize,
        inner_dtype: DataType,
    ) -> Self {
        // A builder is shaped like the array it builds, and the shape of the values is what the
        // physical inner type says — which is why an empty chunk of it is enough to ask for one.
        let values = builder_like(&*new_empty_chunk(&inner_dtype));
        let inner = PlFixedSizeListArrayBuilder::with_capacity(values, width, capacity);
        Self {
            inner,
            width,
            name,
            inner_dtype,
        }
    }
}

impl FixedSizeListBuilder for AnonymousOwnedFixedSizeListBuilder {
    #[inline]
    unsafe fn push_unchecked(&mut self, arr: &dyn PlArray, offset: usize) {
        self.inner.values_mut().subslice_extend(
            arr,
            offset * self.width,
            self.width,
            ShareStrategy::Always,
        );
        self.inner.finish_row();
    }

    #[inline]
    unsafe fn push_null(&mut self) {
        self.inner.extend_nulls(1)
    }

    fn finish(&mut self) -> ArrayChunked {
        let arr = self.inner.freeze_reset();
        // The dtype is the logical one this was asked for. It used to be read back off the Arrow
        // dtype the builder had been handed, which is the same thing for every type that survives
        // the round trip and less than the truth for the ones that do not.
        unsafe {
            ChunkedArray::from_chunks_and_dtype_unchecked(
                self.name.clone(),
                vec![Box::new(arr)],
                DataType::Array(Box::new(self.inner_dtype.clone()), self.width),
            )
        }
    }
}

pub fn get_fixed_size_list_builder(
    inner_type_logical: &DataType,
    capacity: usize,
    width: usize,
    name: PlSmallStr,
) -> PolarsResult<Box<dyn FixedSizeListBuilder>> {
    let phys_dtype = inner_type_logical.to_physical();

    let builder = if phys_dtype.is_primitive_numeric() {
        with_match_physical_numeric_type!(phys_dtype, |$T| {
        // SAFETY: physical type match logical type
        unsafe {
            Box::new(FixedSizeListNumericBuilder::<$T>::new(name, width, capacity,inner_type_logical.clone())) as Box<dyn FixedSizeListBuilder>
        }
        })
    } else {
        Box::new(AnonymousOwnedFixedSizeListBuilder::new(
            name,
            width,
            capacity,
            inner_type_logical.clone(),
        ))
    };
    Ok(builder)
}

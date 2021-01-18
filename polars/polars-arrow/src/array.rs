use crate::vec::AlignedVec;
use arrow::array::{
    Array, ArrayBuilder, ArrayData, ArrayDataRef, ArrayRef, BooleanBufferBuilder, ListArray,
    PrimitiveArray,
};
use arrow::datatypes::ArrowPrimitiveType;
use num::Num;
use std::sync::Arc;
use std::{any::Any, mem};

pub trait GetValues {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num;
}

impl GetValues for ArrayDataRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        debug_assert_eq!(&T::DATA_TYPE, self.data_type());
        // the first buffer is the value array
        let value_buf = &self.buffers()[0];
        let offset = self.offset();
        let vals = unsafe { value_buf.typed_data::<T::Native>() };
        &vals[offset..offset + self.len()]
    }
}

impl GetValues for &dyn Array {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

impl GetValues for ArrayRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

pub trait ToPrimitive {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType;
}

impl ToPrimitive for ArrayDataRef {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        PrimitiveArray::from(self)
    }
}

impl ToPrimitive for &dyn Array {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        self.data().into_primitive_array()
    }
}

/// An arrow primitive builder that is faster than Arrow's native builder because it uses Rust Vec's
/// as buffer
pub struct PrimitiveArrayBuilder<T>
where
    T: ArrowPrimitiveType,
    T::Native: Default,
{
    values: AlignedVec<T::Native>,
    bitmap_builder: BooleanBufferBuilder,
    null_count: usize,
}

impl<T> PrimitiveArrayBuilder<T>
where
    T: ArrowPrimitiveType,
    T::Native: Default,
{
    pub fn new(capacity: usize) -> Self {
        let values = AlignedVec::<T::Native>::with_capacity_aligned(capacity);
        let bitmap_builder = BooleanBufferBuilder::new(capacity);

        Self {
            values,
            bitmap_builder,
            null_count: 0,
        }
    }

    /// Appends a value of type `T::Native` into the builder
    #[inline]
    pub fn append_value(&mut self, v: T::Native) {
        self.values.push(v);
        self.bitmap_builder.append(true);
    }

    #[inline]
    pub fn append_slice(&mut self, other: &[T::Native]) {
        self.values.extend_from_slice(other)
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.bitmap_builder.append(false);
        self.values.push(Default::default());
        self.null_count += 1;
    }

    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit()
    }

    /// Build the array and reset this Builder
    pub fn finish(&mut self) -> PrimitiveArray<T> {
        self.shrink_to_fit();
        let values = mem::take(&mut self.values);
        let null_bit_buffer = self.bitmap_builder.finish();
        let buf = if self.null_count == 0 {
            None
        } else {
            Some(null_bit_buffer)
        };
        values.into_primitive_array(buf)
    }
}

impl<T> ArrayBuilder for PrimitiveArrayBuilder<T>
where
    T: ArrowPrimitiveType,
{
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(PrimitiveArrayBuilder::finish(self))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

pub trait ValueSize {
    /// Useful for a Utf8 or a List to get underlying value size.
    /// During a rechunk this is handy
    fn get_values_size(&self) -> usize;
}

impl ValueSize for ArrayRef {
    fn get_values_size(&self) -> usize {
        self.data_ref().get_values_size()
    }
}

impl ValueSize for ArrayData {
    fn get_values_size(&self) -> usize {
        self.child_data()[0].len() - self.offset()
    }
}

impl ValueSize for ListArray {
    fn get_values_size(&self) -> usize {
        self.data_ref().get_values_size()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::datatypes::UInt32Type;

    #[test]
    fn test_primitive_builder() {
        let mut builder = PrimitiveArrayBuilder::<UInt32Type>::new(10);
        builder.append_value(0);
        builder.append_null();
        let out = builder.finish();
        assert_eq!(out.len(), 2);
        assert_eq!(out.null_count(), 1);
        dbg!(out);
    }
}

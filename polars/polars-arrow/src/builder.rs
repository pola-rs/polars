use crate::bit_util;
use crate::vec::AlignedVec;
use arrow::array::{
    ArrayBuilder, ArrayData, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray,
};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use std::any::Any;
use std::mem;
use std::sync::Arc;

#[derive(Debug)]
pub struct BooleanBufferBuilder {
    buffer: MutableBuffer,
    len: usize,
}

impl BooleanBufferBuilder {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let byte_capacity = bit_util::ceil(capacity, 8);
        let actual_capacity = bit_util::round_upto_multiple_of_64(byte_capacity);
        let mut buffer = MutableBuffer::new(actual_capacity);
        buffer.set_null_bits(0, actual_capacity);

        Self { buffer, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.buffer.capacity() * 8
    }

    pub fn shrink_to_fit(&mut self) {
        let byte_len = bit_util::ceil(self.len(), 8);
        self.buffer.resize(byte_len)
    }

    #[inline]
    pub fn advance(&mut self, i: usize) {
        let new_buffer_len = bit_util::ceil(self.len + i, 8);
        self.buffer.resize(new_buffer_len);
        self.len += i;
    }

    #[inline]
    pub fn reserve(&mut self, n: usize) {
        let new_capacity = self.len + n;
        if new_capacity > self.capacity() {
            let new_byte_capacity = bit_util::ceil(new_capacity, 8);
            let existing_capacity = self.buffer.capacity();
            let new_capacity = self.buffer.reserve(new_byte_capacity);
            self.buffer
                .set_null_bits(existing_capacity, new_capacity - existing_capacity);
        }
    }

    #[inline]
    pub fn append(&mut self, v: bool) {
        self.reserve(1);
        if v {
            let data = unsafe {
                std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.capacity())
            };
            bit_util::set_bit(data, self.len);
        }
        self.len += 1;
    }

    #[inline]
    pub fn append_n(&mut self, n: usize, v: bool) {
        self.reserve(n);
        if n != 0 && v {
            let data = unsafe {
                std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.capacity())
            };
            (self.len..self.len + n).for_each(|i| bit_util::set_bit(data, i))
        }
        self.len += n;
    }

    #[inline]
    pub fn append_slice(&mut self, slice: &[bool]) {
        let array_slots = slice.len();
        self.reserve(array_slots);

        for v in slice {
            if *v {
                // For performance the `len` of the buffer is not
                // updated on each append but is updated in the
                // `into` method instead.
                unsafe {
                    bit_util::set_bit_raw(self.buffer.as_mut_ptr(), self.len);
                }
            }
            self.len += 1;
        }
    }

    #[inline]
    pub fn finish(&mut self) -> Buffer {
        self.shrink_to_fit();
        // `append` does not update the buffer's `len` so do it before `into` is called.
        let new_buffer_len = bit_util::ceil(self.len, 8);
        debug_assert!(new_buffer_len >= self.buffer.len());
        let mut buf = std::mem::replace(&mut self.buffer, MutableBuffer::new(0));
        self.len = 0;
        buf.resize(new_buffer_len);
        buf.into()
    }
}

///  Array builder for fixed-width primitive types
#[derive(Debug)]
pub struct BooleanArrayBuilder {
    values_builder: BooleanBufferBuilder,
    bitmap_builder: BooleanBufferBuilder,
}

impl BooleanArrayBuilder {
    /// Creates a new primitive array builder
    pub fn new(capacity: usize) -> Self {
        Self {
            values_builder: BooleanBufferBuilder::new(capacity),
            bitmap_builder: BooleanBufferBuilder::new(capacity),
        }
    }

    pub fn new_no_nulls(capacity: usize) -> Self {
        Self {
            values_builder: BooleanBufferBuilder::new(capacity),
            bitmap_builder: BooleanBufferBuilder::new(0),
        }
    }

    /// Returns the capacity of this builder measured in slots of type `T`
    pub fn capacity(&self) -> usize {
        self.values_builder.capacity()
    }

    /// Appends a value of type `T` into the builder
    pub fn append_value(&mut self, v: bool) {
        self.bitmap_builder.append(true);
        self.values_builder.append(v);
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        self.bitmap_builder.append(false);
        self.values_builder.advance(1);
    }

    /// Appends an `Option<T>` into the builder
    pub fn append_option(&mut self, v: Option<bool>) {
        match v {
            None => self.append_null(),
            Some(v) => self.append_value(v),
        };
    }

    /// Appends a slice of type `T` into the builder
    pub fn append_slice(&mut self, v: &[bool]) {
        self.bitmap_builder.append_n(v.len(), true);
        self.values_builder.append_slice(v);
    }

    /// Appends values from a slice of type `T` and a validity boolean slice
    pub fn append_values(&mut self, values: &[bool], is_valid: &[bool]) {
        assert_eq!(values.len(), is_valid.len());
        self.bitmap_builder.append_slice(is_valid);
        self.values_builder.append_slice(values);
    }

    pub fn shrink_to_fit(&mut self) {
        self.values_builder.shrink_to_fit();
        self.bitmap_builder.shrink_to_fit();
    }

    pub fn finish_with_null_buffer(&mut self, buffer: Buffer) -> BooleanArray {
        self.shrink_to_fit();
        let len = self.len();
        let data = ArrayData::builder(DataType::Boolean)
            .len(len)
            .add_buffer(self.values_builder.finish())
            .null_bit_buffer(buffer)
            .build();
        BooleanArray::from(data)
    }

    /// Builds the [BooleanArray] and reset this builder.
    pub fn finish(&mut self) -> BooleanArray {
        self.shrink_to_fit();
        let len = self.len();
        let null_bit_buffer = self.bitmap_builder.finish();
        let null_count = len - null_bit_buffer.count_set_bits();
        let mut builder = ArrayData::builder(DataType::Boolean)
            .len(len)
            .add_buffer(self.values_builder.finish());
        if null_count > 0 {
            builder = builder.null_bit_buffer(null_bit_buffer);
        }
        let data = builder.build();
        BooleanArray::from(data)
    }
}

impl ArrayBuilder for BooleanArrayBuilder {
    /// Returns the builder as a non-mutable `Any` reference.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns the builder as a mutable `Any` reference.
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    /// Returns the boxed builder as a box of `Any`.
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    /// Returns the number of array slots in the builder
    fn len(&self) -> usize {
        self.values_builder.len()
    }

    /// Returns whether the number of array slots is zero
    fn is_empty(&self) -> bool {
        self.values_builder.is_empty()
    }

    /// Builds the array and reset this builder.
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.finish())
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

    pub fn new_no_nulls(capacity: usize) -> Self {
        let values = AlignedVec::<T::Native>::with_capacity_aligned(capacity);
        let bitmap_builder = BooleanBufferBuilder::new(0);

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
        self.values.shrink_to_fit();
        self.bitmap_builder.shrink_to_fit();
    }

    pub fn finish_with_null_buffer(&mut self, buffer: Buffer) -> PrimitiveArray<T> {
        self.shrink_to_fit();
        let values = mem::take(&mut self.values);
        values.into_primitive_array(Some(buffer))
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

#[derive(Debug)]
pub struct LargeStringBuilder {
    values: AlignedVec<u8>,
    offsets: AlignedVec<i64>,
    null_buffer: BooleanBufferBuilder,
}

impl LargeStringBuilder {
    pub fn with_capacity(values_capacity: usize, list_capacity: usize) -> Self {
        let mut offsets = AlignedVec::with_capacity_aligned(list_capacity + 1);
        offsets.push(0);
        Self {
            values: AlignedVec::with_capacity_aligned(values_capacity),
            offsets,
            null_buffer: BooleanBufferBuilder::new(list_capacity),
        }
    }

    pub fn append_value(&mut self, value: &str) {
        self.values.extend_from_slice(value.as_bytes());
        self.offsets.push(self.values.len() as i64);
        self.null_buffer.append(true);
    }

    /// Finish the current variable-length list array slot.
    pub fn append(&mut self, is_valid: bool) {
        self.offsets.push(self.values.len() as i64);
        self.null_buffer.append(is_valid);
    }

    /// Append a null value to the array.
    pub fn append_null(&mut self) {
        self.append(false);
    }

    /// Builds the `StringArray` and reset this builder.
    pub fn finish(&mut self) -> LargeStringArray {
        // values are u8 typed
        let values = mem::take(&mut self.values);
        // offsets are i64 typed
        let offsets = mem::take(&mut self.offsets);
        let offsets_len = offsets.len() - 1;
        // buffers are u8 typed
        let buf_offsets = offsets.into_arrow_buffer();
        let buf_values = values.into_arrow_buffer();
        assert_eq!(buf_values.len(), buf_values.capacity());
        assert_eq!(buf_offsets.len(), buf_offsets.capacity());

        // note that the arrays are already shrinked when transformed to an arrow buffer.
        let arraydata = ArrayData::builder(DataType::LargeUtf8)
            .len(offsets_len)
            .add_buffer(buf_offsets)
            .add_buffer(buf_values)
            .null_bit_buffer(self.null_buffer.finish())
            .build();
        LargeStringArray::from(arraydata)
    }
}

impl ArrayBuilder for LargeStringBuilder {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(LargeStringBuilder::finish(self))
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

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::Array;
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

    #[test]
    fn test_string_builder() {
        let mut builder = LargeStringBuilder::with_capacity(1, 3);
        builder.append_value("foo");
        builder.append_null();
        builder.append_value("bar");
        let out = builder.finish();
        let vals = out.iter().collect::<Vec<_>>();
        assert_eq!(vals, &[Some("foo"), None, Some("bar")]);
    }
}

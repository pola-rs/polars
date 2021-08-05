use crate::bit_util;
use crate::vec::AlignedVec;
pub use arrow::array::LargeStringBuilder;
use arrow::array::{ArrayBuilder, ArrayData, ArrayRef, BooleanArray, LargeStringArray};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::DataType;
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
        let buffer = MutableBuffer::from_len_zeroed(byte_capacity);
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

    #[inline]
    pub fn advance(&mut self, additional: usize) {
        let new_len = self.len + additional;
        let new_len_bytes = bit_util::ceil(new_len, 8);
        if new_len_bytes > self.buffer.len() {
            self.buffer.resize(new_len_bytes, 0);
        }
        self.len = new_len;
    }

    /// Reserve space to at least `additional` new bits.
    /// Capacity will be `>= self.len() + additional`.
    /// New bytes are uninitialized and reading them is undefined behavior.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.len + additional;
        if capacity > self.capacity() {
            // convert differential to bytes
            let additional = bit_util::ceil(capacity, 8) - self.buffer.len();
            self.buffer.reserve(additional);
        }
    }

    #[inline]
    pub fn append(&mut self, v: bool) {
        self.advance(1);
        if v {
            unsafe { bit_util::set_bit_raw(self.buffer.as_mut_ptr(), self.len - 1) };
        }
    }

    #[inline]
    pub fn append_n(&mut self, additional: usize, v: bool) {
        self.advance(additional);
        if additional > 0 && v {
            let offset = self.len() - additional;
            (0..additional).for_each(|i| unsafe {
                bit_util::set_bit_raw(self.buffer.as_mut_ptr(), offset + i)
            })
        }
    }

    #[inline]
    pub fn append_slice(&mut self, slice: &[bool]) {
        let additional = slice.len();
        self.advance(additional);

        let offset = self.len() - additional;
        for (i, v) in slice.iter().enumerate() {
            if *v {
                unsafe { bit_util::set_bit_raw(self.buffer.as_mut_ptr(), offset + i) }
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        let byte_len = bit_util::ceil(self.len(), 8);
        self.buffer.resize(byte_len, 0)
    }

    #[inline]
    pub fn finish(&mut self) -> Buffer {
        let buf = std::mem::replace(&mut self.buffer, MutableBuffer::new(0));
        self.len = 0;
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

#[derive(Debug)]
pub struct NoNullLargeStringBuilder {
    values: AlignedVec<u8>,
    offsets: AlignedVec<i64>,
}

impl NoNullLargeStringBuilder {
    pub fn with_capacity(values_capacity: usize, list_capacity: usize) -> Self {
        let mut offsets = AlignedVec::with_capacity(list_capacity + 1);
        offsets.push(0);
        Self {
            values: AlignedVec::with_capacity(values_capacity),
            offsets,
        }
    }

    /// Extends with values and offsets.
    pub fn extend_from_slices(&mut self, values: &[u8], offsets: &[i64]) {
        self.values.extend_from_slice(values);
        self.offsets.extend_from_slice(offsets);
    }

    #[inline]
    pub fn append_value(&mut self, value: &str) {
        self.values.extend_from_slice(value.as_bytes());
        self.offsets.push(self.values.len() as i64);
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

        // note that the arrays are already shrunk when transformed to an arrow buffer.
        let arraydata = ArrayData::builder(DataType::LargeUtf8)
            .len(offsets_len)
            .add_buffer(buf_offsets)
            .add_buffer(buf_values)
            .build();
        LargeStringArray::from(arraydata)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::Array;
    use arrow::datatypes::UInt32Type;

    #[test]
    fn test_string_builder() {
        let mut builder = LargeStringBuilder::with_capacity(1, 3);
        builder.append_value("foo").unwrap();
        builder.append_null().unwrap();
        builder.append_value("bar").unwrap();
        let out = builder.finish();
        let vals = out.iter().collect::<Vec<_>>();
        assert_eq!(vals, &[Some("foo"), None, Some("bar")]);
    }
}

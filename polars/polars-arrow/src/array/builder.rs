//! Forked from arrow-rs so that we don't have unneeded error returns and
//! we can extend with an iterator. This saves allocation in the list builder
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use arrow::array::{BufferBuilder, ArrayBuilder, ArrayRef, PrimitiveArray, ArrayData, DictionaryArray};
use crate::builder::BooleanBufferBuilder;
use std::any::Any;
use arrow::error::ArrowError;
use std::sync::Arc;

///  Array builder for fixed-width primitive types
#[derive(Debug)]
pub struct PrimitiveBuilder<T: ArrowPrimitiveType> {
    values_builder: BufferBuilder<T::Native>,
    /// We only materialize the builder when we add `false`.
    /// This optimization is **very** important for performance of `StringBuilder`.
    bitmap_builder: Option<BooleanBufferBuilder>,
}

impl<T: ArrowPrimitiveType> ArrayBuilder for PrimitiveBuilder<T> {
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

impl<T: ArrowPrimitiveType> PrimitiveBuilder<T> {
    /// Creates a new primitive array builder
    pub fn new(capacity: usize) -> Self {
        Self {
            values_builder: BufferBuilder::<T::Native>::new(capacity),
            bitmap_builder: None,
        }
    }

    /// Returns the capacity of this builder measured in slots of type `T`
    pub fn capacity(&self) -> usize {
        self.values_builder.capacity()
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value(&mut self, v: T::Native) {
        if let Some(b) = self.bitmap_builder.as_mut() {
            b.append(true);
        }
        self.values_builder.append(v);
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.materialize_bitmap_builder();
        self.bitmap_builder.as_mut().unwrap().append(false);
        self.values_builder.advance(1);
    }

    /// Appends an `Option<T>` into the builder
    #[inline]
    pub fn append_option(&mut self, v: Option<T::Native>) {
        match v {
            None => self.append_null(),
            Some(v) => self.append_value(v),
        };
    }

    /// Appends a slice of type `T` into the builder
    #[inline]
    pub fn append_slice(&mut self, v: &[T::Native]) {
        if let Some(b) = self.bitmap_builder.as_mut() {
            b.append_n(v.len(), true);
        }
        self.values_builder.append_slice(v);
    }

    /// Appends values from a slice of type `T` and a validity boolean slice
    #[inline]
    pub fn append_values(
        &mut self,
        values: &[T::Native],
        is_valid: &[bool],
    ) {
        assert_eq!(values.len(), is_valid.len());
        if is_valid.iter().any(|v| !*v) {
            self.materialize_bitmap_builder();
        }
        if let Some(b) = self.bitmap_builder.as_mut() {
            b.append_slice(is_valid);
        }
        self.values_builder.append_slice(values);
    }

    /// Builds the `PrimitiveArray` and reset this builder.
    pub fn finish(&mut self) -> PrimitiveArray<T> {
        let len = self.len();
        let null_bit_buffer = self.bitmap_builder.as_mut().map(|b| b.finish());
        let null_count = len
            - null_bit_buffer
            .as_ref()
            .map(|b| b.count_set_bits())
            .unwrap_or(len);
        let mut builder = ArrayData::builder(T::DATA_TYPE)
            .len(len)
            .add_buffer(self.values_builder.finish());
        if null_count > 0 {
            builder = builder.null_bit_buffer(null_bit_buffer.unwrap());
        }
        let data = builder.build();
        PrimitiveArray::<T>::from(data)
    }

    /// Builds the `DictionaryArray` and reset this builder.
    pub fn finish_dict(&mut self, values: ArrayRef) -> DictionaryArray<T> {
        let len = self.len();
        let null_bit_buffer = self.bitmap_builder.as_mut().map(|b| b.finish());
        let null_count = len
            - null_bit_buffer
            .as_ref()
            .map(|b| b.count_set_bits())
            .unwrap_or(len);
        let data_type = DataType::Dictionary(
            Box::new(T::DATA_TYPE),
            Box::new(values.data_type().clone()),
        );
        let mut builder = ArrayData::builder(data_type)
            .len(len)
            .add_buffer(self.values_builder.finish());
        if null_count > 0 {
            builder = builder.null_bit_buffer(null_bit_buffer.unwrap());
        }
        builder = builder.add_child_data(values.data().clone());
        DictionaryArray::<T>::from(builder.build())
    }

    fn materialize_bitmap_builder(&mut self) {
        if self.bitmap_builder.is_some() {
            return;
        }
        let mut b = BooleanBufferBuilder::new(0);
        b.reserve(self.values_builder.capacity());
        b.append_n(self.values_builder.len(), true);
        self.bitmap_builder = Some(b);
    }
}

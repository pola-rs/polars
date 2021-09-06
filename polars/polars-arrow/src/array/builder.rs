//! Forked from arrow-rs so that we don't have unneeded error returns and
//! we can extend with an iterator. This saves allocation in the list builder
use crate::builder::BooleanBufferBuilder;
use crate::trusted_len::TrustedLen;
use crate::vec::AlignedVec;
use arrow::array::{ArrayBuilder, ArrayData, ArrayRef, DictionaryArray, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use std::any::Any;
use std::sync::Arc;

///  Array builder for fixed-width primitive types
#[derive(Debug)]
pub struct PrimitiveBuilder<T: ArrowPrimitiveType> {
    values_builder: AlignedVec<T::Native>,
    /// We only materialize the builder when we add `false`.
    /// This optimization is **very** important for performance of `StringBuilder`.
    bitmap_builder: Option<BooleanBufferBuilder>,
}

impl<T: ArrowPrimitiveType> ArrayBuilder for PrimitiveBuilder<T>
where
    T::Native: Default,
{
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
        Arc::new(PrimitiveBuilder::finish(self))
    }
}

impl<T: ArrowPrimitiveType> PrimitiveBuilder<T>
where
    T::Native: Default,
{
    /// Creates a new primitive array builder
    pub fn new(capacity: usize) -> Self {
        Self {
            values_builder: AlignedVec::with_capacity(capacity),
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
        self.values_builder.push(v);
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.materialize_bitmap_builder();
        self.bitmap_builder.as_mut().unwrap().append(false);
        self.values_builder.push(T::Native::default());
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
        self.values_builder.extend_memcpy(v);
    }

    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter_values<I: Iterator<Item = T::Native> + TrustedLen>(&mut self, iter: I) {
        let len = iter.size_hint().0;
        if let Some(b) = self.bitmap_builder.as_mut() {
            b.append_n(len, true);
        }
        self.values_builder.extend_trusted_len(iter)
    }

    /// Appends values from a slice of type `T` and a validity boolean slice
    #[inline]
    pub fn append_values(&mut self, values: &[T::Native], is_valid: &[bool]) {
        assert_eq!(values.len(), is_valid.len());
        if is_valid.iter().any(|v| !*v) {
            self.materialize_bitmap_builder();
        }
        if let Some(b) = self.bitmap_builder.as_mut() {
            b.append_slice(is_valid);
        }
        self.values_builder.extend_memcpy(values);
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
        let values_builder = std::mem::take(&mut self.values_builder);
        let mut builder = ArrayData::builder(T::DATA_TYPE)
            .len(len)
            .add_buffer(values_builder.into_arrow_buffer());
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
        let data_type =
            DataType::Dictionary(Box::new(T::DATA_TYPE), Box::new(values.data_type().clone()));
        let values_builder = std::mem::take(&mut self.values_builder);
        let mut builder = ArrayData::builder(data_type)
            .len(len)
            .add_buffer(values_builder.into_arrow_buffer());
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

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Array, Int32Array};
    use arrow::datatypes::*;

    #[test]
    fn test_primitive_array_builder_i32() {
        let mut builder = PrimitiveBuilder::<Int32Type>::new(5);
        for i in 0..5 {
            builder.append_value(i);
        }
        let arr = builder.finish();
        assert_eq!(5, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..5 {
            assert!(!arr.is_null(i));
            assert!(arr.is_valid(i));
            assert_eq!(i as i32, arr.value(i));
        }
    }

    #[test]
    fn test_primitive_array_builder_date32() {
        let mut builder = PrimitiveBuilder::<Date32Type>::new(5);
        for i in 0..5 {
            builder.append_value(i);
        }
        let arr = builder.finish();
        assert_eq!(5, arr.len());
        assert_eq!(0, arr.offset());
        assert_eq!(0, arr.null_count());
        for i in 0..5 {
            assert!(!arr.is_null(i));
            assert!(arr.is_valid(i));
            assert_eq!(i as i32, arr.value(i));
        }
    }

    #[test]
    fn test_primitive_array_builder_append_option() {
        let arr1 = Int32Array::from(vec![Some(0), None, Some(2), None, Some(4)]);

        let mut builder = PrimitiveBuilder::<Int32Type>::new(5);
        builder.append_option(Some(0));
        builder.append_option(None);
        builder.append_option(Some(2));
        builder.append_option(None);
        builder.append_option(Some(4));
        let arr2 = builder.finish();

        assert_eq!(arr1.len(), arr2.len());
        assert_eq!(arr1.offset(), arr2.offset());
        assert_eq!(arr1.null_count(), arr2.null_count());
        for i in 0..5 {
            assert_eq!(arr1.is_null(i), arr2.is_null(i));
            assert_eq!(arr1.is_valid(i), arr2.is_valid(i));
            if arr1.is_valid(i) {
                assert_eq!(arr1.value(i), arr2.value(i));
            }
        }
    }

    #[test]
    fn test_primitive_array_builder_append_null() {
        let arr1 = Int32Array::from(vec![Some(0), Some(2), None, None, Some(4)]);

        let mut builder = PrimitiveBuilder::<Int32Type>::new(5);
        builder.append_value(0);
        builder.append_value(2);
        builder.append_null();
        builder.append_null();
        builder.append_value(4);
        let arr2 = builder.finish();

        assert_eq!(arr1.len(), arr2.len());
        assert_eq!(arr1.offset(), arr2.offset());
        assert_eq!(arr1.null_count(), arr2.null_count());
        for i in 0..5 {
            assert_eq!(arr1.is_null(i), arr2.is_null(i));
            assert_eq!(arr1.is_valid(i), arr2.is_valid(i));
            if arr1.is_valid(i) {
                assert_eq!(arr1.value(i), arr2.value(i));
            }
        }
    }

    #[test]
    fn test_primitive_array_builder_append_slice() {
        let arr1 = Int32Array::from(vec![Some(0), Some(2), None, None, Some(4)]);

        let mut builder = PrimitiveBuilder::<Int32Type>::new(5);
        builder.append_slice(&[0, 2]);
        builder.append_null();
        builder.append_null();
        builder.append_value(4);
        let arr2 = builder.finish();

        assert_eq!(arr1.len(), arr2.len());
        assert_eq!(arr1.offset(), arr2.offset());
        assert_eq!(arr1.null_count(), arr2.null_count());
        for i in 0..5 {
            assert_eq!(arr1.is_null(i), arr2.is_null(i));
            assert_eq!(arr1.is_valid(i), arr2.is_valid(i));
            if arr1.is_valid(i) {
                assert_eq!(arr1.value(i), arr2.value(i));
            }
        }
    }

    #[test]
    fn test_primitive_array_builder_finish() {
        let mut builder = PrimitiveBuilder::<Int32Type>::new(5);
        builder.append_slice(&[2, 4, 6, 8]);
        let mut arr = builder.finish();
        assert_eq!(4, arr.len());
        assert_eq!(0, builder.len());

        builder.append_slice(&[1, 3, 5, 7, 9]);
        arr = builder.finish();
        assert_eq!(5, arr.len());
        assert_eq!(0, builder.len());
    }
}

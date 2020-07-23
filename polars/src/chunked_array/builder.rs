use crate::prelude::*;
use arrow::datatypes::{ArrowPrimitiveType, Field, ToByteSlice};
use arrow::{
    array::{Array, ArrayData, PrimitiveArray, PrimitiveBuilder, StringBuilder},
    buffer::Buffer,
    memory,
    util::bit_util,
};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub struct PrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub builder: PrimitiveBuilder<T>,
    capacity: usize,
    field: Field,
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        PrimitiveChunkedBuilder {
            builder: PrimitiveBuilder::<T>::new(capacity),
            capacity,
            field: Field::new(name, T::get_data_type(), true),
        }
    }

    pub fn new_from_iter(mut self, it: impl Iterator<Item = Option<T::Native>>) -> ChunkedArray<T> {
        it.for_each(|opt| self.append_option(opt).expect("could not append"));
        self.finish()
    }

    pub fn finish(mut self) -> ChunkedArray<T> {
        let arr = Arc::new(self.builder.finish());
        let len = arr.len();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }
}

impl<T: ArrowPrimitiveType> Deref for PrimitiveChunkedBuilder<T> {
    type Target = PrimitiveBuilder<T>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<T: ArrowPrimitiveType> DerefMut for PrimitiveChunkedBuilder<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

pub struct Utf8ChunkedBuilder {
    pub builder: StringBuilder,
    capacity: usize,
    field: Field,
}

impl Utf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilder {
            builder: StringBuilder::new(capacity),
            capacity,
            field: Field::new(name, ArrowDataType::Utf8, true),
        }
    }

    pub fn finish(mut self) -> Utf8Chunked {
        let arr = Arc::new(self.builder.finish());
        let len = arr.len();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }
}

impl Deref for Utf8ChunkedBuilder {
    type Target = StringBuilder;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl DerefMut for Utf8ChunkedBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

pub fn build_primitive_ca_with_opt<T>(s: &[Option<T::Native>], name: &str) -> ChunkedArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let mut builder = PrimitiveChunkedBuilder::new(name, s.len());
    for opt in s {
        builder.append_option(*opt).expect("could not append");
    }
    let ca = builder.finish();
    ca
}

pub fn build_with_existing_null_bitmap<T>(
    null_bit_buffer: Option<Buffer>,
    null_count: usize,
    values: &[T::Native],
) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    let len = values.len();
    // See:
    // https://docs.rs/arrow/0.16.0/src/arrow/array/builder.rs.html#314
    let mut builder = ArrayData::builder(T::get_data_type())
        .len(len)
        .add_buffer(Buffer::from(values.to_byte_slice()));

    if null_count > 0 {
        let null_bit_buffer =
            null_bit_buffer.expect("implementation error. Should not be None if null_count > 0");
        debug_assert!(null_count == len - bit_util::count_set_bits(null_bit_buffer.data()));
        builder = builder
            .null_count(null_count)
            .null_bit_buffer(null_bit_buffer);
    }

    let data = builder.build();
    PrimitiveArray::<T>::from(data)
}

pub fn get_bitmap<T: Array + ?Sized>(arr: &T) -> (usize, Option<Buffer>) {
    let data = arr.data();
    (
        data.null_count(),
        data.null_bitmap().as_ref().map(|bitmap| {
            let buff = bitmap.buffer_ref();
            buff.clone()
        }),
    )
}

/// Returns the nearest number that is `>=` than `num` and is a multiple of 64
#[inline]
pub fn round_upto_multiple_of_64(num: usize) -> usize {
    round_upto_power_of_2(num, 64)
}

/// Returns the nearest multiple of `factor` that is `>=` than `num`. Here `factor` must
/// be a power of 2.
fn round_upto_power_of_2(num: usize, factor: usize) -> usize {
    debug_assert!(factor > 0 && (factor & (factor - 1)) == 0);
    (num + (factor - 1)) & !(factor - 1)
}

/// Take an owned Vec and create a zero copy PrimitiveArray
pub fn vec_to_primitive_array<T: ArrowPrimitiveType>(values: Vec<T::Native>) -> PrimitiveArray<T> {
    let vec_len = values.len();

    let me = ManuallyDrop::new(values);
    let ptr = me.as_ptr() as *const u8;
    let len = me.len() * std::mem::size_of::<T::Native>();
    let capacity = me.capacity() * std::mem::size_of::<T::Native>();

    let buffer = unsafe { Buffer::from_raw_parts(ptr, len, capacity) };

    let data = ArrayData::builder(T::get_data_type())
        .len(vec_len)
        .add_buffer(buffer)
        .build();

    PrimitiveArray::<T>::from(data)
}

pub trait AlignedAlloc<T> {
    fn with_capacity_aligned(size: usize) -> Vec<T>;
}

impl<T> AlignedAlloc<T> for Vec<T> {
    /// Create a new Vec where first bytes memory address has an alignment of 64 bytes, as described
    /// by arrow spec.
    /// Read more:
    /// https://github.com/rust-ndarray/ndarray/issues/771
    fn with_capacity_aligned(size: usize) -> Vec<T> {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let t_size = std::mem::size_of::<T>();
        let capacity = size * t_size;
        let ptr = memory::allocate_aligned(capacity) as *mut T;
        unsafe { Vec::from_raw_parts(ptr, 0, capacity) }
    }
}

pub struct AlignedVec<T>(pub Vec<T>);

impl<T> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let sh = iter.size_hint();
        let size = sh.1.unwrap_or(sh.0);

        let mut inner = Vec::with_capacity_aligned(size);

        while let Some(v) = iter.next() {
            inner.push(v)
        }

        // Iterator size hint wasn't correct and reallocation has occurred
        assert!(inner.len() <= size);
        AlignedVec(inner)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_existing_null_bitmap() {
        let mut builder = PrimitiveBuilder::<UInt32Type>::new(3);
        for val in &[Some(1), None, Some(2)] {
            builder.append_option(*val).unwrap();
        }
        let arr = builder.finish();
        let (null_count, buf) = get_bitmap(&arr);

        let new_arr = build_with_existing_null_bitmap::<UInt32Type>(buf, null_count, &[7, 8, 9]);
        assert!(new_arr.is_valid(0));
        assert!(new_arr.is_null(1));
        assert!(new_arr.is_valid(2));
    }

    #[test]
    fn from_vec() {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let mut v = Vec::with_capacity_aligned(2);
        v.push(1);
        v.push(2);

        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % 64, 0);
        let a = vec_to_primitive_array::<Int32Type>(v);
        assert_eq!(a.value_slice(0, 2), &[1, 2])
    }
}

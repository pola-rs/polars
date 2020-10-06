use crate::prelude::*;
use crate::utils::get_iter_capacity;
use arrow::array::{ArrayBuilder, ArrayDataBuilder, ArrayRef};
use arrow::datatypes::{ArrowPrimitiveType, Field, ToByteSlice};
pub use arrow::memory;
use arrow::{
    array::{Array, ArrayData, LargeListBuilder, PrimitiveArray, PrimitiveBuilder, StringBuilder},
    buffer::Buffer,
    util::bit_util,
};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem;
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

    /// Appends a value of type `T` into the builder
    pub fn append_value(&mut self, v: T::Native) {
        self.builder.append_value(v).expect("could not append");
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        self.builder.append_null().expect("could not append");
    }

    /// Append multiple values at once. This is faster than per value.
    pub fn append_values(&mut self, values: &[T::Native], is_valid: &[bool]) {
        self.builder.append_values(values, is_valid).unwrap();
    }

    /// Appends an `Option<T>` into the builder
    pub fn append_option(&mut self, v: Option<T::Native>) {
        self.builder.append_option(v).expect("could not append");
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

pub type BooleanChunkedBuilder = PrimitiveChunkedBuilder<BooleanType>;

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

    /// Appends a value of type `T` into the builder
    pub fn append_value<S: AsRef<str>>(&mut self, v: S) {
        self.builder
            .append_value(v.as_ref())
            .expect("could not append");
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        self.builder.append_null().expect("could not append");
    }

    pub fn append_option<S: AsRef<str>>(&mut self, opt: Option<S>) {
        match opt {
            Some(s) => self.append_value(s.as_ref()),
            None => self.append_null(),
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
        builder.append_option(*opt);
    }
    let ca = builder.finish();
    ca
}

fn set_null_bits(
    mut builder: ArrayDataBuilder,
    null_bit_buffer: Option<Buffer>,
    null_count: usize,
    len: usize,
) -> ArrayDataBuilder {
    if null_count > 0 {
        let null_bit_buffer =
            null_bit_buffer.expect("implementation error. Should not be None if null_count > 0");
        debug_assert!(null_count == len - bit_util::count_set_bits(null_bit_buffer.data()));
        builder = builder
            .null_count(null_count)
            .null_bit_buffer(null_bit_buffer);
    }
    builder
}

/// Take an existing slice and a null bitmap and construct an arrow array.
pub fn build_with_existing_null_bitmap_and_slice<T>(
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
    // TODO: make implementation for aligned owned vector for zero copy creation.
    let builder = ArrayData::builder(T::get_data_type())
        .len(len)
        .add_buffer(Buffer::from(values.to_byte_slice()));

    let builder = set_null_bits(builder, null_bit_buffer, null_count, len);
    let data = builder.build();
    PrimitiveArray::<T>::from(data)
}

/// Get the null count and the null bitmap of the arrow array
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

// Used in polars/src/chunked_array/apply.rs:24 to collect from aligned vecs and null bitmaps
impl<T> FromIterator<(AlignedVec<T::Native>, (usize, Option<Buffer>))> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = (AlignedVec<T::Native>, (usize, Option<Buffer>))>>(
        iter: I,
    ) -> Self {
        let mut chunks = vec![];

        for (values, (null_count, opt_buffer)) in iter {
            let arr = aligned_vec_to_primitive_array::<T>(values, opt_buffer, null_count);
            chunks.push(Arc::new(arr) as ArrayRef)
        }
        ChunkedArray::new_from_chunks("from_iter", chunks)
    }
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

/// Take an owned Vec that is 64 byte aligned and create a zero copy PrimitiveArray
/// Can also take a null bit buffer into account.
pub fn aligned_vec_to_primitive_array<T: ArrowPrimitiveType>(
    values: AlignedVec<T::Native>,
    null_bit_buffer: Option<Buffer>,
    null_count: usize,
) -> PrimitiveArray<T> {
    let values = unsafe { values.into_inner() };
    let vec_len = values.len();

    let me = mem::ManuallyDrop::new(values);
    let ptr = me.as_ptr() as *const u8;
    let len = me.len() * std::mem::size_of::<T::Native>();
    let capacity = me.capacity() * std::mem::size_of::<T::Native>();
    debug_assert_eq!((ptr as usize) % 64, 0);

    let buffer = unsafe { Buffer::from_raw_parts(ptr, len, capacity) };

    let builder = ArrayData::builder(T::get_data_type())
        .len(vec_len)
        .add_buffer(buffer);

    let builder = set_null_bits(builder, null_bit_buffer, null_count, vec_len);
    let data = builder.build();

    PrimitiveArray::<T>::from(data)
}

#[derive(Debug)]
pub struct AlignedVec<T> {
    inner: Vec<T>,
    capacity: usize,
    // if into_inner is called, this will be true and we can use the default Vec's destructor
    taken: bool,
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if !self.taken {
            let inner = mem::take(&mut self.inner);
            let mut me = mem::ManuallyDrop::new(inner);
            let ptr: *mut T = me.as_mut_ptr();
            let ptr = ptr as *mut u8;
            unsafe { memory::free_aligned(ptr, self.capacity) }
        }
    }
}

impl<T> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let sh = iter.size_hint();
        let size = sh.1.unwrap_or(sh.0);

        let mut av = Self::with_capacity_aligned(size);

        while let Some(v) = iter.next() {
            unsafe { av.push(v) }
        }

        // Iterator size hint wasn't correct and reallocation has occurred
        assert!(av.len() <= size);
        av
    }
}

impl<T> AlignedVec<T> {
    /// Create a new Vec where first bytes memory address has an alignment of 64 bytes, as described
    /// by arrow spec.
    /// Read more:
    /// https://github.com/rust-ndarray/ndarray/issues/771
    pub fn with_capacity_aligned(size: usize) -> Self {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let t_size = std::mem::size_of::<T>();
        let capacity = size * t_size;
        let ptr = memory::allocate_aligned(capacity) as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, 0, capacity) };
        AlignedVec {
            inner: v,
            capacity,
            taken: false,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub unsafe fn from_ptr(ptr: usize, len: usize, capacity: usize) -> Self {
        assert_eq!((ptr as usize) % memory::ALIGNMENT, 0);
        let ptr = ptr as *mut T;
        let v = Vec::from_raw_parts(ptr, len, capacity);
        Self {
            inner: v,
            capacity,
            taken: false,
        }
    }

    /// Take ownership of the Vec. This is UB because the destructor of Vec<T> probably has a different
    /// alignment than what we allocated.
    unsafe fn into_inner(mut self) -> Vec<T> {
        let inner = mem::take(&mut self.inner);
        self.taken = true;
        inner
    }

    /// Push at the end of the Vec. This is unsafe because a push when the capacity of the
    /// inner Vec is reached will reallocate the Vec without the alignment, leaving this destructor's
    /// alignment incorrect
    pub unsafe fn push(&mut self, value: T) {
        debug_assert!(self.inner.len() < self.capacity);
        self.inner.push(value)
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }
}

pub trait NewChunkedArray<T, N> {
    fn new_from_slice(name: &str, v: &[N]) -> Self;
    fn new_from_opt_slice(name: &str, opt_v: &[Option<N>]) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<N>>) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = N>) -> Self;
}

impl<T> NewChunkedArray<T, T::Native> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn new_from_slice(name: &str, v: &[T::Native]) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, v.len());
        v.iter().for_each(|&v| builder.append_value(v));
        builder.finish()
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<T::Native>]) -> Self {
        let mut builder = PrimitiveChunkedBuilder::<T>::new(name, opt_v.len());
        opt_v.iter().for_each(|&opt| builder.append_option(opt));
        builder.finish()
    }

    fn new_from_opt_iter(
        name: &str,
        it: impl Iterator<Item = Option<T::Native>>,
    ) -> ChunkedArray<T> {
        let mut builder = PrimitiveChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = T::Native>) -> ChunkedArray<T> {
        let mut builder = PrimitiveChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_value(opt));
        builder.finish()
    }
}

impl<S> NewChunkedArray<Utf8Type, S> for Utf8Chunked
where
    S: AsRef<str>,
{
    fn new_from_slice(name: &str, v: &[S]) -> Self {
        let mut builder = StringBuilder::new(v.len());
        v.into_iter().for_each(|val| {
            builder
                .append_value(val.as_ref())
                .expect("Could not append value");
        });

        let field = Arc::new(Field::new(name, ArrowDataType::Utf8, true));

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            chunk_id: vec![v.len()],
            phantom: PhantomData,
        }
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<S>]) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, opt_v.len());

        opt_v.iter().for_each(|opt| match opt {
            Some(v) => builder.append_value(v.as_ref()),
            None => builder.append_null(),
        });
        builder.finish()
    }

    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<S>>) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = S>) -> Self {
        let mut builder = Utf8ChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|v| builder.append_value(v));
        builder.finish()
    }
}

pub trait LargListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: &Option<Series>);
    fn append_series(&mut self, s: &Series);
    fn finish(&mut self) -> LargeListChunked;
}

pub struct LargeListPrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub builder: LargeListBuilder<PrimitiveBuilder<T>>,
    field: Field,
}

macro_rules! append_opt_series {
    ($self:ident, $opt_s: ident) => {{
        match $opt_s {
            Some(s) => {
                let data = s.array_data();
                $self
                    .builder
                    .values()
                    .append_data(&data)
                    .expect("should not fail");
                $self.builder.append(true).expect("should not fail");
            }
            None => {
                $self.builder.append(false).expect("should not fail");
            }
        }
    }};
}

macro_rules! append_series {
    ($self:ident, $s: ident) => {{
        let data = $s.array_data();
        $self
            .builder
            .values()
            .append_data(&data)
            .expect("should not fail");
        $self.builder.append(true).expect("should not fail");
    }};
}

macro_rules! finish_largelist_builder {
    ($self:ident) => {{
        let arr = Arc::new($self.builder.finish());
        let len = arr.len();
        LargeListChunked {
            field: Arc::new($self.field.clone()),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }};
}

impl<T> LargeListPrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub fn new(name: &str, values_builder: PrimitiveBuilder<T>, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(
            name,
            ArrowDataType::LargeList(Box::new(T::get_data_type())),
            true,
        );

        LargeListPrimitiveChunkedBuilder { builder, field }
    }

    pub fn append_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(v) => {
                self.builder
                    .values()
                    .append_slice(v)
                    .expect("could not append");
                self.builder.append(true).expect("should not fail");
            }
            None => {
                self.builder.append(false).expect("should not fail");
            }
        }
    }
    pub fn append_opt_slice(&mut self, opt_v: Option<&[Option<T::Native>]>) {
        match opt_v {
            Some(v) => {
                v.iter().for_each(|opt| {
                    self.builder
                        .values()
                        .append_option(*opt)
                        .expect("could not append")
                });
                self.builder.append(true).expect("should not fail");
            }
            None => {
                self.builder.append(false).expect("should not fail");
            }
        }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false).expect("should not fail");
    }
}

impl<T> LargListBuilderTrait for LargeListPrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    fn append_opt_series(&mut self, opt_s: &Option<Series>) {
        append_opt_series!(self, opt_s)
    }

    fn append_series(&mut self, s: &Series) {
        append_series!(self, s);
    }

    fn finish(&mut self) -> LargeListChunked {
        finish_largelist_builder!(self)
    }
}

pub struct LargeListUtf8ChunkedBuilder {
    builder: LargeListBuilder<StringBuilder>,
    field: Field,
}

impl LargeListUtf8ChunkedBuilder {
    pub fn new(name: &str, values_builder: StringBuilder, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(
            name,
            ArrowDataType::LargeList(Box::new(ArrowDataType::Utf8)),
            true,
        );

        LargeListUtf8ChunkedBuilder { builder, field }
    }
}

impl LargListBuilderTrait for LargeListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: &Option<Series>) {
        append_opt_series!(self, opt_s)
    }

    fn append_series(&mut self, s: &Series) {
        append_series!(self, s);
    }

    fn finish(&mut self) -> LargeListChunked {
        finish_largelist_builder!(self)
    }
}

pub fn get_large_list_builder(
    dt: &ArrowDataType,
    capacity: usize,
    name: &str,
) -> Box<dyn LargListBuilderTrait> {
    macro_rules! get_primitive_builder {
        ($type:ty) => {{
            let values_builder = PrimitiveBuilder::<$type>::new(capacity);
            let builder = LargeListPrimitiveChunkedBuilder::new(&name, values_builder, capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_utf8_builder {
        () => {{
            let values_builder = StringBuilder::new(capacity);
            let builder = LargeListUtf8ChunkedBuilder::new(&name, values_builder, capacity);
            Box::new(builder)
        }};
    }
    match_arrow_data_type_apply_macro!(dt, get_primitive_builder, get_utf8_builder)
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::Int32Array;

    #[test]
    fn test_existing_null_bitmap() {
        let mut builder = PrimitiveBuilder::<UInt32Type>::new(3);
        for val in &[Some(1), None, Some(2)] {
            builder.append_option(*val).unwrap();
        }
        let arr = builder.finish();
        let (null_count, buf) = get_bitmap(&arr);

        let new_arr =
            build_with_existing_null_bitmap_and_slice::<UInt32Type>(buf, null_count, &[7, 8, 9]);
        assert!(new_arr.is_valid(0));
        assert!(new_arr.is_null(1));
        assert!(new_arr.is_valid(2));
    }

    #[test]
    fn from_vec() {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let mut v = AlignedVec::with_capacity_aligned(2);
        unsafe {
            v.push(1);
            v.push(2);
        }

        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % 64, 0);
        let a = aligned_vec_to_primitive_array::<Int32Type>(v, None, 0);
        assert_eq!(a.value_slice(0, 2), &[1, 2])
    }

    #[test]
    fn test_list_builder() {
        let values_builder = Int32Array::builder(10);
        let mut builder = LargeListPrimitiveChunkedBuilder::new("a", values_builder, 10);

        // create a series containing two chunks
        let mut s1 = Int32Chunked::new_from_slice("a", &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::new_from_slice("b", &[4, 5, 6]).into_series();
        s1.append(&s2).unwrap();

        builder.append_series(&s1);
        builder.append_series(&s2);
        let ls = builder.finish();
        if let AnyType::LargeList(s) = ls.get_any(0) {
            // many chunks are aggregated to one in the ListArray
            assert_eq!(s.len(), 6)
        } else {
            assert!(false)
        }
        if let AnyType::LargeList(s) = ls.get_any(1) {
            assert_eq!(s.len(), 3)
        } else {
            assert!(false)
        }
    }
}

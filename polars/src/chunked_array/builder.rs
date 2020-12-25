use crate::{
    prelude::*,
    utils::{get_iter_capacity, Xob},
};
use arrow::array::{
    ArrayBuilder, ArrayDataBuilder, ArrayRef, BooleanBufferBuilder, BufferBuilderTrait, ListBuilder,
};
use arrow::datatypes::{Field, ToByteSlice};
pub use arrow::memory;
use arrow::{
    array::{Array, ArrayData, PrimitiveArray, PrimitiveBuilder, StringBuilder},
    buffer::Buffer,
    util::bit_util,
};
use std::borrow::Cow;
use std::fmt::Debug;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem;
use std::mem::ManuallyDrop;
use std::sync::Arc;

/// An arrow primitive builder that is faster than Arrow's native builder because it uses Rust Vec's
/// as buffer
pub struct PrimitiveArrayBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    values: AlignedVec<T::Native>,
    bitmap_builder: BooleanBufferBuilder,
    boolean_builder: PrimitiveBuilder<T>,
    capacity: usize,
    null_count: usize,
}

impl<T> PrimitiveArrayBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    pub fn new(capacity: usize) -> Self {
        let (boolean_builder, values, bitmap_builder) =
            if matches!(T::get_data_type(), ArrowDataType::Boolean) {
                (
                    PrimitiveBuilder::new(capacity),
                    AlignedVec::<T::Native>::with_capacity_aligned(0),
                    BooleanBufferBuilder::new(0),
                )
            } else {
                (
                    PrimitiveBuilder::new(0),
                    AlignedVec::<T::Native>::with_capacity_aligned(capacity),
                    BooleanBufferBuilder::new(capacity),
                )
            };

        Self {
            values,
            bitmap_builder,
            boolean_builder,
            capacity,
            null_count: 0,
        }
    }

    /// Appends a value of type `T::Native` into the builder
    pub fn append_value(&mut self, v: T::Native) {
        if matches!(T::get_data_type(), ArrowDataType::Boolean) {
            self.boolean_builder.append_value(v).unwrap();
        } else {
            self.values.push(v);
            self.bitmap_builder.append(true).unwrap();
        }
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        if matches!(T::get_data_type(), ArrowDataType::Boolean) {
            self.boolean_builder.append_null().unwrap();
        }
        {
            self.bitmap_builder.append(false).unwrap();
            self.values.push(Default::default());
            self.null_count += 1;
        }
    }

    pub fn finish(mut self) -> PrimitiveArray<T> {
        if matches!(T::get_data_type(), ArrowDataType::Boolean) {
            self.boolean_builder.finish()
        } else {
            let null_bit_buffer = self.bitmap_builder.finish();
            aligned_vec_to_primitive_array(
                self.values,
                Some(null_bit_buffer),
                Some(self.null_count),
            )
        }
    }
}

pub trait ChunkedBuilder<N, T> {
    fn append_value(&mut self, val: N);
    fn append_null(&mut self);
    fn append_option(&mut self, opt_val: Option<N>) {
        match opt_val {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }
    fn finish(self) -> ChunkedArray<T>;
}

pub struct PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    array_builder: PrimitiveArrayBuilder<T>,
    field: Field,
}

impl<T> ChunkedBuilder<T::Native, T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    /// Appends a value of type `T` into the builder
    fn append_value(&mut self, v: T::Native) {
        self.array_builder.append_value(v)
    }

    /// Appends a null slot into the builder
    fn append_null(&mut self) {
        self.array_builder.append_null()
    }

    fn finish(self) -> ChunkedArray<T> {
        let arr = Arc::new(self.array_builder.finish());

        let len = arr.len();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        PrimitiveChunkedBuilder {
            array_builder: PrimitiveArrayBuilder::<T>::new(capacity),
            field: Field::new(name, T::get_data_type(), true),
        }
    }
}

pub type BooleanChunkedBuilder = PrimitiveChunkedBuilder<BooleanType>;

pub struct Utf8ChunkedBuilder {
    pub builder: StringBuilder,
    pub capacity: usize,
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

pub struct Utf8ChunkedBuilderCow {
    builder: Utf8ChunkedBuilder,
}

impl Utf8ChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilderCow {
            builder: Utf8ChunkedBuilder::new(name, capacity),
        }
    }
}

impl ChunkedBuilder<Cow<'_, str>, Utf8Type> for Utf8ChunkedBuilderCow {
    fn append_value(&mut self, val: Cow<'_, str>) {
        self.builder.append_value(val.as_ref())
    }

    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(self) -> ChunkedArray<Utf8Type> {
        self.builder.finish()
    }
}

pub fn build_primitive_ca_with_opt<T>(s: &[Option<T::Native>], name: &str) -> ChunkedArray<T>
where
    T: PolarsPrimitiveType,
    T::Native: Copy,
{
    let mut builder = PrimitiveChunkedBuilder::new(name, s.len());
    for opt in s {
        builder.append_option(*opt);
    }
    builder.finish()
}

pub(crate) fn set_null_bits(
    mut builder: ArrayDataBuilder,
    null_bit_buffer: Option<Buffer>,
    null_count: Option<usize>,
    len: usize,
) -> ArrayDataBuilder {
    match null_count {
        Some(null_count) => {
            if null_count > 0 {
                let null_bit_buffer = null_bit_buffer
                    .expect("implementation error. Should not be None if null_count > 0");
                debug_assert!(null_count == len - bit_util::count_set_bits(null_bit_buffer.data()));
                builder = builder
                    .null_count(null_count)
                    .null_bit_buffer(null_bit_buffer);
            }
            builder
        }
        None => match null_bit_buffer {
            None => builder.null_count(0),
            Some(_) => {
                // this should take account into offset and length
                unimplemented!()
            }
        },
    }
}

/// Take an existing slice and a null bitmap and construct an arrow array.
pub fn build_with_existing_null_bitmap_and_slice<T>(
    null_bit_buffer: Option<Buffer>,
    null_count: usize,
    values: &[T::Native],
) -> PrimitiveArray<T>
where
    T: PolarsPrimitiveType,
{
    let len = values.len();
    // See:
    // https://docs.rs/arrow/0.16.0/src/arrow/array/builder.rs.html#314
    let builder = ArrayData::builder(T::get_data_type())
        .len(len)
        .add_buffer(Buffer::from(values.to_byte_slice()));

    let builder = set_null_bits(builder, null_bit_buffer, Some(null_count), len);
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
            let arr = aligned_vec_to_primitive_array::<T>(values, opt_buffer, Some(null_count));
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
pub fn aligned_vec_to_primitive_array<T: PolarsPrimitiveType>(
    values: AlignedVec<T::Native>,
    null_bit_buffer: Option<Buffer>,
    null_count: Option<usize>,
) -> PrimitiveArray<T> {
    let vec_len = values.len();
    let buffer = values.into_arrow_buffer();

    let builder = ArrayData::builder(T::get_data_type())
        .len(vec_len)
        .add_buffer(buffer);

    let builder = set_null_bits(builder, null_bit_buffer, null_count, vec_len);
    let data = builder.build();

    PrimitiveArray::<T>::from(data)
}

/// A `Vec` wrapper with a memory alignment equal to Arrow's primitive arrays.
/// Can be useful in creating a new ChunkedArray or Arrow Primitive array without copying.
#[derive(Debug)]
pub struct AlignedVec<T> {
    pub(crate) inner: Vec<T>,
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
            unsafe { memory::free_aligned(ptr, self.capacity()) }
        }
    }
}

impl<T> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let sh = iter.size_hint();
        let size = sh.1.unwrap_or(sh.0);

        let mut av = Self::with_capacity_aligned(size);

        for v in iter {
            av.push(v)
        }

        // Iterator size hint wasn't correct and reallocation has occurred
        assert!(av.len() <= size);
        av
    }
}

impl<T: Clone> AlignedVec<T> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.inner.resize(new_len, value)
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        let remaining_cap = self.capacity() - self.len();
        let needed_cap = other.len();
        if needed_cap > remaining_cap {
            self.reserve(needed_cap - remaining_cap);
        }
        self.inner.extend_from_slice(other)
    }
}

impl<T> AlignedVec<T> {
    /// Create a new Vec where first bytes memory address has an alignment of 64 bytes, as described
    /// by arrow spec.
    /// Read more:
    /// <https://github.com/rust-ndarray/ndarray/issues/771>
    pub fn with_capacity_aligned(size: usize) -> Self {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let t_size = std::mem::size_of::<T>();
        let capacity = size * t_size;
        let ptr = memory::allocate_aligned(capacity) as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, 0, size) };
        AlignedVec {
            inner: v,
            taken: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn reserve(&mut self, additional: usize) {
        let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
        let ptr = me.as_mut_ptr() as *mut u8;
        let t_size = mem::size_of::<T>();
        let cap = me.capacity();
        let old_capacity = t_size * cap;
        let new_capacity = old_capacity + t_size * additional;
        let ptr = unsafe { memory::reallocate(ptr, old_capacity, new_capacity) as *mut T };
        let v = unsafe { Vec::from_raw_parts(ptr, me.len(), cap + additional) };
        self.inner = v;
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Create a new aligned vec from a ptr.
    ///
    /// # Safety
    /// The ptr should be 64 byte aligned and `len` and `capacity` should be correct otherwise it is UB.
    pub unsafe fn from_ptr(ptr: usize, len: usize, capacity: usize) -> Self {
        assert_eq!((ptr as usize) % memory::ALIGNMENT, 0);
        let ptr = ptr as *mut T;
        let v = Vec::from_raw_parts(ptr, len, capacity);
        Self {
            inner: v,
            taken: false,
        }
    }

    /// Take ownership of the Vec. This is UB because the destructor of Vec<T> probably has a different
    /// alignment than what we allocated.
    unsafe fn into_inner(mut self) -> Vec<T> {
        self.shrink_to_fit();
        self.taken = true;
        mem::take(&mut self.inner)
    }

    /// Push at the end of the Vec. This is unsafe because a push when the capacity of the
    /// inner Vec is reached will reallocate the Vec without the alignment, leaving this destructor's
    /// alignment incorrect
    pub fn push(&mut self, value: T) {
        if self.inner.len() == self.capacity() {
            self.reserve(1);
        }
        self.inner.push(value)
    }

    /// Set the length of the underlying `Vec`.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity`.
    /// - The elements at `old_len..new_len` must be initialized.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.inner.set_len(new_len);
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let mut me = ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len(), me.capacity())
    }

    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.len() {
            let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
            let ptr = me.as_mut_ptr() as *mut u8;

            let t_size = mem::size_of::<T>();
            let new_size = t_size * me.len();
            let old_size = t_size * me.capacity();
            let v = unsafe {
                let ptr = memory::reallocate(ptr, old_size, new_size) as *mut T;
                Vec::from_raw_parts(ptr, me.len(), me.len())
            };
            self.inner = v;
        }
    }

    /// Transform this array to an Arrow Buffer.
    pub fn into_arrow_buffer(self) -> Buffer {
        let values = unsafe { self.into_inner() };

        let me = mem::ManuallyDrop::new(values);
        let ptr = me.as_ptr() as *const u8;
        let len = me.len() * std::mem::size_of::<T>();
        let capacity = me.capacity() * std::mem::size_of::<T>();
        debug_assert_eq!((ptr as usize) % 64, 0);

        unsafe { Buffer::from_raw_parts(ptr, len, capacity) }
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
    T: PolarsPrimitiveType,
{
    fn new_from_slice(name: &str, v: &[T::Native]) -> Self {
        Self::new_from_iter(name, v.iter().copied())
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
        let ca: Xob<ChunkedArray<_>> = it.collect();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }
}

impl<S> NewChunkedArray<Utf8Type, S> for Utf8Chunked
where
    S: AsRef<str>,
{
    fn new_from_slice(name: &str, v: &[S]) -> Self {
        let mut builder = StringBuilder::new(v.len());
        v.iter().for_each(|val| {
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

pub trait ListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>);
    fn append_series(&mut self, s: &Series);
    fn finish(&mut self) -> ListChunked;
}

pub struct ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub builder: ListBuilder<PrimitiveBuilder<T>>,
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

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = Arc::new($self.builder.finish());
        let len = arr.len();
        ListChunked {
            field: Arc::new($self.field.clone()),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
        }
    }};
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub fn new(name: &str, values_builder: PrimitiveBuilder<T>, capacity: usize) -> Self {
        let builder = ListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(
            name,
            ArrowDataType::List(Box::new(T::get_data_type())),
            true,
        );

        ListPrimitiveChunkedBuilder { builder, field }
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

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        append_opt_series!(self, opt_s)
    }

    fn append_series(&mut self, s: &Series) {
        append_series!(self, s);
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListUtf8ChunkedBuilder {
    builder: ListBuilder<StringBuilder>,
    field: Field,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, values_builder: StringBuilder, capacity: usize) -> Self {
        let builder = ListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(
            name,
            ArrowDataType::List(Box::new(ArrowDataType::Utf8)),
            true,
        );

        ListUtf8ChunkedBuilder { builder, field }
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        append_opt_series!(self, opt_s)
    }

    fn append_series(&mut self, s: &Series) {
        append_series!(self, s);
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub fn get_list_builder(
    dt: &ArrowDataType,
    capacity: usize,
    name: &str,
) -> Box<dyn ListBuilderTrait> {
    macro_rules! get_primitive_builder {
        ($type:ty) => {{
            let values_builder = PrimitiveBuilder::<$type>::new(capacity);
            let builder = ListPrimitiveChunkedBuilder::new(&name, values_builder, capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_utf8_builder {
        () => {{
            let values_builder = StringBuilder::new(capacity);
            let builder = ListUtf8ChunkedBuilder::new(&name, values_builder, capacity);
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
    fn test_primitive_builder() {
        let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new("foo", 6);
        let values = &[Some(1), None, Some(2), Some(3), None, Some(4)];
        for val in values {
            builder.append_option(*val);
        }
        let ca = builder.finish();
        assert_eq!(Vec::from(&ca), values);
    }

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
    fn test_aligned_vec_allocations() {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        // check if we can increase above initial capacity and keep the Arrow alignment
        let mut v = AlignedVec::with_capacity_aligned(2);
        v.push(1);
        v.push(2);
        v.push(3);
        v.push(4);

        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % memory::ALIGNMENT, 0);

        // check if we can shrink to fit
        let mut v = AlignedVec::with_capacity_aligned(10);
        v.push(1);
        v.push(2);
        v.shrink_to_fit();
        assert_eq!(v.len(), 2);
        assert_eq!(v.capacity(), 2);
        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % memory::ALIGNMENT, 0);

        let a = aligned_vec_to_primitive_array::<Int32Type>(v, None, Some(0));
        assert_eq!(a.value_slice(0, 2), &[1, 2])
    }

    #[test]
    fn test_list_builder() {
        let values_builder = Int32Array::builder(10);
        let mut builder = ListPrimitiveChunkedBuilder::new("a", values_builder, 10);

        // create a series containing two chunks
        let mut s1 = Int32Chunked::new_from_slice("a", &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::new_from_slice("b", &[4, 5, 6]).into_series();
        s1.append(&s2).unwrap();

        builder.append_series(&s1);
        builder.append_series(&s2);
        let ls = builder.finish();
        if let AnyType::List(s) = ls.get_any(0) {
            // many chunks are aggregated to one in the ListArray
            assert_eq!(s.len(), 6)
        } else {
            assert!(false)
        }
        if let AnyType::List(s) = ls.get_any(1) {
            assert_eq!(s.len(), 3)
        } else {
            assert!(false)
        }
    }
}

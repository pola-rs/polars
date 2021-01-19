use crate::{
    prelude::*,
    use_string_cache,
    utils::{get_iter_capacity, NoNull},
};
use ahash::AHashMap;
use arrow::array::{ArrayDataBuilder, ArrayRef, LargeListBuilder};
use arrow::datatypes::ToByteSlice;
pub use arrow::memory;
use arrow::{
    array::{Array, ArrayData, PrimitiveArray},
    buffer::Buffer,
};
use num::Num;
use polars_arrow::prelude::*;
use std::borrow::Cow;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

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

pub struct BooleanChunkedBuilder {
    array_builder: BooleanArrayBuilder,
    field: Field,
}

impl ChunkedBuilder<bool, BooleanType> for BooleanChunkedBuilder {
    /// Appends a value of type `T` into the builder
    fn append_value(&mut self, v: bool) {
        self.array_builder.append_value(v);
    }

    /// Appends a null slot into the builder
    fn append_null(&mut self) {
        self.array_builder.append_null();
    }

    fn finish(mut self) -> BooleanChunked {
        let arr = Arc::new(self.array_builder.finish());

        let len = arr.len();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
            categorical_map: None,
        }
    }
}

impl BooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        BooleanChunkedBuilder {
            array_builder: BooleanArrayBuilder::new(capacity),
            field: Field::new(name, DataType::Boolean),
        }
    }
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

    fn finish(mut self) -> ChunkedArray<T> {
        let arr = Arc::new(self.array_builder.finish());

        let len = arr.len();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
            categorical_map: None,
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
            field: Field::new(name, T::get_dtype()),
        }
    }
}

pub struct CategoricalChunkedBuilder {
    array_builder: PrimitiveArrayBuilder<UInt32Type>,
    field: Field,
    mapping: AHashMap<String, u32>,
    reverse_mapping: AHashMap<u32, String>,
}

impl CategoricalChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        let mapping = AHashMap::with_capacity(128);
        let reverse_mapping = AHashMap::with_capacity(128);

        CategoricalChunkedBuilder {
            array_builder: PrimitiveArrayBuilder::<UInt32Type>::new(capacity),
            field: Field::new(name, DataType::Categorical),
            mapping,
            reverse_mapping,
        }
    }
}
impl CategoricalChunkedBuilder {
    /// Appends all the values in a single lock of the global string cache.
    pub fn append_values<'a, I>(&mut self, i: I)
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        if use_string_cache() {
            let mut mapping = crate::STRING_CACHE.lock_map();

            for opt_s in i {
                match opt_s {
                    Some(s) => {
                        let idx = match mapping.get(s) {
                            Some(idx) => *idx,
                            None => {
                                let idx = mapping.len() as u32;
                                mapping.insert(s.to_string(), idx);
                                idx
                            }
                        };
                        self.reverse_mapping.insert(idx, s.to_string());
                        self.array_builder.append_value(idx);
                    }
                    None => {
                        self.array_builder.append_null();
                    }
                }
            }
        } else {
            for opt_s in i {
                match opt_s {
                    Some(s) => {
                        let idx = match self.mapping.get(s) {
                            Some(idx) => *idx,
                            None => {
                                let idx = self.mapping.len() as u32;
                                self.mapping.insert(s.to_string(), idx);
                                idx
                            }
                        };
                        self.reverse_mapping.insert(idx, s.to_string());
                        self.array_builder.append_value(idx);
                    }
                    None => {
                        self.array_builder.append_null();
                    }
                }
            }
        }
    }
}

impl ChunkedBuilder<&str, CategoricalType> for CategoricalChunkedBuilder {
    fn append_value(&mut self, val: &str) {
        let idx = if use_string_cache() {
            let mut mapping = crate::STRING_CACHE.lock_map();
            match mapping.get(val) {
                Some(idx) => *idx,
                None => {
                    let idx = mapping.len() as u32;
                    mapping.insert(val.to_string(), idx);
                    idx
                }
            }
        } else {
            match self.mapping.get(val) {
                Some(idx) => *idx,
                None => {
                    let idx = self.mapping.len() as u32;
                    self.mapping.insert(val.to_string(), idx);
                    idx
                }
            }
        };
        self.reverse_mapping.insert(idx, val.to_string());
        self.array_builder.append_value(idx);
    }

    fn append_null(&mut self) {
        self.array_builder.append_null()
    }

    fn finish(mut self) -> ChunkedArray<CategoricalType> {
        if self.mapping.len() > u32::MAX as usize {
            panic!(format!("not more than {} categories supported", u32::MAX))
        };
        let arr = Arc::new(self.array_builder.finish());
        let len = arr.len();
        self.reverse_mapping.shrink_to_fit();
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            chunk_id: vec![len],
            phantom: PhantomData,
            categorical_map: Some(Arc::new(self.reverse_mapping)),
        }
    }
}

pub struct Utf8ChunkedBuilder {
    pub builder: LargeStringBuilder,
    pub capacity: usize,
    field: Field,
}

impl Utf8ChunkedBuilder {
    /// Create a new UtfChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize, bytes_capacity: usize) -> Self {
        Utf8ChunkedBuilder {
            builder: LargeStringBuilder::with_capacity(bytes_capacity, capacity),
            capacity,
            field: Field::new(name, DataType::Utf8),
        }
    }

    /// Appends a value of type `T` into the builder
    pub fn append_value<S: AsRef<str>>(&mut self, v: S) {
        self.builder.append_value(v.as_ref());
    }

    /// Appends a null slot into the builder
    pub fn append_null(&mut self) {
        self.builder.append_null();
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
            categorical_map: None,
        }
    }
}

pub struct Utf8ChunkedBuilderCow {
    builder: Utf8ChunkedBuilder,
}

impl Utf8ChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilderCow {
            builder: Utf8ChunkedBuilder::new(name, capacity, capacity),
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
) -> ArrayDataBuilder {
    match null_count {
        Some(null_count) => {
            if null_count > 0 {
                let null_bit_buffer = null_bit_buffer
                    .expect("implementation error. Should not be None if null_count > 0");

                builder = builder.null_bit_buffer(null_bit_buffer);
            }
            builder
        }
        None => match null_bit_buffer {
            None => builder,
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
    let builder = ArrayData::builder(T::DATA_TYPE)
        .len(len)
        .add_buffer(Buffer::from(values.to_byte_slice()));

    let builder = set_null_bits(builder, null_bit_buffer, Some(null_count));
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
impl<T> FromIterator<(AlignedVec<T::Native>, Option<Buffer>)> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = (AlignedVec<T::Native>, Option<Buffer>)>>(iter: I) -> Self {
        let mut chunks = vec![];

        for (values, opt_buffer) in iter {
            let arr = values.into_primitive_array::<T>(opt_buffer);
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

    let builder = ArrayData::builder(T::DATA_TYPE)
        .len(vec_len)
        .add_buffer(buffer);

    let builder = set_null_bits(builder, null_bit_buffer, null_count);
    let data = builder.build();

    PrimitiveArray::<T>::from(data)
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
        Self::new_from_opt_iter(name, opt_v.iter().copied())
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
        let ca: NoNull<ChunkedArray<_>> = it.collect();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }
}

impl NewChunkedArray<BooleanType, bool> for BooleanChunked {
    fn new_from_slice(name: &str, v: &[bool]) -> Self {
        Self::new_from_iter(name, v.iter().copied())
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<bool>]) -> Self {
        Self::new_from_opt_iter(name, opt_v.iter().copied())
    }

    fn new_from_opt_iter(
        name: &str,
        it: impl Iterator<Item = Option<bool>>,
    ) -> ChunkedArray<BooleanType> {
        let mut builder = BooleanChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = bool>) -> ChunkedArray<BooleanType> {
        let mut ca: ChunkedArray<_> = it.collect();
        ca.rename(name);
        ca
    }
}

impl<S> NewChunkedArray<Utf8Type, S> for Utf8Chunked
where
    S: AsRef<str>,
{
    fn new_from_slice(name: &str, v: &[S]) -> Self {
        let values_size = v.iter().fold(0, |acc, s| acc + s.as_ref().len());

        let mut builder = LargeStringBuilder::with_capacity(values_size, v.len());
        v.iter().for_each(|val| {
            builder.append_value(val.as_ref());
        });

        let field = Arc::new(Field::new(name, DataType::Utf8));

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            chunk_id: vec![v.len()],
            phantom: PhantomData,
            categorical_map: None,
        }
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<S>]) -> Self {
        let values_size = opt_v.iter().fold(0, |acc, s| match s {
            Some(s) => acc + s.as_ref().len(),
            None => acc,
        });
        let mut builder = Utf8ChunkedBuilder::new(name, values_size, opt_v.len());

        opt_v.iter().for_each(|opt| match opt {
            Some(v) => builder.append_value(v.as_ref()),
            None => builder.append_null(),
        });
        builder.finish()
    }

    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<S>>) -> Self {
        let cap = get_iter_capacity(&it);
        let mut builder = Utf8ChunkedBuilder::new(name, cap, cap * 5);
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = S>) -> Self {
        let cap = get_iter_capacity(&it);
        let mut builder = Utf8ChunkedBuilder::new(name, cap, cap * 5);
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
    pub builder: LargeListBuilder<PrimitiveArrayBuilder<T>>,
    field: Field,
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
            categorical_map: None,
        }
    }};
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub fn new(name: &str, values_builder: PrimitiveArrayBuilder<T>, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(T::get_dtype().to_arrow()));

        ListPrimitiveChunkedBuilder { builder, field }
    }

    pub fn append_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(v) => {
                self.builder.values().append_slice(v);
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
    T::Native: Num,
{
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    fn append_series(&mut self, s: &Series) {
        let builder = self.builder.values();
        let arrays = s.chunks();
        for a in arrays {
            let values = a.get_values::<T>();
            if a.null_count() == 0 {
                builder.append_slice(values);
            } else {
                values.iter().enumerate().for_each(|(idx, v)| {
                    if a.is_valid(idx) {
                        builder.append_value(*v);
                    } else {
                        builder.append_null();
                    }
                });
            }
        }
        self.builder.append(true).unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListUtf8ChunkedBuilder {
    builder: LargeListBuilder<LargeStringBuilder>,
    field: Field,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, values_builder: LargeStringBuilder, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::LargeUtf8));

        ListUtf8ChunkedBuilder { builder, field }
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    fn append_series(&mut self, s: &Series) {
        let ca = s.utf8().unwrap();
        let value_builder = self.builder.values();
        for s in ca {
            match s {
                Some(s) => value_builder.append_value(s),
                None => value_builder.append_null(),
            };
        }
        self.builder.append(true).unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBuilder<BooleanArrayBuilder>,
    field: Field,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: &str, values_builder: BooleanArrayBuilder, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::Boolean));

        Self { builder, field }
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    fn append_series(&mut self, s: &Series) {
        let ca = s.bool().unwrap();
        let value_builder = self.builder.values();
        for s in ca {
            match s {
                Some(s) => value_builder.append_value(s),
                None => value_builder.append_null(),
            };
        }
        self.builder.append(true).unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub fn get_list_builder(
    dt: &DataType,
    value_capacity: usize,
    list_capacity: usize,
    name: &str,
) -> Box<dyn ListBuilderTrait> {
    macro_rules! get_primitive_builder {
        ($type:ty) => {{
            let values_builder = PrimitiveArrayBuilder::<$type>::new(value_capacity);
            let builder = ListPrimitiveChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_bool_builder {
        () => {{
            let values_builder = BooleanArrayBuilder::new(value_capacity);
            let builder = ListBooleanChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_utf8_builder {
        () => {{
            let values_builder =
                LargeStringBuilder::with_capacity(value_capacity * 5, value_capacity);
            let builder = ListUtf8ChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    match_arrow_data_type_apply_macro!(
        dt,
        get_primitive_builder,
        get_utf8_builder,
        get_bool_builder
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::PrimitiveBuilder;

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
        assert_eq!(&a.values()[..2], &[1, 2])
    }

    #[test]
    fn test_list_builder() {
        let values_builder = PrimitiveArrayBuilder::<Int32Type>::new(10);
        let mut builder = ListPrimitiveChunkedBuilder::new("a", values_builder, 10);

        // create a series containing two chunks
        let mut s1 = Int32Chunked::new_from_slice("a", &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::new_from_slice("b", &[4, 5, 6]).into_series();
        s1.append(&s2).unwrap();

        builder.append_series(&s1);
        builder.append_series(&s2);
        let ls = builder.finish();
        if let AnyValue::List(s) = ls.get_any_value(0) {
            // many chunks are aggregated to one in the ListArray
            assert_eq!(s.len(), 6)
        } else {
            assert!(false)
        }
        if let AnyValue::List(s) = ls.get_any_value(1) {
            assert_eq!(s.len(), 3)
        } else {
            assert!(false)
        }
        // test list collect
        let out = [&s1, &s2]
            .iter()
            .map(|s| s.clone())
            .collect::<ListChunked>();
        assert_eq!(out.get(0).unwrap().len(), 6);
        assert_eq!(out.get(1).unwrap().len(), 3);
    }

    #[test]
    fn test_categorical_builder() {
        let mut builder = CategoricalChunkedBuilder::new("foo", 10);

        builder.append_value("hello");
        builder.append_null();
        builder.append_value("world");

        let ca = builder.finish();
        let v = AnyValue::Utf8("hello");
        assert_eq!(ca.get_any_value(0), v);
        let v = AnyValue::Null;
        assert_eq!(ca.get_any_value(1), v);
    }
}
